"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

N = 18

# EVOLVE-BLOCK-START
def construct():
    # Safety margin: Evaluator kills at 90s, task requires <60s.
    # We set deadline to 50s to ensure we return well in time.
    deadline = time.time() + 50.0

    # Generate all 2^18 boolean inputs efficiently
    # Using periodic tiling for speed
    inputs = np.zeros((1 << N, N), dtype=np.uint8)
    for i in range(N):
        inputs[:, i] = (np.arange(1 << N, dtype=np.uint32) >> i) & 1

    # Pre-allocate a buffer for state to avoid repeated allocation in verify
    state_buffer = np.empty_like(inputs)
    
    # Pre-allocate temporary buffers for min/max to avoid repeated allocation
    tmp_i = np.empty(1 << N, dtype=np.uint8)
    tmp_j = np.empty(1 << N, dtype=np.uint8)

    def verify(net):
        # Copy inputs to buffer
        np.copyto(state_buffer, inputs)
        
        # Apply network
        for i, j in net:
            # min to i, max to j
            # Using bitwise operations for uint8 0/1
            # mi = si & sj, ma = si | sj
            si = state_buffer[:, i]
            sj = state_buffer[:, j]
            
            # We need to update state_buffer in place.
            # Since we read si and sj before writing, we can compute new values.
            # However, numpy views are tricky if we overwrite one and then use it.
            # Let's use temporary arrays to be safe, or rely on the fact that 
            # we update both i and j simultaneously.
            # Actually, si and sj are views. If we modify state_buffer[:, i], 
            # si changes. So we must compute both new values first.
            
            # mi = si & sj
            # ma = si | sj
            # We can do this without explicit temp arrays if we are careful, 
            # but numpy might evaluate RHS first? No, assignment is immediate.
            # state_buffer[:, i] = si & sj  -> modifies col i
            # state_buffer[:, j] = si | sj  -> uses original si? 
            # In Python, RHS is evaluated before assignment. 
            # So si & sj is computed using current values.
            # Then assigned to col i.
            # Then si | sj is computed using current values (si is still the old view? 
            # No, si is a view to state_buffer[:, i]. If we modify state_buffer[:, i], 
            # si reflects the change? Yes, views reflect changes.
            # So we MUST use temporary storage or compute both before writing.
            
            # Safe way:
            # np.minimum(si, sj, out=tmp_i)
            # np.maximum(si, sj, out=tmp_j)
            # state_buffer[:, i] = tmp_i
            # state_buffer[:, j] = tmp_j
            
            # Bitwise is faster:
            # tmp_i[:] = si & sj
            # tmp_j[:] = si | sj
            # state_buffer[:, i] = tmp_i
            # state_buffer[:, j] = tmp_j
            
            # Let's try to avoid temp arrays if possible by copying first?
            # si_copy = si.copy()
            # sj_copy = sj.copy()
            # That's more memory allocation.
            
            # Let's stick to the temp arrays which are pre-allocated.
            np.bitwise_and(si, sj, out=tmp_i)
            np.bitwise_or(si, sj, out=tmp_j)
            state_buffer[:, i] = tmp_i
            state_buffer[:, j] = tmp_j
        
        # Check sortedness: state[:, k] <= state[:, k+1]
        # This is equivalent to no "10" pattern in the row.
        # We can check if there are any -1s in diff, or simply check the condition.
        # For uint8, subtraction wraps around? No, we can use logical comparison.
        # state[:, :-1] <= state[:, 1:]
        # A faster check: sum of each row must equal the number of 1s in the input?
        # No, that's not sufficient.
        # Sorted binary means all 0s then all 1s.
        # So for each row, once a 1 appears, all subsequent must be 1.
        # Equivalent: diff >= 0 everywhere.
        # Or: np.diff(state_buffer, axis=1) >= 0
        # Or: np.all(state_buffer[:, :-1] <= state_buffer[:, 1:])
        
        # A very fast check:
        # If sorted, the position of the first 1 determines the rest.
        # Actually, just checking diff is fine.
        # np.diff allocates a new array.
        # Let's try:
        # return np.all(state_buffer[:, :-1] <= state_buffer[:, 1:])
        
        # Alternative: 
        # If not sorted, there exists i < j such that state[i]=1 and state[j]=0.
        # This is hard to check quickly without scanning.
        # The diff method is O(N * 2^N) which is fast enough.
        
        return np.all(state_buffer[:, :-1] <= state_buffer[:, 1:])

    # Start with a known good network
    best_network = _odd_even_network()
    best_count = len(best_network)
    
    if not verify(best_network):
        return best_network

    current = list(best_network)
    current_count = len(current)

    # Search Loop
    # We use a greedy approach with random restarts/mutations if stuck.
    stuck_counter = 0
    last_improvement_time = time.time()
    
    while time.time() < deadline:
        # Try to remove a comparator
        removed = False
        # Try random removals first
        for _ in range(10):
            if current_count == 0:
                break
            idx = np.random.randint(current_count)
            candidate = current.copy()
            candidate.pop(idx)
            if verify(candidate):
                current = candidate
                current_count -= 1
                removed = True
                if current_count < best_count:
                    best_network = current
                    best_count = current_count
                    last_improvement_time = time.time()
                break
        
        if not removed:
            # Try systematic removal if network is small enough
            if current_count < 50:
                for idx in range(current_count):
                    candidate = current.copy()
                    candidate.pop(idx)
                    if verify(candidate):
                        current = candidate
                        current_count -= 1
                        removed = True
                        if current_count < best_count:
                            best_network = current
                            best_count = current_count
                            last_improvement_time = time.time()
                        break
            
            if not removed:
                # Try mutation: change a comparator to a random valid one
                # This might allow subsequent removals
                if current_count > 0:
                    idx = np.random.randint(current_count)
                    new_i = np.random.randint(N)
                    new_j = np.random.randint(N)
                    if new_i == new_j:
                        continue
                    if new_i > new_j:
                        new_i, new_j = new_j, new_i
                    candidate = current.copy()
                    candidate[idx] = (new_i, new_j)
                    if verify(candidate):
                        current = candidate
                        # No count change, but potential for future removals
                        stuck_counter += 1
                    else:
                        stuck_counter += 1
                else:
                    break
            
            # If stuck for a long time, reset to best known and try different strategy?
            # For simplicity, we just continue with random mutations.
            # To avoid infinite loop of useless mutations, we can occasionally
            # reset to a slightly perturbed version of the best network.
            if time.time() - last_improvement_time > 5.0:
                # Reset to best network and shuffle?
                # Or just continue. The random mutations are the exploration.
                pass

    return best_network

def _odd_even_network():
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

# EVOLVE-BLOCK-END
