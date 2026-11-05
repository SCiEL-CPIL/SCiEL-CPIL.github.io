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
    # 262144 rows, 18 cols
    # Using np.frombuffer or similar might be faster, but product is okay for one-time cost.
    # To be faster, we can use:
    # inputs = np.array([(i >> j) & 1 for i in range(1 << N) for j in range(N)], dtype=np.uint8).reshape(-1, N)
    # But list comprehension is slow.
    # Let's stick to product but ensure it's fast enough.
    inputs = np.array(list(product([0, 1], repeat=N)), dtype=np.uint8)
    
    # Pre-allocate a buffer for state to avoid repeated allocation in verify
    state_buffer = np.empty_like(inputs)
    
    def verify(net):
        # Copy inputs to buffer
        # np.copyto is faster than .copy() as it doesn't allocate
        np.copyto(state_buffer, inputs)
        
        # Apply network
        for i, j in net:
            mi = state_buffer[:, i] & state_buffer[:, j]
            ma = state_buffer[:, i] | state_buffer[:, j]
            state_buffer[:, i] = mi
            state_buffer[:, j] = ma
        
        # Check sortedness: state[:, k] <= state[:, k+1]
        # This is equivalent to no "10" pattern in the row.
        # We can check if there are any -1s in diff, or simply check the condition.
        # For uint8, subtraction wraps around? No, we can use logical comparison.
        # state[:, :-1] <= state[:, 1:]
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
