"""Size-efficient sorting networks on 17 wires via NumPy vectorized search."""
import time
import random
import numpy as np

N = 17
NUM_INPUTS = 1 << N
DEADLINE = 55.0  # Safety margin before 60s limit
START_TIME = time.time()

# Precompute initial states for all 2^N inputs using NumPy.
# Shape: (NUM_INPUTS, N), dtype: uint8
# input_states[input_idx, wire_idx] = 1 if wire has 1, else 0
def _init_states():
    # Generate all input vectors
    # We can use np.arange and bit operations, or just iterate.
    # np.arange is fast.
    inputs = np.arange(NUM_INPUTS, dtype=np.uint32)
    # Expand to 2D: shape (NUM_INPUTS, N)
    # We want bit k of inputs to be in column k.
    # (inputs[:, None] >> np.arange(N)) & 1
    states = (inputs[:, None] >> np.arange(N)) & 1
    return states.astype(np.uint8)

INITIAL_STATES = _init_states()

def _time_left():
    return DEADLINE - (time.time() - START_TIME)

def verify_network(comps):
    """
    Verify network using NumPy vectorized simulation.
    comps: list of (i, j) tuples.
    Returns True if valid.
    """
    if not comps:
        # Check if initial states are sorted (only true for N=1 or specific cases, usually false for N>1)
        # For N=17, initial state is not sorted.
        return False
        
    # Copy initial states
    # We need a mutable copy. .copy() is fast enough.
    state = INITIAL_STATES.copy()
    
    for i, j in comps:
        wi = state[:, i]
        wj = state[:, j]
        
        # Comparator (i, j): min to i, max to j.
        # Vectorized min/max
        new_wi = np.minimum(wi, wj)
        new_wj = np.maximum(wi, wj)
        
        state[:, i] = new_wi
        state[:, j] = new_wj
        
    # Check if sorted: for all rows, state[:, k] <= state[:, k+1]
    # This is equivalent to checking if any state[:, k] > state[:, k+1]
    # diff = state[:, 1:] - state[:, :-1]
    # if np.any(diff < 0): return False
    # Faster: check if state[:, k] & ~state[:, k+1] has any 1s?
    # Actually, since values are 0 or 1:
    # sorted if state[:, k] <= state[:, k+1] for all k.
    # This is true if there is no index where state[:, k] == 1 and state[:, k+1] == 0.
    # i.e., ~np.any(state[:, :-1] & (1 - state[:, 1:]))
    
    # Cast to signed integer to avoid wrap-around in subtraction
    # uint8 subtraction wraps around (0-1=255), which breaks the < 0 check
    state_signed = state.astype(np.int8)
    diffs = state_signed[:, 1:] - state_signed[:, :-1]
    if np.any(diffs < 0):
        return False
        
    return True

def make_seed():
    """Odd-even transposition network."""
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

def greedy_remove(comps):
    """
    Try to remove each comparator if the network remains valid.
    """
    n = len(comps)
    improved = True
    while improved:
        improved = False
        # Iterate backwards to handle list modifications safely if we were modifying in place, 
        # but we are creating new lists.
        for idx in range(n - 1, -1, -1):
            if _time_left() < 2.0:
                return comps
            candidate = comps[:idx] + comps[idx+1:]
            if verify_network(candidate):
                comps = candidate
                n = len(comps)
                improved = True
                break # Restart scan
    return comps

def simulated_annealing(start_comps, rng, time_limit=30.0):
    """
    Simulated Annealing search for sorting networks.
    """
    current = list(start_comps)
    current_len = len(current)
    best = list(current)
    best_len = current_len
    
    T = 1.0
    cooling = 0.999
    
    # We use a simple energy function: number of comparators.
    # But we need to handle validity.
    # If invalid, penalty is high.
    # Let's stick to the "valid or die" approach for the objective,
    # but accept invalid moves if they are "close" in structure?
    # No, standard SA for sorting networks usually operates on valid networks only
    # or uses a penalty.
    # Given the cost of verification, we should be careful.
    
    # Strategy:
    # 1. Start with a valid network.
    # 2. Propose a move (remove, swap, move, add).
    # 3. If move results in valid network:
    #    - If smaller, accept.
    #    - If same size, accept with probability exp(-delta/T) where delta=0? 
    #      Usually for size minimization, we want to explore same-size valid networks
    #      to find a "hole" for removal.
    #    - If larger, accept with probability exp(-delta/T).
    # 4. If move results in invalid network:
    #    - Reject? Or accept with very low probability?
    #    - If we accept invalid, we can't easily measure "distance" to valid.
    #    - Let's only accept valid networks in the state space, but allow
    #      "sideways" moves (same size valid) to escape local minima.
    
    start_time = time.time()
    
    while _time_left() > time_limit + 2.0 and (time.time() - start_time) < time_limit:
        # Generate mutation
        candidate = list(current)
        n = len(candidate)
        
        op = rng.choice(['remove', 'swap', 'move', 'add', 'remove', 'swap', 'move'])
        
        if op == 'remove':
            if n == 0:
                continue
            idx = rng.randrange(n)
            candidate.pop(idx)
        elif op == 'add':
            # Add a random comparator
            i = rng.randrange(N)
            j = rng.randrange(N)
            if i != j:
                if i > j: i, j = j, i
                candidate.append((i, j))
        elif op == 'swap' and n >= 2:
            i1 = rng.randrange(n)
            i2 = rng.randrange(n)
            if i1 != i2:
                candidate[i1], candidate[i2] = candidate[i2], candidate[i1]
        elif op == 'move' and n >= 1:
            i1 = rng.randrange(n)
            i2 = rng.randrange(n)
            if i1 != i2:
                elem = candidate.pop(i1)
                candidate.insert(i2, elem)
        
        # Evaluate
        is_valid = verify_network(candidate)
        
        if is_valid:
            new_len = len(candidate)
            delta = new_len - current_len
            
            if delta < 0:
                # Improvement
                current = candidate
                current_len = new_len
                if new_len < best_len:
                    best = list(current)
                    best_len = new_len
                    # Greedy remove after improvement
                    current = greedy_remove(current)
                    current_len = len(current)
                    if current_len < best_len:
                        best = list(current)
                        best_len = current_len
            else:
                # Same or worse
                if delta == 0:
                    # Accept sideways moves to explore
                    # Simple acceptance: always accept? Or probabilistic?
                    # Let's accept to explore.
                    current = candidate
                else:
                    # Worse size, accept with probability
                    p = np.exp(-delta / T)
                    if rng.random() < p:
                        current = candidate
        else:
            # Invalid
            # Reject. We stay in current state.
            pass
            
        T *= cooling
        
    return best

def construct():
    """Construct a sorting network for 17 wires, minimizing comparator count."""
    global START_TIME
    START_TIME = time.time()
    
    rng = random.Random(42)
    
    # Start with odd-even transposition
    comps = make_seed()
    
    # First pass: aggressive greedy removal
    comps = greedy_remove(comps)
    
    best = list(comps)
    best_len = len(best)
    
    # Simulated Annealing Phase
    # Use a significant chunk of time for SA
    # Let's give SA 30 seconds, leaving 20s for final polish and safety
    sa_time = 30.0
    if _time_left() > sa_time + 5.0:
        best = simulated_annealing(best, rng, time_limit=sa_time)
        best_len = len(best)
    
    # Final Greedy Polish
    # Use remaining time for another round of greedy removal
    # and perhaps a quick local search
    if _time_left() > 5.0:
        best = greedy_remove(best)
        
        # Quick local search / hill climbing with restarts?
        # Let's just do a few random restarts from best if time permits
        # Actually, just one more pass of greedy is safe.
        # Or maybe a limited SA with very low temp?
        
        # Let's do a short SA run with lower initial temp to polish
        if _time_left() > 10.0:
            # Create a copy to avoid modifying best if it fails
            # SA returns best found, so it's safe
            # But we need to ensure it doesn't timeout badly
            # The SA function checks _time_left()
            
            # We'll run a short SA with T=0.1 to be conservative
            # But the function uses fixed T=1.0.
            # Let's just rely on the main SA and greedy.
            
            # Let's add a simple "shake and greedy" step
            # Take best, apply a few random valid swaps/moves, then greedy remove
            # This can escape local minima where greedy_remove fails.
            
            current = list(best)
            for _ in range(10): # 10 attempts
                if _time_left() < 5.0:
                    break
                # Apply a random valid perturbation
                # We need a valid perturbation.
                # Let's pick a random swap or move and check if valid.
                # If valid, use it as base for greedy.
                
                candidate = list(current)
                n = len(candidate)
                if n < 2:
                    break
                
                op = rng.choice(['swap', 'move'])
                if op == 'swap':
                    i1 = rng.randrange(n)
                    i2 = rng.randrange(n)
                    if i1 != i2:
                        candidate[i1], candidate[i2] = candidate[i2], candidate[i1]
                elif op == 'move':
                    i1 = rng.randrange(n)
                    i2 = rng.randrange(n)
                    if i1 != i2:
                        elem = candidate.pop(i1)
                        candidate.insert(i2, elem)
                
                if verify_network(candidate):
                    # Valid perturbation found
                    # Try to shrink it
                    shrunken = greedy_remove(candidate)
                    if len(shrunken) < best_len:
                        best = shrunken
                        best_len = len(best)
                        current = list(best)
    
    return best
