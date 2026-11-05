"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 18

# EVOLVE-BLOCK-START
def construct():
    # Safety margin: Evaluator kills at 90s, task requires <60s.
    # We set deadline to 50s to ensure we return well in time.
    deadline = time.time() + 50.0
    
    # Generate all 2^18 boolean inputs efficiently
    # 262144 rows, 18 cols
    # Using bit manipulation for speed
    inputs = np.zeros((1 << N, N), dtype=np.uint8)
    for col in range(N):
        # For each column, the bit pattern is periodic with period 2^(col+1)
        # We can use a clever trick:
        # For column j, value is 1 when (i >> j) & 1 == 1
        # We can build this using broadcasting or repeated tiling
        period = 1 << (col + 1)
        half = 1 << col
        # Pattern: 0...0 1...1 repeated
        block = np.zeros(period, dtype=np.uint8)
        block[half:] = 1
        # Tile to fill 2^N rows
        repeats = (1 << N) // period
        inputs[:, col] = np.tile(block, repeats)
    
    # Pre-allocate a buffer for state to avoid repeated allocation in verify
    state_buffer = np.empty_like(inputs)
    
    def verify(net):
        # Copy inputs to buffer
        np.copyto(state_buffer, inputs)
        
        # Apply network
        for i, j in net:
            mi = state_buffer[:, i] & state_buffer[:, j]
            ma = state_buffer[:, i] | state_buffer[:, j]
            state_buffer[:, i] = mi
            state_buffer[:, j] = ma
        
        # Check sortedness: state[:, k] <= state[:, k+1]
        return np.all(state_buffer[:, :-1] <= state_buffer[:, 1:])
    
    def _odd_even_network():
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    # Start with a known good network
    best_network = _odd_even_network()
    best_count = len(best_network)
    
    if not verify(best_network):
        return best_network
    
    current = list(best_network)
    current_count = len(current)
    
    # Search Loop
    # We use a greedy approach with targeted mutations and removals.
    last_improvement_time = time.time()
    
    while time.time() < deadline:
        # Try to remove a comparator
        removed = False
        
        # 1. Try systematic removal first if network is small enough
        # This is more thorough than random when we have time
        if current_count < 60:
            for idx in range(current_count):
                candidate = current.copy()
                candidate.pop(idx)
                if verify(candidate):
                    current = candidate
                    current_count -= 1
                    removed = True
                    if current_count < best_count:
                        best_network = current.copy()
                        best_count = current_count
                        last_improvement_time = time.time()
                    break
        
        # 2. If not removed, try random removals
        if not removed:
            # Try random removals first
            for _ in range(20):
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
                        best_network = current.copy()
                        best_count = current_count
                        last_improvement_time = time.time()
                    break
        
        # 3. If still stuck, try targeted mutation
        # Replace a comparator with a nearby one (end-biased)
        if not removed:
            if current_count > 0:
                # Pick a random index to modify
                idx = np.random.randint(current_count)
                old_i, old_j = current[idx]
                
                # Generate a new comparator that is "near" the old one
                # Bias towards keeping one endpoint and changing the other slightly
                # or towards the ends of the wire range
                strategy = np.random.random()
                
                if strategy < 0.4:
                    # Keep i, change j
                    new_i = old_i
                    # j should be > i
                    # Bias towards nearby or towards end
                    if np.random.random() < 0.5:
                        # Nearby
                        delta = np.random.randint(-3, 4)
                        new_j = old_j + delta
                    else:
                        # Towards end
                        new_j = np.random.randint(old_i + 1, N)
                elif strategy < 0.8:
                    # Keep j, change i
                    new_j = old_j
                    # i should be < j
                    if np.random.random() < 0.5:
                        # Nearby
                        delta = np.random.randint(-3, 4)
                        new_i = old_i + delta
                    else:
                        # Towards start
                        new_i = np.random.randint(0, new_j)
                else:
                    # Random new comparator, biased towards ends
                    # Higher probability for comparators involving wire 0 or N-1
                    if np.random.random() < 0.3:
                        new_i = 0
                        new_j = np.random.randint(1, N)
                    elif np.random.random() < 0.3:
                        new_j = N - 1
                        new_i = np.random.randint(0, N - 1)
                    else:
                        new_i = np.random.randint(N)
                        new_j = np.random.randint(N)
                        if new_i == new_j:
                            new_j = (new_j + 1) % N
                        if new_i > new_j:
                            new_i, new_j = new_j, new_i
                
                # Ensure valid range
                new_i = max(0, min(N - 1, new_i))
                new_j = max(0, min(N - 1, new_j))
                
                if new_i >= new_j:
                    if new_j < N:
                        new_j = min(N - 1, new_j + 1)
                    else:
                        new_i = max(0, new_i - 1)
                
                if new_i >= new_j:
                    continue
                
                candidate = current.copy()
                candidate[idx] = (new_i, new_j)
                
                if verify(candidate):
                    current = candidate
                    # No count change, but potential for future removals
                    # Occasionally update best if we're in a different local minimum
                    if np.random.random() < 0.1:
                        # Check if this is a good base for further reduction
                        # Just keep it as current, don't update best unless we remove
                        pass
                else:
                    # Failed mutation, try a different one or reset
                    pass
            
            # If stuck for a long time, reset to best known
            if time.time() - last_improvement_time > 3.0:
                # Reset to best network
                current = best_network.copy()
                current_count = best_count
                # Small perturbation to escape local minimum
                if current_count > 0:
                    idx = np.random.randint(current_count)
                    old_i, old_j = current[idx]
                    # Try to replace with a nearby comparator
                    new_i = max(0, min(N - 1, old_i + np.random.randint(-2, 3)))
                    new_j = max(0, min(N - 1, old_j + np.random.randint(-2, 3)))
                    if new_i >= new_j:
                        if new_j < N:
                            new_j = min(N - 1, new_j + 1)
                        else:
                            new_i = max(0, new_i - 1)
                    if new_i < new_j:
                        candidate = current.copy()
                        candidate[idx] = (new_i, new_j)
                        if verify(candidate):
                            current = candidate
    
    return best_network

# EVOLVE-BLOCK-END
