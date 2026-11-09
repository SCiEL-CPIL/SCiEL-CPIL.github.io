import numpy as np
import time
import random
import math

N = 18

def _is_sorted_network(comps, N):
    """
    Check if the network sorts all 0-1 inputs using NumPy.
    """
    # Generate all 0-1 inputs
    inputs = np.arange(2**N, dtype=np.uint32)
    
    bits = 1 << np.arange(N)
    matrix = (inputs[:, None] & bits[None, :]) > 0
    
    # Apply network
    for i, j in comps:
        # min to i, max to j
        min_val = np.minimum(matrix[:, i], matrix[:, j])
        max_val = np.maximum(matrix[:, i], matrix[:, j])
        matrix[:, i] = min_val
        matrix[:, j] = max_val
        
    # Check if sorted
    diff = np.diff(matrix, axis=1)
    return np.all(diff >= 0)

def _depth_of_network(comps, N):
    """Calculate parallel depth of the network."""
    depth = [0] * N
    for i, j in comps:
        d = max(depth[i], depth[j]) + 1
        depth[i] = d
        depth[j] = d
    return max(depth) if depth else 0

def _batcher_sort(N):
    """Generate a Batcher sort network."""
    def batcher_merge(N1, N2):
        if N1 == 0 or N2 == 0:
            return []
        depth = math.ceil(math.log2(N1 + N2))
        comps = []
        for j in range(depth):
            step = 1 << j
            for i in range(N1 + N2 - step):
                if j == 0:
                    comps.append((i, i + step))
                else:
                    if (i % step) < (step // 2):
                        comps.append((i, i + step))
        return comps

    if N == 1:
        return []
    N1 = N // 2
    N2 = N - N1
    comps = []
    comps.extend(batcher_sort(N1))
    half2_comps = batcher_sort(N2)
    comps.extend([(i + N1, j + N1) for i, j in half2_comps])
    comps.extend(batcher_merge(N1, N2))
    return comps

def _odd_even_transposition(N):
    """Generate odd-even transposition network."""
    comps = []
    for k in range(N):
        if k % 2 == 0:
            for i in range(0, N-1, 2):
                comps.append((i, i+1))
        else:
            for i in range(1, N-1, 2):
                comps.append((i, i+1))
    return comps

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # 1. Generate initial network (Batcher)
    try:
        comps = _batcher_sort(N)
        comps = list(set(comps))
    except Exception:
        comps = _odd_even_transposition(N)
    
    # Verify initial network
    if not _is_sorted_network(comps, N):
        # Fallback to odd-even transposition
        comps = _odd_even_transposition(N)
    
    best_comps = list(comps)
    best_count = len(comps)
    best_depth = _depth_of_network(best_comps, N)
    
    # 2. Phase 1: Greedy deletion (remove redundant comparators)
    i = 0
    try:
        while i < len(comps) and time.time() < deadline:
            test_comps = comps[:i] + comps[i+1:]
            if _is_sorted_network(test_comps, N):
                comps = test_comps
                if len(comps) < best_count:
                    best_comps = list(comps)
                    best_count = len(comps)
                    best_depth = _depth_of_network(best_comps, N)
            else:
                i += 1
    except Exception:
        pass
    
    # 3. Phase 2: Simulated Annealing / Local Search
    # We can try swapping, moving, or adding/removing comparators
    # Focus on moving comparators to different positions or changing their wires
    
    current_comps = list(best_comps)
    current_count = best_count
    
    temperature = 100.0
    cooling_rate = 0.999
    
    # Pre-compute all possible comparator pairs
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    try:
        while time.time() < deadline and temperature > 1.0:
            # Choose a random move
            move_type = random.choice(['swap', 'move', 'delete', 'insert'])
            
            if move_type == 'swap' and len(current_comps) >= 2:
                # Swap two comparators
                idx1, idx2 = random.sample(range(len(current_comps)), 2)
                new_comps = list(current_comps)
                new_comps[idx1], new_comps[idx2] = new_comps[idx2], new_comps[idx1]
                
            elif move_type == 'move' and len(current_comps) >= 2:
                # Move a comparator to a different position
                idx1 = random.randint(0, len(current_comps) - 1)
                idx2 = random.randint(0, len(current_comps) - 1)
                if idx1 == idx2:
                    continue
                new_comps = list(current_comps)
                comp = new_comps.pop(idx1)
                new_comps.insert(idx2, comp)
                
            elif move_type == 'delete' and len(current_comps) > 1:
                # Delete a comparator
                idx = random.randint(0, len(current_comps) - 1)
                new_comps = current_comps[:idx] + current_comps[idx+1:]
                
            elif move_type == 'insert' and len(current_comps) < 200:
                # Insert a random comparator
                new_comp = random.choice(all_pairs)
                idx = random.randint(0, len(current_comps))
                new_comps = current_comps[:idx] + [new_comp] + current_comps[idx:]
                
            else:
                continue
            
            # Check validity
            if _is_sorted_network(new_comps, N):
                new_count = len(new_comps)
                new_depth = _depth_of_network(new_comps, N)
                
                # Accept if better or with probability based on temperature
                delta_count = new_count - current_count
                delta_depth = new_depth - best_depth
                
                # Prefer lower count, then lower depth
                if new_count < current_count:
                    current_comps = new_comps
                    current_count = new_count
                    if new_count < best_count:
                        best_comps = list(new_comps)
                        best_count = new_count
                        best_depth = new_depth
                elif new_count == current_count and new_depth < best_depth:
                    current_comps = new_comps
                    if new_depth < best_depth:
                        best_comps = list(new_comps)
                        best_depth = new_depth
                else:
                    # Accept worse moves with probability exp(-delta/temperature)
                    # Use count as primary metric for acceptance
                    prob = math.exp(-delta_count / temperature) if delta_count > 0 else 1.0
                    if random.random() < prob:
                        current_comps = new_comps
                        current_count = new_count
            
            temperature *= cooling_rate
            
    except Exception:
        pass
    
    # Final safety check
    if not best_comps:
        best_comps = _odd_even_transposition(N)
    
    return best_comps
