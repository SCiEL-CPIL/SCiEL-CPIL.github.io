"""Improved snake-in-the-box for Q_9 using Simulated Annealing with NumPy vectorization."""
import sys
import time
import numpy as np

N = 9
FULL_MASK = (1 << N) - 1
NUM_VERTICES = 1 << N

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a long induced path (snake) in Q_9 using Simulated Annealing
    with NumPy vectorized constraint checks.
    """
    start_time = time.time()
    time_limit = 115  # Leave buffer for evaluation overhead

    # Precompute distance matrix (Hamming distance)
    # D[i, j] = popcount(i ^ j)
    vertices = np.arange(NUM_VERTICES)
    # Efficient popcount for all pairs
    # (vertices[:, None] ^ vertices[None, :])
    xor_matrix = vertices[:, None] ^ vertices[None, :]
    # Use lookup table or bit tricks for popcount on array
    # np.bitwise_count is not available in all numpy versions, use unpackbits
    # Or simpler: since N=9, we can use a lookup table for 0..511
    popcount_lut = np.array([i.bit_count() for i in range(NUM_VERTICES)], dtype=np.uint8)
    D = popcount_lut[xor_matrix]
    
    # D is 512x512 uint8. Memory ~2MB.

    best_path = []
    best_len = 0
    
    # Helper to check validity of a path (for initial construction)
    def is_valid_path(path):
        if len(path) == 0:
            return True
        if len(path) == 1:
            return True
        p = np.array(path)
        # Consecutive
        if not np.all(D[p[:-1], p[1:]] == 1):
            return False
        # Non-consecutive
        # This is hard to vectorize simply for all pairs.
        # But for SA moves, we check incrementally.
        # For initial greedy, we can do a simple check.
        for i in range(len(path)):
            for j in range(i + 2, len(path)):
                if D[path[i], path[j]] < 2:
                    return False
        return True

    # Greedy initialization
    def greedy_init():
        path = [0]
        used = np.zeros(NUM_VERTICES, dtype=bool)
        used[0] = True
        
        while True:
            cur = path[-1]
            # Find valid next vertices
            # Candidates: neighbors of cur
            candidates = []
            for b in range(N):
                v = cur ^ (1 << b)
                if used[v]:
                    continue
                # Check if v is adjacent to any path[:-1]
                valid = True
                # Vectorized check against path[:-1]
                if len(path) > 1:
                    # D[path[:-1], v] must be >= 2
                    if np.any(D[np.array(path[:-1]), v] == 1):
                        valid = False
                if valid:
                    candidates.append(v)
            
            if not candidates:
                break
            
            # Pick one (random or heuristic)
            # Heuristic: pick one with fewest future options? 
            # For speed, just pick random to diversify, or greedy min-degree.
            # Let's try random to avoid getting stuck in a specific greedy trap immediately.
            # But we want a long start. Greedy min-degree is better for length.
            # Let's compute degree for candidates.
            best_v = -1
            min_deg = 100
            for v in candidates:
                deg = 0
                for b in range(N):
                    w = v ^ (1 << b)
                    if used[w]:
                        continue
                    # Check if w is adjacent to path
                    # Approximation: just count unused neighbors for now?
                    # Proper check:
                    valid_w = True
                    if len(path) > 0:
                        # w must not be adjacent to path[:-1]
                        # w is adjacent to path[-1] (cur).
                        # So check w against path[:-1]
                        if np.any(D[np.array(path[:-1]), w] == 1):
                            valid_w = False
                    if valid_w:
                        deg += 1
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
            
            if best_v == -1:
                break
                
            path.append(best_v)
            used[best_v] = True
            
        return path

    # Simulated Annealing
    def simulated_annealing():
        nonlocal best_path, best_len
        path = greedy_init()
        used = np.zeros(NUM_VERTICES, dtype=bool)
        for v in path:
            used[v] = True
        
        current_len = len(path)
        if current_len > best_len:
            best_len = current_len
            best_path = path[:]
        
        T = 5.0
        cooling = 0.999
        min_T = 0.01
        iterations = 0
        
        while time.time() - start_time < time_limit:
            if T < min_T:
                break
                
            # Choose a move
            move_type = np.random.random()
            
            if move_type < 0.8: # Add
                if current_len == NUM_VERTICES:
                    break
                
                if np.random.random() < 0.5:
                    # Append
                    cur = path[-1]
                    # Find valid v to append
                    # D[path, v] >= 2 for all except last
                    # D[path[-1], v] == 1
                    
                    # Get all unused vertices
                    unused = np.where(~used)[0]
                    if len(unused) == 0:
                        continue
                    
                    # Check neighbors of cur
                    # v must be neighbor of cur
                    # D[cur, v] == 1
                    # And D[path[:-1], v] >= 2
                    
                    # Filter unused by D[cur, v] == 1
                    # D is 512x512. D[cur, :] gives distances from cur to all.
                    neigh_mask = D[cur, :] == 1
                    valid_unused = unused[neigh_mask[unused]]
                    
                    if len(valid_unused) == 0:
                        continue
                    
                    # Check D[path[:-1], v] >= 2
                    # We can check all at once?
                    # Let's pick a random candidate from valid_unused
                    v = valid_unused[np.random.randint(len(valid_unused))]
                    
                    # Verify against path[:-1]
                    if len(path) > 1:
                        if np.any(D[np.array(path[:-1]), v] == 1):
                            continue
                            
                    # Move is valid
                    new_path = path + [v]
                    new_used = used.copy()
                    new_used[v] = True
                    new_len = current_len + 1
                    
                    # Accept (always accept if valid and longer)
                    path = new_path
                    used = new_used
                    current_len = new_len
                    
                    if current_len > best_len:
                        best_len = current_len
                        best_path = path[:]
                        
                else:
                    # Insert
                    if current_len == 0:
                        continue
                    unused = np.where(~used)[0]
                    if len(unused) == 0:
                        continue
                    
                    # Pick random index i to insert before
                    # i in [1, current_len) to avoid ends (append is separate)
                    # Or i in [0, current_len]
                    i = np.random.randint(1, current_len)
                    
                    # Neighbors in path: path[i-1], path[i]
                    v_left = path[i-1]
                    v_right = path[i]
                    
                    # v must be neighbor of v_left and v_right
                    # D[v_left, v] == 1 and D[v_right, v] == 1
                    # Intersection of neighbors
                    neigh_left = D[v_left, :] == 1
                    neigh_right = D[v_right, :] == 1
                    common_neigh = neigh_left & neigh_right
                    
                    valid_unused = unused[common_neigh[unused]]
                    if len(valid_unused) == 0:
                        continue
                    
                    v = valid_unused[np.random.randint(len(valid_unused))]
                    
                    # Check against rest of path
                    # Exclude i-1 and i
                    mask = np.ones(current_len, dtype=bool)
                    mask[i-1] = False
                    mask[i] = False
                    
                    rest_indices = np.where(mask)[0]
                    if len(rest_indices) > 0:
                        if np.any(D[np.array(path[rest_indices]), v] == 1):
                            continue
                    
                    # Move is valid
                    new_path = path[:i] + [v] + path[i:]
                    new_used = used.copy()
                    new_used[v] = True
                    new_len = current_len + 1
                    
                    # Accept
                    path = new_path
                    used = new_used
                    current_len = new_len
                    
                    if current_len > best_len:
                        best_len = current_len
                        best_path = path[:]
                        
            else:
                # Remove
                if current_len <= 1:
                    continue
                # Pick a random index to remove
                # Avoid removing endpoints if possible? 
                # Removing any vertex creates a gap. 
                # To keep it a path, we must reconnect? 
                # No, removing a vertex from a path breaks it into two paths.
                # This move is invalid for maintaining a single path unless we allow "jumping".
                # But the definition is a single sequence.
                # So we can only remove if we can reconnect the ends?
                # If we remove path[i], we connect path[i-1] and path[i+1].
                # This is valid only if D[path[i-1], path[i+1]] == 1.
                # And we must check that path[i-1] and path[i+1] are not adjacent to other vertices in a way that violates induced property?
                # Actually, removing a vertex from an induced path results in a valid induced path 
                # IF the new consecutive pair (i-1, i+1) satisfies:
                # 1. D[path[i-1], path[i+1]] == 1
                # 2. They were not adjacent before (they weren't, distance >= 2 in induced path if |i-1 - (i+1)|=2? No, in induced path, non-consecutive are dist >= 2. So path[i-1] and path[i+1] are non-consecutive, so dist >= 2. 
                # Wait. If dist >= 2, we cannot connect them with an edge (dist 1).
                # So we CANNOT simply remove a middle vertex and close the gap.
                # We can only remove an END vertex.
                # Removing an end vertex always results in a valid induced path.
                
                # So "Remove" move is just "Truncate".
                # This reduces length, so we only accept if it helps escape a local minimum?
                # In SA, we accept worse moves with probability exp(-delta/T).
                # Delta = -1.
                
                # Pick end to remove
                if np.random.random() < 0.5:
                    # Remove last
                    v = path[-1]
                    new_path = path[:-1]
                    new_used = used.copy()
                    new_used[v] = False
                    new_len = current_len - 1
                else:
                    # Remove first
                    v = path[0]
                    new_path = path[1:]
                    new_used = used.copy()
                    new_used[v] = False
                    new_len = current_len - 1
                    
                delta = new_len - current_len
                if delta > 0 or np.random.random() < np.exp(delta / T):
                    path = new_path
                    used = new_used
                    current_len = new_len
                    if current_len > best_len:
                        best_len = current_len
                        best_path = path[:]
                        
            T *= cooling
            iterations += 1

    simulated_annealing()
    
    return best_path
# EVOLVE-BLOCK-END
