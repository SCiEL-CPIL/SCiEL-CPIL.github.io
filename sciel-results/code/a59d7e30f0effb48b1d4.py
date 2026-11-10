"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import itertools
import random
import time

N = 16
POINTS = [(x, y) for x in range(N) for y in range(N)]
NUM_POINTS = N * N

# Precompute squared distances for all pairs
# dist[i][j] where i, j are indices 0..255
dist = [[0] * NUM_POINTS for _ in range(NUM_POINTS)]
for i in range(NUM_POINTS):
    x1, y1 = POINTS[i]
    for j in range(i + 1, NUM_POINTS):
        x2, y2 = POINTS[j]
        d = (x1 - x2) ** 2 + (y1 - y2) ** 2
        dist[i][j] = d
        dist[j][i] = d

def _is_isosceles(i, j, k):
    """Check if points i, j, k form an isosceles triangle."""
    d_ij = dist[i][j]
    d_ik = dist[i][k]
    d_jk = dist[j][k]
    return d_ij == d_ik or d_ij == d_jk or d_ik == d_jk

def _can_add(p_idx, S_indices):
    """Check if point p_idx can be added to set S_indices without forming an isosceles triangle."""
    # S_indices is a list of indices
    # We need to check all pairs in S_indices
    # Optimization: if len(S) < 2, always can add (unless p is already in S, which we assume is not)
    if len(S_indices) < 2:
        return True
    
    # Check against all pairs in S
    # This is O(|S|^2). 
    # For small S, this is fine.
    s_list = list(S_indices)
    for a in s_list:
        d_pa = dist[p_idx][a]
        for b in s_list:
            if a == b:
                continue
            d_pb = dist[p_idx][b]
            d_ab = dist[a][b]
            if d_pa == d_pb or d_pa == d_ab or d_pb == d_ab:
                return False
    return True

def _greedy_complete(S_indices, shuffle_seed=None):
    """
    Complete a set S_indices to a maximal no-isosceles set using a greedy approach.
    S_indices is a list of indices.
    Returns a list of indices.
    """
    S = set(S_indices)
    remaining = [i for i in range(NUM_POINTS) if i not in S]
    
    if shuffle_seed is not None:
        random.seed(shuffle_seed)
    random.shuffle(remaining)
    
    for p in remaining:
        if _can_add(p, S):
            S.add(p)
            
    return list(S)

def _local_search(S_indices, max_iter=50):
    """
    Try to reduce the size of the maximal set S_indices.
    Strategy: Remove a point, then greedily complete the set. If the new set is smaller, accept.
    """
    S = set(S_indices)
    improved = True
    iters = 0
    while improved and iters < max_iter:
        improved = False
        iters += 1
        # Try removing each point in S
        # To speed up, we can just iterate through the list
        S_list = list(S)
        random.shuffle(S_list)
        for p in S_list:
            S_temp = S - {p}
            # Complete S_temp
            S_new = _greedy_complete(list(S_temp))
            if len(S_new) < len(S):
                S = set(S_new)
                improved = True
                break # Restart the loop with the new smaller set
    return list(S)

def construct():
    start_time = time.time()
    deadline = start_time + 300  # 300s limit, leave margin for overhead
    
    best_set = None
    best_size = float('inf')
    
    # 1. Initial Randomized Greedy Runs
    # We run multiple random greedy constructions to find diverse maximal sets.
    # Each run is fast.
    num_initial_runs = 1000 # Adjust based on speed
    
    for i in range(num_initial_runs):
        if time.time() > deadline - 30: # 30s margin for local search
            break
            
        S = _greedy_complete([], shuffle_seed=i)
        size = len(S)
        if size < best_size:
            best_size = size
            best_set = S
            
    # 2. Local Search on the best candidates
    # We can apply local search to the best set found, and maybe a few others.
    # To be safe, we'll apply it to the current best.
    if best_set is not None:
        if time.time() < deadline - 10:
            improved_set = _local_search(best_set, max_iter=20)
            if len(improved_set) < best_size:
                best_set = improved_set
                best_size = len(best_set)
                
    # 3. If we have time, run more random restarts and local search
    # The local search is more expensive. Let's see how much time we have.
    # _greedy_complete is O(N^2 * |S|^2). N=256, |S|~20. 256*400 = 100k ops. ~0.01s?
    # _local_search does |S| greedy completions. 20 * 0.01s = 0.2s per iteration.
    # 20 iterations = 4s.
    # So we can do ~50 local search rounds in 300s?
    
    # Let's do a loop of: Random Greedy -> Local Search
    # But we already did initial greedy.
    # Let's do more random restarts with local search if time permits.
    
    # Strategy: 
    # While time < deadline - 5:
    #   S = _greedy_complete([], shuffle_seed=random.randint(0, 100000))
    #   S = _local_search(S, max_iter=5) # Shallow local search
    #   if len(S) < best_size:
    #       best_set = S
    #       best_size = len(S)
        
    # To avoid getting stuck, we should vary the depth of local search.
    
    while time.time() < deadline - 5:
        # Random restart
        S = _greedy_complete([], shuffle_seed=random.randint(0, 10**6))
        # Shallow local search
        S = _local_search(S, max_iter=10)
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = S
            
    return best_set

# EVOLVE-BLOCK-END
