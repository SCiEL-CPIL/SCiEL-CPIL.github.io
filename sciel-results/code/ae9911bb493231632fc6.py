import itertools
import random
import time
import math
from collections import defaultdict

N = 16
NUM_POINTS = N * N

def _d2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def construct():
    start_time = time.time()
    deadline = start_time + 285
    
    all_points = [(x, y) for x in range(N) for y in range(N)]
    point_to_idx = {p: i for i, p in enumerate(all_points)}
    idx_to_point = {i: p for p, i in point_to_idx.items()}
    
    # Precompute distances using vectorized approach
    # dist[i][j] = squared distance between point i and point j
    dist = [[0] * NUM_POINTS for _ in range(NUM_POINTS)]
    for i in range(NUM_POINTS):
        pi = idx_to_point[i]
        for j in range(i + 1, NUM_POINTS):
            pj = idx_to_point[j]
            d = _d2(pi, pj)
            dist[i][j] = d
            dist[j][i] = d
    
    # Precompute for each point i and each distance d, the set of points at distance d from i
    # dist_from[i][d] = list of points j such that dist[i][j] == d
    dist_from = [defaultdict(list) for _ in range(NUM_POINTS)]
    for i in range(NUM_POINTS):
        for j in range(NUM_POINTS):
            if i != j:
                d = dist[i][j]
                dist_from[i][d].append(j)
    
    # Precompute bad_pair[i][j] as a bitset of points k such that (i, j, k) is isosceles
    # For pair (i,j), k is bad if:
    #   dist(i,k) == dist(i,j)  OR  dist(j,k) == dist(i,j)  OR  dist(i,k) == dist(j,k)
    bad_pair = [[0] * NUM_POINTS for _ in range(NUM_POINTS)]
    
    for i in range(NUM_POINTS):
        for j in range(i + 1, NUM_POINTS):
            d_ij = dist[i][j]
            mask = 0
            # Points k with dist(i,k) == d_ij
            for k in dist_from[i].get(d_ij, []):
                if k != i and k != j:
                    mask |= (1 << k)
            # Points k with dist(j,k) == d_ij
            for k in dist_from[j].get(d_ij, []):
                if k != i and k != j:
                    mask |= (1 << k)
            # Points k with dist(i,k) == dist(j,k)
            # We need to find k where dist(i,k) == dist(j,k)
            # For each point k, check if dist(i,k) == dist(j,k)
            # This is expensive, so we use a different approach:
            # For each distance d, if k is at distance d from both i and j
            # Build a set of distances from i and j
            di = dist[i]
            dj = dist[j]
            # Find common distances
            common_dists = set(di) & set(dj)
            for d in common_dists:
                if d == 0:
                    continue
                for k in dist_from[i].get(d, []):
                    if k != i and k != j and dj[k] == d:
                        mask |= (1 << k)
            bad_pair[i][j] = mask
            bad_pair[j][i] = mask
            
    all_mask = (1 << NUM_POINTS) - 1
    
    best_set = None
    best_size = float('inf')
    
    def _get_avail_list(mask):
        """Convert a bitset to a list of indices."""
        result = []
        temp = mask
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            result.append(idx)
            temp ^= lsb
        return result
    
    def _construct_greedy(seed, use_full_scan=True):
        """Greedy construction with tie-breaking by blocking count."""
        random.seed(seed)
        S = []
        S_mask = 0
        blocked = 0
        
        while True:
            available = all_mask & ~S_mask & ~blocked
            if available == 0:
                break
                
            avail_list = _get_avail_list(available)
            if not avail_list:
                break
                
            best_p = None
            best_key = (-1, -1)  # (gain, -degree)
            
            if use_full_scan and len(avail_list) <= 100:
                candidates = avail_list
            elif len(avail_list) > 60:
                candidates = random.sample(avail_list, min(60, len(avail_list)))
            else:
                candidates = avail_list
                
            for p in candidates:
                new_blocked = 0
                for r in S:
                    new_blocked |= bad_pair[p][r]
                gain = (new_blocked & available).bit_count()
                # Tie-break: prefer points that block more total points (higher degree in conflict graph)
                total_block = new_blocked.bit_count()
                key = (gain, -total_block)
                if key > best_key:
                    best_key = key
                    best_p = p
                    
            if best_p is None:
                best_p = random.choice(avail_list)
                
            S.append(best_p)
            S_mask |= (1 << best_p)
            
            # Update blocked
            for r in S[:-1]:
                blocked |= bad_pair[best_p][r]
                
        return S
    
    def _is_maximal_fast(S_mask, blocked):
        """Check if no available point can be added, given S_mask and blocked bitsets."""
        available = all_mask & ~S_mask & ~blocked
        return available == 0
    
    def _compute_blocked(S_list):
        """Compute the blocked mask for a set S_list."""
        blocked = 0
        n = len(S_list)
        for i in range(n):
            for j in range(i + 1, n):
                blocked |= bad_pair[S_list[i]][S_list[j]]
        return blocked
    
    def _prune(S):
        """Aggressively remove points while maintaining maximality."""
        S_list = S[:]
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            S_mask = 0
            for p in S_list:
                S_mask |= (1 << p)
            
            # Try removing each point and check if still maximal
            for i in range(len(S_list)):
                p = S_list[i]
                # Recompute blocked without point p
                S_temp = S_list[:i] + S_list[i+1:]
                S_temp_mask = S_mask & ~(1 << p)
                blocked_temp = _compute_blocked(S_temp)
                
                if _is_maximal_fast(S_temp_mask, blocked_temp):
                    S_list = S_temp
                    S_mask = S_temp_mask
                    improved = True
                    break
        return S_list
    
    def _local_search_swap(S_list, max_iters=50):
        """Try removing points to reduce set size while maintaining maximality."""
        S_list = S_list[:]
        improved = True
        iterations = 0
        while improved and iterations < max_iters and time.time() < deadline - 10:
            improved = False
            iterations += 1
            S_mask = 0
            for p in S_list:
                S_mask |= (1 << p)
            
            # Try removing each point
            for i in range(len(S_list)):
                p = S_list[i]
                S_temp = S_list[:i] + S_list[i+1:]
                S_temp_mask = S_mask & ~(1 << p)
                blocked_temp = _compute_blocked(S_temp)
                
                if _is_maximal_fast(S_temp_mask, blocked_temp):
                    S_list = S_temp
                    S_mask = S_temp_mask
                    improved = True
                    break
        
        return S_list
    
    # Phase 1: Multiple greedy constructions with different seeds and strategies
    num_trials = 200
    for t in range(num_trials):
        if time.time() > deadline - 20:
            break
        # Alternate between full scan and sampled greedy
        use_full = (t % 3 != 0)
        S = _construct_greedy(t, use_full_scan=use_full)
        S = _prune(S)
        S = _local_search_swap(S)
        if len(S) < best_size:
            best_size = len(S)
            best_set = S[:]
            
    # Phase 2: Local search from the best solution with more iterations
    if best_set is not None and time.time() < deadline - 15:
        S = _local_search_swap(best_set, max_iters=200)
        S = _prune(S)
        if len(S) < best_size:
            best_size = len(S)
            best_set = S[:]
            
    # Phase 3: Try more diverse seeds if time remains
    if time.time() < deadline - 15:
        for t in range(200, 400):
            if time.time() > deadline - 10:
                break
            S = _construct_greedy(t, use_full_scan=True)
            S = _prune(S)
            S = _local_search_swap(S)
            if len(S) < best_size:
                best_size = len(S)
                best_set = S[:]
            
    if best_set is None:
        return []
        
    return [idx_to_point[p] for p in best_set]
