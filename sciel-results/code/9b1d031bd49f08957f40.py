import itertools
import random
import time
import math

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
    
    # Precompute bad_pair[i][j] as a bitset of points k such that (i, j, k) is isosceles.
    bad_pair = [[0] * NUM_POINTS for _ in range(NUM_POINTS)]
    
    # Precompute distances
    dist = [[0] * NUM_POINTS for _ in range(NUM_POINTS)]
    for i in range(NUM_POINTS):
        pi = idx_to_point[i]
        for j in range(i + 1, NUM_POINTS):
            pj = idx_to_point[j]
            d = _d2(pi, pj)
            dist[i][j] = d
            dist[j][i] = d
            
    for i in range(NUM_POINTS):
        for j in range(i + 1, NUM_POINTS):
            mask = 0
            d_ij = dist[i][j]
            for k in range(NUM_POINTS):
                if k == i or k == j:
                    continue
                d_ik = dist[i][k]
                d_jk = dist[j][k]
                if d_ij == d_ik or d_ij == d_jk or d_ik == d_jk:
                    mask |= (1 << k)
            bad_pair[i][j] = mask
            bad_pair[j][i] = mask
            
    all_mask = (1 << NUM_POINTS) - 1
    
    best_set = None
    best_size = float('inf')
    
    def _construct_greedy(seed):
        random.seed(seed)
        S = []
        S_mask = 0
        blocked = 0
        
        while True:
            available = all_mask & ~S_mask & ~blocked
            if available == 0:
                break
                
            # Get list of available indices
            avail_list = []
            temp = available
            while temp:
                lsb = temp & -temp
                idx = lsb.bit_length() - 1
                avail_list.append(idx)
                temp ^= lsb
                
            if not avail_list:
                break
                
            best_p = None
            best_gain = -1
            
            # Sample candidates to speed up
            if len(avail_list) > 40:
                candidates = random.sample(avail_list, 40)
            else:
                candidates = avail_list
                
            for p in candidates:
                new_blocked = 0
                for r in S:
                    new_blocked |= bad_pair[p][r]
                # Count how many currently available points are blocked
                gain = (new_blocked & available).bit_count()
                if gain > best_gain:
                    best_gain = gain
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
        S_list = S[:]
        improved = True
        while improved and time.time() < deadline - 5:
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
    
    def _local_search_swap(S_list, max_iters=100):
        """Try swapping points to reduce set size while maintaining maximality."""
        S_list = S_list[:]
        improved = True
        iterations = 0
        while improved and iterations < max_iters and time.time() < deadline - 5:
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
    
    def _branch_and_bound_reduce(S_list, time_limit=10.0):
        """
        Try to reduce set size using branch-and-bound.
        For each point, check if it can be removed while maintaining maximality.
        Uses a lower bound based on the number of uncovered points.
        """
        S_list = S_list[:]
        start = time.time()
        
        def _can_remove(p, S_mask, blocked):
            """Check if point p can be removed while maintaining maximality."""
            S_temp_mask = S_mask & ~(1 << p)
            
            # Compute new blocked mask without p
            new_blocked = 0
            S_list_temp = [s for s in S_list if s != p]
            for i in range(len(S_list_temp)):
                for j in range(i + 1, len(S_list_temp)):
                    new_blocked |= bad_pair[S_list_temp[i]][S_list_temp[j]]
            
            available = all_mask & ~S_temp_mask & ~new_blocked
            return available == 0
        
        # Iteratively try to remove points
        improved = True
        while improved and time.time() - start < time_limit:
            improved = False
            S_mask = 0
            for p in S_list:
                S_mask |= (1 << p)
            
            # Try removing each point
            for i in range(len(S_list)):
                p = S_list[i]
                if _can_remove(p, S_mask, 0):
                    S_list = S_list[:i] + S_list[i+1:]
                    improved = True
                    break
        
        return S_list
    
    def _smart_prune_with_bnb(S_list, time_limit=15.0):
        """
        More aggressive pruning using branch-and-bound ideas.
        Tries to find a minimal subset that still dominates all points.
        """
        S_list = S_list[:]
        start = time.time()
        
        # First, do standard pruning
        S_list = _prune(S_list)
        
        # Then try to swap points to get a smaller set
        S_list = _local_search_swap(S_list, max_iters=200)
        
        # Finally, use B&B to try further reduction
        if time.time() - start < time_limit - 5:
            S_list = _branch_and_bound_reduce(S_list, time_limit=5.0)
        
        return S_list

    num_trials = 200
    for t in range(num_trials):
        if time.time() > deadline - 30:
            break
        
        # Use different strategies for different trials
        if t % 3 == 0:
            S = _construct_greedy(t)
            S = _prune(S)
            S = _local_search_swap(S)
        elif t % 3 == 1:
            S = _construct_greedy(t + 1000)
            S = _smart_prune_with_bnb(S, time_limit=10.0)
        else:
            S = _construct_greedy(t + 2000)
            S = _branch_and_bound_reduce(S, time_limit=8.0)
            S = _local_search_swap(S)
            
        if len(S) < best_size:
            best_size = len(S)
            best_set = S[:]
            
    # Final polish: try to reduce the best set further
    if best_set is not None and time.time() < deadline - 20:
        best_set = _smart_prune_with_bnb(best_set, time_limit=15.0)
        best_size = len(best_set)
        
    if best_set is None:
        return []
        
    return [idx_to_point[p] for p in best_set]
