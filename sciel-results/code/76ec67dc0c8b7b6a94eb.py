import random
import time
import itertools

N = 16
POINTS = [(x, y) for x in range(N) for y in range(N)]
NUM_POINTS = N * N

# Precompute distances
# dist[i][j] for point i and point j
# Points are indexed 0..255
# Point index i corresponds to (i // N, i % N)

def _point_from_idx(idx):
    return (idx // N, idx % N)

def _idx_from_point(p):
    return p[0] * N + p[1]

def _d2_idx(i, j):
    pi = _point_from_idx(i)
    pj = _point_from_idx(j)
    return (pi[0] - pj[0]) ** 2 + (pi[1] - pj[1]) ** 2

# Precompute all pairwise squared distances
DIST = [[0] * NUM_POINTS for _ in range(NUM_POINTS)]
for i in range(NUM_POINTS):
    for j in range(i + 1, NUM_POINTS):
        d = _d2_idx(i, j)
        DIST[i][j] = d
        DIST[j][i] = d

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin for 300s limit
    
    best_set = None
    best_size = float('inf')
    
    # Precompute conflict info
    # For each point p, we need to know if adding p to a set S is valid.
    # This is fast if we maintain for each a in S:
    #   - set of distances from a to other points in S
    #   - check if d(p, a) is in that set
    #   - check if d(p, a) is a distance from p to any other point in S (vertex p)
    
    def _greedy_maximal(order):
        S = []
        # dist_from[i] is a set of squared distances from S[i] to other points in S
        dist_from = []
        
        for p in order:
            valid = True
            # Compute distances from p to all a in S
            dp = []
            for a in S:
                d = DIST[p][a]
                dp.append(d)
            
            # Check vertex p: all distances from p to S must be distinct
            if len(set(dp)) != len(dp):
                valid = False
            
            if valid:
                # Check vertex a or b: for each a in S, d(p,a) not in dist_from[a]
                for i, a in enumerate(S):
                    if dp[i] in dist_from[i]:
                        valid = False
                        break
            
            if valid:
                S.append(p)
                new_dist_set = set()
                for i, a in enumerate(S[:-1]):
                    d = dp[i]
                    dist_from[i].add(d)
                    new_dist_set.add(d)
                dist_from.append(new_dist_set)
        
        return S

    def _is_maximal(S):
        """Check if S is a maximal no-isosceles set."""
        S_set = set(S)
        for p in range(NUM_POINTS):
            if p in S_set:
                continue
            # Check if p can be added to S
            valid = True
            dp = []
            for a in S:
                d = DIST[p][a]
                dp.append(d)
            
            if len(set(dp)) != len(dp):
                continue  # Can't add p
            
            # Check against dist_from
            # We don't have dist_from here, so recompute
            dist_from = []
            for i, a in enumerate(S):
                ds = set()
                for j, b in enumerate(S):
                    if i != j:
                        ds.add(DIST[a][b])
                dist_from.append(ds)
            
            for i, a in enumerate(S):
                if dp[i] in dist_from[i]:
                    valid = False
                    break
            
            if valid:
                return False  # p can be added, so S is not maximal
        
        return True

    def _local_search(S):
        """Try to find a smaller maximal set by removing points and swapping."""
        current_S = list(S)
        current_size = len(current_S)
        
        # Build dist_from for current_S
        def _build_dist_from(S):
            dist_from = []
            for i, a in enumerate(S):
                ds = set()
                for j, b in enumerate(S):
                    if i != j:
                        ds.add(DIST[a][b])
                dist_from.append(ds)
            return dist_from
        
        def _can_add(p, S, dist_from):
            dp = []
            for a in S:
                d = DIST[p][a]
                dp.append(d)
            if len(set(dp)) != len(dp):
                return False
            for i, a in enumerate(S):
                if dp[i] in dist_from[i]:
                    return False
            return True
        
        dist_from = _build_dist_from(current_S)
        
        improved = True
        while improved:
            improved = False
            # Try removing each point
            for idx in range(len(current_S)):
                p = current_S[idx]
                new_S = current_S[:idx] + current_S[idx+1:]
                
                if not new_S:
                    # Empty set is trivially maximal only if no point can be added
                    # But we want non-empty sets for minimality
                    continue
                
                new_dist_from = _build_dist_from(new_S)
                
                # Check if new_S is maximal
                is_max = True
                S_set = set(new_S)
                for q in range(NUM_POINTS):
                    if q in S_set:
                        continue
                    if _can_add(q, new_S, new_dist_from):
                        is_max = False
                        break
                
                if is_max:
                    current_S = new_S
                    dist_from = new_dist_from
                    current_size = len(current_S)
                    improved = True
                    break
                else:
                    # Try swap: remove p, add some q
                    for q in range(NUM_POINTS):
                        if q in S_set or q == p:
                            continue
                        if _can_add(q, new_S, new_dist_from):
                            swapped_S = new_S + [q]
                            swapped_dist_from = _build_dist_from(swapped_S)
                            
                            # Check if swapped_S is maximal
                            is_max = True
                            SS_set = set(swapped_S)
                            for r in range(NUM_POINTS):
                                if r in SS_set:
                                    continue
                                if _can_add(r, swapped_S, swapped_dist_from):
                                    is_max = False
                                    break
                            
                            if is_max and len(swapped_S) < len(current_S):
                                current_S = swapped_S
                                dist_from = swapped_dist_from
                                current_size = len(current_S)
                                improved = True
                                break
                    if improved:
                        break
        
        return current_S

    # Run multiple random restarts with local search
    num_restarts = 0
    while time.time() < deadline:
        order = list(range(NUM_POINTS))
        random.shuffle(order)
        
        res = _greedy_maximal(order)
        
        # Local search to shrink
        if time.time() < deadline - 10:
            res = _local_search(res)
        
        if len(res) < best_size:
            best_size = len(res)
            best_set = res
            if best_size <= 10:
                break
        
        num_restarts += 1
        if num_restarts % 50 == 0:
            # Periodic check
            pass
    
    if best_set is None:
        best_set = _greedy_maximal(list(range(NUM_POINTS)))
    
    # Convert indices back to points
    return [_point_from_idx(i) for i in best_set]
