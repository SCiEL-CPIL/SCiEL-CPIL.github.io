"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import time
import itertools
import random

# Precomputation:
# 1. Map all grid points to indices 0..N^2-1
# 2. For each point i, precompute a bitmask 'iso_mask[i]' where bit j is set 
#    if there exists a point k such that (i, j, k) is isosceles? 
#    Actually, it's better to precompute for each pair (i, j), which points k 
#    would form an isosceles triangle with i and j.
#    Let's use a different representation for speed.
    
# Total points
TOTAL_POINTS = N * N

# Point to index mapping
def point_to_idx(p):
    return p[0] * N + p[1]

def idx_to_point(i):
    return (i // N, i % N)

# Precompute all isosceles conditions
# For a set S, a point p is valid if for all pairs (a, b) in S, (p, a, b) is NOT isosceles.
# This is equivalent to: p is not in the union of "forbidden zones" for each pair (a, b) in S.
# Forbidden zone for pair (a, b) = { p | (p, a, b) is isosceles }

# Let's precompute `forbidden[pair_index]` as a bitmask of TOTAL_POINTS bits.
# There are N^4 pairs, which is 16^4 = 65536 pairs. 
# 65536 * 256 bits = 65536 * 32 bytes = ~2MB. This is feasible.

# Alternatively, we can just compute on the fly with bitmasks if we precompute 
# `iso_with[i][j]` = bitmask of points k such that (i, j, k) is isosceles.
# Then for a candidate p and set S, we check:
# for all pairs (a, b) in S: is p in iso_with[a][b]?
# This is still O(|S|^2) per check.

# Better approach for DFS:
# Maintain a bitmask `valid_mask` of all points that are currently valid to add.
# Initially, all points are valid.
# When we add a point `p`, we must remove from `valid_mask` any point `q` such that 
# (p, q, r) is isosceles for any `r` already in the set.
# Wait, the condition is: No 3 points in S form an isosceles triangle.
# So if we have set S and we want to add p, we check if (p, a, b) is isosceles for any a,b in S.
# This means p must NOT be in `iso_with[a][b]` for any pair a,b in S.
# So, `valid_mask` should be updated as:
# new_valid_mask = valid_mask & ~(union of iso_with[a][b] for all pairs a,b in S)
# This is expensive to update incrementally if we don't track the union.

# Let's use a simpler but effective heuristic for the DFS:
# We search for a set S of size k.
# We use backtracking.
# State: current set S (list of indices), current index in permutation order.
# To speed up, we can use a "conflict" check.

# Let's try a different angle: 
# The problem is small (N=16, 256 points).
# A maximal no-isosceles set is a maximal independent set in a 3-uniform hypergraph.
# We want the MINIMAL maximal independent set.
# This is NP-hard generally, but N=16 is small enough for smart search.

# Strategy:
# 1. Precompute `iso_pairs[i]` = set of j such that there exists k... no, that's not right.
# 2. Precompute `conflict[i][j]` = bitmask of all k such that (i, j, k) is isosceles.
#    Note: If (i, j, k) is isosceles, then (i, k, j) is too.
#    So if we pick i and j, we cannot pick any k in `conflict[i][j]`.
#    Also, if we pick i and j, we cannot pick any k such that (i, k, j) is isosceles.
#    So `conflict[i][j]` is the set of all points that cannot be in the set if i and j are.
#    
#    Let's verify: 
#    If S contains i and j, then for any k in S, (i, j, k) must NOT be isosceles.
#    So k must NOT be in `conflict[i][j]`.
#    Thus, if we build S incrementally, and we add i, then add j, 
#    we must remove `conflict[i][j]` from the pool of available points.
#    Also, we must ensure that for any previous point p in S, (p, i, j) is not isosceles.
#    This is covered if we maintain a global `valid_mask`.
#    
#    Algorithm:
#    valid_mask = (1 << TOTAL_POINTS) - 1
#    S = []
#    
#    def add_point(p):
#        # Check if p is valid against all pairs in S? 
#        # No, if we maintain valid_mask correctly, p is valid if bit p is set.
#        # How to update valid_mask when adding p?
#        # For each q in S:
#        #   The pair (p, q) forbids all points in conflict[p][q].
#        #   So valid_mask &= ~conflict[p][q]
#        # Also, p itself is added to S, so we remove p from valid_mask (optional, as we track S).
#        
#    This is O(|S|) per addition. 
#    Total complexity of a path of length k is O(k^2).
#    With k ~ 10-20, this is very fast.
#    
#    We will perform a DFS to find a maximal set of size exactly k.
#    We start with k=1, increment until we find a solution.
#    To guide the search, we can use a greedy heuristic to pick the next point, 
#    or randomize to find diverse solutions.
#    
#    Optimization: 
#    We want the MINIMAL maximal set.
#    So we iterate k from 1 upwards.
#    For each k, we try to find a set of size k that is maximal.
#    How to check if a set of size k is maximal?
#    A set S is maximal if no point can be added.
#    This is equivalent to: valid_mask (after adding all points in S) is 0?
#    No, valid_mask contains points that can be added WITHOUT creating an isosceles triangle 
#    with pairs in S.
#    If valid_mask is 0, then no point can be added. So S is maximal.
#    
#    So the algorithm is:
#    1. Precompute conflict[i][j] bitmasks.
#    2. For k in range(1, TOTAL_POINTS):
#        a. Run a search to find a set S of size k such that valid_mask becomes 0.
#        b. If found, return S.
#        
#    Search for fixed k:
#    - Use backtracking.
#    - State: current S, current valid_mask, remaining points to pick (k - len(S)).
#    - Pruning: 
#        - If len(S) == k: check if valid_mask == 0. If yes, success.
#        - If valid_mask has fewer bits set than (k - len(S)): fail (not enough candidates).
#        - Also, we can prune if the maximum number of points we can possibly add is < k - len(S).
#          (This is just the popcount of valid_mask).
#          
#    To make the search efficient, we should order the candidates.
#    We can try a few random orderings or use a heuristic (e.g., pick point that removes most options first? 
#    Actually, for minimal maximal sets, we want to cover the space quickly. 
#    Picking a point that forbids many other points might help reach "maximal" state faster, 
#    but we need exactly k points. 
#    Let's just use random shuffling for the candidates at each step to avoid bias, 
#    and limit the depth of search per k to a time budget.

def construct():
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    # 1. Precomputation
    # conflict[i][j] is a bitmask of length TOTAL_POINTS
    # conflict[i][j] bit k is 1 if (i, j, k) is isosceles.
    # Note: (i, j, k) is isosceles if dist(i,j)==dist(i,k) or dist(i,j)==dist(j,k) or dist(j,k)==dist(i,k).
    
    # Precompute squared distances? No, just compute on the fly for precomputation.
    # 256 points. 256*256 pairs. 256 checks per pair. 256^3 = 16M ops. Fast enough in Python? 
    # 16M simple integer ops might take ~1-2 seconds in Python. Acceptable.
    
    points = [idx_to_point(i) for i in range(TOTAL_POINTS)]
    
    # Precompute distances between all pairs to speed up isosceles check?
    # dist[i][j] = squared distance
    dist = [[0]*TOTAL_POINTS for _ in range(TOTAL_POINTS)]
    for i in range(TOTAL_POINTS):
        xi, yi = points[i]
        for j in range(i+1, TOTAL_POINTS):
            xj, yj = points[j]
            d = (xi-xj)**2 + (yi-yj)**2
            dist[i][j] = d
            dist[j][i] = d
    
    conflict = [[0]*TOTAL_POINTS for _ in range(TOTAL_POINTS)]
    
    for i in range(TOTAL_POINTS):
        for j in range(i+1, TOTAL_POINTS):
            mask = 0
            dij = dist[i][j]
            for k in range(TOTAL_POINTS):
                if k == i or k == j:
                    continue
                dik = dist[i][k]
                djk = dist[j][k]
                if dij == dik or dij == djk or dik == djk:
                    mask |= (1 << k)
            conflict[i][j] = mask
            conflict[j][i] = mask
            
    # 2. Search for minimal k
    best_solution = None
    
    # We need to find a set S of size k such that no point can be added.
    # This means the "forbidden" set covers all other points.
    # Forbidden set for S is union of conflict[a][b] for all a,b in S.
    # Plus S itself.
    # If (Forbidden | S) == All Points, then S is maximal.
    
    # Let's use a recursive search with pruning.
    
    def search(k, time_limit):
        # Returns a list of indices of size k that form a maximal set, or None
        initial_mask = (1 << TOTAL_POINTS) - 1
        
        # We will use a stack-based DFS or recursion.
        # To allow randomization, we shuffle the candidate list at each node.
        
        def dfs(S, valid_mask, depth):
            # S: list of indices in current set
            # valid_mask: bitmask of points that can be added to S without violating isosceles constraint
            # depth: number of points still to add
            
            if time.time() > time_limit:
                return None
            
            if depth == 0:
                # Check if S is maximal
                # Maximal means no point outside S can be added.
                # Points outside S that can be added are those in valid_mask.
                # Note: valid_mask does not include points in S? 
                # We should remove S from valid_mask when adding to S.
                # Let's ensure valid_mask excludes S.
                if valid_mask == 0:
                    return list(S)
                return None
            
            # Pruning: if number of available points < depth, fail
            # popcount in Python 3.10+ is int.bit_count()
            if valid_mask.bit_count() < depth:
                return None
                
            # Get list of candidates
            candidates = []
            m = valid_mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                candidates.append(idx)
                m ^= lsb
                
            # Randomize order to find diverse solutions
            random.shuffle(candidates)
            
            for p in candidates:
                # Try adding p
                new_S = S + [p]
                
                # Update valid_mask
                # New valid_mask = valid_mask & ~(union of conflict[p][q] for q in S)
                # Also remove p from valid_mask (though it's already there, we add to S)
                # Actually, p is in valid_mask. We remove it.
                # And we remove all points forbidden by pairs (p, q) for q in S.
                
                new_valid_mask = valid_mask & ~(1 << p)
                
                for q in S:
                    new_valid_mask &= ~conflict[p][q]
                    
                # Optimization: If new_valid_mask becomes 0 and depth > 1, 
                # we can't add more points, so this branch fails unless depth==1 (handled above)
                # Actually, if new_valid_mask is 0, we can only succeed if depth==1 (which is checked at start of next recursion? No, depth decreases by 1).
                # If depth-1 == 0, we check in next call.
                # If depth-1 > 0, we need more points, but valid_mask is 0, so fail.
                if new_valid_mask == 0 and depth > 1:
                    continue
                
                res = dfs(new_S, new_valid_mask, depth - 1)
                if res is not None:
                    return res
                    
            return None

        return dfs([], initial_mask, k)

    # Iterate k from 1 upwards
    # We expect the answer to be small. 
    # For N=16, a random greedy set might be ~20-30. Minimal maximal is likely smaller.
    # Let's try up to 20 first, then increase if time permits.
    
    current_k = 1
    while time.time() < deadline:
        # Allocate time budget for this k
        # If k is small, search is fast. If k is larger, search is harder.
        # Let's give 5 seconds per k, or scale it.
        time_budget = 5.0
        t_start = time.time()
        
        # Try multiple random restarts for this k if it fails quickly?
        # Or just one long search?
        # One long search with randomization is usually better for finding *a* solution.
        
        sol = search(current_k, t_start + time_budget)
        
        if sol is not None:
            # Verify solution just in case
            # Check isosceles
            is_valid = True
            for i in range(len(sol)):
                for j in range(i+1, len(sol)):
                    for k in range(j+1, len(sol)):
                        a, b, c = sol[i], sol[j], sol[k]
                        if dist[a][b] == dist[a][c] or dist[a][b] == dist[b][c] or dist[a][c] == dist[b][c]:
                            is_valid = False
                            break
                    if not is_valid: break
                if not is_valid: break
            
            # Check maximality
            if is_valid:
                # Compute forbidden mask
                forbidden = 0
                for i in range(len(sol)):
                    for j in range(i+1, len(sol)):
                        forbidden |= conflict[sol[i]][sol[j]]
                # Add points in sol to forbidden
                for p in sol:
                    forbidden |= (1 << p)
                
                if forbidden == (1 << TOTAL_POINTS) - 1:
                    # Found a valid minimal maximal set of size current_k
                    # Convert indices to points
                    return [idx_to_point(i) for i in sol]
        
        current_k += 1
        
        # If we've tried up to 30 and didn't find, we might need a fallback
        # But the loop continues until deadline.
        
    # Fallback: if no solution found via search, use a greedy construction
    # This should not happen for N=16, but good for safety.
    # Greedy random
    best_greedy = []
    for _ in range(100):
        order = list(range(TOTAL_POINTS))
        random.shuffle(order)
        chosen = []
        valid_mask = (1 << TOTAL_POINTS) - 1
        for p in order:
            if valid_mask & (1 << p):
                chosen.append(p)
                new_mask = valid_mask & ~(1 << p)
                for q in chosen[:-1]:
                    new_mask &= ~conflict[p][q]
                valid_mask = new_mask
                if valid_mask == 0:
                    break
        if len(chosen) < len(best_greedy) or not best_greedy:
            best_greedy = chosen
            
    return [idx_to_point(i) for i in best_greedy]

# EVOLVE-BLOCK-END
