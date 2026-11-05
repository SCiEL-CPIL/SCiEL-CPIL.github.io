import random
import time
import itertools
import math
import numpy as np

N = 16
ALL_POINTS = list(itertools.product(range(N), repeat=2))
N_PTS = len(ALL_POINTS)

# Precompute squared distances
coords = np.array(ALL_POINTS)
diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
dist2 = np.sum(diff**2, axis=2)

# Precompute isosceles conditions
# For each point p, we want to know which pairs (u, v) form an isosceles triangle with p.
# Specifically, we need to efficiently check if a set S contains a pair (u, v) such that {p, u, v} is isosceles.
# This is equivalent to: Does S intersect the set of points that form an isosceles triangle with p and some other point in S?
# Let's precompute for each p, a list of neighbors u such that dist(p,u) == dist(p,v) for some v? No.
# Condition for {p, u, v} to be isosceles:
# dist(p,u)^2 == dist(p,v)^2 OR dist(p,u)^2 == dist(u,v)^2 OR dist(p,v)^2 == dist(u,v)^2

# To speed up "is p addable to S":
# We need to check if there exist u, v in S such that {p, u, v} is isosceles.
# Let's precompute for each p, a set of "conflict pairs" or use a bitset approach.
# Given N=256, a bitset is 4 uint64s.
# For each p, let's precompute a list of u's.
# Actually, simpler: Precompute for each p, a boolean matrix M_p[u][v] = True if {p,u,v} is isosceles.
# This is 256 * 256 * 256 booleans = 16 million booleans. 2MB. Very manageable.
# M[p] is a 256x256 boolean matrix.

M = np.zeros((N_PTS, N_PTS, N_PTS), dtype=bool)
for p in range(N_PTS):
    dp = dist2[p, :]
    for u in range(N_PTS):
        if u == p:
            continue
        du = dist2[u, :]
        # Check dist(p,u) == dist(p,v)
        # This is dp[u] == dp[v]
        # Check dist(p,u) == dist(u,v)
        # This is dp[u] == du[v]
        # Check dist(p,v) == dist(u,v)
        # This is dp[v] == du[v]
        
        # Vectorized check for fixed p, u
        # v ranges 0..N_PTS
        # mask1: dp[u] == dp[v]
        # mask2: dp[u] == du[v]
        # mask3: dp[v] == du[v]
        
        # We only care if there exists v in S.
        # So for a given S (bitset), we check if (M[p, u, :] & S).any()
        
        M[p, u, :] = (dp[u] == dp) | (dp[u] == du) | (dp == du)
        # Exclude u and p themselves from being the "third point" in a degenerate sense? 
        # The problem says "distinct A, B, C". 
        # If v == u, dist(u,v)=0. 
        # If v == p, dist(p,v)=0.
        # We should mask out u and p in the v dimension for M[p, u, :].
        M[p, u, u] = False
        M[p, u, p] = False

# For faster lookup, we can convert M[p, u] to a list of v's or keep as numpy array.
# Since we will do bitset operations, let's convert to Python integers (bitsets).
# 256 bits fits in 4 uint64 or just Python int.
# M_bit[p][u] is an int where bit v is 1 if {p,u,v} is isosceles.

M_bit = [[0 for _ in range(N_PTS)] for _ in range(N_PTS)]
for p in range(N_PTS):
    for u in range(N_PTS):
        mask = M[p, u, :]
        val = 0
        for v in range(N_PTS):
            if mask[v]:
                val |= (1 << v)
        M_bit[p][u] = val

def _is_addable(p, S_bit):
    """
    Check if p can be added to S (represented as a bitset int).
    Returns True if no isosceles triangle is formed.
    """
    for u in range(N_PTS):
        if (S_bit >> u) & 1:
            # Check if there is any v in S such that {p, u, v} is isosceles
            # M_bit[p][u] has bits set for all such v.
            # We need to check if (M_bit[p][u] & S_bit) != 0
            # Note: M_bit[p][u] might have bit u or p set? We masked them out in precomp.
            # But S_bit contains u. If M_bit[p][u] had bit u, it would be a false positive?
            # We set M[p,u,u]=False and M[p,u,p]=False.
            # So if (M_bit[p][u] & S_bit) != 0, there is a v in S (v != u, v != p) such that {p,u,v} is iso.
            if M_bit[p][u] & S_bit:
                return False
    return True

def _get_addable_set(S_bit):
    """
    Get a set of points that can be added to S.
    """
    addable = set()
    for p in range(N_PTS):
        if not ((S_bit >> p) & 1):
            if _is_addable(p, S_bit):
                addable.add(p)
    return addable

def _greedy_complete(S_bit, deadline):
    """
    Greedily add points to S to make it maximal.
    Strategy: Pick the point p that blocks the most other addable points.
    """
    addable = _get_addable_set(S_bit)
    
    while addable:
        if time.time() > deadline:
            break
        
        best_p = -1
        max_blocked = -1
        best_new_addable = set()
        
        # Heuristic: Try a few random candidates if the set is large, or all if small
        # For N=256, iterating all is O(N^2 * cost_check)
        # cost_check is O(N) for _is_addable? No, _is_addable is O(N) bit ops.
        # So total O(N^3) bit ops? 
        # Let's just iterate all for now, it's fast enough in C via numpy/int ops?
        # Python int bit ops are fast.
        
        candidates = list(addable)
        random.shuffle(candidates)
        
        for p in candidates:
            # Simulate adding p
            new_S_bit = S_bit | (1 << p)
            new_addable = set()
            for q in addable:
                if q == p:
                    continue
                if _is_addable(q, new_S_bit):
                    new_addable.add(q)
            
            # We want to minimize the size of new_addable (maximize blocking)
            # Or maximize reduction in len(addable)
            reduction = len(addable) - len(new_addable)
            if reduction > max_blocked:
                max_blocked = reduction
                best_p = p
                best_new_addable = new_addable
            
            # Early break if we blocked everyone?
            if not best_new_addable:
                break

        if best_p == -1:
            break
            
        S_bit |= (1 << best_p)
        addable = best_new_addable
        
    return S_bit

def construct():
    start_time = time.time()
    deadline = start_time + 300.0
    
    best_S_bit = 0
    best_size = float('inf')
    
    # Initial solution: Greedy from empty
    S_bit = _greedy_complete(0, deadline - 10.0)
    best_S_bit = S_bit
    best_size = bin(S_bit).count('1')
    
    # Local Search: Delete-and-Reconstruct
    while time.time() < deadline - 5.0:
        # Get list of points in current best set
        S_list = [i for i in range(N_PTS) if (best_S_bit >> i) & 1]
        
        if not S_list:
            break
            
        # Pick a random point to remove
        # Or iterate through all? Iterating all is O(N * GreedyCost).
        # GreedyCost is high. Let's try a few random removals per iteration.
        num_trials = 5
        improved = False
        
        for _ in range(num_trials):
            if time.time() > deadline - 5.0:
                break
                
            if not S_list:
                break
                
            idx = random.randint(0, len(S_list) - 1)
            p_remove = S_list[idx]
            
            # Remove p_remove
            S_trial = best_S_bit & ~(1 << p_remove)
            
            # Complete to maximal
            S_new = _greedy_complete(S_trial, deadline - 2.0)
            
            size_new = bin(S_new).count('1')
            
            if size_new < best_size:
                best_size = size_new
                best_S_bit = S_new
                improved = True
                # Update S_list for next iteration in this loop?
                # We can break and restart the while loop to get fresh S_list
                break
        
        if improved:
            continue
            
        # If no improvement in this batch, we might get stuck.
        # Try a random restart or perturbation?
        # For now, just keep trying random removals.
        # To escape local minima, we could accept equal size moves sometimes?
        # Or try removing 2 points?
        pass

    # Convert best_S_bit to list of points
    result = [ALL_POINTS[i] for i in range(N_PTS) if (best_S_bit >> i) & 1]
    return result
