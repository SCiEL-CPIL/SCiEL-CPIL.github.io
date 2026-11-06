"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

def _precompute_dist2():
    """Precompute squared distances between all pairs of grid points."""
    points = list(itertools.product(range(N), repeat=2))
    V = len(points)
    dist2 = np.zeros((V, V), dtype=np.int64)
    for i in range(V):
        xi, yi = points[i]
        for j in range(i + 1, V):
            xj, yj = points[j]
            d = (xi - xj) ** 2 + (yi - yj) ** 2
            dist2[i, j] = d
            dist2[j, i] = d
    return points, dist2

# Precompute at module load time
POINTS, DIST2 = _precompute_dist2()
V = len(POINTS)

def _is_blocked(u, S_indices):
    """
    Check if point u is blocked by the set S_indices.
    u is blocked if there exist a, b in S_indices such that triangle (u, a, b) is isosceles.
    Note: If u is in S_indices, it is trivially "handled" (part of the set), 
    but for the purpose of finding a maximal set, we need every point NOT in S to be blocked.
    """
    S_list = list(S_indices)
    n = len(S_list)
    if n < 2:
        return False
    
    # We need to check if for any pair (a, b) in S, d(u,a)==d(u,b) or d(u,a)==d(a,b) or d(u,b)==d(a,b)
    # This is O(n^2) per check.
    # Optimization: Precompute distances from u to all points in S.
    # d_u = [DIST2[u, a] for a in S_list]
    # Check 1: Any duplicate in d_u?
    # Check 2: For each a in S, is d(u,a) in {d(a,b) for b in S, b!=a}?
    
    # Let's just do the direct check, it's fast enough for small S.
    for i in range(n):
        a = S_list[i]
        d_ua = DIST2[u, a]
        for j in range(i + 1, n):
            b = S_list[j]
            d_ub = DIST2[u, b]
            d_ab = DIST2[a, b]
            if d_ua == d_ub or d_ua == d_ab or d_ub == d_ab:
                return True
    return False

def _count_blocked(S_indices):
    """
    Count how many points in the grid are blocked by S_indices.
    A point u is considered 'covered' if u is in S_indices OR if u is blocked by S_indices.
    """
    S_set = set(S_indices)
    covered_count = len(S_set)
    for u in range(V):
        if u in S_set:
            continue
        if _is_blocked(u, S_indices):
            covered_count += 1
    return covered_count

def construct():
    start_time = time.time()
    deadline = start_time + 295.0 # 5s buffer for safety
    
    best_size = V
    best_set = []
    
    # Strategy: Greedy Set Cover with Randomized Restarts
    # We want to pick points that cover (block) the most uncovered points.
    
    for restart in range(10000):
        if time.time() > deadline:
            break
            
        current_set = []
        current_set_set = set()
        # Track which points are currently covered (in set or blocked)
        # Initially, nothing is covered.
        
        # To make it randomized, we can shuffle the order of tie-breaking or start with random seeds.
        # Let's try a simple greedy: at each step, pick the point p that maximizes the number of 
        # currently UNCOVERED points that would become covered if p were added.
        
        # Pre-compute potential gains? 
        # gain(p) = number of u not in current_set_set and not blocked by current_set 
        #           such that u becomes blocked by (current_set U {p}) OR u == p.
        
        # Since V=256, and set size is small (<20), we can afford O(V * |S|^2) per step.
        
        while True:
            if time.time() > deadline:
                break
                
            # Check if all points are covered
            # If yes, we have a maximal set.
            all_covered = True
            for u in range(V):
                if u not in current_set_set:
                    if not _is_blocked(u, current_set):
                        all_covered = False
                        break
            
            if all_covered:
                if len(current_set) < best_size:
                    best_size = len(current_set)
                    best_set = current_set[:]
                break
            
            # Find the point p that maximizes coverage gain
            best_gain = -1
            best_p = -1
            
            # We can optimize by only considering points that are not yet in the set.
            # Also, if a point is already blocked, adding it doesn't help cover others directly 
            # (it might help block others, but usually the points that block the most are those 
            # that form many isosceles triangles with existing points).
            
            # Iterate through all candidate points p not in current_set
            for p in range(V):
                if p in current_set_set:
                    continue
                
                # Calculate how many new points are covered if we add p
                # A point u is newly covered if:
                # 1. u == p
                # 2. u was not covered before, but is blocked by (current_set U {p})
                
                # This check is expensive if done for all u.
                # Approximation: 
                # The primary way p helps is by forming isosceles triangles with pairs in current_set.
                # Let's just compute the gain for a subset of candidates or use a heuristic.
                
                # Heuristic: Pick p that is "far" from existing points in a way that creates many equal distances?
                # Actually, let's just do the full check for all p, but optimize the inner loop.
                
                # To speed this up, we can precompute for the current set S, 
                # the set of points that are "almost blocked" or just check the gain.
                
                # Let's stick to the simple O(V^2 * |S|^2) per step, which is 256^2 * 400 ~ 25M ops.
                # Might be too slow for 10000 restarts.
                
                # Optimization: 
                # Only evaluate p if it is not already "useless".
                # Let's limit the search space. 
                # Actually, let's just try a few random candidates plus the best greedy choice.
                
                pass # Placeholder, will implement specific logic below
            
            # Implementation of the greedy step with optimization:
            # We will evaluate all p not in S.
            # For each p, we check how many u (not in S) become blocked.
            
            # To make it faster, we can note that if |S| is small, _is_blocked is fast.
            # Let's just run it. If it's too slow, we reduce restarts.
            
            for p in range(V):
                if p in current_set_set:
                    continue
                
                gain = 1 # p itself is covered
                # Check which other points u become blocked
                # u is blocked by S U {p} if:
                # - u is blocked by S (already covered, so gain 0)
                # - OR u forms isosceles with a,b in S where one is p? No, p is new.
                # - OR u forms isosceles with p and a in S: d(u,p)==d(u,a) or d(u,p)==d(p,a) or d(u,a)==d(p,a)
                
                # Let's check for each u not in S:
                for u in range(V):
                    if u in current_set_set or u == p:
                        continue
                    
                    # If u was already blocked by S, it doesn't count as new gain.
                    # But checking "was already blocked" is expensive.
                    # Let's just check if u is blocked by S U {p}.
                    # And assume if it wasn't blocked by S, it's a gain.
                    # We can track a `covered` boolean array to avoid re-checking.
                    
                    # Let's maintain a `covered` array outside the loop.
                    pass
            
            # This inner loop is too slow. 
            # Alternative strategy: 
            # Instead of full greedy set cover, use a "Randomized Local Search" on the set S.
            # Start with a random valid set. Try to remove points. If the set remains maximal, keep the removal.
            # This is "Minimization by Deletion".
            
            # Let's switch to Deletion Heuristic.
            # 1. Generate a maximal set using the old greedy method (fast).
            # 2. Try to remove each point one by one.
            # 3. If removing a point keeps the set maximal (i.e., the point was redundant for blocking), remove it.
            # 4. Repeat until no more points can be removed.
            
            # This is a standard heuristic for minimal maximal sets.
            
            # Step 1: Generate an initial maximal set
            # Use the previous logic to get a maximal set quickly.
            order = list(range(V))
            random.shuffle(order)
            init_set = []
            init_set_set = set()
            
            for idx in order:
                if _check_isosceles_with_new_point(idx, init_set_set):
                    init_set.append(idx)
                    init_set_set.add(idx)
            
            # Now, try to minimize init_set
            S = init_set[:]
            S_set = set(S)
            changed = True
            while changed:
                changed = False
                for i in range(len(S)):
                    p = S[i]
                    # Try removing p
                    S_set.remove(p)
                    
                    # Check if the set S_set is still maximal.
                    # i.e., is every point u not in S_set blocked by S_set?
                    is_maximal = True
                    for u in range(V):
                        if u in S_set:
                            continue
                        if not _is_blocked(u, S_set):
                            is_maximal = False
                            break
                    
                    if is_maximal:
                        # We can remove p
                        S.pop(i)
                        changed = True
                        break # Restart the loop from the beginning as indices changed
                    else:
                        # Restore p
                        S_set.add(p)
                
                # If we went through the whole list without removing, we are done.
            
            if len(S) < best_size:
                best_size = len(S)
                best_set = S[:]
                if best_size <= 8:
                    break

    return [POINTS[i] for i in best_set]

def _check_isosceles_with_new_point(new_idx, selected_indices):
    """
    Check if adding new_idx to selected_indices creates any isosceles triangle.
    """
    if len(selected_indices) < 2:
        return True
    
    S = list(selected_indices)
    
    # Check condition 1: d(new, a) distinct for all a in S
    d_new_to_S = {}
    for a in S:
        d = DIST2[new_idx, a]
        if d in d_new_to_S:
            return False
        d_new_to_S[d] = a
        
    # Check condition 2: d(new, a) != d(a, b) for all a, b in S, a != b
    for a in S:
        d_na = DIST2[new_idx, a]
        for b in S:
            if a == b:
                continue
            d_ab = DIST2[a, b]
            if d_na == d_ab:
                return False
                
    return True

# EVOLVE-BLOCK-END
