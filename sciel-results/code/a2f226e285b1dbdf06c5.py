import random
from math import gcd
import time
from itertools import combinations

N = 128

def compute_bad_triples():
    """
    Precompute all bad triples (a, b, c) with a < b < c
    where gcd(a,b) = gcd(b,c) = gcd(a,c)
    """
    bad_triples = set()
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g_ab = gcd(a, b)
            for c in range(b + 1, N + 1):
                if gcd(a, c) == g_ab and gcd(b, c) == g_ab:
                    bad_triples.add((a, b, c))
    return bad_triples

BAD_TRIPLES = compute_bad_triples()

# Build for each element, the bitmask of pairs that form a bad triple with it
# conflict_mask[x] has bits set for elements y such that (x,y) is part of a bad triple
# Actually, we need: for element x, which pairs (u,v) in the current set would conflict?
# Better: for each x, precompute a list of (u, v) pairs where x,u,v form a bad triple
element_pairs = {i: [] for i in range(1, N + 1)}
for (a, b, c) in BAD_TRIPLES:
    element_pairs[a].append((b, c))
    element_pairs[b].append((a, c))
    element_pairs[c].append((a, b))

# For bitmask-based checking:
# For each x, precompute bitmask where bit i is set if i is in some pair with x
# Actually, we need: given current set S (as bitmask), can we add x?
# x conflicts with S if there exist u,v in S such that (x,u,v) is a bad triple
# This means: for some (u,v) in element_pairs[x], both u and v are in S
# We can precompute for each x a list of bitmasks representing the pairs

# Precompute: for each x, a list of (mask_u, mask_v) where mask_u = 1<<u, mask_v = 1<<v
# Then x can be added to S if for all (mu, mv) in pairs: not (S & mu and S & mv)
# Even better: precompute for each x a single bitmask conflict_mask[x] such that
# x conflicts with S iff (S & conflict_mask[x]) has at least 2 bits set from the same pair
# This is complex. Let's use a different approach:
# For each x, precompute list of pair_bitmasks where pair_bitmask = (1<<u) | (1<<v)
# x can be added to S iff for all pb in pair_bitmasks[x]: not (S & pb == pb)

pair_bitmasks = {i: [] for i in range(1, N + 1)}
for x in range(1, N + 1):
    for (u, v) in element_pairs[x]:
        pair_bitmasks[x].append((1 << u) | (1 << v))

def can_add_bitmask(x, current_mask):
    """Check if x can be added to the set represented by current_mask."""
    for pb in pair_bitmasks[x]:
        if (current_mask & pb) == pb:
            return False
    return True

def greedy_build_bitmask(order):
    """Build a set greedily following the given order, using bitmask."""
    mask = 0
    for x in order:
        if can_add_bitmask(x, mask):
            mask |= (1 << x)
    return mask

def local_search_bitmask(mask, deadline, max_iter=100):
    """
    Improve set represented by bitmask using local search.
    """
    improved = True
    iter_count = 0
    while improved and time.time() < deadline and iter_count < max_iter:
        improved = False
        iter_count += 1
        
        # 1. Try to add any single element
        candidates = [x for x in range(1, N + 1) if not (mask & (1 << x))]
        random.shuffle(candidates)
        for x in candidates:
            if time.time() > deadline:
                return mask
            if can_add_bitmask(x, mask):
                mask |= (1 << x)
                improved = True
                break
        
        if improved:
            continue
            
        # 2. Try 1-for-1 swap: remove one, add one
        list_s = [x for x in range(1, N + 1) if (mask & (1 << x))]
        random.shuffle(list_s)
        
        for u in list_s:
            if time.time() > deadline:
                return mask
            
            mask_without_u = mask & ~(1 << u)
            
            addable = [x for x in range(1, N + 1) if not (mask & (1 << x)) and x != u]
            random.shuffle(addable)
            
            for x in addable:
                if time.time() > deadline:
                    return mask
                if can_add_bitmask(x, mask_without_u):
                    mask = mask_without_u | (1 << x)
                    improved = True
                    break
            
            if improved:
                break
        
        # 3. Try 2-for-3 swap: remove two, add three (increase size by 1)
        if not improved and time.time() < deadline:
            list_s = [x for x in range(1, N + 1) if (mask & (1 << x))]
            random.shuffle(list_s)
            for i in range(min(len(list_s), 12)):
                u = list_s[i]
                for j in range(i + 1, min(len(list_s), 12)):
                    v = list_s[j]
                    if time.time() > deadline:
                        return mask
                    mask_without_uv = mask & ~(1 << u) & ~(1 << v)
                    addable = [x for x in range(1, N + 1) if not (mask & (1 << x)) and x != u and x != v]
                    random.shuffle(addable)
                    
                    # Try to find three elements that can all be added
                    found_triple = False
                    for i1 in range(min(len(addable), 25)):
                        if time.time() > deadline:
                            return mask
                        x1 = addable[i1]
                        if can_add_bitmask(x1, mask_without_uv):
                            mask_with_x1 = mask_without_uv | (1 << x1)
                            for i2 in range(i1 + 1, min(len(addable), 25)):
                                x2 = addable[i2]
                                if can_add_bitmask(x2, mask_with_x1):
                                    mask_with_x1x2 = mask_with_x1 | (1 << x2)
                                    for i3 in range(i2 + 1, min(len(addable), 25)):
                                        x3 = addable[i3]
                                        if can_add_bitmask(x3, mask_with_x1x2):
                                            mask = mask_without_uv | (1 << x1) | (1 << x2) | (1 << x3)
                                            found_triple = True
                                            improved = True
                                            break
                                    if improved:
                                        break
                                if improved:
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        
        # 4. Try 3-for-4 swap: remove three, add four (increase size by 1)
        if not improved and time.time() < deadline:
            list_s = [x for x in range(1, N + 1) if (mask & (1 << x))]
            random.shuffle(list_s)
            for i in range(min(len(list_s), 8)):
                u = list_s[i]
                for j in range(i + 1, min(len(list_s), 8)):
                    v = list_s[j]
                    for k in range(j + 1, min(len(list_s), 8)):
                        w = list_s[k]
                        if time.time() > deadline:
                            return mask
                        mask_without_uvw = mask & ~(1 << u) & ~(1 << v) & ~(1 << w)
                        addable = [x for x in range(1, N + 1) if not (mask & (1 << x)) and x != u and x != v and x != w]
                        random.shuffle(addable)
                        
                        # Try to find four elements that can all be added
                        found_quad = False
                        for i1 in range(min(len(addable), 15)):
                            if time.time() > deadline:
                                return mask
                            x1 = addable[i1]
                            if can_add_bitmask(x1, mask_without_uvw):
                                mask_with_x1 = mask_without_uvw | (1 << x1)
                                for i2 in range(i1 + 1, min(len(addable), 15)):
                                    x2 = addable[i2]
                                    if can_add_bitmask(x2, mask_with_x1):
                                        mask_with_x1x2 = mask_with_x1 | (1 << x2)
                                        for i3 in range(i2 + 1, min(len(addable), 15)):
                                            x3 = addable[i3]
                                            if can_add_bitmask(x3, mask_with_x1x2):
                                                mask_with_x1x2x3 = mask_with_x1x2 | (1 << x3)
                                                for i4 in range(i3 + 1, min(len(addable), 15)):
                                                    x4 = addable[i4]
                                                    if can_add_bitmask(x4, mask_with_x1x2x3):
                                                        mask = mask_without_uvw | (1 << x1) | (1 << x2) | (1 << x3) | (1 << x4)
                                                        found_quad = True
                                                        improved = True
                                                        break
                                                if improved:
                                                    break
                                            if improved:
                                                break
                                        if improved:
                                            break
                                    if improved:
                                        break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

    return mask

def branch_and_bound_polish(mask, deadline, target_size):
    """
    Try to increase the set size from len(mask) to target_size using branch and bound.
    Returns improved mask or original if no improvement found.
    """
    current_size = bin(mask).count('1')
    if current_size >= target_size:
        return mask
    
    # Get elements not in current set
    candidates = [x for x in range(1, N + 1) if not (mask & (1 << x))]
    
    # Sort candidates by how "compatible" they are (fewer conflicts with current set)
    def conflict_count(x, m):
        count = 0
        for pb in pair_bitmasks[x]:
            if (m & pb) == pb:
                count += 1
        return count
    
    candidates.sort(key=lambda x: conflict_count(x, mask))
    
    # Try adding elements one by one, then local search
    best_mask = mask
    best_size = current_size
    
    for x in candidates:
        if time.time() > deadline:
            break
        if can_add_bitmask(x, mask):
            new_mask = mask | (1 << x)
            new_mask = local_search_bitmask(new_mask, deadline, max_iter=20)
            if bin(new_mask).count('1') > best_size:
                best_size = bin(new_mask).count('1')
                best_mask = new_mask
                if best_size >= target_size:
                    break
    
    return best_mask

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    best_mask = 0
    
    # Strategy 1: Greedy with different orderings
    orderings = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        sorted(range(1, N + 1), key=lambda x: len(pair_bitmasks[x])),
        sorted(range(1, N + 1), key=lambda x: -len(pair_bitmasks[x])),
    ]
    
    for order in orderings:
        if time.time() > deadline:
            break
        m = greedy_build_bitmask(order)
        m = local_search_bitmask(m, deadline, max_iter=50)
        if bin(m).count('1') > bin(best_mask).count('1'):
            best_mask = m
    
    # Strategy 2: Random restarts with local search
    while time.time() < deadline - 5.0:
        random_order = list(range(1, N + 1))
        random.shuffle(random_order)
        
        m = greedy_build_bitmask(random_order)
        m = local_search_bitmask(m, deadline, max_iter=30)
        
        if bin(m).count('1') > bin(best_mask).count('1'):
            best_mask = m
    
    # Strategy 3: Branch and bound polish in remaining time
    if time.time() < deadline:
        target_size = bin(best_mask).count('1') + 1
        best_mask = branch_and_bound_polish(best_mask, deadline, target_size)
        # Try again if we improved
        if time.time() < deadline:
            target_size = bin(best_mask).count('1') + 1
            best_mask = branch_and_bound_polish(best_mask, deadline, target_size)
    
    # Convert bitmask back to list
    result = [x for x in range(1, N + 1) if (best_mask & (1 << x))]
    return result
