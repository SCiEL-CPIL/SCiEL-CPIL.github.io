import time
import itertools
import sys
from collections import defaultdict

N = 16
TOTAL_POINTS = N * N

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 330s kill

    # Indexing: point (x, y) -> x * N + y
    points = list(itertools.product(range(N), repeat=2))
    idx_to_pt = {i: p for i, p in enumerate(points)}
    pt_to_idx = {p: i for i, p in enumerate(points)}
    
    # Precompute squared distances between all pairs
    # dist2[i][j] = squared distance between point i and point j
    dist2 = [[0] * TOTAL_POINTS for _ in range(TOTAL_POINTS)]
    
    for i in range(TOTAL_POINTS):
        ax, ay = idx_to_pt[i]
        for j in range(i + 1, TOTAL_POINTS):
            bx, by = idx_to_pt[j]
            d = (ax - bx)**2 + (ay - by)**2
            dist2[i][j] = d
            dist2[j][i] = d

    # Precompute isosceles conflicts
    # For each pair (i, j), find all k such that {i, j, k} is isosceles
    # iso_mask[i][j] is a bitmask of points k (k != i, k != j) such that
    # d(i,k) == d(i,j) or d(j,k) == d(i,j) or d(i,k) == d(j,k)
    
    iso_mask = [[0] * TOTAL_POINTS for _ in range(TOTAL_POINTS)]
    
    for i in range(TOTAL_POINTS):
        for j in range(i + 1, TOTAL_POINTS):
            d_ij = dist2[i][j]
            if d_ij == 0:
                continue  # Same point, skip
            
            mask = 0
            for k in range(TOTAL_POINTS):
                if k == i or k == j:
                    continue
                d_ik = dist2[i][k]
                d_jk = dist2[j][k]
                if d_ik == d_ij or d_jk == d_ij or d_ik == d_jk:
                    mask |= (1 << k)
            
            iso_mask[i][j] = mask
            iso_mask[j][i] = mask

    all_mask = (1 << TOTAL_POINTS) - 1
    
    best_set_mask = None
    best_size = TOTAL_POINTS + 1

    # Precompute for each point p, the list of points it conflicts with
    # conflict_mask[p] = bitmask of all points q such that adding p would block q
    # But this depends on the chosen set. Let's precompute pairwise conflicts.
    
    # For a given chosen set S, a point p can be added if:
    # 1. p not in S
    # 2. For all q in S, p is not in iso_mask[q][p] (equivalently q is not in iso_mask[p][q])
    #    Actually, we need to check if {p, q, r} is isosceles for any r in S.
    #    This is equivalent to: for all q in S, the mask iso_mask[p][q] should not contain any other point in S?
    #    No, that's not right. The condition is that no THREE points in the final set form an isosceles triangle.
    #    So when we add p to S, we need to check that for all pairs (q, r) in S, p is not in iso_mask[q][r].
    #    Also, for all q in S, the pair (p, q) should not form an isosceles triangle with any other point in S?
    #    Actually, the condition is: for all q, r in S (q != r), p is not in iso_mask[q][r].
    #    AND for all q in S, the pair (p, q) is fine as long as no third point completes it.
    #    Wait, the conflict mask iso_mask[i][j] already contains all k that complete an isosceles triangle with i and j.
    #    So if we have a set S, it is valid (no-isosceles) iff for all i, j in S (i != j), 
    #    iso_mask[i][j] & (mask of S) == 0 (excluding i and j themselves, but they are not in the mask anyway).
    
    # So when adding p to S, we need to check:
    # For all q in S: (iso_mask[p][q] & mask(S)) == 0
    # AND (iso_mask[q][p] & mask(S)) == 0 -- but iso_mask is symmetric, so just one check.
    # Actually, iso_mask[p][q] is symmetric in the sense that the set of conflicts is the same.
    # Let's just check: for all q in S, iso_mask[p][q] & mask(S) == 0.
    
    # Precompute: for each point p, and for each possible "blocked by" scenario?
    # Let's define a function to check if p can be added to S.
    
    def can_add(p, chosen_mask):
        # Check if adding p to chosen_mask keeps the set no-isosceles
        # For all q in chosen_mask, iso_mask[p][q] should not intersect with chosen_mask
        tmp = chosen_mask
        while tmp:
            lsb = tmp & -tmp
            q = lsb.bit_length() - 1
            if iso_mask[p][q] & chosen_mask:
                return False
            tmp ^= lsb
        return True

    # Greedy construction to get an initial valid maximal set
    def greedy_construct():
        chosen_mask = 0
        chosen_list = []
        
        # Shuffle points for diversity? Or just iterate?
        # Let's try multiple greedy passes with different orderings.
        import random
        random.seed(42)
        
        best_greedy_mask = None
        best_greedy_size = TOTAL_POINTS + 1
        
        for trial in range(20):
            order = list(range(TOTAL_POINTS))
            random.shuffle(order)
            
            c_mask = 0
            c_list = []
            for p in order:
                if c_mask & (1 << p):
                    continue
                if can_add(p, c_mask):
                    c_mask |= (1 << p)
                    c_list.append(p)
            
            # Now make it maximal by adding any point that can be added
            changed = True
            while changed:
                changed = False
                for p in range(TOTAL_POINTS):
                    if c_mask & (1 << p):
                        continue
                    if can_add(p, c_mask):
                        c_mask |= (1 << p)
                        c_list.append(p)
                        changed = True
            
            size = len(c_list)
            if size < best_greedy_size:
                best_greedy_size = size
                best_greedy_mask = c_mask
        
        return best_greedy_mask, best_greedy_size

    initial_mask, initial_size = greedy_construct()
    best_set_mask = initial_mask
    best_size = initial_size

    # Local search: try to remove points while maintaining maximality
    # If we remove p from S, S' = S \ {p}.
    # S' is maximal if no point can be added to S'.
    # A point q can be added to S' if can_add(q, S') is true.
    # If S' is not maximal, we must keep p (or find a replacement).
    # Strategy: try to remove each point. If removal makes it non-maximal, 
    # try to add a different point to restore maximality, then check size.
    
    # Actually, a simpler approach:
    # 1. Start with a valid maximal set S.
    # 2. For each p in S, try to remove it.
    #    If S\{p} is still maximal, keep the removal.
    #    If not, check if we can add some q not in S\{p} such that S\{p} U {q} is maximal.
    #    If so, we have the same size. No gain.
    #    So we only gain if S\{p} is maximal.
    
    # This is a "descent" method. Let's implement it.
    
    def is_maximal(chosen_mask):
        # Check if no point can be added
        for p in range(TOTAL_POINTS):
            if chosen_mask & (1 << p):
                continue
            if can_add(p, chosen_mask):
                return False
        return True

    # Iterative removal
    changed = True
    while changed:
        changed = False
        S_list = []
        tmp = best_set_mask
        while tmp:
            lsb = tmp & -tmp
            idx = lsb.bit_length() - 1
            S_list.append(idx)
            tmp ^= lsb
        
        for p in S_list:
            if time.time() > deadline:
                break
            
            new_mask = best_set_mask & ~(1 << p)
            if is_maximal(new_mask):
                best_set_mask = new_mask
                best_size -= 1
                changed = True
                break  # Restart the loop with the new set
        
        if time.time() > deadline:
            break

    # Local search with swaps
    # Try to swap p (in S) with q (not in S) if |S| decreases? No, swap keeps size.
    # We want to decrease size. So we only care about removals.
    # But maybe a swap enables a future removal?
    # Let's try: for each p in S, for each q not in S, if S\{p} U {q} is maximal, 
    # and then we can remove some r from S\{p} U {q}?
    # This is getting complex. Let's stick to simple removals for now.
    
    # Another idea: Simulated Annealing on the set of points.
    # State: chosen_mask
    # Move: add a point, remove a point, or swap
    # Cost: size of set + penalty * (1 if not maximal else 0)
    # But we must maintain validity (no-isosceles).
    
    # Let's try a simple random walk with restarts.
    # 1. Generate a random valid set (not necessarily maximal).
    # 2. Make it maximal.
    # 3. Try to remove points.
    
    import random
    random.seed(43)
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Generate random valid set
        chosen_mask = 0
        available = list(range(TOTAL_POINTS))
        random.shuffle(available)
        
        for p in available:
            if can_add(p, chosen_mask):
                chosen_mask |= (1 << p)
        
        # Make maximal
        changed = True
        while changed:
            changed = False
            for p in range(TOTAL_POINTS):
                if chosen_mask & (1 << p):
                    continue
                if can_add(p, chosen_mask):
                    chosen_mask |= (1 << p)
                    changed = True
        
        size = bin(chosen_mask).count('1')
        
        # Try to reduce size
        for _ in range(100):
            if time.time() > deadline:
                break
            
            S_list = []
            tmp = chosen_mask
            while tmp:
                lsb = tmp & -tmp
                idx = lsb.bit_length() - 1
                S_list.append(idx)
                tmp ^= lsb
            
            if not S_list:
                break
            
            p = random.choice(S_list)
            new_mask = chosen_mask & ~(1 << p)
            
            if is_maximal(new_mask):
                chosen_mask = new_mask
                size -= 1
            else:
                # Try to add a point to restore maximality?
                # If we can add a point q such that new_mask U {q} is maximal, 
                # we have the same size. No gain.
                # So we just skip this removal.
                pass
        
        if size < best_size:
            best_size = size
            best_set_mask = chosen_mask

    # Convert mask to list of tuples
    result = []
    tmp = best_set_mask
    while tmp:
        lsb = tmp & -tmp
        idx = lsb.bit_length() - 1
        result.append(idx_to_pt[idx])
        tmp ^= lsb
    
    return result
