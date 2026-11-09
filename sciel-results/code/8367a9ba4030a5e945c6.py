import time
import itertools
import random
import sys

N = 16
TOTAL_POINTS = N * N

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    # Indexing: point (x, y) -> x * N + y
    points = list(itertools.product(range(N), repeat=2))
    idx_to_pt = {i: p for i, p in enumerate(points)}
    pt_to_idx = {p: i for i, p in enumerate(points)}
    
    # Precompute covers
    # covers[i][j] is a bitmask of points k such that {i, j, k} is isosceles.
    # Note: k must be distinct from i and j.
    covers = [[0] * TOTAL_POINTS for _ in range(TOTAL_POINTS)]
    
    # Distance squared
    def d2(i, j):
        ax, ay = idx_to_pt[i]
        bx, by = idx_to_pt[j]
        return (ax - bx)**2 + (ay - by)**2

    # Precompute distances for speed? 256x256 matrix of ints.
    dist = [[0] * TOTAL_POINTS for _ in range(TOTAL_POINTS)]
    for i in range(TOTAL_POINTS):
        for j in range(i+1, TOTAL_POINTS):
            d = d2(i, j)
            dist[i][j] = d
            dist[j][i] = d

    for i in range(TOTAL_POINTS):
        for j in range(i + 1, TOTAL_POINTS):
            d_ij = dist[i][j]
            mask = 0
            for k in range(TOTAL_POINTS):
                if k == i or k == j:
                    continue
                d_ik = dist[i][k]
                d_jk = dist[j][k]
                # Isosceles if any two sides are equal
                if d_ik == d_ij or d_jk == d_ij or d_ik == d_jk:
                    mask |= (1 << k)
            covers[i][j] = mask
            covers[j][i] = mask

    all_mask = (1 << TOTAL_POINTS) - 1
    
    def is_valid(chosen_mask):
        # Check if any point in chosen is covered by a pair in chosen
        # covered_by_chosen = union of covers[p][q] for p,q in chosen
        # If chosen & covered_by_chosen != 0, invalid.
        covered = 0
        tmp = chosen_mask
        # To optimize, we can just check pairs.
        # But we need the union.
        # Iterate over set bits in chosen_mask
        # This is O(K^2 * bit_ops). K is small (~10-20).
        # We can precompute "covers_union" incrementally? 
        # For now, direct check.
        # Optimization: 
        # covered = 0
        # For each p in chosen:
        #    For each q in chosen, q > p:
        #       covered |= covers[p][q]
        # return (chosen & covered) == 0
        
        # Faster:
        # Since K is small, we can just list the points.
        pts = []
        tmp = chosen_mask
        while tmp:
            lsb = tmp & -tmp
            idx = lsb.bit_length() - 1
            pts.append(idx)
            tmp ^= lsb
            
        covered = 0
        for idx_a in range(len(pts)):
            a = pts[idx_a]
            for idx_b in range(idx_a + 1, len(pts)):
                b = pts[idx_b]
                covered |= covers[a][b]
                
        return (chosen_mask & covered) == 0

    def is_maximal(chosen_mask):
        # A set is maximal if no point can be added.
        # A point v can be added if it is NOT in chosen AND it is NOT covered by chosen.
        # i.e., v is available if v not in chosen and v not in covered_by_chosen.
        # If available == 0, it is maximal.
        
        # Compute covered_by_chosen
        pts = []
        tmp = chosen_mask
        while tmp:
            lsb = tmp & -tmp
            idx = lsb.bit_length() - 1
            pts.append(idx)
            tmp ^= lsb
            
        covered = 0
        for idx_a in range(len(pts)):
            a = pts[idx_a]
            for idx_b in range(idx_a + 1, len(pts)):
                b = pts[idx_b]
                covered |= covers[a][b]
                
        # Points that can be added are those not in chosen and not in covered
        # available = all_mask & ~chosen_mask & ~covered
        available = all_mask & ~chosen_mask & ~covered
        return available == 0

    def get_covered(chosen_mask):
        pts = []
        tmp = chosen_mask
        while tmp:
            lsb = tmp & -tmp
            idx = lsb.bit_length() - 1
            pts.append(idx)
            tmp ^= lsb
            
        covered = 0
        for idx_a in range(len(pts)):
            a = pts[idx_a]
            for idx_b in range(idx_a + 1, len(pts)):
                b = pts[idx_b]
                covered |= covers[a][b]
        return covered

    def greedy_construct():
        # Start with empty set, add points greedily to maintain validity, stop when maximal?
        # No, we want a maximal set.
        # Greedy: Pick points that don't create isosceles triangles.
        # To ensure maximality, we must add points until no more can be added.
        # This will produce a maximal set.
        chosen = 0
        covered = 0 # Points covered by chosen pairs
        # Actually, we need to know which points are "blocked" from being added to maintain validity.
        # A point p can be added to S if S U {p} is valid.
        # S U {p} is valid iff p is not covered by S (i.e., p not in covered_by_S) AND 
        # adding p doesn't cover any existing point in S?
        # Wait. Validity of S U {p}:
        # No isosceles triple in S U {p}.
        # Triples in S are fine.
        # New triples involve p and two points a,b in S.
        # So {p,a,b} must not be isosceles.
        # This means p must not be in covers[a][b] for any a,b in S.
        # AND a must not be in covers[p][b] and b must not be in covers[p][a]?
        # Yes, symmetric.
        # So condition: p is not in covered_by_S.
        # AND for all a in S, a is not in covers[p][b] for any b in S?
        # Actually, if p is not in covered_by_S, it means for all a,b in S, {p,a,b} is not isosceles.
        # This is exactly the condition.
        # So we just need to maintain `covered` (points that would form isosceles with a pair in S).
        # If we add p, we must ensure p not in covered.
        # Then we update covered: new_covered = covered | union(covers[p][a] for a in S).
        
        available = all_mask
        while available:
            # Pick a random point from available
            # To make it more random, shuffle? Or just pick lowest?
            # Let's pick random to allow different structures.
            tmp = available
            # Get list of available indices
            avail_list = []
            t = tmp
            while t:
                lsb = t & -t
                idx = lsb.bit_length() - 1
                avail_list.append(idx)
                t ^= lsb
            
            if not avail_list:
                break
            
            p = random.choice(avail_list)
            
            # Check if p is valid to add
            if (covered >> p) & 1:
                # p is covered, cannot add
                continue
            
            # Add p
            chosen |= (1 << p)
            
            # Update covered
            # New pairs (p, a) for all a in S (old S)
            # We need to OR covers[p][a] for all a in old_S.
            tmp_s = chosen & ~(1 << p) # old S
            while tmp_s:
                lsb_s = tmp_s & -tmp_s
                a_idx = lsb_s.bit_length() - 1
                covered |= covers[p][a_idx]
                tmp_s ^= lsb_s
            
            # Remove p from available
            available &= ~(1 << p)
            # Remove newly covered points from available?
            # If a point q is covered by S U {p}, it cannot be added later.
            # So we can remove them from available to speed up.
            # available &= ~covered  -- Wait, covered contains points covered by ANY pair in S.
            # Yes, if q is in covered, it can't be added.
            available &= ~covered
            
            # Check if maximal?
            # If available == 0, we are done.
            
        return chosen

    best_set = None
    best_size = TOTAL_POINTS + 1
    
    # Run multiple restarts
    num_restarts = 100000 # Will be limited by time
    
    for _ in range(num_restarts):
        if time.time() > deadline:
            break
        
        # 1. Generate a random maximal set
        S = greedy_construct()
        
        if S == 0:
            continue
            
        size = bin(S).count('1')
        
        # 2. Local Search: Try to reduce size
        # We maintain S.
        # We try to remove points.
        # If removal makes it non-maximal, we try to swap to escape local minima.
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            pts = []
            tmp = S
            while tmp:
                lsb = tmp & -tmp
                idx = lsb.bit_length() - 1
                pts.append(idx)
                tmp ^= lsb
            
            random.shuffle(pts)
            
            for p in pts:
                if time.time() > deadline:
                    break
                
                # Try removing p
                S_new = S & ~(1 << p)
                
                # Check if S_new is maximal
                # S_new is always valid.
                if is_maximal(S_new):
                    # Accept removal
                    S = S_new
                    size -= 1
                    improved = True
                    break # Restart the while loop with new S
                
            # If no removal was successful, try a random swap to change the structure
            if not improved and time.time() < deadline:
                # Pick a random p in S and random q not in S
                # Try to swap p with q
                # Condition: S' = (S - {p}) U {q} must be valid and maximal.
                
                # To avoid O(N) checks for every q, we can just try a few.
                # Or, we can find a q that covers the "gaps" left by p?
                # If S - {p} is not maximal, there are uncovered points.
                # We need q to help cover them.
                
                # Let's just try random swaps.
                pts_S = []
                tmp = S
                while tmp:
                    lsb = tmp & -tmp
                    idx = lsb.bit_length() - 1
                    pts_S.append(idx)
                    tmp ^= lsb
                    
                pts_not_S = []
                tmp = all_mask & ~S
                while tmp:
                    lsb = tmp & -tmp
                    idx = lsb.bit_length() - 1
                    pts_not_S.append(idx)
                    tmp ^= lsb
                    
                if pts_S and pts_not_S:
                    # Try a few random swaps
                    for _ in range(10):
                        if time.time() > deadline:
                            break
                        p = random.choice(pts_S)
                        q = random.choice(pts_not_S)
                        
                        S_swap = (S & ~(1 << p)) | (1 << q)
                        
                        # Check validity
                        if is_valid(S_swap):
                            # Check maximality
                            if is_maximal(S_swap):
                                S = S_swap
                                # Size remains same
                                improved = True
                                break
        
        # Update best
        if size < best_size:
            best_size = size
            best_set = S

    # If no set found (unlikely), fallback to greedy
    if best_set is None:
        best_set = greedy_construct()
        
    # Convert mask to list of tuples
    result = []
    tmp = best_set
    while tmp:
        lsb = tmp & -tmp
        idx = lsb.bit_length() - 1
        result.append(idx_to_pt[idx])
        tmp ^= lsb
        
    return result

if __name__ == "__main__":
    res = construct()
    print(len(res))
