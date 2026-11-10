import numpy as np
import time
from itertools import combinations

def kissing_number11() -> np.ndarray:
    d = 11
    deadline = time.time() + 580
    
    # Generate all 4-subsets of {0, ..., 10}
    four_subsets = list(combinations(range(d), 4))
    n_subsets = len(four_subsets)  # 330
    
    # Map each 4-subset to an index
    subset_to_idx = {s: i for i, s in enumerate(four_subsets)}
    
    # For each 4-subset, find its 3-subsets
    # sub3s[i] is a list of (3-subset, index_in_subset)
    # Actually, we just need the 3-subset tuple to look up other 4-sets containing it
    
    # Precompute for each 3-subset, the list of 4-subsets containing it
    three_subsets = list(combinations(range(d), 3))
    three_to_fours = {t: [] for t in three_subsets}
    
    for i, s in enumerate(four_subsets):
        # Generate 3-subsets of s
        for j in range(4):
            t = tuple(x for k, x in enumerate(s) if k != j)
            three_to_fours[t].append(i)
            
    # Precompute sign vectors for each 4-subset
    # signs[i] is a list of 16 vectors of length 11 (int64)
    # Actually, we can store the 4-bit pattern and reconstruct if needed, 
    # but for speed in distance check, maybe store the vector?
    # No, we only need the restriction to 3-sets for conflict check.
    # We can store the 4-bit integer representation of the signs.
    # 0: +1, 1: -1.
    
    # For each 4-subset i, and each 3-subset t of i, 
    # we want to know the 3-bit restriction of each of the 16 signs.
    # sign_to_restriction[i][t_idx_in_i][sign_bitmask] = 3-bit value
    
    # Let's precompute:
    # restrict[i][j] where j is 0..3 (which element is missing from the 3-subset)
    # restrict[i][j] is an array of 16 values, each 0..7
    
    restrict = np.zeros((n_subsets, 4, 16), dtype=np.uint8)
    for i, s in enumerate(four_subsets):
        for j in range(4):
            # t is the 3-subset obtained by removing s[j]
            # The positions in the 3-subset correspond to s[0..3] except s[j]
            # We need to map the 4-bit sign to the 3-bit restriction
            # Bit k of sign corresponds to s[k]
            # The 3-bit value is formed by bits k where k != j
            for sign in range(16):
                val = 0
                bit_pos = 0
                for k in range(4):
                    if k == j:
                        continue
                    if (sign >> k) & 1:
                        val |= (1 << bit_pos)
                    bit_pos += 1
                restrict[i, j, sign] = val
                
    # Greedy selection
    # We will maintain `used[i][j]` as a bitmask of 8 bits, 
    # indicating which 3-bit patterns have been used for support i, 3-subset j.
    
    best_points = []
    best_size = 0
    rng = np.random.default_rng(42)
    
    # Create list of all potential points: (subset_idx, sign_idx)
    all_points = [(i, s) for i in range(n_subsets) for s in range(16)]
    
    def greedy(order):
        used = np.zeros((n_subsets, 4), dtype=np.uint8) # bitmask of used patterns
        selected = []
        
        for i, s in order:
            # Check conflicts
            ok = True
            s_tuple = four_subsets[i]
            
            # For each of the 4 three-subsets of i
            for j in range(4):
                # The 3-subset is s_tuple without s_tuple[j]
                t = tuple(x for k, x in enumerate(s_tuple) if k != j)
                
                # Get the 3-bit restriction of sign s for this 3-subset
                r = restrict[i, j, s]
                
                # Check all other 4-subsets T containing t
                for t_idx in three_to_fours[t]:
                    if t_idx == i:
                        continue
                    # Check if pattern r is already used in t_idx for the corresponding 3-subset
                    # We need to find which 'j' index in t_idx corresponds to t
                    # This is tricky. 
                    # Instead, let's precompute for each (i, j) the list of (t_idx, j_t) pairs.
                    pass
                    
            # This implementation is getting complex due to index mapping.
            # Let's simplify: 
            # For each point (i, s), we can precompute the list of conflicts.
            # Conflict with (t_idx, s_t) if they share a 3-subset and signs match on it.
            
            # Let's precompute conflict lists for all 5280 points.
            # This is 5280 * 5280 / 2 pairs? No, only if they share 3 elements.
            # Number of pairs sharing 3 elements:
            # Each 3-set is in 8 4-sets. Pairs in 8 4-sets: C(8,2)=28.
            # 330 3-sets? No, C(11,3)=165 3-sets.
            # 165 * 28 = 4620 pairs of 4-sets sharing 3 elements.
            # For each pair of 4-sets, how many sign pairs conflict?
            # 16 signs for i, 16 for t.
            # Conflict if restriction matches.
            # For a fixed 3-set, 8 patterns. 
            # Each pattern appears in 2 signs for i and 2 signs for t.
            # So 2*2=4 conflicting pairs per pattern.
            # 8 patterns * 4 = 32 conflicting sign pairs per 4-set pair.
            # Total conflicting point pairs = 4620 * 32 = 147840.
            # This is sparse! We can store a list of conflicting points for each point.
            
            pass
            
    # Let's rebuild with precomputed conflicts.
    # Point ID: i * 16 + s
    # Conflict list: conf[id] = list of conflicting IDs
    
    conf = [[] for _ in range(n_subsets * 16)]
    
    # Iterate over all pairs of 4-sets sharing 3 elements
    for t in three_to_fours:
        fours = three_to_fours[t]
        # For each pair of fours in this list
        for a in range(len(fours)):
            for b in range(a + 1, len(fours)):
                i1 = fours[a]
                i2 = fours[b]
                
                # Find the 3-subset index j1 in i1 and j2 in i2 that corresponds to t
                # t is a tuple of 3 elements.
                # i1 is a tuple of 4.
                # j1 is the index of the element in i1 not in t.
                s1 = four_subsets[i1]
                s2 = four_subsets[i2]
                
                # Find j1
                for j1 in range(4):
                    if s1[j1] not in t:
                        break
                for j2 in range(4):
                    if s2[j2] not in t:
                        break
                        
                # Now, for each sign s1_sign in 0..15 and s2_sign in 0..15
                # Check if restrict[i1, j1, s1_sign] == restrict[i2, j2, s2_sign]
                for s1_sign in range(16):
                    r1 = restrict[i1, j1, s1_sign]
                    id1 = i1 * 16 + s1_sign
                    for s2_sign in range(16):
                        r2 = restrict[i2, j2, s2_sign]
                        if r1 == r2:
                            id2 = i2 * 16 + s2_sign
                            conf[id1].append(id2)
                            conf[id2].append(id1)
                            
    # Now run greedy on the 5280 points
    def greedy_search(order):
        selected = [False] * (n_subsets * 16)
        count = 0
        for pid in order:
            if selected[pid]:
                continue
            # Check if any neighbor is selected
            ok = True
            for nb in conf[pid]:
                if selected[nb]:
                    ok = False
                    break
            if ok:
                selected[pid] = True
                count += 1
        return selected, count
        
    # Run multiple restarts
    all_ids = list(range(n_subsets * 16))
    
    best_sel = None
    best_cnt = 0
    
    num_restarts = 0
    while time.time() < deadline - 100:
        num_restarts += 1
        order = rng.permutation(all_ids)
        sel, cnt = greedy_search(order)
        if cnt > best_cnt:
            best_cnt = cnt
            best_sel = sel
            
        if best_cnt > 4000: # Heuristic stop
            break
            
    # Construct points from best_sel
    points = []
    for pid in range(n_subsets * 16):
        if best_sel[pid]:
            i = pid // 16
            s = pid % 16
            v = np.zeros(d, dtype=np.int64)
            support = four_subsets[i]
            for k, pos in enumerate(support):
                if (s >> k) & 1:
                    v[pos] = -1
                else:
                    v[pos] = 1
            points.append(v)
            
    if len(points) < 2:
        points = [np.ones(d, dtype=np.int64), -np.ones(d, dtype=np.int64)]
        
    pts_array = np.array(points, dtype=np.int64)
    
    # Verification
    n_pts = len(pts_array)
    if n_pts > 1:
        norms = np.sum(pts_array ** 2, axis=1)
        max_norm = norms.max()
        # Sample check if too large
        if n_pts < 5000:
            min_dist = float('inf')
            for i in range(n_pts):
                for j in range(i + 1, n_pts):
                    diff = pts_array[i] - pts_array[j]
                    dist_sq = int(np.sum(diff ** 2))
                    if dist_sq < min_dist:
                        min_dist = dist_sq
            if min_dist < max_norm:
                # Fallback
                points = [np.ones(d, dtype=np.int64), -np.ones(d, dtype=np.int64)]
                pts_array = np.array(points, dtype=np.int64)
                
    return np.around(pts_array).astype(np.int64)
