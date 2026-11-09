import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    N = 20
    
    def are_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])
    
    def check_no_four_collinear(pts):
        n = len(pts)
        # Check all 4-tuples
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    # If i, j, k are collinear, check if any l is collinear with them
                    if are_collinear(pts[i], pts[j], pts[k]):
                        for l in range(k + 1, n):
                            if are_collinear(pts[i], pts[j], pts[l]):
                                return False
        return True
    
    def compute_collinear_masks(pts):
        """
        Returns a list of lists of bitmasks.
        masks[i][j] is a bitmask of points k (k != i, k != j) collinear with i and j.
        """
        n = len(pts)
        masks = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                mask = 0
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if are_collinear(pts[i], pts[j], pts[k]):
                        mask |= (1 << k)
                masks[i][j] = mask
                masks[j][i] = mask
        return masks
    
    def max_no_three_collinear(pts):
        """
        Exact branch and bound to find largest subset with no 3 collinear.
        """
        n = len(pts)
        masks = compute_collinear_masks(pts)
        
        best = 0
        # State: candidates (bitmask), selected (bitmask), current_size
        # We can use recursion with bitmasks
        
        def solve(candidates, selected, current_size):
            nonlocal best
            if current_size + bin(candidates).count('1') <= best:
                return
            
            if candidates == 0:
                if current_size > best:
                    best = current_size
                return
            
            # Heuristic: pick a point from candidates
            # Pick the one that conflicts with the most other candidates?
            # Or just the lowest set bit for simplicity, but heuristic helps pruning.
            # Let's pick the first one for now, or one with high degree.
            # To speed up, we can just pick the lowest index.
            # But a better heuristic: pick a point that is in many collinear triples within candidates.
            
            # Simple heuristic: pick lowest index
            # k = (candidates & -candidates).bit_length() - 1
            # Let's try to pick a point that eliminates many others.
            
            best_k = -1
            max_conflicts = -1
            # Iterate through candidates
            temp_cand = candidates
            while temp_cand:
                lsb = temp_cand & -temp_cand
                k = lsb.bit_length() - 1
                # Calculate conflicts if we add k
                conflicts = 0
                sel_temp = selected
                while sel_temp:
                    lsb_s = sel_temp & -sel_temp
                    i = lsb_s.bit_length() - 1
                    conflicts |= masks[i][k]
                    sel_temp ^= lsb_s
                # Number of candidates removed
                removed = candidates & conflicts
                cnt_removed = bin(removed).count('1')
                if cnt_removed > max_conflicts:
                    max_conflicts = cnt_removed
                    best_k = k
                temp_cand ^= lsb
            
            if best_k == -1:
                return
                
            k_bit = 1 << best_k
            
            # Branch 1: Skip best_k
            solve(candidates & ~k_bit, selected, current_size)
            
            # Branch 2: Add best_k
            conflicts = 0
            sel_temp = selected
            while sel_temp:
                lsb_s = sel_temp & -sel_temp
                i = lsb_s.bit_length() - 1
                conflicts |= masks[i][best_k]
                sel_temp ^= lsb_s
            
            new_candidates = candidates & ~conflicts & ~k_bit
            solve(new_candidates, selected | k_bit, current_size + 1)
            
        # Initial call
        all_candidates = (1 << n) - 1
        solve(all_candidates, 0, 0)
        return best

    # Initialize with a valid set
    # Try to find a valid set with low score
    # Start with a grid and perturb
    
    best_points = None
    best_score = 21
    
    # Generate initial candidates
    # Grid 4x5
    base_grid = [[i, j] for i in range(4) for j in range(5)]
    
    # We need to break 4-collinearities.
    # Randomly perturb points in a small range.
    
    found_initial = False
    for _ in range(500):
        cand = []
        for p in base_grid:
            # Perturb by -1, 0, 1
            dx = random.randint(-2, 2)
            dy = random.randint(-2, 2)
            cand.append([p[0] + dx, p[1] + dy])
        
        # Ensure distinct
        if len(set(map(tuple, cand))) < 20:
            continue
            
        # Ensure bounds
        if any(abs(x) > 100 or abs(y) > 100 for x, y in cand):
            continue
            
        if check_no_four_collinear(cand):
            score = max_no_three_collinear(cand)
            if score < best_score:
                best_score = score
                best_points = cand
            if not found_initial:
                found_initial = True
                
    if not found_initial:
        # Fallback: Random points
        while True:
            cand = [[random.randint(-50, 50), random.randint(-50, 50)] for _ in range(20)]
            if len(set(map(tuple, cand))) < 20:
                continue
            if check_no_four_collinear(cand):
                score = max_no_three_collinear(cand)
                best_points = cand
                best_score = score
                break

    # Simulated Annealing
    current_points = best_points[:]
    current_score = best_score
    
    T = 50.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, 19)
        old_p = current_points[idx]
        
        # Try a random move
        # Move within a small neighborhood
        radius = random.randint(1, 5)
        new_p = [old_p[0] + random.randint(-radius, radius), 
                 old_p[1] + random.randint(-radius, radius)]
        
        # Check bounds
        if abs(new_p[0]) > 100 or abs(new_p[1]) > 100:
            continue
            
        # Check distinct
        is_distinct = True
        for i in range(20):
            if i == idx: continue
            if current_points[i] == new_p:
                is_distinct = False
                break
        if not is_distinct:
            continue
            
        # Apply move
        current_points[idx] = new_p
        
        # Check validity
        if not check_no_four_collinear(current_points):
            # Revert
            current_points[idx] = old_p
            continue
            
        # Calculate score
        new_score = max_no_three_collinear(current_points)
        
        if new_score < current_score:
            # Accept
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_points = current_points[:]
        else:
            # Accept with probability exp(-delta / T)
            delta = new_score - current_score
            if delta == 0:
                # Sideways move, accept with some prob
                if random.random() < 0.5:
                    pass # Keep move
                else:
                    current_points[idx] = old_p
            else:
                prob = random.exp(-delta / T) if T > 0 else 0
                if random.random() < prob:
                    current_score = new_score
                    # Keep move
                else:
                    current_points[idx] = old_p
        
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min

    return best_points

