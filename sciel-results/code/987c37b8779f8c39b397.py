import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin for 60s limit
    
    n = 128
    
    # Precompute 4-term progressions: (a, a+d, a+2d, a+3d) with d > 0, all < n
    progressions = []
    for a in range(n):
        for d in range(1, n):
            b = a + d
            c = a + 2*d
            e = a + 3*d
            if e < n:
                progressions.append((a, b, c, e))
    
    num_progs = len(progressions)
    prog_array = np.array(progressions, dtype=np.int32)
    
    # For each position, list of indices of progressions it belongs to
    prog_at = [[] for _ in range(n)]
    for idx, (a, b, c, e) in enumerate(progressions):
        prog_at[a].append(idx)
        prog_at[b].append(idx)
        prog_at[c].append(idx)
        prog_at[e].append(idx)
        
    def get_violations_for_prog(prog_idx, colors):
        """
        Returns 1 if the progression has < 3 distinct colors, 0 otherwise.
        """
        a, b, c, e = progressions[prog_idx]
        ca = colors[a]
        cb = colors[b]
        cc = colors[c]
        ce = colors[e]
        
        # Count distinct colors
        # 4 elements. If < 3 distinct, then max count of any color is >= 2.
        # Actually, just check the conditions for 1 or 2 distinct colors.
        
        # 1 distinct color
        if ca == cb and cb == cc and cc == ce:
            return 1
            
        # 2 distinct colors
        # This happens if the set of colors has size 2.
        # We can check if any pair is equal, and not all are equal (already handled), 
        # but we must ensure it's not 3 or 4 distinct.
        # A simpler way: 
        # If (ca==cb) or (ca==cc) or (ca==ce) or (cb==cc) or (cb==ce) or (cc==ce):
        #   It might be 2 or 3 distinct.
        # To be 2 distinct, the "other" colors must match one of the pair.
        
        # Let's just use a set, it's only 4 elements.
        # But we want speed.
        
        # Logic for 2 distinct colors:
        # There exist two values x, y such that the multiset is {x,x,y,y}, {x,x,x,y}, {x,x,y,y}...
        # Basically, if we pick ca:
        # If ca == cb:
        #   If cc == ca or ce == ca:
        #       If cc == ca and ce == ca: 1 distinct (handled)
        #       If cc == ca and ce != ca: 2 distinct (ca, ce) -> VIOLATION
        #       If cc != ca and ce == ca: 2 distinct (ca, cc) -> VIOLATION
        #       If cc != ca and ce != ca:
        #           If cc == ce: 2 distinct (ca, cc) -> VIOLATION
        #           If cc != ce: 3 distinct (ca, cc, ce) -> OK
        #   If cc != ca and ce != ca:
        #       If cc == ce: 2 distinct (ca, cc) -> VIOLATION
        #       If cc != ce: 3 distinct -> OK
        # If ca != cb:
        #   If ca == cc:
        #       If ce == ca: 2 distinct (ca, cb) -> VIOLATION
        #       If ce != ca:
        #           If ce == cb: 2 distinct (ca, cb) -> VIOLATION
        #           If ce != cb: 3 distinct (ca, cb, ce) -> OK
        #   If ca != cc:
        #       If ca == ce:
        #           If cb == ca: handled
        #           If cb != ca:
        #               If cb == cc: 2 distinct (ca, cb) -> VIOLATION (ca=ce, cb=cc)
        #               If cb != cc: 3 distinct (ca, cb, cc) -> OK
        #       If ca != ce:
        #           If cb == cc: 2 distinct (cb, ca) -> VIOLATION (cb=cc, ca, ce distinct? No, if cb=cc, colors are ca, cb, cb, ce. If ca!=cb and ca!=ce and cb!=ce -> 3 distinct. If ca==ce -> 2 distinct.)
        #           # Let's refine.
        
        # Easier:
        distinct = len({ca, cb, cc, ce})
        return 1 if distinct < 3 else 0

    def count_violations(colors):
        """
        Vectorized count of violations for the entire array.
        Used for initial checks and final verification.
        """
        c0 = colors[prog_array[:, 0]]
        c1 = colors[prog_array[:, 1]]
        c2 = colors[prog_array[:, 2]]
        c3 = colors[prog_array[:, 3]]
        
        # All 4 same
        all_same = (c0 == c1) & (c1 == c2) & (c2 == c3)
        
        # 3 same
        three_same = (
            ((c0 == c1) & (c1 == c2) & (c2 != c3)) |
            ((c0 == c1) & (c1 == c3) & (c2 != c3)) |
            ((c0 == c2) & (c2 == c3) & (c0 != c1)) |
            ((c1 == c2) & (c2 == c3) & (c1 != c0))
        )
        
        # 2 pairs
        two_pairs = (
            ((c0 == c1) & (c2 == c3) & (c0 != c2)) |
            ((c0 == c2) & (c1 == c3) & (c0 != c1)) |
            ((c0 == c3) & (c1 == c2) & (c0 != c1))
        )
        
        violations = all_same | three_same | two_pairs
        return np.sum(violations)

    def delta_violation(pos, old_color, new_color, colors):
        """
        Calculate change in number of violations when changing colors[pos] from old_color to new_color.
        Only checks progressions involving pos.
        """
        if old_color == new_color:
            return 0
            
        delta = 0
        for p_idx in prog_at[pos]:
            # Check before
            v_before = get_violations_for_prog(p_idx, colors)
            
            # Change color
            colors[pos] = new_color
            
            # Check after
            v_after = get_violations_for_prog(p_idx, colors)
            
            # Revert for other checks (unless we commit, but we just want delta)
            # We'll accumulate delta
            delta += v_after - v_before
            
            # Revert
            colors[pos] = old_color
            
        return delta

    # Strategy:
    # 1. Greedy construction to get a valid 3-coloring.
    # 2. Local search to minimize distinct colors (merge classes).
    
    # --- Phase 1: Greedy Construction ---
    # We want a valid 3-coloring.
    # Initialize with 0s.
    colors = np.zeros(n, dtype=np.int32)
    
    # To make greedy work well, we should process positions in an order that constrains future choices?
    # Or just fill left to right, picking the color that minimizes violations?
    # Since we start with 0s, we have many violations.
    # Better: Random restart with Greedy Repair.
    
    best_colors = None
    best_violations = float('inf')
    
    for restart in range(20):
        if time.time() > deadline - 10.0:
            break
            
        # Initialize randomly with 3 colors
        colors = np.array([random.randint(0, 2) for _ in range(n)], dtype=np.int32)
        current_violations = count_violations(colors)
        
        if current_violations == 0:
            best_colors = colors.copy()
            break
            
        if current_violations < best_violations:
            best_violations = current_violations
            best_colors = colors.copy()
            
        # Greedy Repair: Iterate through positions, change color to minimize local violations.
        # Do a few passes.
        for pass_num in range(5):
            if time.time() > deadline - 10.0:
                break
                
            improved = False
            # Shuffle order to avoid bias
            order = list(range(n))
            random.shuffle(order)
            
            for pos in order:
                old_color = colors[pos]
                best_local_viol = current_violations
                best_new_color = old_color
                
                for new_color in range(3):
                    if new_color == old_color:
                        continue
                    d = delta_violation(pos, old_color, new_color, colors)
                    if d < best_local_viol - current_violations: # d is change, so new_total = current + d
                        # We want min new_total
                        pass 
                    
                # Simpler: just try each color and pick the one with lowest resulting total
                min_delta = 0
                min_color = old_color
                for new_color in range(3):
                    if new_color == old_color:
                        continue
                    d = delta_violation(pos, old_color, new_color, colors)
                    if d < min_delta:
                        min_delta = d
                        min_color = new_color
                        
                if min_color != old_color:
                    colors[pos] = min_color
                    current_violations += min_delta
                    improved = True
                    if current_violations == 0:
                        break
                        
            if not improved:
                break
                
        if current_violations < best_violations:
            best_violations = current_violations
            best_colors = colors.copy()
            
        if best_violations == 0:
            break
            
    if best_violations == 0:
        # We have a valid 3-coloring.
        # Can we do better? Minimize number of labels used.
        # With 3 colors, we use 3 labels.
        # Can we use 2 labels? No, 2 colors always fails for some 4-AP?
        # The problem says "Minimize the number of labels actually used".
        # If a valid 3-coloring exists, the answer is 3.
        # Is it possible to have a valid coloring with < 3 colors?
        # "Every nonconstant four-term progression ... must use AT LEAST three distinct colors."
        # If we only use 2 colors total, then every progression uses at most 2 distinct colors.
        # So it's impossible to have a valid coloring with < 3 colors.
        # Thus, the minimum is 3.
        # We just need to return ANY valid 3-coloring.
        
        # Ensure labels are in 0..127. They are 0,1,2.
        return [int(c) for c in best_colors]
        
    # If greedy failed to find 0 violations (unlikely for n=128 with 3 colors, but possible if stuck)
    # Try Simulated Annealing on the best found so far
    
    if best_colors is None:
        best_colors = np.array([i % 3 for i in range(n)], dtype=np.int32)
        best_violations = count_violations(best_colors)
        
    colors = best_colors.copy()
    current_violations = best_violations
    
    # SA to fix remaining violations
    T = 10.0
    cooling = 0.995
    min_T = 0.01
    
    while time.time() < deadline - 5.0 and T > min_T:
        pos = random.randint(0, n-1)
        old_color = colors[pos]
        new_color = random.randint(0, 2)
        while new_color == old_color:
            new_color = random.randint(0, 2)
            
        d = delta_violation(pos, old_color, new_color, colors)
        
        if d < 0 or random.random() < np.exp(-d / T):
            colors[pos] = new_color
            current_violations += d
            if current_violations == 0:
                return [int(c) for c in colors]
            if current_violations < best_violations:
                best_violations = current_violations
                best_colors = colors.copy()
        # else: revert (colors[pos] is still old_color because delta_violation reverts)
        
        T *= cooling
        
    # If still not 0, return best effort (though it will be invalid)
    # But for n=128, 3-coloring should exist.
    # If we fail, return the best we have.
    return [int(c) for c in best_colors]

