import time
import random
import math

def construct():
    n = 128
    deadline = time.time() + 85.0  # Safety margin before 90s kill
    
    # Precompute all nonconstant 4-term APs (0-indexed positions)
    aps = []
    for d in range(1, (n - 3) // 3 + 1):
        for i in range(0, n - 3 * d):
            aps.append((i, i + d, i + 2 * d, i + 3 * d))
    
    # Map each position to the APs it belongs to
    pos_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(idx)
    
    def count_violations(colors):
        """Count violations: APs with < 3 distinct colors."""
        v = 0
        for ap in aps:
            c0 = colors[ap[0]]
            c1 = colors[ap[1]]
            c2 = colors[ap[2]]
            c3 = colors[ap[3]]
            # Check distinct count
            if c0 == c1 == c2 == c3:
                v += 1
            elif c0 == c1 and c0 == c2:
                if c3 == c0: v += 1
                # else 2 distinct, valid? No, 2 distinct is < 3.
                # Wait. Condition: AT LEAST 3 distinct.
                # So 1 or 2 distinct is a violation.
                # My logic above:
                # if distinct < 3: v += 1
                pass # Handled by general check below for clarity in thought, but code will be specific
            # Let's just use a set for simplicity in the fast loop? 
            # For n=128 and ~1000 APs, set creation is fast enough.
            # But we can optimize:
            # distinct = 1
            # if c1 != c0: distinct += 1
            # if c2 != c0 and c2 != c1: distinct += 1
            # if c3 != c0 and c3 != c1 and c3 != c2: distinct += 1
            # if distinct < 3: v += 1
            
            distinct = 1
            if c1 != c0:
                distinct += 1
            if c2 != c0 and c2 != c1:
                distinct += 1
            if c3 != c0 and c3 != c1 and c3 != c2:
                distinct += 1
            if distinct < 3:
                v += 1
        return v

    def local_search_for_k(K, time_limit):
        """Find a valid coloring with K colors."""
        # Initialize with random colors in 0..K-1
        colors = [random.randint(0, K - 1) for _ in range(n)]
        
        best_colors = colors[:]
        best_violations = count_violations(colors)
        
        if best_violations == 0:
            return best_colors
        
        temp = 2.0
        cooling = 0.995
        min_temp = 0.01
        
        while time.time() < time_limit and temp > min_temp:
            pos = random.randint(0, n - 1)
            old_color = colors[pos]
            
            # Choose a new color different from old
            new_color = random.randint(0, K - 1)
            while new_color == old_color:
                new_color = random.randint(0, K - 1)
            
            # Calculate delta violations
            # Only APs containing pos are affected
            curr_invalid = 0
            for ap_idx in pos_to_aps[pos]:
                ap = aps[ap_idx]
                c0 = colors[ap[0]]
                c1 = colors[ap[1]]
                c2 = colors[ap[2]]
                c3 = colors[ap[3]]
                distinct = 1
                if c1 != c0: distinct += 1
                if c2 != c0 and c2 != c1: distinct += 1
                if c3 != c0 and c3 != c1 and c3 != c2: distinct += 1
                if distinct < 3:
                    curr_invalid += 1
            
            colors[pos] = new_color
            
            new_invalid = 0
            for ap_idx in pos_to_aps[pos]:
                ap = aps[ap_idx]
                c0 = colors[ap[0]]
                c1 = colors[ap[1]]
                c2 = colors[ap[2]]
                c3 = colors[ap[3]]
                distinct = 1
                if c1 != c0: distinct += 1
                if c2 != c0 and c2 != c1: distinct += 1
                if c3 != c0 and c3 != c1 and c3 != c2: distinct += 1
                if distinct < 3:
                    new_invalid += 1
            
            delta_v = new_invalid - curr_invalid
            
            # Acceptance
            if delta_v <= 0:
                accept = True
            else:
                energy = delta_v
                if temp > 0:
                    prob = math.exp(-energy / temp)
                    accept = random.random() < prob
                else:
                    accept = False
            
            if accept:
                current_violations = best_violations + delta_v # This is an approximation, better to track global
                # Actually, we should maintain a global violation count
                # For simplicity and speed, let's just recalculate best_violations occasionally
                # Or maintain it.
                pass # Logic below handles it
                
            if accept:
                # Update global state
                # We don't have a global counter variable in scope easily without passing it.
                # Let's just recalculate count_violations every N steps?
                # No, delta is exact for the changed APs.
                # But we need to know the *current* total violations.
                # Let's restructure: keep `current_violations` variable.
                
                # I will rewrite the loop to maintain current_violations.
                pass
                
            temp *= cooling
        
        # The above loop is a bit messy with state. 
        # Let's implement a cleaner version.
        return best_colors

    # Clean implementation of Local Search
    def solve_with_k(K, time_limit):
        colors = [random.randint(0, K - 1) for _ in range(n)]
        current_violations = count_violations(colors)
        best_violations = current_violations
        best_colors = colors[:]
        
        temp = 1.0
        cooling = 0.995
        min_temp = 0.01
        
        while time.time() < time_limit and temp > min_temp:
            pos = random.randint(0, n - 1)
            old_color = colors[pos]
            
            new_color = random.randint(0, K - 1)
            while new_color == old_color:
                new_color = random.randint(0, K - 1)
            
            # Calculate delta
            # Get affected APs
            affected = pos_to_aps[pos]
            
            curr_invalid = 0
            for ap_idx in affected:
                ap = aps[ap_idx]
                c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                distinct = 1
                if c1 != c0: distinct += 1
                if c2 != c0 and c2 != c1: distinct += 1
                if c3 != c0 and c3 != c1 and c3 != c2: distinct += 1
                if distinct < 3:
                    curr_invalid += 1
            
            colors[pos] = new_color
            
            new_invalid = 0
            for ap_idx in affected:
                ap = aps[ap_idx]
                c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                distinct = 1
                if c1 != c0: distinct += 1
                if c2 != c0 and c2 != c1: distinct += 1
                if c3 != c0 and c3 != c1 and c3 != c2: distinct += 1
                if distinct < 3:
                    new_invalid += 1
            
            delta_v = new_invalid - curr_invalid
            
            if delta_v < 0:
                accept = True
            elif delta_v == 0:
                accept = True
            else:
                energy = delta_v
                if temp > 0:
                    prob = math.exp(-energy / temp)
                    accept = random.random() < prob
                else:
                    accept = False
            
            if accept:
                current_violations += delta_v
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_colors = colors[:]
                    if best_violations == 0:
                        return best_colors
            else:
                colors[pos] = old_color
                
            temp *= cooling
            
        return best_colors

    # Try K=4, 5, 6
    # Time budget: 80s total.
    # Give 20s to K=4, 20s to K=5, 20s to K=6, 10s buffer.
    
    # First, check if K=4 is possible.
    # If found, return.
    
    t0 = time.time()
    time_for_4 = 20.0
    time_for_5 = 20.0
    time_for_6 = 20.0
    
    best_result = None
    best_k = 100
    
    # Try K=4
    limit_4 = t0 + time_for_4
    res_4 = solve_with_k(4, limit_4)
    v_4 = count_violations(res_4)
    if v_4 == 0:
        return res_4
    else:
        best_result = res_4
        best_k = 4
        best_v = v_4

    # Try K=5
    limit_5 = t0 + time_for_4 + time_for_5
    res_5 = solve_with_k(5, limit_5)
    v_5 = count_violations(res_5)
    if v_5 == 0:
        return res_5
    else:
        if v_5 < best_v:
            best_result = res_5
            best_k = 5
            best_v = v_5

    # Try K=6
    limit_6 = t0 + time_for_4 + time_for_5 + time_for_6
    res_6 = solve_with_k(6, limit_6)
    v_6 = count_violations(res_6)
    if v_6 == 0:
        return res_6
    else:
        if v_6 < best_v:
            best_result = res_6
            best_k = 6
            best_v = v_6

    # If none are perfect, return the one with fewest colors among valid?
    # No, we must return a valid solution.
    # The task says "Every nonconstant... must use AT LEAST three".
    # If we can't find a valid one, we should return a valid one with more colors?
    # Or does the evaluator penalize invalid ones?
    # "an invalid result... scores 0".
    # So we MUST return a valid coloring.
    
    # If K=4,5,6 all failed to reach 0 violations, we need a fallback.
    # Use K=8 or K=10 which is easier.
    
    # Fallback: Greedy with larger K
    # Or just run solve_with_k(8, deadline - 2)
    
    remaining = deadline - time.time()
    if remaining > 2.0:
        res_8 = solve_with_k(8, deadline - 1.0)
        v_8 = count_violations(res_8)
        if v_8 == 0:
            return res_8
        # If still invalid, try K=10
        res_10 = solve_with_k(10, deadline - 0.5)
        v_10 = count_violations(res_10)
        if v_10 == 0:
            return res_10
    
    # Last resort: Identity mapping (128 colors) is always valid
    # But we want to minimize.
    # If we found a valid one in 4,5,6,8,10, return it.
    # The logic above returns immediately if 0 violations found.
    # If we reach here, we have no valid solution from the searches.
    # This shouldn't happen with K=10.
    
    return list(range(n))
