import time
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # safety margin before 60s kill
    
    N = 128
    
    # Precompute all 4-term APs
    # 4-term AP: (i, i+d, i+2d, i+3d)
    aps4 = []
    ap_indices_per_pos = [[] for _ in range(N)]
    
    for d in range(1, (N - 3) // 3 + 1):
        for i in range(0, N - 3 * d):
            idx = len(aps4)
            ap = (i, i + d, i + 2 * d, i + 3 * d)
            aps4.append(ap)
            # Add this AP index to the list of APs for each position in it
            ap_indices_per_pos[ap[0]].append(idx)
            ap_indices_per_pos[ap[1]].append(idx)
            ap_indices_per_pos[ap[2]].append(idx)
            ap_indices_per_pos[ap[3]].append(idx)
    
    total_aps = len(aps4)
    
    def is_valid(colors):
        for ap in aps4:
            if len({colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}) < 3:
                return False
        return True

    def local_search(k, time_limit):
        """
        Try to find a valid coloring using exactly k colors (0 to k-1).
        Returns colors list if found, else None.
        """
        # Random initialization
        colors = [random.randint(0, k - 1) for _ in range(N)]
        
        # Initialize violation status
        ap_violation = [False] * total_aps
        total_violations = 0
        
        for idx, ap in enumerate(aps4):
            if len({colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}) < 3:
                ap_violation[idx] = True
                total_violations += 1
                
        if total_violations == 0:
            return colors
        
        # Local search
        # We use a "best improvement" or "first improvement" strategy with some randomness.
        # To avoid getting stuck, we can use a simulated annealing-like approach or just random restarts.
        # Given the small N, random restarts are effective.
        
        max_restarts = 500
        restart_count = 0
        
        while time.time() - start_time < time_limit and restart_count < max_restarts:
            if total_violations == 0:
                return colors
            
            # Pick a random position involved in a violation
            # We can pick a random violating AP, then a random position in it
            violating_aps = [i for i in range(total_aps) if ap_violation[i]]
            if not violating_aps:
                return colors
            
            target_ap_idx = random.choice(violating_aps)
            target_ap = aps4[target_ap_idx]
            target_pos = random.choice(target_ap)
            
            old_color = colors[target_pos]
            
            # Try all k colors for target_pos
            best_color = old_color
            best_violations = total_violations + 1 # Force change if no improvement
            best_delta = 1 # +1 means worse
            
            for new_color in range(k):
                if new_color == old_color:
                    continue
                
                # Calculate delta in violations
                delta = 0
                for ap_idx in ap_indices_per_pos[target_pos]:
                    ap = aps4[ap_idx]
                    # Get current colors of the AP
                    # We need to check old violation status and new violation status
                    old_viol = ap_violation[ap_idx]
                    
                    # Construct new color set for this AP
                    # Replace color at target_pos with new_color
                    c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                    if ap[0] == target_pos:
                        c0 = new_color
                    elif ap[1] == target_pos:
                        c1 = new_color
                    elif ap[2] == target_pos:
                        c2 = new_color
                    elif ap[3] == target_pos:
                        c3 = new_color
                    
                    new_viol = len({c0, c1, c2, c3}) < 3
                    
                    if old_viol and not new_viol:
                        delta -= 1
                    elif not old_viol and new_viol:
                        delta += 1
                
                if delta < best_delta:
                    best_delta = delta
                    best_color = new_color
                
                # Early break if we find a move that fixes all violations for this pos?
                # Not exactly, because it might break others. But if delta is very negative, it's good.
            
            # If best_color is still old_color (no improvement), we might want to allow a random move
            # to escape local minima.
            if best_color == old_color:
                # With some probability, make a random move
                if random.random() < 0.1:
                    best_color = random.randint(0, k - 1)
                    if best_color == old_color:
                        best_color = (best_color + 1) % k
                    # Recalculate delta for this random move? Or just apply and update?
                    # For simplicity, let's just apply the best improvement. If none, restart.
                    # Actually, random walk is better. Let's implement a simple random walk step if no improvement.
                    # But for now, let's just restart if no improvement found.
                    pass 
                else:
                    # No improvement, restart with new random initialization
                    restart_count += 1
                    colors = [random.randint(0, k - 1) for _ in range(N)]
                    ap_violation = [False] * total_aps
                    total_violations = 0
                    for idx, ap in enumerate(aps4):
                        if len({colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}) < 3:
                            ap_violation[idx] = True
                            total_violations += 1
                    continue
            
            # Apply the best move
            colors[target_pos] = best_color
            
            # Update violation counts
            for ap_idx in ap_indices_per_pos[target_pos]:
                ap = aps4[ap_idx]
                old_viol = ap_violation[ap_idx]
                
                c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                new_viol = len({c0, c1, c2, c3}) < 3
                
                if old_viol and not new_viol:
                    total_violations -= 1
                elif not old_viol and new_viol:
                    total_violations += 1
                
                ap_violation[ap_idx] = new_viol
        
        return None

    # Try with increasing number of colors
    for k in [3, 4, 5, 6]:
        if time.time() - start_time > DEADLINE:
            break
        
        # Allocate time based on k
        # Give more time to smaller k
        time_alloc = 10.0 if k == 3 else 5.0
        end_time = time.time() + time_alloc
        
        result = local_search(k, end_time)
        
        if result is not None:
            # Verify
            if is_valid(result):
                return result
    
    # Fallback: Greedy approach (guaranteed valid, but more colors)
    colors = []
    for x in range(1, 129):
        found = False
        for color in range(128):
            valid = True
            # Check 4-APs ending at x
            for d in range(1, (x - 1) // 3 + 1):
                if x - 3 * d < 1:
                    break
                c1 = colors[x - 1 - 3 * d]
                c2 = colors[x - 1 - 2 * d]
                c3 = colors[x - 1 - d]
                if len({color, c1, c2, c3}) < 3:
                    valid = False
                    break
            if not valid:
                continue
            
            # Check 3-APs ending at x? 
            # The constraint is only on 4-APs. 
            # Wait, does "every nonconstant four-term progression" imply constraints on 3-APs?
            # No. But if a 3-AP is monochromatic, does it force a 4-AP to be bad?
            # Not necessarily.
            # However, the seed program checked 3-APs. Why?
            # "Every nonconstant four-term progression of positions must use AT LEAST three distinct colors."
            # It does NOT say 3-term progressions.
            # Let's re-read carefully.
            # "Every nonconstant four-term progression of positions must use AT LEAST three distinct colors."
            # The seed program's fallback checks:
            # all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in ...) -> 4-APs
            # and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in ...) -> This ensures no 3-term AP is monochromatic?
            # If a 3-term AP (a, b, c) is monochromatic (color C), does that violate the 4-AP constraint?
            # Consider a 4-AP (a, b, c, d). If a=b=c=C, then the set is {C, color(d)}.
            # If color(d) != C, set size is 2. This violates "at least 3".
            # So, if a 3-AP is monochromatic, and it is the first 3 terms of a 4-AP, the 4-AP can only have 2 colors max.
            # Thus, we MUST avoid monochromatic 3-APs that can be extended to a 4-AP.
            # Actually, ANY monochromatic 3-AP (a, a+d, a+2d) can be extended to (a-d, a, a+d, a+2d) or (a, a+d, a+2d, a+3d) if within bounds.
            # If (a, a+d, a+2d) are all C, then (a, a+d, a+2d, a+3d) has colors {C, C, C, C_next}.
            # If C_next != C, size is 2. Invalid.
            # If C_next == C, size is 1. Invalid.
            # So, NO monochromatic 3-APs are allowed if they are part of any 4-AP.
            # Since N=128, most 3-APs are part of some 4-AP.
            # So we must ensure no 3-AP is monochromatic.
            
            # So the check in the seed was correct. We need to ensure no 3-AP is monochromatic.
            # Let's add that check to the fallback.
            
            for d in range(1, (x - 1) // 2 + 1):
                if x - 2 * d < 1:
                    break
                c1 = colors[x - 1 - 2 * d]
                c2 = colors[x - 1 - d]
                if c1 == color and c2 == color:
                    valid = False
                    break
            
            if valid:
                colors.append(color)
                found = True
                break
        if not found:
            # This should not happen with 128 colors
            colors.append(0)
    
    return colors
