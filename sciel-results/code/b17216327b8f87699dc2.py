"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    n = 128
    
    # Precompute all 4-term APs
    aps = []
    for a in range(n):
        for d in range(1, (n - a) // 3 + 1):
            i3 = a + 3*d
            if i3 < n:
                aps.append((a, a+d, a+2*d, i3))
    
    # For each position, list of AP indices that include it
    pos_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(idx)
    
    def ap_violation(ap, colors):
        return len({colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}) < 3
    
    # Try increasing number of colors
    for k in range(3, 129):
        if time.time() > deadline - 10:
            break
        
        # Random initialization
        colors = [random.randint(0, k-1) for _ in range(n)]
        
        # Count violations
        current_violations = sum(1 for ap in aps if ap_violation(ap, colors))
        
        if current_violations == 0:
            return colors
        
        best_colors = colors[:]
        best_violations = current_violations
        
        # Local search
        temp = 1.0
        cooling = 0.99995
        steps = 0
        max_steps = 100000
        
        while steps < max_steps and time.time() < deadline - 5:
            steps += 1
            i = random.randint(0, n-1)
            old_color = colors[i]
            new_color = random.randint(0, k-1)
            if new_color == old_color:
                continue
            
            # Compute delta
            delta = 0
            for ap_idx in pos_to_aps[i]:
                ap = aps[ap_idx]
                was = ap_violation(ap, colors)
                colors[i] = new_color
                now = ap_violation(ap, colors)
                colors[i] = old_color
                if was and not now:
                    delta -= 1
                elif not was and now:
                    delta += 1
            
            new_violations = current_violations + delta
            
            if new_violations <= current_violations or random.random() < pow(2.718281828, -(new_violations - current_violations) / temp):
                colors[i] = new_color
                current_violations = new_violations
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_colors = colors[:]
                if current_violations == 0:
                    return colors
            
            temp *= cooling
        
        if best_violations == 0:
            return best_colors
    
    # Fallback: greedy
    colors = [0] * n
    for x in range(n):
        found = False
        for c in range(128):
            valid = True
            for d in range(1, x // 3 + 1):
                i3 = x - 3*d
                i2 = x - 2*d
                i1 = x - d
                if i3 < 0:
                    continue
                if len({colors[i3], colors[i2], colors[i1], c}) < 3:
                    valid = False
                    break
            if valid:
                colors[x] = c
                found = True
                break
        if not found:
            colors[x] = 0
    
    return colors
# EVOLVE-BLOCK-END
