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
    
    def count_violations(colors):
        """Count the number of 4-term APs with fewer than 3 distinct colors."""
        violations = 0
        for ap in aps:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            # Count distinct colors
            distinct = 1
            if c1 != c0: distinct += 1
            if c2 != c0 and c2 != c1: distinct += 1
            if c3 != c0 and c3 != c1 and c3 != c2: distinct += 1
            if distinct < 3:
                violations += 1
        return violations
    
    def delta_violations(colors, pos, new_color):
        """Compute the change in violation count if we change colors[pos] to new_color."""
        old_color = colors[pos]
        if old_color == new_color:
            return 0
        
        delta = 0
        for ap_idx in pos_to_aps[pos]:
            ap = aps[ap_idx]
            # Find the index of pos in ap
            # Get the other 3 positions
            others = []
            for j, p in enumerate(ap):
                if p != pos:
                    others.append(colors[p])
            
            # Before change: colors at pos is old_color
            before_colors = [old_color] + others
            # Count distinct
            distinct_before = len(set(before_colors))
            
            # After change: colors at pos is new_color
            after_colors = [new_color] + others
            distinct_after = len(set(after_colors))
            
            was_violation = distinct_before < 3
            is_violation = distinct_after < 3
            
            if was_violation and not is_violation:
                delta -= 1
            elif not was_violation and is_violation:
                delta += 1
        
        return delta
    
    # Try increasing number of colors
    best_result = None
    best_violations = float('inf')
    
    for k in range(3, 129):
        if time.time() > deadline - 15:
            break
        
        # Try multiple random initializations
        attempts = 0
        max_attempts = 50 if k < 10 else 20
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            colors = [random.randint(0, k-1) for _ in range(n)]
            
            # Count violations
            current_violations = count_violations(colors)
            
            if current_violations == 0:
                return colors
            
            best_colors = colors[:]
            best_violations_local = current_violations
            
            # Local search with simulated annealing
            temp = 2.0
            cooling = 0.99998
            steps = 0
            max_steps = 50000
            
            while steps < max_steps and time.time() < deadline - 5:
                steps += 1
                i = random.randint(0, n-1)
                old_color = colors[i]
                new_color = random.randint(0, k-1)
                if new_color == old_color:
                    continue
                
                # Compute delta
                delta = delta_violations(colors, i, new_color)
                new_violations = current_violations + delta
                
                if new_violations <= current_violations or random.random() < pow(2.718281828, -(new_violations - current_violations) / temp):
                    colors[i] = new_color
                    current_violations = new_violations
                    if current_violations < best_violations_local:
                        best_violations_local = current_violations
                        best_colors = colors[:]
                    if current_violations == 0:
                        return colors
                
                temp *= cooling
            
            if best_violations_local == 0:
                return best_colors
            
            # Track global best
            if best_violations_local < best_violations:
                best_violations = best_violations_local
                best_result = best_colors[:]
    
    # If we found a valid coloring, return it
    if best_result is not None and best_violations == 0:
        return best_result
    
    # Fallback: greedy approach
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
