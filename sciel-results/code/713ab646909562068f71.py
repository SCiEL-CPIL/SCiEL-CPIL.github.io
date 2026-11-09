<<<<<<< SEARCH
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 128
    
    # Precompute all 4-term APs (0-indexed positions)
    aps = []
    for d in range(1, 43):
        for a in range(0, n - 3*d):
            aps.append((a, a+d, a+2*d, a+3*d))
    
    num_aps = len(aps)
    
    # For each position, list of AP indices that contain it
    pos_to_aps = [[] for _ in range(n)]
    for i, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(i)
    
    def count_violations(colors):
        count = 0
        for ap in aps:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            # Check if < 3 distinct
            if c0 == c1 == c2 == c3:
                count += 1
            elif c0 == c1 == c2 or c0 == c1 == c3 or c0 == c2 == c3 or c1 == c2 == c3:
                count += 1
            elif c0 == c1 and c2 == c3 and c0 != c2:
                count += 1
            elif c0 == c2 and c1 == c3 and c0 != c1:
                count += 1
            elif c0 == c3 and c1 == c2 and c0 != c1:
                count += 1
        return count
    
    def count_violations_fast(colors):
        """Faster: use set length check"""
        count = 0
        for ap in aps:
            if len({colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}) < 3:
                count += 1
        return count
    
    best_colors = None
    best_violations = float('inf')
    
    for k in range(3, 5):
        for attempt in range(30):
            if time.time() > deadline - 2:
                break
            
            random.seed(attempt * 1000 + k * 7)
            colors = np.random.randint(0, k, size=n, dtype=np.int8)
            violations = count_violations_fast(colors.tolist())
            
            if violations == 0:
                best_colors = colors
                best_violations = 0
                break
            
            if violations < best_violations:
                best_violations = violations
                best_colors = colors.copy()
            
            # Local search with simulated annealing
            temperature = 1.0
            for step in range(8000):
                if time.time() > deadline - 2:
                    break
                
                pos = random.randint(0, n-1)
                old_color = colors[pos]
                new_color = random.randint(0, k-1)
                if new_color == old_color:
                    continue
                
                # Only recompute violations for APs containing pos
                colors[pos] = new_color
                new_violations = violations
                for ap_idx in pos_to_aps[pos]:
                    ap = aps[ap_idx]
                    c_vals = [colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]]
                    was_bad = len(set(c_vals)) < 3
                    # We need to know if it was bad before... this is tricky
                    # Let me just recompute all for simplicity since n is small
                    pass
                
                # Actually, let's just recompute all violations - 2667 APs is fast
                new_violations = count_violations_fast(colors.tolist())
                
                delta = new_violations - violations
                if delta < 0 or (temperature > 0.01 and random.random() < np.exp(-delta / temperature)):
                    violations = new_violations
                    if violations < best_violations:
                        best_violations = violations
                        best_colors = colors.copy()
                        if violations == 0:
                            break
                else:
                    colors[pos] = old_color
                
                temperature *= 0.9995
                if temperature < 0.01:
                    break
            
            if best_violations == 0:
                break
        
        if best_violations == 0:
            break
    
    if best_colors is not None and best_violations == 0:
        return list(best_colors)
    
    # Fallback: return best found (may have violations, but better than nothing)
    if best_colors is not None:
        return list(best_colors)
    
    return list(range(n))
=======
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    n = 128
    
    # Precompute all 4-term APs (0-indexed positions)
    aps = []
    ap_indices = {} # position -> list of ap indices
    for d in range(1, n // 3 + 1):
        for a in range(n - 3 * d):
            idx = len(aps)
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
            for pos in aps[-1]:
                if pos not in ap_indices:
                    ap_indices[pos] = []
                ap_indices[pos].append(idx)
    
    num_aps = len(aps)
    
    def is_violation(colors, ap):
        c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
        if c0 == c1 == c2 == c3:
            return True
        if c0 == c1 == c2 or c0 == c1 == c3 or c0 == c2 == c3 or c1 == c2 == c3:
            return True
        if (c0 == c1 and c2 == c3 and c0 != c2) or \
           (c0 == c2 and c1 == c3 and c0 != c1) or \
           (c0 == c3 and c1 == c2 and c0 != c1):
            return True
        return False

    best_colors = None
    best_num_colors = n + 1
    best_violations = float('inf')
    
    # We want to minimize the number of colors used.
    # Try k=3 first. If found, it's optimal (min possible is 3).
    for k in range(3, 6):
        if time.time() > deadline - 5:
            break
            
        for attempt in range(20):
            if time.time() > deadline - 2:
                break
                
            random.seed(attempt * 100 + k)
            np.random.seed(attempt * 100 + k)
            
            # Initialize colors
            colors = np.random.randint(0, k, size=n, dtype=np.int8)
            
            # Calculate initial violations and status
            ap_bad = np.zeros(num_aps, dtype=bool)
            violations = 0
            for i, ap in enumerate(aps):
                if is_violation(colors, ap):
                    ap_bad[i] = True
                    violations += 1
            
            if violations == 0:
                unique_colors = set(colors.tolist())
                num_used = len(unique_colors)
                if num_used < best_num_colors:
                    best_num_colors = num_used
                    best_colors = colors.copy()
                break # Found valid with this k
            
            # Local Search (Simulated Annealing / Hill Climbing)
            temp = 1.0
            steps = 0
            max_steps = 5000
            
            while steps < max_steps:
                if time.time() > deadline - 1:
                    break
                
                steps += 1
                pos = random.randint(0, n - 1)
                old_c = colors[pos]
                new_c = random.randint(0, k - 1)
                if new_c == old_c:
                    continue
                
                # Identify affected APs
                affected_aps = ap_indices.get(pos, [])
                
                # Calculate delta violations
                delta_v = 0
                for ap_idx in affected_aps:
                    ap = aps[ap_idx]
                    was_bad = ap_bad[ap_idx]
                    
                    # Temporarily change color to check new state
                    colors[pos] = new_c
                    is_now_bad = is_violation(colors, ap)
                    
                    if was_bad and not is_now_bad:
                        delta_v -= 1
                    elif not was_bad and is_now_bad:
                        delta_v += 1
                    # Update ap_bad status for this AP
                    ap_bad[ap_idx] = is_now_bad
                    
                new_violations = violations + delta_v
                
                # Acceptance criterion
                accept = False
                if delta_v < 0:
                    accept = True
                elif delta_v == 0:
                    accept = random.random() < 0.5
                else:
                    if temp > 0.01:
                        accept = random.random() < np.exp(-delta_v / temp)
                
                if accept:
                    violations = new_violations
                    if violations < best_violations:
                        best_violations = violations
                        if violations == 0:
                            unique_colors = set(colors.tolist())
                            num_used = len(unique_colors)
                            if num_used < best_num_colors:
                                best_num_colors = num_used
                                best_colors = colors.copy()
                            break # Stop this attempt
                else:
                    # Revert color and ap_bad statuses
                    colors[pos] = old_c
                    for ap_idx in affected_aps:
                        ap = aps[ap_idx]
                        ap_bad[ap_idx] = is_violation(colors, ap)
                
                temp *= 0.999
                if temp < 0.01:
                    break
            
            if best_violations == 0:
                break
        
        if best_violations == 0 and best_num_colors == k:
            # Found a solution with k colors.
            # If k==3, it's optimal, so we can stop entirely.
            if k == 3:
                break
    
    if best_colors is not None:
        return list(best_colors)
    
    # Fallback
    return list(np.random.randint(0, 3, size=n, dtype=np.int8))
>>>>>>> REPLACE
