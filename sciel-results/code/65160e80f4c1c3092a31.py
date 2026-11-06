import numpy as np
import random
import time

def construct():
    n = 128
    # Precompute all 4-term arithmetic progressions
    aps = []
    for a in range(n):
        for d in range(1, (n - a) // 3 + 1):
            i1 = a
            i2 = a + d
            i3 = a + 2 * d
            i4 = a + 3 * d
            if i4 < n:
                aps.append((i1, i2, i3, i4))
    
    # Map index to list of AP indices containing it
    index_to_aps = [[] for _ in range(n)]
    for ap_idx, ap in enumerate(aps):
        for idx in ap:
            index_to_aps[idx].append(ap_idx)
            
    def count_violations(colors):
        """Count number of APs with < 3 distinct colors."""
        viol = 0
        for ap in aps:
            c = set(colors[ap])
            if len(c) < 3:
                viol += 1
        return viol

    def delta_violations(colors, pos, new_color):
        """Calculate change in violation count if colors[pos] = new_color."""
        old_color = colors[pos]
        if old_color == new_color:
            return 0
        
        delta = 0
        for ap_idx in index_to_aps[pos]:
            ap = aps[ap_idx]
            # Get colors of the other 3 elements
            other_indices = [i for i in ap if i != pos]
            other_colors = [colors[i] for i in other_indices]
            
            # Check old state
            old_set = set([old_color] + other_colors)
            old_viol = 1 if len(old_set) < 3 else 0
            
            # Check new state
            new_set = set([new_color] + other_colors)
            new_viol = 1 if len(new_set) < 3 else 0
            
            delta += new_viol - old_viol
            
        return delta

    best_colors = None
    best_score = float('inf')
    
    # Try multiple random restarts with 3 colors
    for restart in range(100):
        colors = np.random.randint(0, 3, n)
        current_violations = count_violations(colors)
        
        if current_violations == 0:
            return colors.tolist()

        # Local search
        stuck_count = 0
        for step in range(20000):
            if current_violations == 0:
                break
            
            # Pick a random position
            pos = random.randint(0, n - 1)
            
            best_delta = float('inf')
            best_new_color = colors[pos]
            
            for c in range(3):
                if c == colors[pos]:
                    continue
                d = delta_violations(colors, pos, c)
                if d < best_delta:
                    best_delta = d
                    best_new_color = c
            
            if best_delta < 0:
                colors[pos] = best_new_color
                current_violations += best_delta
                stuck_count = 0
            else:
                # Stuck or no improvement
                if stuck_count > 100 or random.random() < 0.2:
                    new_c = random.randint(0, 2)
                    d = delta_violations(colors, pos, new_c)
                    colors[pos] = new_c
                    current_violations += d
                    stuck_count = 0
                else:
                    stuck_count += 1
                    
        if current_violations < best_score:
            best_score = current_violations
            best_colors = colors.copy()
            
        if best_score == 0:
            break
            
    if best_colors is not None and best_score == 0:
        return best_colors.tolist()
        
    # Fallback: Try 4 colors if 3 failed
    for restart in range(50):
        colors = np.random.randint(0, 4, n)
        current_violations = count_violations(colors)
        
        if current_violations == 0:
            return colors.tolist()
            
        stuck_count = 0
        for step in range(20000):
            if current_violations == 0:
                break
            
            pos = random.randint(0, n - 1)
            best_delta = float('inf')
            best_new_color = colors[pos]
            
            for c in range(4):
                if c == colors[pos]:
                    continue
                d = delta_violations(colors, pos, c)
                if d < best_delta:
                    best_delta = d
                    best_new_color = c
            
            if best_delta < 0:
                colors[pos] = best_new_color
                current_violations += best_delta
                stuck_count = 0
            else:
                if stuck_count > 100 or random.random() < 0.2:
                    new_c = random.randint(0, 3)
                    d = delta_violations(colors, pos, new_c)
                    colors[pos] = new_c
                    current_violations += d
                    stuck_count = 0
                else:
                    stuck_count += 1
                    
        if current_violations == 0:
            return colors.tolist()

    # Last resort
    return best_colors.tolist() if best_colors is not None else [0]*n
