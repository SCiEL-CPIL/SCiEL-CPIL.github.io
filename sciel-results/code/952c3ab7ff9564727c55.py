import time
import random
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s
    
    n = 128
    
    # Generate all 4-term APs (a, a+d, a+2d, a+3d) with 1-indexed positions
    # Convert to 0-indexed: positions 0..127
    aps = []
    for a in range(n):
        for d in range(1, (n - 1 - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e < n:
                aps.append((a, b, c, e))
    
    def check_validity(colors):
        """Check if coloring satisfies: every 4-AP has at least 3 distinct colors."""
        for (a, b, c, e) in aps:
            if len({colors[a], colors[b], colors[c], colors[e]}) < 3:
                return False
        return True
    
    def count_colors(colors):
        return len(set(colors))
    
    # Try with 3 colors first (0, 1, 2)
    # Use randomized local search / simulated annealing
    
    def local_search(num_colors, max_iters=50000):
        """Local search to find valid coloring with given number of colors."""
        best_score = float('inf')
        best_colors = None
        
        for attempt in range(5):
            if time.time() > deadline:
                break
                
            # Random initial coloring
            colors = [random.randint(0, num_colors - 1) for _ in range(n)]
            
            # Count violations
            def get_violations(cols):
                viol = 0
                for (a, b, c, e) in aps:
                    if len({cols[a], cols[b], cols[c], cols[e]}) < 3:
                        viol += 1
                return viol
            
            cur_viol = get_violations(colors)
            if cur_viol == 0:
                return colors
            
            temp = 1.0
            cooling = 0.9995
            
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                    
                if cur_viol == 0:
                    return colors
                    
                # Pick random position to change
                pos = random.randint(0, n - 1)
                old_color = colors[pos]
                new_color = random.randint(0, num_colors - 1)
                if new_color == old_color:
                    continue
                
                colors[pos] = new_color
                new_viol = get_violations(colors)
                
                delta = new_viol - cur_viol
                if delta <= 0 or random.random() < pow(2.718281828, -delta / temp):
                    cur_viol = new_viol
                    if cur_viol == 0:
                        return colors
                else:
                    colors[pos] = old_color
                
                temp *= cooling
                
                if it % 10000 == 0 and cur_viol < best_score:
                    best_score = cur_viol
                    best_colors = list(colors)
        
        return best_colors
    
    # Try 3 colors
    for attempt in range(20):
        if time.time() > deadline:
            break
        result = local_search(3, max_iters=30000)
        if result is not None and check_validity(result):
            return result
    
    # Try 4 colors
    for attempt in range(20):
        if time.time() > deadline:
            break
        result = local_search(4, max_iters=30000)
        if result is not None and check_validity(result):
            return result
    
    # Fallback: use the seed program approach
    colors = []
    for x in range(1, 129):
        found = False
        for color in range(128):
            ok = True
            # Check 4-term APs ending at x
            for d in range(1, (x - 1) // 3 + 1):
                if x - 3 * d < 1:
                    break
                a = x - 3 * d
                b = x - 2 * d
                c = x - d
                if len({color} | {colors[a - 1], colors[b - 1], colors[c - 1]}) < 3:
                    ok = False
                    break
            if ok:
                colors.append(color)
                found = True
                break
        if not found:
            # Shouldn't happen, but fallback
            colors.append(0)
    
    return colors
