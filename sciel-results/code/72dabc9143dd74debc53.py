import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 128
    aps = []
    for d in range(1, (n - 1) // 3 + 1):
        for p in range(0, n - 3 * d):
            aps.append((p, p + d, p + 2 * d, p + 3 * d))
    
    def count_violations(colors):
        count = 0
        for ap in aps:
            p0, p1, p2, p3 = ap
            if len({colors[p0], colors[p1], colors[p2], colors[p3]}) < 3:
                count += 1
        return count
    
    # Try increasing number of colors
    for num_colors in range(3, 7):
        if time.time() > deadline - 10.0:
            break
        
        best = None
        best_v = float('inf')
        
        for attempt in range(200):
            if time.time() > deadline - 5.0:
                break
            
            colors = [random.randint(0, num_colors - 1) for _ in range(n)]
            v = count_violations(colors)
            
            if v == 0:
                return colors
            
            if v < best_v:
                best_v = v
                best = colors[:]
            
            # Local search
            for step in range(300):
                if time.time() > deadline - 5.0:
                    break
                pos = random.randint(0, n - 1)
                old = colors[pos]
                new = random.randint(0, num_colors - 1)
                if new == old:
                    continue
                colors[pos] = new
                nv = count_violations(colors)
                if nv < v:
                    v = nv
                    if v == 0:
                        return colors
                    if v < best_v:
                        best_v = v
                        best = colors[:]
                else:
                    colors[pos] = old
        
        if best_v == 0 and best:
            return best
    
    # Fallback
    return [i % 3 for i in range(n)]
