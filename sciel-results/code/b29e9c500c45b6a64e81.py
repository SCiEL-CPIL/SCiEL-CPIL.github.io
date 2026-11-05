import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 115.0
    n = 20
    ncolours = 20
    left = list(range(10))
    right = list(range(10, 20))
    
    # Precompute all 4-cycles
    four_cycles = []
    for i1, i2 in combinations(range(10), 2):
        for j1, j2 in combinations(range(10), 2):
            four_cycles.append(((i1, j1), (i2, j1), (i2, j2), (i1, j2)))
    
    def count_violations(colors):
        viol = 0
        for cycle in four_cycles:
            cs = [colors.get(e) for e in cycle]
            if None in cs:
                continue
            if len(set(cs)) < 4:
                viol += 1
        return viol
    
    # Try algebraic colourings f(i,j) = (a*i + b*j) % 20
    best_colors = None
    best_viol = float('inf')
    
    for a in range(1, 20):
        for b in range(1, 20):
            if time.time() > deadline - 5:
                break
            colors = {}
            for i in range(10):
                for j in range(10):
                    colors[(i, j)] = (a * i + b * j) % ncolours
            
            viol = count_violations(colors)
            if viol < best_viol:
                best_viol = viol
                best_colors = colors.copy()
            if viol == 0:
                result = [[left[i], right[j], c] for (i, j), c in best_colors.items()]
                return result
        if time.time() > deadline - 5:
            break
    
    if best_viol == 0 and best_colors:
        return [[left[i], right[j], c] for (i, j), c in best_colors.items()]
    
    # Local search to fix remaining violations
    colors = best_colors.copy()
    
    for it in range(200000):
        if time.time() > deadline:
            break
        
        viol = count_violations(colors)
        if viol == 0:
            break
        
        viol_cycle = None
        for cycle in four_cycles:
            cs = [colors.get(e) for e in cycle]
            if None not in cs and len(set(cs)) < 4:
                viol_cycle = cycle
                break
        
        if not viol_cycle:
            break
        
        edge = random.choice(viol_cycle)
        old_c = colors[edge]
        new_c = random.randint(0, ncolours - 1)
        if new_c == old_c:
            continue
        colors[edge] = new_c
        
        new_viol = count_violations(colors)
        if new_viol >= viol:
            colors[edge] = old_c
    
    result = [[left[i], right[j], c] for (i, j), c in colors.items()]
    return result
