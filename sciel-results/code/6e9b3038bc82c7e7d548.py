import random

def construct():
    n = 128
    best = None
    best_distinct = 128
    
    for seed in range(200):
        random.seed(seed)
        colors = [random.randint(0, 2) for _ in range(n)]
        
        improved = True
        while improved:
            improved = False
            # Find all violations
            for a in range(n):
                max_d = (n - 1 - a) // 3
                for d in range(1, max_d + 1):
                    idxs = [a, a+d, a+2*d, a+3*d]
                    cs = [colors[i] for i in idxs]
                    if len(set(cs)) < 3:
                        # Try to fix
                        for idx in idxs:
                            old = colors[idx]
                            for new_c in range(3):
                                if new_c == old:
                                    continue
                                colors[idx] = new_c
                                cs2 = [colors[i] for i in idxs]
                                if len(set(cs2)) >= 3:
                                    improved = True
                                    break
                                colors[idx] = old
                            if improved:
                                break
                        if improved:
                            break
                if improved:
                    break
        
        distinct = len(set(colors))
        if distinct < best_distinct:
            best_distinct = distinct
            best = colors[:]
            
            # Verify
            valid = True
            for a in range(n):
                max_d = (n - 1 - a) // 3
                for d in range(1, max_d + 1):
                    idxs = [a, a+d, a+2*d, a+3*d]
                    if len(set(colors[i] for i in idxs)) < 3:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                break
    
    if best is not None:
        return best
    return [i % 3 for i in range(n)]
