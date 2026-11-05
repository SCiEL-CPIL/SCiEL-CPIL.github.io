import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    def check_cyclic_validity(n, dist_colors):
        allowed_steps = [set() for _ in range(4)]
        for d, c in enumerate(dist_colors, 1):
            allowed_steps[c].add(d)
            allowed_steps[c].add(-d % n)
            
        for c in range(4):
            steps = allowed_steps[c]
            if not steps:
                continue
            
            found = False
            for s1 in steps:
                for s2 in steps:
                    v2 = (s1 + s2) % n
                    if v2 == 0: continue
                    for s3 in steps:
                        v3 = (v2 + s3) % n
                        if v3 == 0 or v3 == s1: continue
                        for s4 in steps:
                            v4 = (v3 + s4) % n
                            if v4 == 0 or v4 == s1 or v4 == s2 or v4 == v2: continue
                            for s5 in steps:
                                v5 = (v4 + s5) % n
                                if v5 == 0 or v5 == s1 or v5 == s2 or v5 == v2 or v5 == v3 or v5 == v4: continue
                                s6 = (-v5) % n
                                if s6 in steps:
                                    found = True
                                    break
                                if found: break
                            if found: break
                        if found: break
                    if found: break
                if found: break
            if found:
                return False
        return True

    def try_cyclic(n):
        max_dist = n // 2
        # Optimization: fix first distance to 0 to break symmetry
        # But we iterate all, so it's fine.
        for dist_colors in product(range(4), repeat=max_dist):
            if check_cyclic_validity(n, dist_colors):
                colors = []
                for i in range(n):
                    for j in range(i+1, n):
                        d = j - i
                        if d > max_dist: d = n - d
                        colors.append(dist_colors[d-1])
                return {'n': n, 'colors': colors}
        return None

    # Try n=18
    res = try_cyclic(18)
    if res:
        return res
        
    # Try n=17
    if time.time() < deadline - 5:
        res = try_cyclic(17)
        if res:
            return res

    # Fallback to seed
    return {'n': 9, 'colors': [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]}
