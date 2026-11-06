import math
import itertools
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    def is_prime(n):
        if n < 2: return False
        if n % 2 == 0: return n == 2
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True

    def build_paley(p):
        squares = set()
        for x in range(1, p):
            squares.add((x*x) % p)
        
        orientation = []
        for i in range(p):
            for j in range(i+1, p):
                diff = (j - i) % p
                if diff in squares:
                    orientation.append(0)
                else:
                    orientation.append(1)
        
        return p, orientation

    def verify_paley(p, orientation):
        A = np.zeros((p, p), dtype=np.uint8)
        idx = 0
        for i in range(p):
            for j in range(i+1, p):
                if orientation[idx] == 0:
                    A[i][j] = 1
                else:
                    A[j][i] = 1
                idx += 1
        
        # Precompute pair intersections of predecessor sets
        # Pred(u) = {v | v -> u} = A.T[u]
        pair_intersects = {}
        for i in range(p):
            for j in range(i+1, p):
                pair_intersects[(i,j)] = A.T[i] & A.T[j]
        
        for quad in itertools.combinations(range(p), 4):
            a, b, c, d = quad
            # a < b and c < d are guaranteed by combinations
            inter1 = pair_intersects[(a,b)]
            inter2 = pair_intersects[(c,d)]
            if not np.any(inter1 & inter2):
                return False
        return True

    candidates = [31, 43, 47, 59, 67, 79, 83]
    
    for p in candidates:
        if time.time() > deadline:
            break
        if p % 4 != 3 or not is_prime(p):
            continue
        
        try:
            _, orientation = build_paley(p)
            if verify_paley(p, orientation):
                return {'n': p, 'orientation': orientation}
        except Exception:
            continue

    # Fallback to 67
    p = 67
    _, orientation = build_paley(p)
    return {'n': p, 'orientation': orientation}

# EVOLVE-BLOCK-END
