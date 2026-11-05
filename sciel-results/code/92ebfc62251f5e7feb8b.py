import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # 5s margin
    
    N = 128
    
    # Precompute LCMs using gcd to be safe
    lcm_table = [[1] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = math.gcd(i, j)
            lcm_table[i][j] = (i // g) * j
            lcm_table[j][i] = lcm_table[i][j]
    
    # Precompute conflicts
    conflicts = [[[] for _ in range(N + 1)] for _ in range(N + 1)]
    
    for x in range(1, N + 1):
        for y in range(x, N + 1):
            lxy = lcm_table[x][y]
            for z in range(1, N + 1):
                if z == x or z == y:
                    continue
                if lcm_table[x][z] == lxy and lcm_table[y][z] == lxy:
                    conflicts[x][y].append(z)
                    conflicts[y][x].append(z)
    
    # Heuristic: Sort by largest prime factor
    def largest_prime_factor(n):
        m = n
        max_p = 1
        d = 2
        while d * d <= m:
            while m % d == 0:
                max_p = d
                m //= d
            d += 1
        if m > 1:
            max_p = m
        return max_p

    candidates = list(range(1, N + 1))
    candidates.sort(key=lambda n: largest_prime_factor(n), reverse=True)
    
    S = []
    forbidden = set()
    
    for x in candidates:
        if time.time() > DEADLINE:
            break
        if x in forbidden or x in S:
            continue
        
        S.append(x)
        
        for y in S[:-1]:
            for z in conflicts[x][y]:
                if z not in S:
                    forbidden.add(z)
    
    return S
