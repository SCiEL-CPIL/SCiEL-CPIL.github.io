import math

def find_solution(n):
    # x must be > n/4 and <= 3n/4
    x_min = n // 4 + 1
    x_max = 3 * n // 4
    
    for x in range(x_min, x_max + 1):
        A = 4 * x - n
        B = n * x
        
        # (y+B)(z+B) = B(A+B)
        # K = B(A+B)
        K = B * (A + B)
        
        # We need divisors U of K such that U > B and U < K/U (so y < z)
        # U < sqrt(K)
        sqrt_K = int(math.isqrt(K))
        
        # Iterate through potential U values
        # Instead of iterating all divisors, we can iterate U from B+1 to sqrt_K
        # But checking if K%U==0 is O(1)
        # Range size: sqrt(K) - B
        # K = 4nx^2. sqrt(K) = 2x*sqrt(n).
        # B = nx.
        # Range approx 2x*sqrt(n) - nx.
        # For large n, this might be larger than the previous y-range?
        # Previous y-range: y_max - y_min ~ 2B/A - B/A = B/A ~ n.
        # New range: ~ x*sqrt(n).
        # Since x ~ n, new range ~ n^1.5.
        # Old range ~ n.
        # Old is faster!
        
        # Stick to original y-loop but optimize?
        # The original is fine.
        
        # Let's just use the original logic but ensure it's robust.
        y_min = max(x + 1, B // A + 1)
        y_max = (2 * B) // A
        
        for y in range(y_min, y_max + 1):
            D = A * y - B
            if D <= 0:
                continue
            if (B * y) % D == 0:
                z = (B * y) // D
                if z > y:
                    if x != y and y != z and x != z:
                        if x <= 10**12 and y <= 10**12 and z <= 10**12:
                            return [x, y, z]
    return None

_TABLE = {}
for _n in range(3, 257):
    _TABLE[_n] = find_solution(_n)

def construct():
    return [_TABLE[n] for n in range(3, 257)]
