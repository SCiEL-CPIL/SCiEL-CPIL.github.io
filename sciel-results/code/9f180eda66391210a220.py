import numpy as np

def nikodym_set() -> np.ndarray:
    p = 7
    k = 3  # alpha^2 = 3
    
    # Field element representation: (a, b) for a + b*alpha
    # Index 0..48
    # a = idx // 7, b = idx % 7
    
    def f_add(i, j):
        a1, b1 = i // 7, i % 7
        a2, b2 = j // 7, j % 7
        return ((a1 + a2) % p) * 7 + ((b1 + b2) % p)
        
    def f_mul(i, j):
        a1, b1 = i // 7, i % 7
        a2, b2 = j // 7, j % 7
        # (a1 + b1*alpha) * (a2 + b2*alpha) = a1a2 + (a1b2 + a2b1)alpha + b1b2*alpha^2
        # alpha^2 = 3
        a3 = (a1*a2 + 3*b1*b2) % p
        b3 = (a1*b2 + a2*b1) % p
        return a3 * 7 + b3
        
    def f_pow(i, n):
        res = 1  # 1 is index 7 (1 + 0*alpha) -> 1*7 + 0 = 7
        base = i
        while n:
            if n % 2:
                res = f_mul(res, base)
            base = f_mul(base, base)
            n //= 2
        return res
        
    # Point representation: index 0..2400
    # x index 0..48, y index 0..48
    # point index = x * 49 + y
    
    n_points = 2401
    S = np.zeros(n_points, dtype=bool)
    
    # Construct S = { (x, x^2) }
    for x in range(49):
        y = f_pow(x, 2)
        idx = x * 49 + y
        S[idx] = True
        
    # N is the complement
    N_indices = np.where(~S)[0]
    
    result = np.zeros((len(N_indices), 2, 2), dtype=np.int64)
    for idx, i in enumerate(N_indices):
        i = int(i)
        x = i // 49
        y = i % 49
        xa, xb = x // 7, x % 7
        ya, yb = y // 7, y % 7
        result[idx, 0, 0] = xa
        result[idx, 0, 1] = xb
        result[idx, 1, 0] = ya
        result[idx, 1, 1] = yb
        
    return result
