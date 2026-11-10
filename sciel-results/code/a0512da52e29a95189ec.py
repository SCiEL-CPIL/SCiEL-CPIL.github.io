import numpy as np

def nikodym_set() -> np.ndarray:
    p = 7
    k = 3 # alpha^2 = 3
    
    def mul(u, v):
        ua, ub = u
        va, vb = v
        return ((ua*va + k*ub*vb) % p, (ua*vb + ub*va) % p)
        
    def pow8(u):
        # u^8
        u2 = mul(u, u)
        u4 = mul(u2, u2)
        u8 = mul(u4, u4)
        return u8
        
    one = (1, 0)
    
    S = []
    for xa in range(p):
        for xb in range(p):
            x = (xa, xb)
            x8 = pow8(x)
            for ya in range(p):
                for yb in range(p):
                    y = (ya, yb)
                    y8 = pow8(y)
                    # x8 + y8 + 1 = 0
                    sum_a = (x8[0] + y8[0] + 1) % p
                    sum_b = (x8[1] + y8[1] + 0) % p
                    if sum_a == 0 and sum_b == 0:
                        S.append((xa, xb, ya, yb))
                        
    # S is the complement. N is everything else.
    # We need to return N.
    N = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (xa, xb, ya, yb) not in S:
                        N.append([[xa, xb], [ya, yb]])
                        
    return np.array(N, dtype=np.int64)
