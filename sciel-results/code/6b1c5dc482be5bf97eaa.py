import time
import numpy as np
from fractions import Fraction

def construct():
    t0 = time.time()
    deadline = t0 + 135.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8
    rank = 29

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen 2x2x2 decomposition
    # U_local: 2x2 matrix, index [i, a]
    # V_local: 2x2 matrix, index [a, k]
    # W_local: 2x2 matrix, index [k, i]
    
    S_U = [
        [1, 0],  # r=0: A00
        [0, 1],  # r=0: A11 -> Wait, U[i,a]. 
        # r=0: A00 + A11. U[0,0]=1, U[1,1]=1.
        # My previous list was flattened or something. Let's use lists of dicts or 2D lists.
    ]
    
    # Let's define the 7 terms explicitly for 2x2x2
    # Term r:
    # U: list of (i, a, val)
    # V: list of (a, k, val)
    # W: list of (k, i, val)
    
    terms = [
        {
            "U": { (0,0): 1, (1,1): 1 },
            "V": { (0,0): 1, (1,1): 1 },
            "W": { (0,0): 1, (1,1): 1 }
        },
        {
            "U": { (1,0): 1, (1,1): 1 },
            "V": { (0,0): 1 },
            "W": { (0,1): -1, (1,1): -1 } # C10 is i=1,k=0. C11 is i=1,k=1.
            # W index is k*n+i. 
            # C10: k=0, i=1 -> idx 1.
            # C11: k=1, i=1 -> idx 3.
            # So W[1]=-1, W[3]=-1.
            # In dict: (0,1):-1, (1,1):-1.
        },
        {
            "U": { (0,0): 1 },
            "V": { (0,1): 1, (1,1): -1 },
            "W": { (1,0): 1, (1,1): 1 } # C01 (k=1,i=0) idx 2. C11 (k=1,i=1) idx 3.
            # Dict: (1,0):1, (1,1):1.
        },
        {
            "U": { (1,1): 1 },
            "V": { (1,0): 1, (0,0): -1 },
            "W": { (0,0): 1, (0,1): 1 } # C00 (k=0,i=0) idx 0. C10 (k=0,i=1) idx 1.
            # Dict: (0,0):1, (0,1):1.
        },
        {
            "U": { (0,0): 1, (0,1): 1 },
            "V": { (1,1): 1 },
            "W": { (0,0): -1, (1,0): 1 } # -C00 (idx 0) + C01 (idx 2).
            # Dict: (0,0):-1, (1,0):1.
        },
        {
            "U": { (1,0): 1, (0,0): -1 },
            "V": { (0,0): 1, (0,1): 1 },
            "W": { (1,1): 1 } # C11 (idx 3).
            # Dict: (1,1):1.
        },
        {
            "U": { (0,1): 1, (1,1): -1 },
            "V": { (1,0): 1, (1,1): 1 },
            "W": { (0,0): 1 } # C00 (idx 0).
            # Dict: (0,0):1.
        }
    ]

    # Fill U, V, W for 4 blocks of 2x2x2
    for b in range(4):
        k0 = b * 2
        for r in range(7):
            col = b * 7 + r
            t = terms[r]
            
            for (i, a), val in t["U"].items():
                U[i * m + a][col] = val
            
            for (a, k_loc), val in t["V"].items():
                k_glob = k0 + k_loc
                V[a * p + k_glob][col] = val
            
            for (k_loc, i), val in t["W"].items():
                k_glob = k0 + k_loc
                W[k_glob * n + i][col] = val
        
        if not _elapsed_ok():
            return None

    # Fill the last term for a=2 (2x1x8 part)
    col = 28
    for i in range(n):
        U[i * m + 2][col] = 1
    for k in range(p):
        V[2 * p + k][col] = 1
    for k in range(p):
        for i in range(n):
            W[k * n + i][col] = 1

    candidate = {"U": U, "V": V, "W": W}
    
    # Validation
    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if acc != 1:
                        return False
        return True

    if _validate(candidate):
        return candidate
    else:
        # Fallback to schoolbook if validation fails (should not happen)
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
