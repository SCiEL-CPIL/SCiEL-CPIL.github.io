import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    U_forms = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ])
    
    for u in range(6):
        for t in range(7):
            c = u * 7 + t
            
            # Fill U
            U[:, c] = U_forms[t]
            
            # Fill V
            # V indices: row = j * p + k
            # j in 0..1, k in 0..11
            # Block u corresponds to columns 2u, 2u+1
            
            if t == 0:
                V[0 * p + 2*u, c] += 1
                V[1 * p + 2*u + 1, c] += 1
            elif t == 1:
                V[0 * p + 2*u, c] += 1
            elif t == 2:
                V[0 * p + 2*u + 1, c] += 1
                V[1 * p + 2*u + 1, c] -= 1
            elif t == 3:
                V[1 * p + 2*u, c] += 1
                V[0 * p + 2*u, c] -= 1
            elif t == 4:
                V[1 * p + 2*u + 1, c] += 1
            elif t == 5:
                V[0 * p + 2*u, c] += 1
                V[0 * p + 2*u + 1, c] += 1
            elif t == 6:
                V[1 * p + 2*u, c] += 1
                V[1 * p + 2*u + 1, c] += 1
                
            # Fill W
            # W indices: row = k * n + i
            # k in 0..11, i in 0..1
            
            if t == 0:
                W[(2*u) * n + 0, c] += 1
                W[(2*u+1) * n + 1, c] += 1
            elif t == 1:
                W[(2*u) * n + 1, c] += 1
                W[(2*u+1) * n + 1, c] -= 1
            elif t == 2:
                W[(2*u+1) * n + 0, c] += 1
                W[(2*u+1) * n + 1, c] += 1
            elif t == 3:
                W[(2*u) * n + 0, c] += 1
                W[(2*u) * n + 1, c] += 1
            elif t == 4:
                W[(2*u) * n + 0, c] -= 1
                W[(2*u+1) * n + 0, c] += 1
            elif t == 5:
                W[(2*u+1) * n + 1, c] += 1
            elif t == 6:
                W[(2*u) * n + 0, c] += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
