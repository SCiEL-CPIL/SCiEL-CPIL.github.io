import numpy as np
from itertools import product

# Verify a candidate decomposition (U,V,W) against the multiplication tensor.
# Convention: U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] summed over r == C[i,j] = sum_k A[i,k]*B[k,j]
def _verify(U, V, W, n=2, m=3, p=6, tol=1e-9):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    if V.shape != (m*p, rank) or W.shape != (p*n, rank) or U.shape != (n*m, rank):
        return False
    for i in range(n):
        for j in range(p):
            total = 0.0
            for k in range(m):
                # accumulate U[i*m+k, r]*V[k*p+j, r]*W[j*n+i, r]
                u = U[i*m+k, :]
                v = V[k*p+j, :]
                w = W[j*n+i, :]
                total += np.dot(u*v, w)
            if abs(total - 1.0) > tol:
                return False
    return True

# The current program's terms, transcribed exactly.
TERMS = [
    {'u': {0: 1, 4: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}},
    {'u': {3: 1, 4: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1}},
    {'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1}},
    {'u': {4: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1}},
    {'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1}},
    {'u': {3: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}},
    {'u': {1: 1, 4: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1}},
    {'u': {2: 1}, 'v': {4: 1}, 'w': {0: 1}},
    {'u': {2: 1}, 'v': {5: 1}, 'w': {1: 1}},
    {'u': {5: 1}, 'v': {4: 1}, 'w': {2: 1}},
    {'u': {5: 1}, 'v': {5: 1}, 'w': {3: 1}},
]

def _build_current():
    n, m, p = 2, 3, 6
    rank = 33
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    for block in range(3):
        for t_idx, term in enumerate(TERMS):
            col = block*11 + t_idx
            for idx, val in term['u'].items():
                U[idx][col] = val
            for l_idx, val in term['v'].items():
                j_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = j_local*p + (block*2 + k_local)
                V[g_idx][col] = val
            for l_idx, val in term['w'].items():
                i_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = (block*2 + k_local)*n + i_local
                W[g_idx][col] = val
    return U, V, W

U0, V0, W0 = _build_current()
print("current valid:", _verify(U0, V0, W0))
