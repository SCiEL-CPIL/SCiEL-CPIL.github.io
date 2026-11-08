"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    deadline = time.time() + 110  # Safety margin

    # Naive rank-60 decomposition
    rank_naive = n * m * p  # 60
    U_naive = [[0] * rank_naive for _ in range(n * m)]
    V_naive = [[0] * rank_naive for _ in range(m * p)]
    W_naive = [[0] * rank_naive for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_naive[i * m + j][col] = 1
                V_naive[j * p + k][col] = 1
                W_naive[k * n + i][col] = 1
                col += 1

    best_rank = rank_naive
    best_result = {"U": U_naive, "V": V_naive, "W": W_naive}

    def verify(U, V, W, tol=1e-6):
        """Check if the decomposition is valid.
        Condition: for all i in [0,n), j in [0,m), k in [0,p):
            sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        """
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    # Build a Strassen-based rank-54 decomposition
    # Split j into blocks: {0,1}, {2,3}, {4}
    # Split k into blocks: {0,1}, {2,3}, {4,5}
    # For each (j_block, k_block) pair where j_block has size 2 and k_block has size 2,
    # use Strassen (7 terms instead of 8).
    # For j_block size 1, k_block size 2: 4 terms (naive).
    
    j_blocks = [(0, 1), (2, 3), (4,)]
    k_blocks = [(0, 1), (2, 3), (4, 5)]
    
    # Collect all terms
    terms = []  # each term is (U_dict, V_dict, W_dict)
    
    for jb in j_blocks:
        for kb in k_blocks:
            if len(jb) == 2 and len(kb) == 2:
                # Strassen: 7 terms
                j0, j1 = jb
                k0, k1 = kb
                # a=A[0,j0], b=A[0,j1], c=A[1,j0], d=A[1,j1]
                # e=B[j0,k0], f=B[j0,k1], g=B[j1,k0], h=B[j1,k1]
                # M1 = (a+d)(e+h)
                terms.append(({0*m+j0: 1, 1*m+j1: 1}, {j0*p+k0: 1, j1*p+k1: 1}, {k0*n+0: 1, k1*n+1: 1, k1*n+0: -1, k0*n+1: -1}))
                # M2 = (c+d)e = (A[1,j0]+A[1,j1]) B[j0,k0]
                terms.append(({1*m+j0: 1, 1*m+j1: 1}, {j0*p+k0: 1}, {k0*n+1: 1}))
                # M3 = a(f-h) = A[0,j0](B[j0,k1]-B[j1,k1])
                terms.append(({0*m+j0: 1}, {j0*p+k1: 1, j1*p+k1: -1}, {k1*n+0: 1}))
                # M4 = d(g-e) = A[1,j1](B[j1,k0]-B[j0,k0])
                terms.append(({1*m+j1: 1}, {j1*p+k0: 1, j0*p+k0: -1}, {k0*n+1: 1}))
                # M5 = (a+b)h = (A[0,j0]+A[0,j1]) B[j1,k1]
                terms.append(({0*m+j0: 1, 0*m+j1: 1}, {j1*p+k1: 1}, {k0*n+0: -1, k1*n+0: 1}))
                # M6 = (c+a)g = (A[1,j0]+A[0,j0]) B[j1,k0]
                terms.append(({1*m+j0: 1, 0*m+j0: 1}, {j1*p+k0: 1}, {k0*n+1: 1}))
                # M7 = (b-d)(e+f) = (A[0,j1]-A[1,j1])(B[j0,k0]+B[j1,k0])
                terms.append(({0*m+j1: 1, 1*m+j1: -1}, {j0*p+k0: 1, j1*p+k0: 1}, {k0*n+0: 1}))
            elif len(jb) == 1 and len(kb) == 2:
                # Naive: n * len(kb) terms = 2 * 2 = 4 terms
                j0 = jb[0]
                for i in range(n):
                    for k in kb:
                        terms.append(({i*m+j0: 1}, {j0*p+k: 1}, {k*n+i: 1}))
            elif len(jb) == 2 and len(kb) == 1:
                # Naive: n * len(jb) terms = 2 * 2 = 4 terms
                k0 = kb[0]
                for i in range(n):
                    for j in jb:
                        terms.append(({i*m+j: 1}, {j*p+k0: 1}, {k0*n+i: 1}))
            else:
                # Both size 1: n terms = 2 terms
                j0 = jb[0]
                k0 = kb[0]
                for i in range(n):
                    terms.append(({i*m+j0: 1}, {j0*p+k0: 1}, {k0*n+i: 1}))
    
    rank_strassen = len(terms)
    
    if rank_strassen < best_rank:
        U_s = [[0] * rank_strassen for _ in range(n * m)]
        V_s = [[0] * rank_strassen for _ in range(m * p)]
        W_s = [[0] * rank_strassen for _ in range(p * n)]
        
        for c, (u_dict, v_dict, w_dict) in enumerate(terms):
            for idx, val in u_dict.items():
                U_s[idx][c] = val
            for idx, val in v_dict.items():
                V_s[idx][c] = val
            for idx, val in w_dict.items():
                W_s[idx][c] = val
        
        if verify(U_s, V_s, W_s):
            best_rank = rank_strassen
            best_result = {"U": U_s, "V": V_s, "W": W_s}
    
    # Try Kronecker product decomposition: split j into {0,1} and {2,3,4}
    # Sub-problem 1: 2x2x6 (j in {0,1}, all k)
    # Sub-problem 2: 2x3x6 (j in {2,3,4}, all k)
    
    if time.time() < deadline - 30:
        terms_kron = []
        
        # Sub-problem 1: 2x2x6, split k into {0,1}, {2,3}, {4,5}
        for kb in [(0,1), (2,3), (4,5)]:
            j0, j1 = 0, 1
            k0, k1 = kb
            # Strassen 7 terms
            terms_kron.append(({0*m+j0: 1, 1*m+j1: 1}, {j0*p+k0: 1, j1*p+k1: 1}, {k0*n+0: 1, k1*n+1: 1, k1*n+0: -1, k0*n+1: -1}))
            terms_kron.append(({1*m+j0: 1, 1*m+j1: 1}, {j0*p+k0: 1}, {k0*n+1: 1}))
            terms_kron.append(({0*m+j0: 1}, {j0*p+k1: 1, j1*p+k1: -1}, {k1*n+0: 1}))
            terms_kron.append(({1*m+j1: 1}, {j1*p+k0: 1, j0*p+k0: -1}, {k0*n+1: 1}))
            terms_kron.append(({0*m+j0: 1, 0*m+j1: 1}, {j1*p+k1: 1}, {k0*n+0: -1, k1*n+0: 1}))
            terms_kron.append(({1*m+j0: 1, 0*m+j0: 1}, {j1*p+k0: 1}, {k0*n+1: 1}))
            terms_kron.append(({0*m+j1: 1, 1*m+j1: -1}, {j0*p+k0: 1, j1*p+k0: 1}, {k0*n+0: 1}))
        
        # Sub-problem 2: 2x3x6, split j into {2,3} and {4}, k into {0,1}, {2,3}, {4,5}
        for kb in [(0,1), (2,3), (4,5)]:
            # 2x2x2 Strassen for j={2,3}, k=kb
            j0, j1 = 2, 3
            k0, k1 = kb
            terms_kron.append(({0*m+j0: 1, 1*m+j1: 1}, {j0*p+k0: 1, j1*p+k1: 1}, {k0*n+0: 1, k1*n+1: 1, k1*n+0: -1, k0*n+1: -1}))
            terms_kron.append(({1*m+j0: 1, 1*m+j1: 1}, {j0*p+k0: 1}, {k0*n+1: 1}))
            terms_kron.append(({0*m+j0: 1}, {j0*p+k1: 1, j1*p+k1: -1}, {k1*n+0: 1}))
            terms_kron.append(({1*m+j1: 1}, {j1*p+k0: 1, j0*p+k0: -1}, {k0*n+1: 1}))
            terms_kron.append(({0*m+j0: 1, 0*m+j1: 1}, {j1*p+k1: 1}, {k0*n+0: -1, k1*n+0: 1}))
            terms_kron.append(({1*m+j0: 1, 0*m+j0: 1}, {j1*p+k0: 1}, {k0*n+1: 1}))
            terms_kron.append(({0*m+j1: 1, 1*m+j1: -1}, {j0*p+k0: 1, j1*p+k0: 1}, {k0*n+0: 1}))
            # 2x1x2 naive for j={4}, k=kb
            for i in range(n):
                for k in kb:
                    terms_kron.append(({i*m+4: 1}, {4*p+k: 1}, {k*n+i: 1}))
        
        rank_kron = len(terms_kron)
        if rank_kron < best_rank:
            U_k = [[0] * rank_kron for _ in range(n * m)]
            V_k = [[0] * rank_kron for _ in range(m * p)]
            W_k = [[0] * rank_kron for _ in range(p * n)]
            for c, (u_dict, v_dict, w_dict) in enumerate(terms_kron):
                for idx, val in u_dict.items():
                    U_k[idx][c] = val
                for idx, val in v_dict.items():
                    V_k[idx][c] = val
                for idx, val in w_dict.items():
                    W_k[idx][c] = val
            if verify(U_k, V_k, W_k):
                best_rank = rank_kron
                best_result = {"U": U_k, "V": V_k, "W": W_k}
    
    return best_result
# EVOLVE-BLOCK-END
