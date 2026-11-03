"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition exactly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # The decomposition should satisfy:
    # For all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # (since the multiplication tensor has all-ones entries)
    
    # Check a few random (i,j,k) triples
    import random
    random.seed(42)
    for _ in range(100):
        i = random.randint(0, n-1)
        j = random.randint(0, m-1)
        k = random.randint(0, p-1)
        val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
        if abs(val - 1) > 1e-9:
            return False
    return True

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # 110 second deadline
    
    # We will construct a rank-99 decomposition using 6 Strassen blocks and 57 schoolbook terms.
    # n=3: split into {0,1} and {2}
    # m=5: split into {0,1}, {2,3}, and {4}
    # p=7: split into {0,1}, {2,3}, {4,5}, and {6}
    
    n_pairs = [(0, 1)]
    m_pairs = [(0, 1), (2, 3)]
    p_pairs = [(0, 1), (2, 3), (4, 5)]
    
    rank = 99
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    for (a, b) in n_pairs:
        for (c, d) in m_pairs:
            for (e, f) in p_pairs:
                # M1
                U[a*m+c][col] += 1
                U[b*m+d][col] += 1
                V[c*p+e][col] += 1
                V[d*p+f][col] += 1
                W[e*n+a][col] += 1
                W[f*n+b][col] += 1
                col += 1
                
                # M2
                U[b*m+c][col] += 1
                U[b*m+d][col] += 1
                V[c*p+e][col] += 1
                W[e*n+b][col] += 1
                col += 1
                
                # M3
                U[a*m+c][col] += 1
                V[d*p+f][col] += 1
                V[d*p+e][col] -= 1
                W[f*n+a][col] += 1
                col += 1
                
                # M4
                U[b*m+d][col] += 1
                V[c*p+f][col] += 1
                V[c*p+e][col] -= 1
                W[e*n+a][col] += 1
                col += 1
                
                # M5
                U[a*m+c][col] += 1
                U[a*m+d][col] += 1
                V[d*p+f][col] += 1
                W[f*n+a][col] += 1
                W[e*n+a][col] -= 1
                col += 1
                
                # M6
                U[b*m+c][col] += 1
                U[a*m+c][col] -= 1
                V[c*p+e][col] += 1
                V[c*p+f][col] += 1
                W[e*n+b][col] += 1
                col += 1
                
                # M7
                U[b*m+d][col] += 1
                U[a*m+d][col] -= 1
                V[d*p+e][col] += 1
                V[d*p+f][col] += 1
                W[e*n+a][col] -= 1
                W[f*n+b][col] += 1
                col += 1
                
    for i in range(n):
        for j in range(m):
            for k in range(p):
                covered = False
                for (a, b) in n_pairs:
                    if i in (a, b):
                        for (c, d) in m_pairs:
                            if j in (c, d):
                                for (e, f) in p_pairs:
                                    if k in (e, f):
                                        covered = True
                if not covered:
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
                    
    assert col == rank, f"Rank mismatch: {col} vs {rank}"
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(val - 1) > 1e-9:
                    rank_school = n * m * p
                    U_school = [[0] * rank_school for _ in range(n * m)]
                    V_school = [[0] * rank_school for _ in range(m * p)]
                    W_school = [[0] * rank_school for _ in range(p * n)]
                    c = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U_school[ii*m+jj][c] = 1
                                V_school[jj*p+kk][c] = 1
                                W_school[kk*n+ii][c] = 1
                                c += 1
                    return {"U": U_school, "V": V_school, "W": W_school}
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
