import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # Target rank based on optimal block tiling
    # 105 (schoolbook) - 6 (savings from 6 Strassen-like blocks) = 99
    # We will construct the rank 99 decomposition.
    
    rank = 99
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    def add_strassen(i0, j0, k0):
        nonlocal col
        # M1
        U[i0 * m + j0][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[j0 * p + k0][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[k0 * n + i0][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        
        # M2
        U[(i0 + 1) * m + j0][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[j0 * p + k0][col] += 1
        W[k0 * n + i0 + 1][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] -= 1
        col += 1
        
        # M3
        U[i0 * m + j0][col] += 1
        V[j0 * p + k0 + 1][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] -= 1
        W[(k0 + 1) * n + i0][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        
        # M4
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[(j0 + 1) * p + k0][col] += 1
        V[j0 * p + k0][col] -= 1
        W[k0 * n + i0][col] += 1
        W[k0 * n + i0 + 1][col] += 1
        col += 1
        
        # M5
        U[i0 * m + j0][col] += 1
        U[i0 * m + j0 + 1][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[(k0 + 1) * n + i0][col] += 1
        W[k0 * n + i0][col] -= 1
        col += 1
        
        # M6
        U[(i0 + 1) * m + j0][col] += 1
        U[i0 * m + j0][col] -= 1
        V[j0 * p + k0][col] += 1
        V[j0 * p + k0 + 1][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        
        # M7
        U[i0 * m + j0 + 1][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] -= 1
        V[(j0 + 1) * p + k0][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[k0 * n + i0][col] += 1
        col += 1

    def add_schoolbook(i_list, j_list, k_list):
        nonlocal col
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1

    # Tiling Strategy:
    # i in {0, 1}: Use Strassen for j-pairs and k-pairs
    # j in {0, 1}, k in {0, 1}, {2, 3}, {4, 5} -> 3 Strassen blocks
    # j in {0, 1}, k in {6} -> Schoolbook
    # j in {2, 3}, k in {0, 1}, {2, 3}, {4, 5} -> 3 Strassen blocks
    # j in {2, 3}, k in {6} -> Schoolbook
    # j in {4}, k in {0..6} -> Schoolbook
    # i in {2}: All Schoolbook

    # i=0,1; j=0,1
    for k0 in [0, 2, 4]:
        add_strassen(0, 0, k0)
    add_schoolbook([0, 1], [0, 1], [6])
    
    # i=0,1; j=2,3
    for k0 in [0, 2, 4]:
        add_strassen(0, 2, k0)
    add_schoolbook([0, 1], [2, 3], [6])
    
    # i=0,1; j=4
    add_schoolbook([0, 1], [4], range(7))
    
    # i=2; j=0..4
    add_schoolbook([2], range(5), range(7))

    # Verify with NumPy
    if time.time() > deadline - 10:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Tensor T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i
    # We check T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    
    # Create indices
    # a in [0, 15), b in [0, 35), c in [0, 21)
    # We can compute the full tensor product sum using einsum
    # result[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    
    # U is (15, 99), V is (35, 99), W is (21, 99)
    # We want to check against the target tensor.
    
    # Target tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Compute decomposition
    # U @ V.T @ W.T ? No.
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # This is a tensor contraction.
    # We can do:
    # UV = U @ V.T  (15, 35)
    # This doesn't work directly because W is involved.
    # We can iterate over r? No, that's slow.
    # We can use einsum: 'ar,br,cr->abc'
    
    try:
        computed = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        if not np.allclose(computed, T, atol=1e-9):
            # Fallback to schoolbook if verification fails (should not happen)
            rank_fb = n * m * p
            U_fb = np.zeros((n * m, rank_fb))
            V_fb = np.zeros((m * p, rank_fb))
            W_fb = np.zeros((p * n, rank_fb))
            col_fb = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i * m + j, col_fb] = 1
                        V_fb[j * p + k, col_fb] = 1
                        W_fb[k * n + i, col_fb] = 1
                        col_fb += 1
            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    except Exception:
        # If einsum fails or takes too long, return current best
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
