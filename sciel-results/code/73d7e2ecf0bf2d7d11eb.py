import time
import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit

    n, m, p = DIMENSIONS

    # Primary: Recursive Strassen (Rank 49)
    try:
        U, V, W = _strassen_4x4(n, m, p)
        # Verify correctness before returning
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass

    # Fallback: Schoolbook decomposition (Rank 64)
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that sum_r U[:,r] * V[:,r] * W[:,r] == matmul tensor.
    We check a few random integer matrices to ensure exactness.
    """
    # Convert to numpy for speed
    U_np = np.array(U, dtype=np.int64)
    V_np = np.array(V, dtype=np.int64)
    W_np = np.array(W, dtype=np.int64)
    
    rank = U_np.shape[1]
    
    # Random test matrices
    np.random.seed(42)
    for _ in range(5):
        A = np.random.randint(-10, 10, (n, m), dtype=np.int64)
        B = np.random.randint(-10, 10, (m, p), dtype=np.int64)
        C_expected = A @ B
        
        # Compute C via decomposition
        # C[i,k] = sum_j A[i,j] B[j,k]
        # Decomp: C[i,k] = sum_r (sum_j A[i,j] U[i*m+j, r]) * (sum_l B[j,k] V[j*p+k, r]) * W[k*n+i, r]
        
        # Flatten A and B
        A_flat = A.flatten() # row-major: i*m + j
        B_flat = B.flatten() # row-major: j*p + k
        
        # Compute intermediate products
        # P1[r] = sum_j A_flat[i*m+j] * U[i*m+j, r] for fixed i? No, U depends on i,j.
        # Let's compute the full tensor contraction.
        
        # Efficient way:
        # For each r:
        #   u_vec = U[:, r]  (shape n*m)
        #   v_vec = V[:, r]  (shape m*p)
        #   w_vec = W[:, r]  (shape p*n)
        #   contrib[i,k] = sum_j A[i,j] u_vec[i*m+j] * (sum_l B[j,k] v_vec[j*p+k]) * w_vec[k*n+i]
        
        # This is O(n*m*p*rank). 4*4*4*49 = 3136 ops per test. Very fast.
        
        C_decomp = np.zeros((n, p), dtype=np.int64)
        
        for r in range(rank):
            u_r = U_np[:, r]
            v_r = V_np[:, r]
            w_r = W_np[:, r]
            
            # u_r is indexed by i*m+j
            # v_r is indexed by j*p+k
            # w_r is indexed by k*n+i
            
            # Let's build the contribution
            # For each i, k:
            #   term = w_r[k*n+i] * ( sum_j A[i,j] * u_r[i*m+j] * ( sum_l B[j,l] ... wait B[j,k] ) )
            
            # Actually, the bilinear form is:
            # C[i,k] += w_r[k*n+i] * ( sum_j A[i,j] * u_r[i*m+j] * v_r[j*p+k] )
            
            # Let's compute M[i,j] = A[i,j] * u_r[i*m+j]
            # Let's compute N[j,k] = B[j,k] * v_r[j*p+k]
            # Then C[i,k] += w_r[k*n+i] * sum_j M[i,j] * N[j,k]
            
            M = A * u_r.reshape(n, m)
            N = B * v_r.reshape(m, p)
            
            # Sum over j
            S = M @ N  # shape (n, p)
            
            # Scale by w_r and add to C
            # w_r is shape (p*n), indexed by k*n + i
            # We need to reshape w_r to (p, n) and transpose to (n, p)
            w_mat = w_r.reshape(p, n).T
            
            C_decomp += S * w_mat
        
        if not np.array_equal(C_decomp, C_expected):
            return False
            
    return True

def _strassen_4x4(n, m, p):
    """
    Recursive Strassen: 4x4 = 2x2 blocks, each 2x2 block mult uses 7 products.
    Total rank = 7 (outer Strassen) * 7 (inner Strassen) = 49
    """
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Inner Strassen 2x2 decomposition (rank 7)
    # For multiplying 2x2 X * Y = Z:
    # X rows: 0,1 ; cols: 0,1
    # Y rows: 0,1 ; cols: 0,1
    # Z rows: 0,1 ; cols: 0,1
    
    # M0: (X00+X11)(Y00+Y11) -> Z00 += 1, Z11 += 1
    # M1: (X10+X11)Y00       -> Z10 += 1, Z11 -= 1
    # M2: X00(Y01-Y11)       -> Z01 += 1, Z11 += 1
    # M3: X11(Y10-Y00)       -> Z00 += 1, Z10 += 1
    # M4: (X00+X01)Y11       -> Z01 += 1, Z00 -= 1
    # M5: (X10-X00)(Y00+Y01) -> Z11 += 1
    # M6: (X01-X11)(Y10+Y11) -> Z00 += 1

    inner_defs = [
        {'u': [(0,0,1),(1,1,1)], 'v': [(0,0,1),(1,1,1)], 'w': [(0,0,1),(1,1,1)]},
        {'u': [(1,0,1),(1,1,1)], 'v': [(0,0,1)], 'w': [(1,0,1),(1,1,-1)]},
        {'u': [(0,0,1)], 'v': [(0,1,1),(1,1,-1)], 'w': [(0,1,1),(1,1,1)]},
        {'u': [(1,1,1)], 'v': [(1,0,1),(0,0,-1)], 'w': [(0,0,1),(1,0,1)]},
        {'u': [(0,0,1),(0,1,1)], 'v': [(1,1,1)], 'w': [(0,1,1),(0,0,-1)]},
        {'u': [(1,0,1),(0,0,-1)], 'v': [(0,0,1),(0,1,1)], 'w': [(1,1,1)]},
        {'u': [(0,1,1),(1,1,-1)], 'v': [(1,0,1),(1,1,1)], 'w': [(0,0,1)]},
    ]

    # Outer Strassen for 4x4 (treating as 2x2 blocks of size 2x2)
    outer_defs = [
        {'a_blocks': [(0,0,1),(1,1,1)], 'b_blocks': [(0,0,1),(1,1,1)], 'c_blocks': [(0,0,1),(1,1,1)]},
        {'a_blocks': [(1,0,1),(1,1,1)], 'b_blocks': [(0,0,1)], 'c_blocks': [(1,0,1),(1,1,-1)]},
        {'a_blocks': [(0,0,1)], 'b_blocks': [(0,1,1),(1,1,-1)], 'c_blocks': [(0,1,1),(1,1,1)]},
        {'a_blocks': [(1,1,1)], 'b_blocks': [(1,0,1),(0,0,-1)], 'c_blocks': [(0,0,1),(1,0,1)]},
        {'a_blocks': [(0,0,1),(0,1,1)], 'b_blocks': [(1,1,1)], 'c_blocks': [(0,1,1),(0,0,-1)]},
        {'a_blocks': [(1,0,1),(0,0,-1)], 'b_blocks': [(0,0,1),(0,1,1)], 'c_blocks': [(1,1,1)]},
        {'a_blocks': [(0,1,1),(1,1,-1)], 'b_blocks': [(1,0,1),(1,1,1)], 'c_blocks': [(0,0,1)]},
    ]

    for s, outer in enumerate(outer_defs):
        for t, inner in enumerate(inner_defs):
            r = s * 7 + t

            # Build U contributions
            for (P, Q, sign_a) in outer['a_blocks']:
                for (ir, jc, sign_x) in inner['u']:
                    coeff = sign_a * sign_x
                    if coeff != 0:
                        row = 2 * P + ir
                        col = 2 * Q + jc
                        u_idx = row * m + col
                        U[u_idx][r] += coeff

            # Build V contributions
            for (P, Q, sign_b) in outer['b_blocks']:
                for (ir, jc, sign_y) in inner['v']:
                    coeff = sign_b * sign_y
                    if coeff != 0:
                        row = 2 * P + ir
                        col = 2 * Q + jc
                        v_idx = row * p + col
                        V[v_idx][r] += coeff

            # Build W contributions
            for (P, Q, sign_c) in outer['c_blocks']:
                for (ir, jc, sign_z) in inner['w']:
                    coeff = sign_c * sign_z
                    if coeff != 0:
                        row = 2 * P + ir
                        col = 2 * Q + jc
                        w_idx = col * n + row
                        W[w_idx][r] += coeff

    return U, V, W
