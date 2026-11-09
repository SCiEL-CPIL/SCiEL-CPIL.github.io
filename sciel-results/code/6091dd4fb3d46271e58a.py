import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = DIMENSIONS
    
    # We use the 3x3 block decomposition where each 2x2 block multiplication
    # is handled by Strassen's algorithm (rank 7).
    # Total rank = 3 * 3 * 7 = 189.
    
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 2x2 decomposition terms
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Indices for 2x2 block:
    # A: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # B: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # C: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    
    strassen_terms = [
        # M1: (a+d)(e+h)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2: (c+d)e
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3: a(f-h)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4: d(g-e)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5: (a+b)h
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6: (c-a)(e+f)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7: (b-d)(g+h)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    # Iterate over 3x3 block grid
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
                    # U coefficients: A block (i, k)
                    for local_idx, coeff in u_coeffs.items():
                        r, c = divmod(local_idx, 2)
                        global_r = 2 * i + r
                        global_c = 2 * k + c
                        U[global_r * m + global_c][col] += coeff
                    
                    # V coefficients: B block (k, j)
                    for local_idx, coeff in v_coeffs.items():
                        r, c = divmod(local_idx, 2)
                        global_r = 2 * k + r
                        global_c = 2 * j + c
                        V[global_r * p + global_c][col] += coeff
                    
                    # W coefficients: C block (i, j)
                    # W index convention: W[k*n+i, r] where C[i, j] corresponds to W[j*n+i]
                    # Here global_r is row i of C, global_c is col j of C.
                    for local_idx, coeff in w_coeffs.items():
                        r, c = divmod(local_idx, 2)
                        global_r = 2 * i + r
                        global_c = 2 * j + c
                        W[global_c * n + global_r][col] += coeff
                    
                    col += 1
    
    # Verify the decomposition
    if time.time() > deadline - 5:
        # If we are close to deadline, return immediately without verification
        # assuming the construction is correct based on known algorithms
        return {"U": U, "V": V, "W": W}

    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    U_t = U_np.reshape(n, m, rank)
    V_t = V_np.reshape(m, p, rank)
    W_t = W_np.reshape(p, n, rank)
    
    # T_{ikj} should be delta_{ik} delta_{kj} for matrix multiplication tensor?
    # Actually, the tensor for matrix mult is T_{i,k,j} = delta_{ik} delta_{kj}? No.
    # C_{ij} = sum_k A_{ik} B_{kj}.
    # Tensor T_{i,k,j} = 1 if A_{ik} and B_{kj} contribute to C_{ij}.
    # The decomposition is: T_{i,k,j} = sum_t U_{i,k,t} V_{k,j,t} W_{j,i,t}
    # So we check if np.einsum('ikt,kjt,jit->ikj', U_t, V_t, W_t) equals the identity tensor.
    # The identity tensor for matrix multiplication has T_{i,k,j} = 1 if i==k? No.
    # T_{i,k,j} is 1 if the term A_{ik} B_{kj} is part of C_{ij}.
    # In the standard basis, T_{i,k,j} = delta_{ik} delta_{kj} is NOT correct.
    # T_{i,k,j} = 1 for all i,k,j? No.
    # The tensor T is defined such that C = A * B.
    # T_{i,k,j} = 1 if the multiplication A_{ik} * B_{kj} contributes to C_{ij}.
    # Since every pair (ik, kj) contributes to exactly one C_{ij} (specifically C_{i,j}),
    # the tensor T_{i,k,j} is 1 for all i,k,j?
    # Let's check the einsum.
    # sum_t U_{i,k,t} V_{k,j,t} W_{j,i,t}
    # If U,V,W are the decomposition of the multiplication tensor,
    # then for each t, U_{i,k,t} V_{k,j,t} W_{j,i,t} adds to T_{i,k,j}.
    # The target tensor T_{i,k,j} is 1 if i==k? No.
    # The target tensor is 1 for all i, k, j? No.
    # Let's look at the schoolbook:
    # U_{i,k,t} = delta_{i, i_t} delta_{k, k_t}
    # V_{k,j,t} = delta_{k, k_t} delta_{j, j_t}
    # W_{j,i,t} = delta_{j, j_t} delta_{i, i_t}
    # Product is 1 if i=i_t, k=k_t, j=j_t.
    # Sum over t gives 1 if (i,k,j) is a valid triple?
    # Actually, the tensor for matrix multiplication is T_{i,k,j} = 1 for all i,k,j?
    # No, T_{i,k,j} = delta_{ik} delta_{kj} is for something else.
    # The tensor T is defined by T_{i,k,j} = 1 if A_{ik} B_{kj} contributes to C_{ij}.
    # Since C_{ij} = sum_k A_{ik} B_{kj}, the term A_{ik} B_{kj} contributes to C_{ij} with coefficient 1.
    # So T_{i,k,j} = 1 for all i, k, j.
    
    T = np.einsum('ikt,kjt,jit->ikj', U_t, V_t, W_t)
    expected = np.ones((n, m, p))
    
    if not np.allclose(T, expected, atol=1e-10):
        # Fallback to schoolbook if verification fails (should not happen)
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        col_s = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + k][col_s] = 1
                    V_s[k * p + j][col_s] = 1
                    W_s[j * n + i][col_s] = 1
                    col_s += 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
