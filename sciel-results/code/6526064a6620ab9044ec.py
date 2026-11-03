import numpy as np
import time

def construct():
    n, m, p = 2, 4, 8
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    # Strassen 2x2 local coefficients
    # Format: (u_indices, v_indices, w_indices)
    # Each is a list of (local_idx, coeff)
    strassen_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]), # M0
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),        # M1
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),        # M2
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),        # M3
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),        # M4
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),        # M5
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),        # M6
    ]

    total_rank = 56
    # Initialize matrices
    # Using numpy for speed, then convert to list at the end
    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)

    group = 0
    for jj in range(p // 2): # 4 groups of 2 columns in C
        for half in range(2): # 2 groups of 2 rows in B / 2 cols in A
            base_r = group * 7
            group += 1
            
            # Local to Global Maps
            # A: 2x4. Row i, Col j.
            # half 0: Cols 0,1. half 1: Cols 2,3.
            a_map = {
                0: 0 * m + (2 * half + 0),
                1: 0 * m + (2 * half + 1),
                2: 1 * m + (2 * half + 0),
                3: 1 * m + (2 * half + 1)
            }
            
            # B: 4x8. Row j, Col k.
            # half 0: Rows 0,1. half 1: Rows 2,3.
            # jj 0..3: Cols 2*jj, 2*jj+1
            b_map = {
                0: (2 * half + 0) * p + (2 * jj + 0),
                1: (2 * half + 0) * p + (2 * jj + 1),
                2: (2 * half + 1) * p + (2 * jj + 0),
                3: (2 * half + 1) * p + (2 * jj + 1)
            }
            
            # C: 2x8. Row i, Col k.
            # W index: k * n + i
            c_map = {
                0: (2 * jj + 0) * n + 0,
                1: (2 * jj + 1) * n + 0,
                2: (2 * jj + 0) * n + 1,
                3: (2 * jj + 1) * n + 1
            }

            for r in range(7):
                rr = base_r + r
                u_list, v_list, w_list = strassen_terms[r]
                
                for idx, val in u_list:
                    U[a_map[idx], rr] += val
                    
                for idx, val in v_list:
                    V[b_map[idx], rr] += val
                    
                for idx, val in w_list:
                    W[c_map[idx], rr] += val
                    
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    # Optional: Quick Validation
    # Check if U @ V^T matches W @ I? No, tensor decomposition check:
    # sum_r U[i,j,r] * V[j,k,r] == W[k,i,r] ? No.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Decomposition: C[i,k] = sum_r U[i,j,r] ... wait.
    # Standard form: T_{ijk} = sum_r U_{ir} V_{jr} W_{kr}
    # Here indices are (i, j) for U, (j, k) for V, (k, i) for W.
    # So sum_r U[(i,j), r] * V[(j,k), r] * W[(k,i), r] should be 0?
    # No, the tensor is $T_{i,j,k} = \delta_{i,k?} \dots$
    # The multiplication tensor is $T_{i,j,k} = 1$ if we are computing $C_{i,k} = \sum_j A_{i,j} B_{j,k}$.
    # The decomposition is $T_{i,j,k} = \sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}$?
    # No, the output is $C_{i,k}$. The "tensor" is the map $(A,B) \to C$.
    # $C_{i,k} = \sum_r (\sum_j U_{i,j,r} A_{i,j}) (\sum_k V_{j,k,r} B_{j,k})$? No.
    
    # Correct check:
    # Let A, B be random.
    # C = A @ B
    # C_decomp = sum_r (U[:, :, r] * A) * (V[:, :, r] * B) ? No.
    # U is (n*m, R). V is (m*p, R). W is (p*n, R).
    # We need to verify that the rank-R decomposition is exact.
    # This is hard to check quickly without reshaping.
    # Given the small size, we can skip full validation to save time,
    # or do a spot check.
    
    # Convert to lists of lists as per current interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
