import sys
import time
import random
import numpy as np

def construct():
    """
    Search for rank-49 tensor decomposition for 2x4 by 4x7 matrix multiplication.
    Starts from rank-50 (Strassen+schoolbook) and searches for rank reduction.
    Falls back to rank 50 if no improvement found within time budget.
    """
    n, m, p = 2, 4, 7
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # Build rank-50 decomposition
    def build_rank50():
        rank = 50
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        
        def add_strassen_block(A_indices, B_indices, C_indices):
            nonlocal col
            def process_mk(A_coeffs, B_coeffs, C_coeffs):
                nonlocal col
                for local_a, coeff in A_coeffs.items():
                    global_a = A_indices[local_a]
                    U[global_a[0] * m + global_a[1]][col] += coeff
                for local_b, coeff in B_coeffs.items():
                    global_b = B_indices[local_b]
                    V[global_b[0] * p + global_b[1]][col] += coeff
                for local_c, coeff in C_coeffs.items():
                    global_c = C_indices[local_c]
                    W[global_c[1] * n + global_c[0]][col] += coeff
                col += 1

            process_mk({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1})
            process_mk({2:1, 3:1}, {0:1}, {2:1, 3:-1})
            process_mk({0:1}, {1:1, 3:-1}, {1:1, 3:1})
            process_mk({3:1}, {2:1, 0:-1}, {0:1, 2:1})
            process_mk({0:1, 1:1}, {3:1}, {0:-1, 1:1})
            process_mk({2:1, 0:-1}, {0:1, 1:1}, {3:1})
            process_mk({1:1, 3:-1}, {2:1, 3:1}, {0:1})
        
        def add_schoolbook_221(A_indices, B_indices, C_indices):
            nonlocal col
            def process_mult(A_idx, B_idx, C_idx, coeff=1):
                nonlocal col
                global_a = A_indices[A_idx]
                global_b = B_indices[B_idx]
                global_c = C_indices[C_idx]
                U[global_a[0] * m + global_a[1]][col] += coeff
                V[global_b[0] * p + global_b[1]][col] += coeff
                W[global_c[1] * n + global_c[0]][col] += coeff
                col += 1

            process_mult(0, 0, 0, 1)
            process_mult(1, 1, 0, 1)
            process_mult(2, 0, 1, 1)
            process_mult(3, 1, 1, 1)
        
        A_L = [(0,0), (0,1), (1,0), (1,1)]
        A_R = [(0,2), (0,3), (1,2), (1,3)]
        
        B_1_L = [(0,0), (0,1), (1,0), (1,1)]
        B_1_R = [(2,0), (2,1), (3,0), (3,1)]
        C_1 = [(0,0), (0,1), (1,0), (1,1)]
        add_strassen_block(A_L, B_1_L, C_1)
        add_strassen_block(A_R, B_1_R, C_1)
        
        B_2_L = [(0,2), (0,3), (1,2), (1,3)]
        B_2_R = [(2,2), (2,3), (3,2), (3,3)]
        C_2 = [(0,2), (0,3), (1,2), (1,3)]
        add_strassen_block(A_L, B_2_L, C_2)
        add_strassen_block(A_R, B_2_R, C_2)
        
        B_3_L_L = [(0,4), (0,5), (1,4), (1,5)]
        B_3_L_R = [(2,4), (2,5), (3,4), (3,5)]
        C_3_L = [(0,4), (0,5), (1,4), (1,5)]
        add_strassen_block(A_L, B_3_L_L, C_3_L)
        add_strassen_block(A_R, B_3_L_R, C_3_L)
        
        B_4_L = [(0,6), (1,6)]
        B_4_R = [(2,6), (3,6)]
        C_4 = [(0,6), (1,6)]
        add_schoolbook_221(A_L, B_4_L, C_4)
        add_schoolbook_221(A_R, B_4_R, C_4)
        
        return U, V, W
    
    U50, V50, W50 = build_rank50()
    
    # Verify rank-50 is correct
    def verify(U, V, W, rank):
        # Check tensor decomposition: for all i,j,k:
        # C[i,j] = sum_l U[i,l] * V[j,l] * W[k,l]
        # where C[i,j] = sum_k A[i,k]*B[k,j]
        # Tensor entry T[i,j,k] = delta(i,k) for matmul
        # Actually for matmul: T[i,j,k] = 1 if j==k else 0 (C[i,j] = sum_k A[i,k]*B[k,j])
        # So T[i,j,k] = delta(j,k)
        for i in range(n):
            for j in range(p):
                for k in range(m):
                    expected = 1.0 if j == k else 0.0
                    actual = 0.0
                    for l in range(rank):
                        actual += U[i*m+k][l] * V[k*p+j][l] * W[j*n+i][l]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True
    
    # Try to find a rank-49 decomposition by searching for a rank-10 2x2x3 block
    # If found, we can build rank-48 total
    
    def build_rank48(U23, V23, W23):
        """Build full rank-48 decomposition given a rank-10 2x2x3 decomposition."""
        rank = 48
        U = [[0] * rank for _ in range(8)]   # (2*4, 48)
        V = [[0] * rank for _ in range(28)]  # (4*7, 48)
        W = [[0] * rank for _ in range(14)]  # (7*2, 48)
        col = 0
        
        # A is 2x4: rows i=0,1; cols k=0,1,2,3
        # B is 4x7: rows k=0,1,2,3; cols j=0,1,2,3,4,5,6
        # C is 2x7: rows i=0,1; cols j=0,1,2,3,4,5,6
        
        # Split A cols: A0 = cols 0,1; A1 = cols 2,3
        # Split B rows: B0 = rows 0,1; B1 = rows 2,3
        # Split B cols: B_left = cols 0-3; B_right = cols 4-6
        
        # Part 1: A0 * B0[:, 0:4] -> 2x4 (use 2x2x4 = two Strassen 2x2x2)
        # A0 rows: 0,1; cols: 0,1
        # B0 rows: 0,1; cols: 0,1,2,3
        # C cols: 0,1,2,3
        
        # Strassen for B0 cols 0,1:
        def add_strassen(A_col_offset, B_row_offset, C_col_offset):
            nonlocal col
            # A: rows 0,1; cols A_col_offset, A_col_offset+1
            # B: rows B_row_offset, B_row_offset+1; cols C_col_offset, C_col_offset+1
            # C: rows 0,1; cols C_col_offset, C_col_offset+1
            
            terms = [
                # (U_coeffs, V_coeffs, W_coeffs)
                # U: (i, k) -> i*4 + k
                # V: (k, j) -> k*7 + j
                # W: (j, i) -> j*2 + i
                ({(0, A_col_offset): 1, (1, A_col_offset+1): 1},
                 {(B_row_offset, C_col_offset): 1, (B_row_offset+1, C_col_offset+1): 1},
                 {(C_col_offset, 0): 1, (C_col_offset+1, 1): 1, (C_col_offset+1, 0): -1, (C_col_offset, 1): -1}),
                ({(1, A_col_offset): 1, (1, A_col_offset+1): 1},
                 {(B_row_offset, C_col_offset): 1},
                 {(C_col_offset, 1): 1, (C_col_offset+1, 1): 1}),
                ({(0, A_col_offset): 1},
                 {(B_row_offset+1, C_col_offset): 1, (B_row_offset+1, C_col_offset+1): -1},
                 {(C_col_offset+1, 0): 1, (C_col_offset+1, 1): 1}),
                ({(1, A_col_offset+1): 1},
                 {(B_row_offset, C_col_offset+1): 1, (B_row_offset, C_col_offset): -1},
                 {(C_col_offset, 0): -1, (C_col_offset+1, 1): -1, (C_col_offset+1, 0): 1, (C_col_offset, 1): 1}),
                ({(0, A_col_offset): 1, (0, A_col_offset+1): 1},
                 {(B_row_offset+1, C_col_offset+1): 1},
                 {(C_col_offset, 0): -1, (C_col_offset, 1): -1, (C_col_offset+1, 0): 1, (C_col_offset+1, 1): 1}),
                ({(1, A_col_offset): 1, (0, A_col_offset): -1},
                 {(B_row_offset, C_col_offset): 1, (B_row_offset, C_col_offset+1): 1},
                 {(C_col_offset+1, 0): 1, (C_col_offset+1, 1): -1}),
                ({(0, A_col_offset+1): 1, (1, A_col_offset+1): -1},
                 {(B_row_offset+1, C_col_offset): 1, (B_row_offset+1, C_col_offset+1): 1},
                 {(C_col_offset, 0): 1, (C_col_offset, 1): 1}),
            ]
            
            for u_c, v_c, w_c in terms:
                for (ai, ak), coeff in u_c.items():
                    U[ai * 4 + ak][col] += coeff
                for (bk, bj), coeff in v_c.items():
                    V[bk * 7 + bj][col] += coeff
                for (cj, ci), coeff in w_c.items():
                    W[cj * 2 + ci][col] += coeff
                col += 1
        
        # Part 1: A0 (cols 0,1) * B0 (rows 0,1) for C cols 0,1 and 2,3
        add_strassen(0, 0, 0)  # A0 * B0[:,0:2] -> C[:,0:2]
        add_strassen(0, 0, 2)  # A0 * B0[:,2:4] -> C[:,2:4]
        
        # Part 2: A1 (cols 2,3) * B1 (rows 2,3) for C cols 0,1 and 2,3
        add_strassen(2, 2, 0)  # A1 * B1[:,0:2] -> C[:,0:2]
        add_strassen(2, 2, 2)  # A1 * B1[:,2:4] -> C[:,2:4]
        
        # Part 3: A0 (cols 0,1) * B0 (rows 0,1) for C cols 4,5,6 (2x2x3)
        # A0: rows 0,1; cols 0,1
        # B0: rows 0,1; cols 4,5,6
        # C: rows 0,1; cols 4,5,6
        def add_2x2x3(A_col_offset, B_row_offset, C_col_offset):
            nonlocal col
            # U23: (4, 10) indexed by (i,k) -> i*2+k
            # V23: (6, 10) indexed by (k,j) -> k*3+j
            # W23: (6, 10) indexed by (j,i) -> j*2+i
            for l in range(10):
                for i in range(2):
                    for k in range(2):
                        val = U23[i*2+k, l]
                        if abs(val) > 1e-9:
                            U[i * 4 + (A_col_offset + k)][col] += val
                for k in range(2):
                    for j in range(3):
                        val = V23[k*3+j, l]
                        if abs(val) > 1e-9:
                            V[(B_row_offset + k) * 7 + (C_col_offset + j)][col] += val
                for j in range(3):
                    for i in range(2):
                        val = W23[j*2+i, l]
                        if abs(val) > 1e-9:
                            W[(C_col_offset + j) * 2 + i][col] += val
                col += 1
        
        add_2x2x3(0, 0, 4)  # A0 * B0[:,4:7] -> C[:,4:7]
        
        # Part 4: A1 (cols 2,3) * B1 (rows 2,3) for C cols 4,5,6 (2x2x3)
        add_2x2x3(2, 2, 4)  # A1 * B1[:,4:7] -> C[:,4:7]
        
        # Total: 4*7 + 2*10 = 28 + 20 = 48
        return U, V, W
    
    # Search for rank-10 2x2x3
    def find_2x2x3_rank10():
        """Search for rank-10 decomposition of 2x2x3 matmul."""
        r = 10
        best_loss = float('inf')
        best_params = None
        
        for attempt in range(500):
            if time.time() > deadline:
                break
            
            np.random.seed(attempt * 12345 + 67890)
            
            # Start from rank-11 solution and try to perturb to rank 10
            U = np.zeros((4, r))
            V = np.zeros((6, r))
            W = np.zeros((6, r))
            
            # Strassen 2x2x2 for j=0,1
            # U: (i,k) -> i*2+k
            # V: (k,j) -> k*3+j
            # W: (j,i) -> j*2+i
            
            # M1: (A00+A11)(B00+B11)
            U[0,0] = 1; U[3,0] = 1
            V[0,0] = 1; V[4,0] = 1
            W[0,0] = 1; W[2,0] = 1; W[1,0] = -1; W[3,0] = 1
            
            # M2: (A10+A11)B00
            U[2,1] = 1; U[3,1] = 1
            V[0,1] = 1
            W[2,0] = 1; W[3,0] = 1
            
            # M3: A00(B10-B11)
            U[0,2] = 1
            V[3,2] = 1; V[4,2] = -1
            W[1,0] = 1; W[3,0] = 1
            
            # M4: A11(B01-B00)
            U[3,3] = 1
            V[1,3] = 1; V[0,3] = -1
            W[0,0] = -1; W[2,0] = 1; W[1,0] = -1; W[3,0] = 1
            
            # M5: (A00+A01)B11
            U[0,4] = 1; U[1,4] = 1
            V[4,4] = 1
            W[0,0] = -1; W[2,0] = -1; W[1,0] = 1; W[3,0] = 1
            
            # M6: (A10-A00)(B00+B01)
            U[2,5] = 1; U[0,5] = -1
            V[0,5] = 1; V[1,5] = 1
            W[1,0] = 1; W[3,0] = -1
            
            # M7: (A01-A11)(B10+B11)
            U[1,6] = 1; U[3,6] = -1
            V[3,6] = 1; V[4,6] = 1
            W[0,0] = 1; W[2,0] = 1
            
            # Schoolbook for j=2
            # A00*B02
            U[0,7] = 1
            V[2,7] = 1
            W[4,0] = 1
            
            # A10*B02
            U[2,8] = 1
            V[2,8] = 1
            W[5,0] = 1
            
            # A01*B12
            U[1,9] = 1
            V[5,9] = 1
            W[4,0] = 1
            
            # A11*B12 - this would be term 10, but we only have 10 terms (0-9)
            # So we need to absorb this into the existing terms
            
            # Perturb the solution
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def compute_error(p):
                Uf = p[0:40].reshape(4, 10)
                Vf = p[40:100].reshape(6, 10)
                Wf = p[100:160].reshape(6, 10)
                err = 0.0
                for i in range(2):
                    for j in range(3):
                        for k in range(2):
                            expected = 1.0 if j == k else 0.0
                            actual = sum(Uf[i*2+k, l] * Vf[k*3+j, l] * Wf[j*2+i, l] for l in range(10))
                            err += (actual - expected) ** 2
                return err
            
            err = compute_error(params)
            
            # Local search
            for step in range(200):
                if time.time() > deadline:
                    break
                delta = np.random.normal(0, 0.5, size=160)
                new_params = params + delta
                new_err = compute_error(new_params)
                if new_err < err:
                    err = new_err
                    params = new_params
                    if err < 1e-10:
                        break
            
            if err < best_loss:
                best_loss = err
                best_params = params.copy()
            
            if best_loss < 1e-10:
                break
        
        if best_loss < 1e-10:
            Uf = best_params[0:40].reshape(4, 10)
            Vf = best_params[40:100].reshape(6, 10)
            Wf = best_params[100:160].reshape(6, 10)
            Uf = np.round(Uf * 2) / 2
            Vf = np.round(Vf * 2) / 2
            Wf = np.round(Wf * 2) / 2
            for i in range(2):
                for j in range(3):
                    for k in range(2):
                        expected = 1.0 if j == k else 0.0
                        actual = sum(Uf[i*2+k, l] * Vf[k*3+j, l] * Wf[j*2+i, l] for l in range(10))
                        if abs(actual - expected) > 1e-9:
                            return None
            return Uf, Vf, Wf
        return None
    
    result = find_2x2x3_rank10()
    if result is not None:
        U23, V23, W23 = result
        U48, V48, W48 = build_rank48(U23, V23, W23)
        # Verify
        if verify(U48, V48, W48, 48):
            return {"U": U48, "V": V48, "W": W48}
    
    return {"U": U50, "V": V50, "W": W50}
