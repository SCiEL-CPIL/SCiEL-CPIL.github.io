"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 8, 8
    
    def naive_decomp():
        rank = n * m * p  # 16
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W):
        """Verify the tensor decomposition is exact."""
        R = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def try_strassen_rank14():
        """
        Rank-14 decomposition for 2x2 * 2x4 using two Strassen 2x2 blocks.
        """
        R = 14
        U = [[0]*R for _ in range(4)]
        V = [[0]*R for _ in range(8)]
        W = [[0]*R for _ in range(8)]
        
        # Left block (cols 0,1 of B) - standard Strassen
        # r=0: M1 = (a00+a11)(b00+b11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[5][0] = 1
        W[0][0] = 1; W[3][0] = 1
        # r=1: M2 = (a10+a11)b00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        # r=2: M3 = a00(b01-b11)
        U[0][2] = 1
        V[1][2] = 1; V[5][2] = -1
        W[2][2] = 1; W[3][2] = 1
        # r=3: M4 = a11(b10-b00)
        U[3][3] = 1
        V[4][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        # r=4: M5 = (a00+a01)b11
        U[0][4] = 1; U[1][4] = 1
        V[5][4] = 1
        W[0][4] = -1; W[2][4] = 1
        # r=5: M6 = (a10-a00)(b00+b01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        # r=6: M7 = (a01-a11)(b10+b11)
        U[1][6] = 1; U[3][6] = -1
        V[4][6] = 1; V[5][6] = 1
        W[0][6] = 1
        
        # Right block (cols 2,3 of B) - standard Strassen
        # r=7: N1 = (a00+a11)(b02+b13)
        U[0][7] = 1; U[3][7] = 1
        V[2][7] = 1; V[7][7] = 1
        W[4][7] = 1; W[7][7] = 1
        # r=8: N2 = (a10+a11)b02
        U[2][8] = 1; U[3][8] = 1
        V[2][8] = 1
        W[5][8] = 1; W[7][8] = -1
        # r=9: N3 = a00(b03-b13)
        U[0][9] = 1
        V[3][9] = 1; V[7][9] = -1
        W[6][9] = 1; W[7][9] = 1
        # r=10: N4 = a11(b12-b02)
        U[3][10] = 1
        V[6][10] = 1; V[2][10] = -1
        W[4][10] = 1; W[5][10] = 1
        # r=11: N5 = (a00+a01)b13
        U[0][11] = 1; U[1][11] = 1
        V[7][11] = 1
        W[4][11] = -1; W[6][11] = 1
        # r=12: N6 = (a10-a00)(b02+b03)
        U[2][12] = 1; U[0][12] = -1
        V[2][12] = 1; V[3][12] = 1
        W[7][12] = 1
        # r=13: N7 = (a01-a11)(b12+b13)
        U[1][13] = 1; U[3][13] = -1
        V[6][13] = 1; V[7][13] = 1
        W[4][13] = 1
        
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
        return None
    
    def try_rank13_search():
        """
        Search for rank-13 decomposition using linear algebra.
        """
        if time.time() > deadline - 10:
            return None

        # Define Strassen terms for 2x2 block
        # U: [a00, a01, a10, a11]
        # V: [b00, b01]
        # W: [C00, C10, C01, C11]
        left_terms = [
            ([1,0,0,1], [1,1], [1,0,0,-1]),    # M1
            ([0,0,1,1], [1,0], [0,1,-1,0]),    # M2
            ([1,0,0,0], [0,1], [0,0,1,1]),     # M3
            ([0,0,0,1], [1,0], [1,-1,0,0]),    # M4
            ([1,1,0,0], [0,1], [-1,0,1,0]),    # M5
            ([1,-1,0,0], [1,1], [0,0,0,1]),    # M6
            ([0,1,-1,0], [1,1], [1,0,0,0]),    # M7
        ]
        
        right_terms = [
            ([1,0,0,1], [1,1], [0,0,0,0,1,0,0,-1]),    # N1
            ([0,0,1,1], [1,0], [0,0,0,0,0,1,-1,0]),    # N2
            ([1,0,0,0], [0,1], [0,0,0,0,0,0,1,1]),     # N3
            ([0,0,0,1], [1,0], [0,0,0,0,1,-1,0,0]),    # N4
            ([1,1,0,0], [0,1], [0,0,0,0,-1,0,0,1]),    # N5
            ([1,-1,0,0], [1,1], [0,0,0,0,0,0,0,1]),    # N6
            ([0,1,-1,0], [1,1], [0,0,0,0,1,0,0,0]),    # N7
        ]

        # Try merging pairs of left and right terms
        for li in range(7):
            if time.time() > deadline - 5:
                return None
            for ri in range(7):
                U_left = np.array(left_terms[li][0], dtype=float)
                U_right = np.array(right_terms[ri][0], dtype=float)
                
                # Check if U vectors are linearly independent
                U_mat = np.column_stack([U_left, U_right])
                if np.linalg.matrix_rank(U_mat) < 2:
                    continue
                    
                R = 13
                U_mat_full = np.zeros((4, R))
                V_mat = np.zeros((8, R))
                W_mat = np.zeros((8, R))
                
                # Merged term
                U_mat_full[:, 0] = U_left
                V_mat[:2, 0] = left_terms[li][1]
                V_mat[2:4, 0] = right_terms[ri][1]
                W_mat[:4, 0] = left_terms[li][2]
                W_mat[4:8, 0] = right_terms[ri][2]
                
                # Remaining left terms
                col = 1
                for l_idx in range(7):
                    if l_idx == li: continue
                    U_mat_full[:, col] = left_terms[l_idx][0]
                    V_mat[:2, col] = left_terms[l_idx][1]
                    W_mat[:4, col] = left_terms[l_idx][2]
                    col += 1
                
                # Remaining right terms
                for r_idx in range(7):
                    if r_idx == ri: continue
                    U_mat_full[:, col] = right_terms[r_idx][0]
                    V_mat[2:4, col] = right_terms[r_idx][1]
                    W_mat[4:8, col] = right_terms[r_idx][2]
                    col += 1
                
                # Build the linear system for V
                # For each (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
                # This is linear in V
                A_V = np.zeros((16, 8 * R))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            eq_idx = i * m * p + j * p + k
                            for r in range(R):
                                col_idx = j * p * R + k * R + r
                                A_V[eq_idx, col_idx] = U_mat_full[i * m + j, r] * W_mat[k * n + i, r]
                
                rhs = np.ones(16)
                try:
                    V_flat, residuals, rank, sv = np.linalg.lstsq(A_V, rhs, rcond=None)
                    if np.allclose(A_V @ V_flat, rhs, atol=1e-9):
                        V_sol = V_flat.reshape(8, R).T
                        if verify(U_mat_full.tolist(), V_sol.tolist(), W_mat.tolist()):
                            return {"U": U_mat_full.tolist(), "V": V_sol.tolist(), "W": W_mat.tolist()}
                except:
                    pass
        return None
    
    def try_rank13_als_search():
        """
        Alternating least squares search for rank-13 decomposition.
        """
        if time.time() > deadline - 15:
            return None
        
        R = 13
        rng = np.random.RandomState(42)
        
        # Define the 16 standard basis terms
        # For each (i,j,k), the term is e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
        basis_terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_vec = np.zeros(nm)
                    v_vec = np.zeros(mp)
                    w_vec = np.zeros(pn)
                    u_vec[i*m+j] = 1
                    v_vec[j*p+k] = 1
                    w_vec[k*n+i] = 1
                    basis_terms.append((u_vec, v_vec, w_vec))
        
        # Try random combinations of 13 basis terms
        for trial in range(200):
            if time.time() > deadline - 5:
                return None
            
            # Pick 13 out of 16 terms
            selected = rng.choice(16, 13, replace=False)
            
            U_mat = np.zeros((nm, R))
            V_mat = np.zeros((mp, R))
            W_mat = np.zeros((pn, R))
            
            for r, idx in enumerate(selected):
                u_vec, v_vec, w_vec = basis_terms[idx]
                U_mat[:, r] = u_vec
                V_mat[:, r] = v_vec
                W_mat[:, r] = w_vec
            
            # Check if this gives a valid decomposition
            if verify(U_mat.tolist(), V_mat.tolist(), W_mat.tolist()):
                return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
            
            # Try alternating least squares
            for als_iter in range(20):
                if time.time() > deadline - 5:
                    return None
                
                # Fix U and W, solve for V
                A_V = np.zeros((16, mp * R))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            eq_idx = i * m * p + j * p + k
                            for r in range(R):
                                col_idx = j * p * R + k * R + r
                                A_V[eq_idx, col_idx] = U_mat[i * m + j, r] * W_mat[k * n + i, r]
                
                rhs = np.ones(16)
                try:
                    V_flat, _, _, _ = np.linalg.lstsq(A_V, rhs, rcond=None)
                    V_mat = V_flat.reshape(mp, R).T
                except:
                    continue
                
                # Fix U and V, solve for W
                A_W = np.zeros((16, pn * R))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            eq_idx = i * m * p + j * p + k
                            for r in range(R):
                                col_idx = k * n * R + i * R + r
                                A_W[eq_idx, col_idx] = U_mat[i * m + j, r] * V_mat[j * p + k, r]
                
                try:
                    W_flat, _, _, _ = np.linalg.lstsq(A_W, rhs, rcond=None)
                    W_mat = W_flat.reshape(pn, R).T
                except:
                    continue
                
                # Fix V and W, solve for U
                A_U = np.zeros((16, nm * R))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            eq_idx = i * m * p + j * p + k
                            for r in range(R):
                                col_idx = i * m * R + j * R + r
                                A_U[eq_idx, col_idx] = V_mat[j * p + k, r] * W_mat[k * n + i, r]
                
                try:
                    U_flat, _, _, _ = np.linalg.lstsq(A_U, rhs, rcond=None)
                    U_mat = U_flat.reshape(nm, R).T
                except:
                    continue
                
                # Check convergence
                if als_iter >= 15:
                    if verify(U_mat.tolist(), V_mat.tolist(), W_mat.tolist()):
                        return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
        
        return None
    
    # Try rank-13 search first
    result = try_rank13_search()
    if result is not None:
        return result
    
    # Try ALS search for rank 13
    result = try_rank13_als_search()
    if result is not None:
        return result
    
    # Try the structured rank-14 decomposition
    result = try_strassen_rank14()
    if result is not None:
        return result
    
    # Fallback to naive rank 16
    return naive_decomp()
# EVOLVE-BLOCK-END
