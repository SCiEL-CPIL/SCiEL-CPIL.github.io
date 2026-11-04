"""Matrix multiplication 2x3x4 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 4]
DEADLINE = 150.0
START_TIME = time.time()

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 4

    def verify(U, V, W):
        """Check exact decomposition: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1"""
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(R):
                        total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if not np.isclose(total, 1.0, atol=1e-9):
                        return False
        return True

    def build_rank22():
        """
        Systematic rank-22 construction:
        C = A(2x3) * B(3x4)
        Split: C = A[:,0:2]*B[0:2,:] + A[:,2:3]*B[2:3,:]
        First term: 2x2 * 2x4 = two independent 2x2*2x2 blocks (Strassen rank 7 each) = 14
        Second term: 2x1 * 1x4 = outer product, rank 8
        Total: 22
        """
        rank = 22
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        # Strassen for 2x2*2x2: returns 7 (U_row, V_row, W_row) triples
        # Each triple: (u_indices, v_indices, w_indices) with coefficients
        # Standard Strassen for A(2x2)*B(2x2)->C(2x2):
        # m1=(a00+a11)(b00+b11): C00+=m1, C11+=m1
        # m2=(a10+a11)(b00): C10+=m2, C11-=m2
        # m3=a00*(b01-b11): C01+=m3, C11+=m3
        # m4=a11*(b10-b00): C00+=m4, C10+=m4
        # m5=(a00+a01)*b11: C00-=m5, C01+=m5
        # m6=(a10-a00)*(b00+b10): C11+=m6
        # m7=(a01-a11)*(b10+b11): C00+=m7

        def strassen_2x2(offset, a_row0, a_row1, b_col0, b_col1, c_row0, c_row1):
            """
            a_row0, a_row1: row indices in A (the 2 rows of the 2x2 sub-block)
            b_col0, b_col1: col indices in B (the 2 cols of the 2x2 sub-block)
            c_row0, c_row1: row indices in C
            The sub-block is A[a_row0:a_row0+2, a_row0:a_row0+2] * B[b_col0:b_col0+2, b_col0:b_col0+2]
            Wait, that's not right. Let me think more carefully.
            
            Actually for A(2x3)*B(3x4):
            A has rows 0,1 and cols 0,1,2
            B has rows 0,1,2 and cols 0,1,2,3
            C has rows 0,1 and cols 0,1,2,3
            
            C[i,k] = sum_j A[i,j]*B[j,k]
            
            For the 2x2*2x2 block using A rows {0,1}, A cols {0,1}, B rows {0,1}, B cols {0,1}:
            This contributes to C[0,0], C[0,1], C[1,0], C[1,1]
            """
            # A sub-block indices: A[0,0],A[0,1],A[1,0],A[1,1] -> flat: 0,1,3,4 (i*m+j)
            # B sub-block indices: B[0,0],B[0,1],B[1,0],B[1,1] -> flat: 0,1,4,5 (j*p+k)
            # C sub-block: C[0,0],C[0,1],C[1,0],C[1,1] -> W[k*n+i]: C[0,0]->W[0], C[0,1]->W[1], C[1,0]->W[2], C[1,1]->W[3]
            
            a00 = 0*3+0  # A[0,0]
            a01 = 0*3+1  # A[0,1]
            a10 = 1*3+0  # A[1,0]
            a11 = 1*3+1  # A[1,1]
            
            b00 = 0*4+0  # B[0,0]
            b01 = 0*4+1  # B[0,1]
            b10 = 1*4+0  # B[1,0]
            b11 = 1*4+1  # B[1,1]
            
            c00 = 0*2+0  # C[0,0] -> W[k*n+i] = W[0*2+0]
            c01 = 0*2+1  # C[0,1] -> W[0*2+1]
            c10 = 1*2+0  # C[1,0] -> W[1*2+0]
            c11 = 1*2+1  # C[1,1] -> W[1*2+1]
            
            # m1
            U[a00, offset+0] += 1; U[a11, offset+0] += 1
            V[b00, offset+0] += 1; V[b11, offset+0] += 1
            W[c00, offset+0] += 1; W[c11, offset+0] += 1
            # m2
            U[a10, offset+1] += 1; U[a11, offset+1] += 1
            V[b00, offset+1] += 1
            W[c10, offset+1] += 1; W[c11, offset+1] += -1
            # m3
            U[a00, offset+2] += 1
            V[b01, offset+2] += 1; V[b11, offset+2] += -1
            W[c01, offset+2] += 1; W[c11, offset+2] += 1
            # m4
            U[a11, offset+3] += 1
            V[b10, offset+3] += 1; V[b00, offset+3] += -1
            W[c00, offset+3] += 1; W[c10, offset+3] += 1
            # m5
            U[a00, offset+4] += 1; U[a01, offset+4] += 1
            V[b11, offset+4] += 1
            W[c00, offset+4] += -1; W[c01, offset+4] += 1
            # m6
            U[a10, offset+5] += 1; U[a00, offset+5] += -1
            V[b00, offset+5] += 1; V[b10, offset+5] += 1
            W[c11, offset+5] += 1
            # m7
            U[a01, offset+6] += 1; U[a11, offset+6] += -1
            V[b10, offset+6] += 1; V[b11, offset+6] += 1
            W[c00, offset+6] += 1

        def strassen_2x2_offset(offset, a_cols, b_cols, c_cols):
            """
            Generalized Strassen for A[:, a_cols[0]:a_cols[0]+2] * B[a_cols[0]:a_cols[0]+2, b_cols[0]:b_cols[0]+2]
            contributing to C[:, c_cols[0]:c_cols[0]+2]
            
            a_cols: the 2 column indices of A used (and correspondingly row indices of B)
            b_cols: the 2 column indices of B
            c_cols: the 2 column indices of C
            """
            # A rows are always 0,1 (since n=2)
            # A[i, a_cols[0]], A[i, a_cols[1]] for i=0,1
            # B[a_cols[0], b_cols[0]], B[a_cols[0], b_cols[1]], B[a_cols[1], b_cols[0]], B[a_cols[1], b_cols[1]]
            # C[i, c_cols[0]], C[i, c_cols[1]] for i=0,1
            
            a00 = 0*m + a_cols[0]  # A[0, a_cols[0]]
            a01 = 0*m + a_cols[1]  # A[0, a_cols[1]]
            a10 = 1*m + a_cols[0]  # A[1, a_cols[0]]
            a11 = 1*m + a_cols[1]  # A[1, a_cols[1]]
            
            b00 = a_cols[0]*p + b_cols[0]  # B[a_cols[0], b_cols[0]]
            b01 = a_cols[0]*p + b_cols[1]  # B[a_cols[0], b_cols[1]]
            b10 = a_cols[1]*p + b_cols[0]  # B[a_cols[1], b_cols[0]]
            b11 = a_cols[1]*p + b_cols[1]  # B[a_cols[1], b_cols[1]]
            
            c00 = c_cols[0]*n + 0  # C[0, c_cols[0]]
            c01 = c_cols[1]*n + 0  # C[0, c_cols[1]]
            c10 = c_cols[0]*n + 1  # C[1, c_cols[0]]
            c11 = c_cols[1]*n + 1  # C[1, c_cols[1]]
            
            # m1 = (a00+a11)(b00+b11): C00+=m1, C11+=m1
            U[a00, offset+0] += 1; U[a11, offset+0] += 1
            V[b00, offset+0] += 1; V[b11, offset+0] += 1
            W[c00, offset+0] += 1; W[c11, offset+0] += 1
            # m2 = (a10+a11)(b00): C10+=m2, C11-=m2
            U[a10, offset+1] += 1; U[a11, offset+1] += 1
            V[b00, offset+1] += 1
            W[c10, offset+1] += 1; W[c11, offset+1] += -1
            # m3 = a00*(b01-b11): C01+=m3, C11+=m3
            U[a00, offset+2] += 1
            V[b01, offset+2] += 1; V[b11, offset+2] += -1
            W[c01, offset+2] += 1; W[c11, offset+2] += 1
            # m4 = a11*(b10-b00): C00+=m4, C10+=m4
            U[a11, offset+3] += 1
            V[b10, offset+3] += 1; V[b00, offset+3] += -1
            W[c00, offset+3] += 1; W[c10, offset+3] += 1
            # m5 = (a00+a01)*b11: C00-=m5, C01+=m5
            U[a00, offset+4] += 1; U[a01, offset+4] += 1
            V[b11, offset+4] += 1
            W[c00, offset+4] += -1; W[c01, offset+4] += 1
            # m6 = (a10-a00)*(b00+b10): C11+=m6
            U[a10, offset+5] += 1; U[a00, offset+5] += -1
            V[b00, offset+5] += 1; V[b10, offset+5] += 1
            W[c11, offset+5] += 1
            # m7 = (a01-a11)*(b10+b11): C00+=m7
            U[a01, offset+6] += 1; U[a11, offset+6] += -1
            V[b10, offset+6] += 1; V[b11, offset+6] += 1
            W[c00, offset+6] += 1

        # Block 1: A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2]
        strassen_2x2_offset(0, [0, 1], [0, 1], [0, 1])
        
        # Block 2: A[:, 0:2] * B[0:2, 2:4] -> C[:, 2:4]
        strassen_2x2_offset(7, [0, 1], [2, 3], [2, 3])
        
        # Block 3: A[:, 2:3] * B[2:3, 0:4] -> C[:, 0:4]
        # This is (2x1) * (1x4), rank 8
        # C[i,k] += A[i,2] * B[2,k] for i in {0,1}, k in {0,1,2,3}
        offset = 14
        for i in range(n):
            for k in range(p):
                U[i*m + 2, offset] += 1
                V[2*p + k, offset] += 1
                W[k*n + i, offset] += 1
                offset += 1

        return U, V, W

    # Build and verify rank-22
    U22, V22, W22 = build_rank22()
    
    if not verify(U22, V22, W22):
        # Fallback to schoolbook rank-24
        U_school = np.zeros((n * m, 24))
        V_school = np.zeros((m * p, 24))
        W_school = np.zeros((p * n, 24))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j, idx] = 1
                    V_school[j * p + k, idx] = 1
                    W_school[k * n + i, idx] = 1
                    idx += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    best_result = {"U": U22.tolist(), "V": V22.tolist(), "W": W22.tolist()}

    # Try rank 21 with alternating least squares and multiple restarts
    target_rank = 21
    deadline_guard = DEADLINE - 15  # safety margin
    
    # Precompute index structures for efficiency
    # For solving U[i*m+j, :]: need V[j*p+k, r] * W[k*n+i, r] for all k, r
    # For solving V[j*p+k, :]: need U[i*m+j, r] * W[k*n+i, r] for all i, r
    # For solving W[k*n+i, :]: need U[i*m+j, r] * V[j*p+k, r] for all j, r
    
    def als_one_step(U, V, W, R):
        """One full ALS step: update U, then V, then W."""
        # Update U
        for i in range(n):
            for j in range(m):
                # A_mat[k, r] = V[j*p+k, r] * W[k*n+i, r]
                A_mat = np.zeros((p, R))
                for k in range(p):
                    A_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
                b_vec = np.ones(p)
                sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                U[i*m+j, :] = sol
        
        # Update V
        for j in range(m):
            for k in range(p):
                A_mat = np.zeros((n, R))
                for i in range(n):
                    A_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                b_vec = np.ones(n)
                sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                V[j*p+k, :] = sol
        
        # Update W
        for k in range(p):
            for i in range(n):
                A_mat = np.zeros((m, R))
                for j in range(m):
                    A_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
                b_vec = np.ones(m)
                sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                W[k*n+i, :] = sol

    def compute_residual(U, V, W):
        """Compute max absolute residual over all (i,j,k)."""
        max_res = 0.0
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(R):
                        total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    res = abs(total - 1.0)
                    if res > max_res:
                        max_res = res
        return max_res

    found_21 = False
    max_restarts = 30
    max_iters = 80
    
    for attempt in range(max_restarts):
        if time.time() > deadline_guard:
            break
        
        # Diverse random initialization
        np.random.seed(attempt * 7919 + 42)
        scale = 1.0
        U = np.random.randn(n*m, target_rank) * scale
        V = np.random.randn(m*p, target_rank) * scale
        W = np.random.randn(p*n, target_rank) * scale
        
        prev_res = np.inf
        for it in range(max_iters):
            if time.time() > deadline_guard:
                break
            als_one_step(U, V, W, target_rank)
            res = compute_residual(U, V, W)
            if res < 1e-10:
                # Check if we can round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                if verify(U_r, V_r, W_r):
                    best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    found_21 = True
                    break
                elif verify(U, V, W):
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    found_21 = True
                    break
            if abs(res - prev_res) < 1e-12:
                break
            prev_res = res
        else:
            # Loop completed without break
            res = compute_residual(U, V, W)
            if res < 1e-10:
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                if verify(U_r, V_r, W_r):
                    best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    found_21 = True
                elif verify(U, V, W):
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    found_21 = True
        
        if found_21:
            break

    return best_result
# EVOLVE-BLOCK-END
