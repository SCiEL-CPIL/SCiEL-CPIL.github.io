import numpy as np
import time

def construct():
    n, m, p = 2, 2, 7
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 150s kill

    def verify(U_mat, V_mat, W_mat):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    val = np.dot(U_mat[a_idx, :], V_mat[b_idx, :] * W_mat[c_idx, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for j2 in range(m):
                        if j2 != j:
                            a_idx = i * m + j
                            b_idx = j2 * p + k
                            c_idx = k * n + i
                            val = np.dot(U_mat[a_idx, :], V_mat[b_idx, :] * W_mat[c_idx, :])
                            if abs(val) > 1e-9:
                                return False
        return True

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, r] = 1.0
                    V[j * p + k, r] = 1.0
                    W[k * n + i, r] = 1.0
                    r += 1
        return U, V, W

    # Build rank-25: three Strassen blocks (7 each) + 4 schoolbook terms for last column
    def build_rank25():
        rank = 25
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)

        def add_strassen_block(col_start, k_start):
            # Term 0: (A[0,0]+A[1,1])*(B[0,k0]+B[1,k1])
            U[0, col_start+0] += 1
            U[3, col_start+0] += 1
            V[7*0 + k_start, col_start+0] += 1
            V[7*1 + (k_start+1), col_start+0] += 1
            W[2*k_start + 0, col_start+0] += 1
            W[2*(k_start+1) + 1, col_start+0] += 1

            # Term 1: (A[1,0]+A[1,1])*(B[0,k0])
            U[2, col_start+1] += 1
            U[3, col_start+1] += 1
            V[7*0 + k_start, col_start+1] += 1
            W[2*k_start + 1, col_start+1] += 1
            W[2*(k_start+1) + 1, col_start+1] -= 1

            # Term 2: A[0,0]*(B[0,k1] - B[1,k1])
            U[0, col_start+2] += 1
            V[7*0 + (k_start+1), col_start+2] += 1
            V[7*1 + (k_start+1), col_start+2] -= 1
            W[2*(k_start+1) + 0, col_start+2] += 1
            W[2*(k_start+1) + 1, col_start+2] += 1

            # Term 3: A[1,1]*(B[1,k0] - B[0,k0])
            U[3, col_start+3] += 1
            V[7*1 + k_start, col_start+3] += 1
            V[7*0 + k_start, col_start+3] -= 1
            W[2*k_start + 0, col_start+3] += 1
            W[2*k_start + 1, col_start+3] += 1

            # Term 4: (A[0,0]+A[0,1])*(B[1,k1])
            U[0, col_start+4] += 1
            U[1, col_start+4] += 1
            V[7*1 + (k_start+1), col_start+4] += 1
            W[2*k_start + 0, col_start+4] -= 1
            W[2*(k_start+1) + 0, col_start+4] += 1

            # Term 5: (A[1,0]-A[0,0])*(B[0,k0]+B[0,k1])
            U[2, col_start+5] += 1
            U[0, col_start+5] -= 1
            V[7*0 + k_start, col_start+5] += 1
            V[7*0 + (k_start+1), col_start+5] += 1
            W[2*(k_start+1) + 1, col_start+5] += 1

            # Term 6: (A[0,1]-A[1,1])*(B[1,k0]+B[1,k1])
            U[1, col_start+6] += 1
            U[3, col_start+6] -= 1
            V[7*1 + k_start, col_start+6] += 1
            V[7*1 + (k_start+1), col_start+6] += 1
            W[2*k_start + 0, col_start+6] += 1

        # Apply Strassen blocks for column pairs (0,1), (2,3), (4,5)
        add_strassen_block(0, 0)
        add_strassen_block(7, 2)
        add_strassen_block(14, 4)

        # Handle column 6 with 4 separate terms
        col_start = 21
        k = 6
        U[0, col_start+0] += 1
        V[7*0 + k, col_start+0] += 1
        W[2*k + 0, col_start+0] += 1

        U[1, col_start+1] += 1
        V[7*1 + k, col_start+1] += 1
        W[2*k + 0, col_start+1] += 1

        U[2, col_start+2] += 1
        V[7*0 + k, col_start+2] += 1
        W[2*k + 1, col_start+2] += 1

        U[3, col_start+3] += 1
        V[7*1 + k, col_start+3] += 1
        W[2*k + 1, col_start+3] += 1

        return U, V, W

    # Try to find a rank-24 decomposition using a different block structure
    # Idea: Use a rank-8 decomposition for 2x2x3 and combine with rank-7 for 2x2x2
    # Total: 8 + 7 + 7 + 4 = 26? No, that's worse.
    # Alternative: 7 + 7 + 10 = 24 if we can do 2x2x4 in rank 10
    
    def build_rank24():
        """Try rank-24: two Strassen blocks (7+7) + rank-10 for 2x2x4"""
        rank = 24
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)

        def add_strassen_block(col_start, k_start):
            U[0, col_start+0] += 1
            U[3, col_start+0] += 1
            V[7*0 + k_start, col_start+0] += 1
            V[7*1 + (k_start+1), col_start+0] += 1
            W[2*k_start + 0, col_start+0] += 1
            W[2*(k_start+1) + 1, col_start+0] += 1

            U[2, col_start+1] += 1
            U[3, col_start+1] += 1
            V[7*0 + k_start, col_start+1] += 1
            W[2*k_start + 1, col_start+1] += 1
            W[2*(k_start+1) + 1, col_start+1] -= 1

            U[0, col_start+2] += 1
            V[7*0 + (k_start+1), col_start+2] += 1
            V[7*1 + (k_start+1), col_start+2] -= 1
            W[2*(k_start+1) + 0, col_start+2] += 1
            W[2*(k_start+1) + 1, col_start+2] += 1

            U[3, col_start+3] += 1
            V[7*1 + k_start, col_start+3] += 1
            V[7*0 + k_start, col_start+3] -= 1
            W[2*k_start + 0, col_start+3] += 1
            W[2*k_start + 1, col_start+3] += 1

            U[0, col_start+4] += 1
            U[1, col_start+4] += 1
            V[7*1 + (k_start+1), col_start+4] += 1
            W[2*k_start + 0, col_start+4] -= 1
            W[2*(k_start+1) + 0, col_start+4] += 1

            U[2, col_start+5] += 1
            U[0, col_start+5] -= 1
            V[7*0 + k_start, col_start+5] += 1
            V[7*0 + (k_start+1), col_start+5] += 1
            W[2*(k_start+1) + 1, col_start+5] += 1

            U[1, col_start+6] += 1
            U[3, col_start+6] -= 1
            V[7*1 + k_start, col_start+6] += 1
            V[7*1 + (k_start+1), col_start+6] += 1
            W[2*k_start + 0, col_start+6] += 1

        # Strassen for cols (0,1) and (2,3)
        add_strassen_block(0, 0)
        add_strassen_block(7, 2)

        # For cols (4,5,6), try a rank-10 decomposition
        # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for k=4,5,6
        # This is 2x2x3, which needs at least 7+3=10 terms if we can share.
        
        # Let me try a specific rank-10 construction for 2x2x3:
        # Using a modified approach where we share some terms.
        
        col_start = 14
        # Terms for cols 4,5,6 (indices 4,5,6 in B)
        
        # Term 0: (A[0,0]+A[1,1])*(B[0,4]+B[1,5])
        U[0, col_start+0] += 1
        U[3, col_start+0] += 1
        V[7*0 + 4, col_start+0] += 1
        V[7*1 + 5, col_start+0] += 1
        W[2*4 + 0, col_start+0] += 1
        W[2*5 + 1, col_start+0] += 1

        # Term 1: (A[1,0]+A[1,1])*(B[0,4])
        U[2, col_start+1] += 1
        U[3, col_start+1] += 1
        V[7*0 + 4, col_start+1] += 1
        W[2*4 + 1, col_start+1] += 1
        W[2*5 + 1, col_start+1] -= 1

        # Term 2: A[0,0]*(B[0,5] - B[1,5])
        U[0, col_start+2] += 1
        V[7*0 + 5, col_start+2] += 1
        V[7*1 + 5, col_start+2] -= 1
        W[2*5 + 0, col_start+2] += 1
        W[2*5 + 1, col_start+2] += 1

        # Term 3: A[1,1]*(B[1,4] - B[0,4])
        U[3, col_start+3] += 1
        V[7*1 + 4, col_start+3] += 1
        V[7*0 + 4, col_start+3] -= 1
        W[2*4 + 0, col_start+3] += 1
        W[2*4 + 1, col_start+3] += 1

        # Term 4: (A[0,0]+A[0,1])*(B[1,5])
        U[0, col_start+4] += 1
        U[1, col_start+4] += 1
        V[7*1 + 5, col_start+4] += 1
        W[2*4 + 0, col_start+4] -= 1
        W[2*5 + 0, col_start+4] += 1

        # Term 5: (A[1,0]-A[0,0])*(B[0,4]+B[0,5])
        U[2, col_start+5] += 1
        U[0, col_start+5] -= 1
        V[7*0 + 4, col_start+5] += 1
        V[7*0 + 5, col_start+5] += 1
        W[2*5 + 1, col_start+5] += 1

        # Term 6: (A[0,1]-A[1,1])*(B[1,4]+B[1,5])
        U[1, col_start+6] += 1
        U[3, col_start+6] -= 1
        V[7*1 + 4, col_start+6] += 1
        V[7*1 + 5, col_start+6] += 1
        W[2*4 + 0, col_start+6] += 1

        # Terms 7-9: Handle col 6 with 3 terms (not 4, sharing with above)
        # C[0,6] = A[0,0]*B[0,6] + A[0,1]*B[1,6]
        # C[1,6] = A[1,0]*B[0,6] + A[1,1]*B[1,6]
        
        # Term 7: (A[0,0]+A[1,1])*(B[0,6]+B[1,6])
        U[0, col_start+7] += 1
        U[3, col_start+7] += 1
        V[7*0 + 6, col_start+7] += 1
        V[7*1 + 6, col_start+7] += 1
        W[2*6 + 0, col_start+7] += 1
        W[2*6 + 1, col_start+7] += 1

        # Term 8: (A[1,0]-A[0,0])*(B[0,6])
        U[2, col_start+8] += 1
        U[0, col_start+8] -= 1
        V[7*0 + 6, col_start+8] += 1
        W[2*6 + 1, col_start+8] += 1
        W[2*6 + 0, col_start+8] -= 1

        # Term 9: (A[0,1]-A[1,1])*(B[1,6])
        U[1, col_start+9] += 1
        U[3, col_start+9] -= 1
        V[7*1 + 6, col_start+9] += 1
        W[2*6 + 0, col_start+9] += 1
        W[2*6 + 1, col_start+9] -= 1

        return U, V, W

    # Try rank-24 first
    if time.time() < deadline:
        U, V, W = build_rank24()
        if verify(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to rank-25
    U, V, W = build_rank25()
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Final fallback: schoolbook
    U, V, W = schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
