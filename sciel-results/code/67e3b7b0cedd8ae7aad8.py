import numpy as np
import time
import sys

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p

    start_time = time.time()
    deadline = start_time + 135.0

    # Precompute the target tensor unfoldings
    T1 = np.zeros((n*m, m*p * p*n))
    T2 = np.zeros((m*p, n*m * p*n))
    T3 = np.zeros((p*n, n*m * m*p))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                
                T1[a, b * (p*n) + c] = 1.0
                T2[b, a * (p*n) + c] = 1.0
                T3[c, a * (m*p) + b] = 1.0
                
    T1_T = T1.T
    T2_T = T2.T
    T3_T = T3.T

    def khatri_rao(A, B):
        # A: (dim1, R), B: (dim2, R)
        # Returns: (dim1*dim2, R) where each column is kron(A[:,r], B[:,r])
        dim1, R = A.shape
        dim2, _ = B.shape
        prod = A.T[:, :, np.newaxis] * B.T[:, np.newaxis, :]
        return prod.reshape(R, dim1 * dim2).T

    def check_exact(U, V, W):
        MWV = khatri_rao(W, V)
        T1_recon = U @ MWV.T
        if not np.allclose(T1_recon, T1, atol=1e-9):
            return False
        MUW = khatri_rao(U, W)
        T2_recon = V @ MUW.T
        if not np.allclose(T2_recon, T2, atol=1e-9):
            return False
        MVU = khatri_rao(V, U)
        T3_recon = W @ MVU.T
        if not np.allclose(T3_recon, T3, atol=1e-9):
            return False
        return True

    def snap_and_verify(U, V, W):
        # Snap to half-integers
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        if check_exact(U_s, V_s, W_s):
            return U_s, V_s, W_s
        # Also try integer snapping
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        if check_exact(U_i, V_i, W_i):
            return U_i, V_i, W_i
        return None

    def polish(U, V, W, max_iter=50):
        """High-precision polish after ALS convergence"""
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            # Update U
            MWV = khatri_rao(W, V)
            U = np.linalg.lstsq(MWV, T1_T, rcond=None)[0].T
            
            # Update V
            MUW = khatri_rao(U, W)
            V = np.linalg.lstsq(MUW, T2_T, rcond=None)[0].T
            
            # Update W
            MVU = khatri_rao(V, U)
            W = np.linalg.lstsq(MVU, T3_T, rcond=None)[0].T
            
            # Check convergence
            MWV = khatri_rao(W, V)
            T1_recon = U @ MWV.T
            err = np.linalg.norm(T1_recon - T1)
            if err < 1e-12:
                res = snap_and_verify(U, V, W)
                if res:
                    return res
        return None

    def solve_als(U, V, W, max_iter=30):
        prev_err = np.inf
        for _ in range(max_iter):
            if time.time() > deadline:
                break
                
            # Update U
            MWV = khatri_rao(W, V)
            U = np.linalg.lstsq(MWV, T1_T, rcond=None)[0].T
            
            # Update V
            MUW = khatri_rao(U, W)
            V = np.linalg.lstsq(MUW, T2_T, rcond=None)[0].T
            
            # Update W
            MVU = khatri_rao(V, U)
            W = np.linalg.lstsq(MVU, T3_T, rcond=None)[0].T
            
            # Calculate error
            MWV = khatri_rao(W, V)
            T1_recon = U @ MWV.T
            err = np.linalg.norm(T1_recon - T1)
            
            if err < 1e-8:
                # Try polishing
                res = polish(U, V, W)
                if res:
                    return res, True
            
            if err > prev_err * 1.5 and _ > 5:
                break
            prev_err = err
            
        return (U, V, W), False

    def strassen_2x2x2():
        """
        Return Strassen decomposition for 2x2 * 2x2.
        Rank 7 decomposition.
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # p1 = (a+d)(e+h)
        U[0, 0] = 1  # a
        U[3, 0] = 1  # d
        V[0, 0] = 1  # e
        V[3, 0] = 1  # h
        W[0, 0] = 1  # c00
        W[3, 0] = 1  # c11
        
        # p2 = (c+d)e
        U[2, 1] = 1  # c
        U[3, 1] = 1  # d
        V[0, 1] = 1  # e
        W[1, 1] = 1  # c01
        W[3, 1] = 1  # c11
        
        # p3 = a(f-h)
        U[0, 2] = 1  # a
        V[1, 2] = 1  # f
        V[3, 2] = -1 # h
        W[2, 2] = 1  # c10
        
        # p4 = d(g-f)
        U[3, 3] = 1  # d
        V[2, 3] = 1  # g
        V[1, 3] = -1 # f
        W[0, 3] = 1  # c00
        W[3, 3] = 1  # c11
        
        # p5 = (a+b)h
        U[0, 4] = 1  # a
        U[1, 4] = 1  # b
        V[3, 4] = 1  # h
        W[0, 4] = -1 # c00
        W[2, 4] = 1  # c10
        
        # p6 = (c+d)(f-h)
        U[2, 5] = 1  # c
        U[3, 5] = 1  # d
        V[1, 5] = 1  # f
        V[3, 5] = -1 # h
        W[0, 5] = 1  # c00
        
        # p7 = (a-b)(e+g)
        U[0, 6] = 1  # a
        U[1, 6] = -1 # b
        V[0, 6] = 1  # e
        V[2, 6] = 1  # g
        W[3, 6] = 1  # c11
        
        return U, V, W

    def build_strassen_based_solution():
        """
        Build a Strassen-based solution for 2x5 * 5x6.
        Rank 43: 2 blocks of Strassen (2x2x2) + 1 block of rank-1 (2x1x6).
        """
        # Global dimensions
        # A: 2x5, indices a = i*5 + j
        # B: 5x6, indices b = j*6 + k
        # C: 2x6, indices c = k*2 + i
        
        # Blocks:
        # Block 1: j in {0, 1}, k in {0, 1}. A is 2x2, B is 2x2, C is 2x2.
        # Block 2: j in {2, 3}, k in {2, 3}. A is 2x2, B is 2x2, C is 2x2.
        # Block 3: j in {4}, k in {4, 5}. A is 2x1, B is 1x2, C is 2x2.
        # Wait, the previous plan was:
        # Block 1: j in {0, 1}, k in {0, 1}.
        # Block 2: j in {2, 3}, k in {2, 3}.
        # Block 3: j in {4}, k in {4, 5}.
        
        # Let's verify indices for Block 3.
        # j=4. k=4, 5.
        # A indices: i in {0,1}, j=4. a = i*5+4. So a in {4, 9}.
        # B indices: j=4, k in {4,5}. b = 4*6+k. So b in {28, 29}.
        # C indices: k in {4,5}, i in {0,1}. c = k*2+i. So c in {8, 9, 10, 11}.
        
        # This is a 2x1 * 1x2 = 2x2 problem.
        # Rank 1 decomposition?
        # A = [[a0], [a1]], B = [[b0, b1]].
        # C = [[a0*b0, a0*b1], [a1*b0, a1*b1]] = [a0, a1]^T * [b0, b1].
        # This is rank 1!
        # U: a0=1, a1=1. (Wait, U is 2x1 vector).
        # V: b0=1, b1=1. (Wait, V is 2x1 vector).
        # W: 2x2 matrix of 1s.
        
        # So Block 3 has rank 1.
        # Total rank = 7 + 7 + 1 = 15?
        # No, the blocks are disjoint in A and B indices, but they write to C.
        # Do they write to disjoint C indices?
        # Block 1 C indices: k in {0,1}, i in {0,1}. c in {0,1,2,3}.
        # Block 2 C indices: k in {2,3}, i in {0,1}. c in {4,5,6,7}.
        # Block 3 C indices: k in {4,5}, i in {0,1}. c in {8,9,10,11}.
        # Yes! The C indices are disjoint.
        # So we can simply sum the decompositions.
        
        U_total = np.zeros((n*m, 15))
        V_total = np.zeros((m*p, 15))
        W_total = np.zeros((p*n, 15))
        
        # Block 1
        U_s, V_s, W_s = strassen_2x2x2()
        
        # Map indices for Block 1
        # A: i in {0,1}, j in {0,1}. a = i*5+j.
        # Local A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1).
        # Global A indices: 0, 1, 5, 6.
        a_map_1 = [0, 1, 5, 6]
        
        # B: j in {0,1}, k in {0,1}. b = j*6+k.
        # Local B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1).
        # Global B indices: 0, 1, 6, 7.
        b_map_1 = [0, 1, 6, 7]
        
        # C: k in {0,1}, i in {0,1}. c = k*2+i.
        # Local C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1).
        # Global C indices: 0, 1, 2, 3.
        c_map_1 = [0, 1, 2, 3]
        
        for r in range(7):
            for local_a, global_a in enumerate(a_map_1):
                U_total[global_a, r] = U_s[local_a, r]
            for local_b, global_b in enumerate(b_map_1):
                V_total[global_b, r] = V_s[local_b, r]
            for local_c, global_c in enumerate(c_map_1):
                W_total[global_c, r] = W_s[local_c, r]
        
        # Block 2
        # A: i in {0,1}, j in {2,3}. a = i*5+j.
        # Local A indices: 0:(0,2), 1:(0,3), 2:(1,2), 3:(1,3).
        # Global A indices: 2, 3, 7, 8.
        a_map_2 = [2, 3, 7, 8]
        
        # B: j in {2,3}, k in {2,3}. b = j*6+k.
        # Local B indices: 0:(2,2), 1:(2,3), 2:(3,2), 3:(3,3).
        # Global B indices: 14, 15, 20, 21.
        b_map_2 = [14, 15, 20, 21]
        
        # C: k in {2,3}, i in {0,1}. c = k*2+i.
        # Local C indices: 0:(2,0), 1:(2,1), 2:(3,0), 3:(3,1).
        # Global C indices: 4, 5, 6, 7.
        c_map_2 = [4, 5, 6, 7]
        
        for r in range(7, 14):
            for local_a, global_a in enumerate(a_map_2):
                U_total[global_a, r] = U_s[local_a, r-7]
            for local_b, global_b in enumerate(b_map_2):
                V_total[global_b, r] = V_s[local_b, r-7]
            for local_c, global_c in enumerate(c_map_2):
                W_total[global_c, r] = W_s[local_c, r-7]
        
        # Block 3
        # A: i in {0,1}, j=4. a = i*5+4.
        # Global A indices: 4, 9.
        a_map_3 = [4, 9]
        
        # B: j=4, k in {4,5}. b = 4*6+k.
        # Global B indices: 28, 29.
        b_map_3 = [28, 29]
        
        # C: k in {4,5}, i in {0,1}. c = k*2+i.
        # Global C indices: 8, 9, 10, 11.
        c_map_3 = [8, 9, 10, 11]
        
        # Rank 1 decomposition
        # U: [1, 1]^T
        # V: [1, 1]^T
        # W: [[1, 1], [1, 1]]
        
        r = 14
        U_total[a_map_3[0], r] = 1
        U_total[a_map_3[1], r] = 1
        V_total[b_map_3[0], r] = 1
        V_total[b_map_3[1], r] = 1
        W_total[c_map_3[0], r] = 1
        W_total[c_map_3[1], r] = 1
        W_total[c_map_3[2], r] = 1
        W_total[c_map_3[3], r] = 1
        
        return U_total, V_total, W_total

    # First, try the constructive Strassen-based solution
    U_c, V_c, W_c = build_strassen_based_solution()
    if check_exact(U_c, V_c, W_c):
        return {"U": U_c.tolist(), "V": V_c.tolist(), "W": W_c.tolist()}

    # If that fails (shouldn't), fall back to ALS
    min_rank = 16
    max_rank = 59
    
    found_solution = None
    
    for R in range(min_rank, max_rank + 1):
        if time.time() > deadline:
            break
            
        restarts = 10
        for r in range(restarts):
            if time.time() > deadline:
                break
            
            scale = np.sqrt(1.0 / R)
            U = np.random.randn(n*m, R) * scale
            V = np.random.randn(m*p, R) * scale
            W = np.random.randn(p*n, R) * scale
            
            sol, success = solve_als(U, V, W)
            if success:
                found_solution = sol
                break
        if found_solution:
            break

    if found_solution:
        U, V, W = found_solution
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to Schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
