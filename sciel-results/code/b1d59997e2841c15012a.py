import numpy as np
import time
import sys

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin before 150s kill

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
        # Vectorized: for each r, outer product of A[:,r] and B[:,r]
        # A.T is (R, dim1), B.T is (R, dim2)
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
        # Strassen's 7 products for 2x2 matrices
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # C = [[p1+p4-p5+p6, p2+p4], [p3+p5, p1-p3+p4+p7]]
        
        # U: (4, 7) - rows indexed by (i,j) for A
        # V: (4, 7) - rows indexed by (j,k) for B  
        # W: (4, 7) - rows indexed by (k,i) for C
        
        # Let's use the standard Strassen formulas
        # p1 = (a+d)(e+h)
        # p2 = (c+d)e
        # p3 = a(f-h)
        # p4 = d(g-f)
        # p5 = (a+b)h
        # p6 = (c+d)(f-h)
        # p7 = (a-b)(e+g)
        
        # U rows: a=0, b=1, c=2, d=3
        # V rows: e=0, f=1, g=2, h=3
        # W rows: c00=0, c01=1, c10=2, c11=3
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # p1 = (a+d)(e+h)
        U[0, 0] = 1  # a
        U[3, 0] = 1  # d
        V[0, 0] = 1  # e
        V[3, 0] = 1  # h
        W[0, 0] = 1  # c00
        W[1, 0] = 0
        W[2, 0] = 0
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
        W[1, 5] = 0
        W[2, 5] = 0
        W[3, 5] = 0
        
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
        Split the 5x6 matrix into blocks and use Strassen for 2x2 subproblems.
        
        For 2x5 * 5x6:
        A is 2x5, B is 5x6, C is 2x6.
        
        We can split B into three 5x2 blocks: B1 (cols 0-1), B2 (cols 2-3), B3 (cols 4-5)
        C = A * B = [A*B1, A*B2, A*B3]
        
        Each A*B_i is 2x2 * 2x5... wait, A is 2x5 and B_i is 5x2, so A*B_i is 2x2.
        
        Actually, let me think differently. Let's split A into 2x5 = [A1, A2, A3, A4, A5] where each Ai is 2x1.
        And B into 5x6 = [B1; B2; B3; B4; B5] where each Bi is 1x6.
        
        C = sum_i Ai * Bi
        
        That's 5 terms of rank 2*6=12 each, total rank 60. Not helpful.
        
        Better approach: Use block structure.
        Split A (2x5) into [A_left (2x3), A_right (2x2)]
        Split B (5x6) into [B_top (3x6); B_bottom (2x6)]
        
        C = A_left * B_top + A_right * B_bottom
        
        Each product is 2x3 * 3x6 = 2x6.
        Rank of 2x3*3x6: schoolbook is 18.
        
        Hmm, this is getting complicated. Let me try a different approach.
        
        Actually, for 2x5 * 5x6, we can think of it as:
        C[i,k] = sum_j A[i,j] * B[j,k]
        
        One approach: Use the fact that 2x2 has rank 7 (Strassen).
        Split the 5 columns into 2+2+1 and 6 rows into 2+2+2.
        
        Let's try: Split A into 3 blocks: A1 (2x2), A2 (2x2), A3 (2x1)
        Split B into 3 blocks: B1 (2x6), B2 (2x6), B3 (1x6)
        
        C = A1*B1 + A2*B2 + A3*B3
        
        A1*B1: 2x2 * 2x6 = 2x6, rank <= 7*3 = 21 (using Strassen on each 2x2*2x2 subblock)
        A2*B2: same, rank <= 21
        A3*B3: 2x1 * 1x6 = 2x6, rank = 2*6 = 12 (outer product structure)
        
        Total rank <= 21 + 21 + 12 = 54. Still worse than schoolbook 60? No, 54 < 60.
        
        But this is complex to implement correctly. Let me just use the ALS approach with better parameters.
        """
        return None

    min_rank = 20
    max_rank = 59
    
    found_solution = None
    
    # Try ranks from low to high
    for R in range(min_rank, max_rank + 1):
        if time.time() > deadline:
            break
            
        # Number of restarts: more for lower ranks (harder), fewer for higher
        if R < 30:
            restarts = 30
        elif R < 45:
            restarts = 15
        else:
            restarts = 10
        
        for r in range(restarts):
            if time.time() > deadline:
                break
            
            # Random initialization with varying scales
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
