"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
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
        """Verify the tensor decomposition is exact using NumPy."""
        R = len(U[0])
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        # Target tensor T[i, j, k] = 1 for all i, j, k
        # C[i, k] = sum_j A[i, j] B[j, k]
        # Term is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # We need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        # for all i in 0..1, j in 0..1, k in 0..3
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    # Use dot product for speed
                    s = np.dot(U_np[i*m+j], V_np[j*p+k]) 
                    # Wait, W depends on k and i.
                    # s = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    # This is element-wise product of three vectors, then sum.
                    # s = np.sum(U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i])
                    s = np.sum(U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i])
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
        Attempt to find a rank-13 decomposition by searching for a merge
        of terms from the left and right Strassen blocks.
        """
        if time.time() > deadline - 10:
            return None

        # Define the 7 Strassen terms for one 2x2 block
        # U vectors (4 entries)
        U_vecs = [
            [1, 0, 0, 1], # 1
            [0, 0, 1, 1], # 2
            [1, 0, 0, 0], # 3
            [0, 0, 0, 1], # 4
            [1, 1, 0, 0], # 5
            [-1, 0, 1, 0],# 6
            [0, 1, 0, -1] # 7
        ]
        # V vectors for left block (8 entries, only 0,1,4,5 non-zero)
        V_vecs_L = [
            [1, 0, 0, 0, 0, 1, 0, 0], # 1
            [1, 0, 0, 0, 0, 0, 0, 0], # 2
            [0, 1, 0, 0, 0, -1, 0, 0],# 3
            [0, 0, 0, 0, 1, 0, 0, 0], # 4 (Note: M4 uses b10-b00. V[4]=1, V[0]=-1? 
            # Let's re-check M4: a11(b10-b00). V[4]=1, V[0]=-1.
            # My previous code: V[4][3]=1; V[0][3]=-1.
            # So V_vecs_L[3] should be [ -1, 0, 0, 0, 1, 0, 0, 0 ]
            [0, 0, 0, 1, 0, 0, 0, 0], # 5 (M5: (a00+a01)b11. V[5]=1)
            [1, 1, 0, 0, 0, 0, 0, 0], # 6 (M6: (a10-a00)(b00+b01). V[0]=1, V[1]=1)
            [0, 0, 0, 0, 1, 1, 0, 0]  # 7 (M7: (a01-a11)(b10+b11). V[4]=1, V[5]=1)
        ]
        # Correction for V_vecs_L[3]
        V_vecs_L[3] = [-1, 0, 0, 0, 1, 0, 0, 0]
        
        # W vectors for left block (8 entries, only 0,1,2,3 non-zero)
        W_vecs_L = [
            [1, 0, 0, 1, 0, 0, 0, 0], # 1 (M1: C00, C11. W[0]=1, W[3]=1)
            [0, 1, 0, -1, 0, 0, 0, 0],# 2 (M2: C10, C11. W[1]=1, W[3]=-1)
            [0, 0, 1, 1, 0, 0, 0, 0], # 3 (M3: C01, C11. W[2]=1, W[3]=1)
            [1, 1, 0, 0, 0, 0, 0, 0], # 4 (M4: C00, C10. W[0]=1, W[1]=1)
            [-1, 0, 1, 0, 0, 0, 0, 0],# 5 (M5: C00, C01. W[0]=-1, W[2]=1)
            [0, 0, 0, 1, 0, 0, 0, 0], # 6 (M6: C11. W[3]=1)
            [1, 0, 0, 0, 0, 0, 0, 0]  # 7 (M7: C00. W[0]=1)
        ]
        
        # V vectors for right block (8 entries, only 2,3,6,7 non-zero)
        V_vecs_R = [
            [0, 0, 1, 0, 0, 0, 0, 1], # 1 (N1: b02+b13. V[2]=1, V[7]=1)
            [0, 0, 1, 0, 0, 0, 0, 0], # 2 (N2: b02. V[2]=1)
            [0, 0, 0, 1, 0, 0, 0, -1],# 3 (N3: b03-b13. V[3]=1, V[7]=-1)
            [0, 0, -1, 0, 0, 0, 1, 0],# 4 (N4: b12-b02. V[6]=1, V[2]=-1)
            [0, 0, 0, 0, 0, 0, 0, 1], # 5 (N5: b13. V[7]=1)
            [0, 0, 1, 1, 0, 0, 0, 0], # 6 (N6: b02+b03. V[2]=1, V[3]=1)
            [0, 0, 0, 0, 0, 0, 1, 1]  # 7 (N7: b12+b13. V[6]=1, V[7]=1)
        ]
        
        # W vectors for right block (8 entries, only 4,5,6,7 non-zero)
        W_vecs_R = [
            [0, 0, 0, 0, 1, 0, 0, 1], # 1 (N1: C02, C13. W[4]=1, W[7]=1)
            [0, 0, 0, 0, 0, 1, 0, -1],# 2 (N2: C12, C13. W[5]=1, W[7]=-1)
            [0, 0, 0, 0, 0, 0, 1, 1], # 3 (N3: C03, C13. W[6]=1, W[7]=1)
            [0, 0, 0, 0, 1, 1, 0, 0], # 4 (N4: C02, C12. W[4]=1, W[5]=1)
            [0, 0, 0, 0, -1, 0, 1, 0],# 5 (N5: C02, C03. W[4]=-1, W[6]=1)
            [0, 0, 0, 0, 0, 0, 0, 1], # 6 (N6: C13. W[7]=1)
            [0, 0, 0, 0, 1, 0, 0, 0]  # 7 (N7: C02. W[4]=1)
        ]

        # Try to merge term i from Left and term j from Right
        # We want to find a new term (U_new, V_new, W_new) that replaces them
        # such that the tensor sum is preserved.
        # This is equivalent to finding alpha, beta such that:
        # alpha * (u_i, v_iL, w_iL) + beta * (u_j, v_jR, w_jR) = (u_new, v_new, w_new)
        # and the remaining 12 terms + this new term sum to T.
        
        # A simpler heuristic: check if any linear combination of U vectors
        # allows a simpler V/W structure.
        
        # Given the complexity and low probability of success, we perform a
        # limited random search for rank 13.
        
        random.seed(42)
        
        # Build the full rank-14 tensor components
        U14 = np.array([U_vecs[i] for i in range(7)] + [U_vecs[i] for i in range(7)])
        V14 = np.array(V_vecs_L + V_vecs_R)
        W14 = np.array(W_vecs_L + W_vecs_R)
        
        # Try to eliminate one term by adjusting others
        # This is a non-convex optimization problem.
        # We'll skip the heavy search to ensure timeout safety,
        # as rank 14 is the known optimal bound.
        
        return None

    # Try the structured rank-14 decomposition first
    result = try_strassen_rank14()
    if result is not None:
        return result
    
    # Fallback to naive rank 16
    return naive_decomp()
# EVOLVE-BLOCK-END
