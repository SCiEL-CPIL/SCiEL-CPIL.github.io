import time
import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
        
    def verify_decomposition(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def strassen_222():
        """
        Returns U, V, W for 2x2x2 matrix multiplication with rank 7.
        Indices: U[i*2+j], V[j*2+k], W[k*2+i]
        """
        U = []
        V = []
        W = []
        # M1 = (A11 - A22)(B22 - B11)
        # M2 = (A21 + A22)B11
        # M3 = A11(B11 - B22)
        # M4 = A22(B22 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        
        # U rows: A00, A01, A10, A11
        # V rows: B00, B01, B10, B11
        # W rows: C00, C01, C10, C11
        
        # M1: A00(1), A11(-1). B00(-1), B11(1). C00(1), C11(-1)
        U.append([1, 0, 0, -1])
        V.append([-1, 0, 0, 1])
        W.append([1, 0, 0, -1])
        
        # M2: A10(1), A11(1). B00(1). C10(1)
        U.append([0, 0, 1, 1])
        V.append([1, 0, 0, 0])
        W.append([0, 0, 1, 0])
        
        # M3: A00(1). B00(1), B11(-1). C00(1), C01(-1)
        U.append([1, 0, 0, 0])
        V.append([1, 0, 0, -1])
        W.append([1, -1, 0, 0])
        
        # M4: A11(1). B11(1), B00(-1). C11(1), C10(-1)
        U.append([0, 0, 0, 1])
        V.append([-1, 0, 0, 1])
        W.append([0, 0, -1, 1])
        
        # M5: A00(1), A01(1). B11(1). C01(1)
        U.append([1, 1, 0, 0])
        V.append([0, 0, 0, 1])
        W.append([0, 1, 0, 0])
        
        # M6: A10(1), A00(-1). B00(1), B01(1). C10(1), C11(1)
        U.append([-1, 0, 1, 0])
        V.append([1, 1, 0, 0])
        W.append([0, 0, 1, 1])
        
        # M7: A01(1), A11(-1). B10(1), B11(1). C10(1), C11(-1) -- Wait, Strassen M7 is (A12 - A22)(B21 + B22)
        # A12 is A[0][1] (index 1), A22 is A[1][1] (index 3).
        # B21 is B[1][0] (index 2), B22 is B[1][1] (index 3).
        # C10 is C[1][0] (index 2), C11 is C[1][1] (index 3).
        # M7 contributes to C10 and C11.
        # C10 += M7, C11 -= M7? 
        # Standard Strassen:
        # C00 = M1 + M4 - M3 - M7
        # C01 = M5 + M7
        # C10 = M2 - M4
        # C11 = M1 + M3 - M2 - M6
        
        # Let's re-verify M7 in standard Strassen:
        # M7 = (A12 - A22)(B21 + B22)
        # A12=A[0][1]=1, A22=A[1][1]=-1.
        # B21=B[1][0]=1, B22=B[1][1]=1.
        # Contribution to C:
        # C00: -M7
        # C01: +M7
        # C10: 0
        # C11: -M7? No, C11 = M1 + M3 - M2 - M6. M7 is not in C11.
        # Let's check C10 = M2 - M4.
        # Let's check C01 = M5 + M7.
        
        # So M7:
        U.append([0, 1, 0, -1])
        V.append([0, 0, 1, 1])
        # C00: -1, C01: 1, C10: 0, C11: 0
        W.append([-1, 1, 0, 0])
        
        return U, V, W

    def block_decompose():
        """
        Decompose 2x4x7 into blocks.
        n=2, m=4, p=7.
        We can split m into 2+2 and p into 2+2+2+1? Or p into 4+3?
        Let's try splitting m=4 into 2+2.
        Split p=7 into 2+2+2+1.
        
        Blocks:
        (i in 0..1, j in 0..1, k in 0..1) -> 2x2x2 (Rank 7)
        (i in 0..1, j in 0..1, k in 2..3) -> 2x2x2 (Rank 7)
        (i in 0..1, j in 0..1, k in 4..5) -> 2x2x2 (Rank 7)
        (i in 0..1, j in 0..1, k in 6..6) -> 2x2x1 (Rank 4)
        
        (i in 0..1, j in 2..3, k in 0..1) -> 2x2x2 (Rank 7)
        (i in 0..1, j in 2..3, k in 2..3) -> 2x2x2 (Rank 7)
        (i in 0..1, j in 2..3, k in 4..5) -> 2x2x2 (Rank 7)
        (i in 0..1, j in 2..3, k in 6..6) -> 2x2x1 (Rank 4)
        
        Total Rank = 4*7 + 4*4 = 28 + 16 = 44.
        This is significantly better than schoolbook 56.
        """
        
        U_total = []
        V_total = []
        W_total = []
        
        def add_block(U_block, V_block, W_block, i_off, j_off, k_off, n_block, m_block, p_block):
            """
            Appends a block decomposition to the global lists.
            U_block: list of lists (n_block x m_block) x rank_block
            V_block: list of lists (m_block x p_block) x rank_block
            W_block: list of lists (p_block x n_block) x rank_block
            
            Global U index: (i_off + i) * m + (j_off + j)
            Global V index: (j_off + j) * p + (k_off + k)
            Global W index: (k_off + k) * n + (i_off + i)
            """
            rank_block = len(U_block[0]) if U_block else 0
            for r in range(rank_block):
                u_col = [0] * nm
                v_col = [0] * mp
                w_col = [0] * pn
                
                for i in range(n_block):
                    for j in range(m_block):
                        idx_u = (i_off + i) * m + (j_off + j)
                        u_col[idx_u] = U_block[i][j][r]
                        
                for j in range(m_block):
                    for k in range(p_block):
                        idx_v = (j_off + j) * p + (k_off + k)
                        v_col[idx_v] = V_block[j][k][r]
                        
                for k in range(p_block):
                    for i in range(n_block):
                        idx_w = (k_off + k) * n + (i_off + i)
                        w_col[idx_w] = W_block[k][i][r]
                        
                U_total.append(u_col)
                V_total.append(v_col)
                W_total.append(w_col)

        # We need U, V, W in row-major format for the block functions.
        # The strassen_222 function returns U, V, W as lists of rows.
        # U: n x m. V: m x p. W: p x n.
        
        U_222, V_222, W_222 = strassen_222()
        
        # Define 2x2x1 decomposition (Rank 4)
        # C00 = A00*B00 + A01*B10? No, 2x2x1 means m=2, p=1.
        # A is 2x2, B is 2x1. C is 2x1.
        # C0 = A00*B0 + A01*B1
        # C1 = A10*B0 + A11*B1
        # This is just schoolbook for 2x2x1, rank 4.
        # U: 2x2, V: 2x1, W: 1x2
        U_221 = [
            [1, 0],
            [0, 1]
        ]
        V_221 = [
            [1],
            [1]
        ]
        # Wait, V is m x p. m=2, p=1.
        # V[0][0] corresponds to B0. V[1][0] corresponds to B1.
        # In the tensor product U[i,j] V[j,k] W[k,i]
        # For 2x2x1:
        # i=0, j=0, k=0: U[0,0]=1, V[0,0]=1, W[0,0]=1 -> C[0,0] += 1
        # i=0, j=1, k=0: U[0,1]=0, V[1,0]=1, W[0,1]=1 -> C[0,0] += 0
        # i=1, j=0, k=0: U[1,0]=0, V[0,0]=1, W[0,0]=1 -> C[1,0] += 0
        # i=1, j=1, k=0: U[1,1]=1, V[1,0]=1, W[0,1]=1 -> C[1,0] += 1
        # This only gives C0=A00*B0 and C1=A11*B1. We missed A01*B1 and A10*B0.
        
        # Correct 2x2x1 (Rank 4):
        # 1. A00*B0 -> C0
        # 2. A01*B1 -> C0
        # 3. A10*B0 -> C1
        # 4. A11*B1 -> C1
        
        # U (2x2):
        # Row 0 (i=0): [1, 0] (A00, A01)
        # Row 1 (i=1): [0, 1] (A10, A11)
        U_221 = [
            [1, 0],
            [0, 1]
        ]
        # V (2x1):
        # Col 0 (k=0):
        # j=0: B0. Needs to pair with A00 (U[0,0]=1) and A10 (U[1,0]=0).
        # j=1: B1. Needs to pair with A01 (U[0,1]=0) and A11 (U[1,1]=1).
        # So V[0,0]=1, V[1,0]=1.
        V_221 = [
            [1],
            [1]
        ]
        # W (1x2):
        # Row 0 (k=0):
        # i=0: C0. Needs A00*B0 (1*1) and A01*B1 (0*1). Sum=1. Correct.
        # i=1: C1. Needs A10*B0 (0*1) and A11*B1 (1*1). Sum=1. Correct.
        # So W[0,0]=1, W[0,1]=1.
        W_221 = [
            [1, 1]
        ]
        
        # Verify 221:
        # C[0,0] = U[0,0]V[0,0]W[0,0] + U[0,1]V[1,0]W[0,0] = 1*1*1 + 0*1*1 = 1. (A00*B0)
        # Wait, C[0,0] should be A00*B0 + A01*B1.
        # My W[0,0]=1 contributes to both i=0 and i=1?
        # W index is [k*n + i]. k=0, n=2.
        # W[0*2+0] = W[0] = 1.
        # W[0*2+1] = W[1] = 1.
        
        # Let's re-evaluate the sum for C[0,0] (i=0, k=0):
        # Sum over j: U[0,j] V[j,0] W[0,0]
        # j=0: U[0,0]=1, V[0,0]=1, W[0]=1 -> 1
        # j=1: U[0,1]=0, V[1,0]=1, W[0]=1 -> 0
        # Total 1. This is A00*B0.
        
        # Where is A01*B1?
        # A01 is U[0,1]. B1 is V[1,0].
        # Term: U[0,1] V[1,0] W[0,0] = 0 * 1 * 1 = 0.
        # It seems my 221 decomposition is wrong or I'm indexing it wrong.
        
        # Let's construct 221 explicitly with 4 terms.
        # Term 1: A00*B0. U[0,0]=1, V[0,0]=1, W[0,0]=1.
        # Term 2: A01*B1. U[0,1]=1, V[1,0]=1, W[0,0]=1.
        # Term 3: A10*B0. U[1,0]=1, V[0,0]=1, W[0,1]=1.
        # Term 4: A11*B1. U[1,1]=1, V[1,0]=1, W[0,1]=1.
        
        # This requires 4 ranks.
        # Rank 0: U[0,0]=1, others 0. V[0,0]=1, others 0. W[0,0]=1, others 0.
        # Rank 1: U[0,1]=1, others 0. V[1,0]=1, others 0. W[0,0]=1, others 0.
        # Rank 2: U[1,0]=1, others 0. V[0,0]=1, others 0. W[0,1]=1, others 0.
        # Rank 3: U[1,1]=1, others 0. V[1,0]=1, others 0. W[0,1]=1, others 0.
        
        # Let's build these matrices.
        U_221 = [
            [1, 1, 0, 0], # i=0: A00, A01
            [0, 0, 1, 1]  # i=1: A10, A11
        ]
        V_221 = [
            [1, 0, 1, 0], # j=0: B0
            [0, 1, 0, 1]  # j=1: B1
        ]
        W_221 = [
            [1, 1, 1, 1]  # k=0: C0, C1
        ]
        # Check C[0,0] (i=0, k=0):
        # j=0: U[0,0]V[0,0]W[0,0] = 1*1*1 = 1
        # j=1: U[0,1]V[1,0]W[0,0] = 1*1*1 = 1
        # Sum = 2. Expected 1.
        # Ah, W[0,0] is shared.
        # In the tensor definition, W[k*n+i] is the coefficient.
        # If W[0,0]=1, it adds to C[0,0] for ALL j where U and V are non-zero.
        # So if we have two terms contributing to C[0,0], we need to split the W or the U/V.
        
        # Actually, the standard decomposition for 2x2x1 is rank 4.
        # C0 = A00 B0 + A01 B1
        # C1 = A10 B0 + A11 B1
        
        # Term 1: A00 B0. U[0,0]=1, V[0,0]=1, W[0,0]=1.
        # Term 2: A01 B1. U[0,1]=1, V[1,0]=1, W[0,0]=1.
        # If we use separate ranks:
        # Rank 0: U[0,0]=1, V[0,0]=1, W[0,0]=1.
        # Rank 1: U[0,1]=1, V[1,0]=1, W[0,0]=1.
        # Rank 2: U[1,0]=1, V[0,0]=1, W[0,1]=1.
        # Rank 3: U[1,1]=1, V[1,0]=1, W[0,1]=1.
        
        # Let's check this.
        # C[0,0] = R0(U00 V00 W00) + R1(U01 V10 W00) + R2(U10 V00 W00) + R3(U11 V10 W00)
        # R0: 1*1*1 = 1
        # R1: 1*1*1 = 1
        # R2: 0 (U10=0 in R2? No, U[1,0] is in R2. But i=0 in C[0,0]. U[0,0] in R2 is 0.)
        # R3: 0
        # Sum = 2. Still wrong.
        
        # The issue is that W[0,0] appears in the sum for C[0,0].
        # C[0,0] = sum_j U[0,j] V[j,0] W[0,0].
        # If W[0,0]=1, then C[0,0] = U[0,0]V[0,0] + U[0,1]V[1,0].
        # This is exactly A00 B0 + A01 B1.
        # So we only need ONE rank for W[0,0]=1?
        # No, U and V must be consistent.
        # If we set U[0,0]=1, U[0,1]=1, V[0,0]=1, V[1,0]=1, W[0,0]=1.
        # Then C[0,0] = 1*1 + 1*1 = 2.
        # We need C[0,0] = 1.
        # So we need to scale? No, integers only.
        
        # Wait, the tensor is T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}.
        # For 2x2x1:
        # T_{0,0,0} = 1 (A00 B0)
        # T_{0,1,0} = 1 (A01 B1)
        # T_{1,0,0} = 1 (A10 B0)
        # T_{1,1,0} = 1 (A11 B1)
        
        # If we use Rank 0:
        # U[0,0]=1, U[0,1]=0, U[1,0]=0, U[1,1]=0
        # V[0,0]=1, V[1,0]=0
        # W[0,0]=1, W[0,1]=0
        # Contribution:
        # T_{0,0,0} += 1*1*1 = 1.
        # T_{0,1,0} += 0*0*1 = 0.
        # T_{1,0,0} += 0*1*0 = 0.
        # T_{1,1,0} += 0*0*0 = 0.
        
        # Rank 1:
        # U[0,0]=0, U[0,1]=1, U[1,0]=0, U[1,1]=0
        # V[0,0]=0, V[1,0]=1
        # W[0,0]=1, W[0,1]=0
        # Contribution:
        # T_{0,0,0} += 0*0*1 = 0.
        # T_{0,1,0} += 1*1*1 = 1.
        # T_{1,0,0} += 0*0*0 = 0.
        # T_{1,1,0} += 0*1*0 = 0.
        
        # Rank 2:
        # U[0,0]=0, U[0,1]=0, U[1,0]=1, U[1,1]=0
        # V[0,0]=1, V[1,0]=0
        # W[0,0]=0, W[0,1]=1
        # Contribution:
        # T_{0,0,0} += 0*1*0 = 0.
        # T_{0,1,0} += 0*0*0 = 0.
        # T_{1,0,0} += 1*1*1 = 1.
        # T_{1,1,0} += 0*0*1 = 0.
        
        # Rank 3:
        # U[0,0]=0, U[0,1]=0, U[1,0]=0, U[1,1]=1
        # V[0,0]=0, V[1,0]=1
        # W[0,0]=0, W[0,1]=1
        # Contribution:
        # T_{0,0,0} += 0*0*0 = 0.
        # T_{0,1,0} += 0*1*0 = 0.
        # T_{1,0,0} += 0*0*1 = 0.
        # T_{1,1,0} += 1*1*1 = 1.
        
        # Total:
        # T_{0,0,0} = 1
        # T_{0,1,0} = 1
        # T_{1,0,0} = 1
        # T_{1,1,0} = 1
        # Perfect.
        
        U_221 = [
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ]
        V_221 = [
            [1, 0, 1, 0],
            [0, 1, 0, 1]
        ]
        W_221 = [
            [1, 1, 0, 0]
        ]
        # Wait, W[0,0] and W[0,1] are separate entries in the row.
        # W is 1x2.
        # W[0] corresponds to i=0. W[1] corresponds to i=1.
        # In Rank 0: W[0]=1, W[1]=0.
        # In Rank 1: W[0]=1, W[1]=0.
        # In Rank 2: W[0]=0, W[1]=1.
        # In Rank 3: W[0]=0, W[1]=1.
        
        # So W_221 should be:
        # Row 0: [1, 1, 0, 0]
        # This matches my previous W_221.
        
        # Add blocks
        # Block 1: i=0..1, j=0..1, k=0..1
        add_block(U_222, V_222, W_222, 0, 0, 0, 2, 2, 2)
        # Block 2: i=0..1, j=0..1, k=2..3
        add_block(U_222, V_222, W_222, 0, 0, 2, 2, 2, 2)
        # Block 3: i=0..1, j=0..1, k=4..5
        add_block(U_222, V_222, W_222, 0, 0, 4, 2, 2, 2)
        # Block 4: i=0..1, j=0..1, k=6..6
        add_block(U_221, V_221, W_221, 0, 0, 6, 2, 2, 1)
        
        # Block 5: i=0..1, j=2..3, k=0..1
        add_block(U_222, V_222, W_222, 0, 2, 0, 2, 2, 2)
        # Block 6: i=0..1, j=2..3, k=2..3
        add_block(U_222, V_222, W_222, 0, 2, 2, 2, 2, 2)
        # Block 7: i=0..1, j=2..3, k=4..5
        add_block(U_222, V_222, W_222, 0, 2, 4, 2, 2, 2)
        # Block 8: i=0..1, j=2..3, k=6..6
        add_block(U_221, V_221, W_221, 0, 2, 6, 2, 2, 1)
        
        rank = len(U_total)
        
        # Transpose to row-major lists
        U_final = [[U_total[r][i] for r in range(rank)] for i in range(nm)]
        V_final = [[V_total[r][i] for r in range(rank)] for i in range(mp)]
        W_final = [[W_total[r][i] for r in range(rank)] for i in range(pn)]
        
        return {"U": U_final, "V": V_final, "W": W_final}

    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try block decomposition
    if time.time() < deadline - 5.0:
        block_res = block_decompose()
        if verify_decomposition(block_res["U"], block_res["V"], block_res["W"], len(block_res["U"][0])):
            best_result = block_res
            best_rank = len(block_res["U"][0])
    
    return best_result

