"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    import time
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    def build_strassen_4x4x4():
        """
        Build 4x4x4 matrix multiplication using recursive Strassen.
        4x4 matrices split into 2x2 blocks.
        Each 2x2 product uses 7 scalar multiplications (Strassen).
        Total: 7 * 4 = 28? No - we need to be more careful.
        
        Actually for 4x4x4:
        Split A, B, C into 2x2 blocks of 2x2 matrices.
        C = A*B where each block is 2x2.
        We can use Strassen on the block level: 7 block multiplications.
        Each block multiplication is a 2x2x2 problem, which itself can use Strassen (rank 7).
        Total rank = 7 * 7 = 49.
        """
        nn, mm, pp = 4, 4, 4
        
        # Strassen for 2x2: 
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # For 4x4x4, we apply Strassen at the 2x2 block level.
        # Each "scalar" in the Strassen formula is actually a 2x2 matrix.
        # So we need 7 "multiplications" of 2x2 matrices, each with rank 7.
        # Total rank = 49.
        
        # Let me build this explicitly.
        # The 4x4x4 tensor can be decomposed into 49 rank-1 terms.
        
        # I'll use a known explicit construction.
        # For simplicity, let me use a verified decomposition.
        
        # Actually, let me just use schoolbook for 4x4x4 (rank 64) 
        # and focus on optimizing the 4x4x3 part.
        # 64 + 12 = 76, which is already better than 112.
        
        # But let me try to get 49 for the 4x4x4 part.
        
        # Here's a construction using recursive Strassen:
        # We'll build U, V, W matrices for the 4x4x4 case.
        
        r = 49
        U = np.zeros((16, r))
        V = np.zeros((16, r))
        W = np.zeros((16, r))
        
        # Index: U row = i*4+j (i,j in 0..3), V row = j*4+k, W row = k*4+i
        # For the recursive Strassen construction:
        # Level 1: Split into 2x2 blocks of 2x2 matrices
        # Block (bi, bj) where bi, bj in {0,1}
        # A[bi*2:(bi+1)*2, bj*2:(bj+1)*2]
        
        # Strassen formulas at block level:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Each Mx is a 2x2 product, decomposed with rank 7.
        
        # For a 2x2x2 Strassen:
        # Given 2x2 matrices P, Q, compute R = P*Q
        # R00 = M1 + M4 - M5 + M7
        # R01 = M3 + M5
        # R10 = M2 + M4
        # R11 = M1 - M2 + M3 + M6
        # where:
        # M1 = (P00+P11)(Q00+Q11)
        # M2 = (P10+P11)Q00
        # M3 = P00(Q10-Q11)
        # M4 = P11(Q01-Q00)
        # M5 = (P00+P01)Q11
        # M6 = (P10-P00)(Q00+Q01)
        # M7 = (P01-P11)(Q10+Q11)
        
        # For the 4x4x4 case, each "scalar multiplication" in the outer Strassen
        # is a 2x2 matrix multiplication.
        
        # Let me define the 7 outer Strassen multiplications:
        # For each, I need to specify which 2x2 blocks of A and B are multiplied,
        # with what signs.
        
        # Outer Strassen terms (at 2x2 block level):
        # T1: (A00+A11) * (B00+B11)
        # T2: (A10+A11) * B00
        # T3: A00 * (B10-B11)
        # T4: A11 * (B01-B00)
        # T5: (A00+A01) * B11
        # T6: (A10-A00) * (B00+B01)
        # T7: (A01-A11) * (B10+B11)
        
        # Each Tx is a 2x2x2 problem.
        
        # Let me encode each term.
        # A block (bi, bj) has rows 2*bi to 2*bi+1, cols 2*bj to 2*bj+1
        # So A[i,j] where i = 2*bi + ii, j = 2*bj + jj
        
        # For each outer term, I'll generate 7 inner Strassen terms.
        
        def get_2x2_strassen_terms():
            """Return the 7 Strassen terms for 2x2x2.
            Each term is (P_coeffs, Q_coeffs, R_coeffs)
            where P_coeffs is a dict mapping (ii,jj) to coefficient,
            Q_coeffs is a dict mapping (ii,jj) to coefficient,
            R_coeffs is a dict mapping (ii,jj) to coefficient.
            """
            terms = []
            # M1 = (P00+P11)(Q00+Q11)
            terms.append((
                {(0,0): 1, (1,1): 1},
                {(0,0): 1, (1,1): 1},
                None  # We'll handle R separately
            ))
            # M2 = (P10+P11)Q00
            terms.append((
                {(1,0): 1, (1,1): 1},
                {(0,0): 1},
                None
            ))
            # M3 = P00(Q10-Q11)
            terms.append((
                {(0,0): 1},
                {(1,0): 1, (1,1): -1},
                None
            ))
            # M4 = P11(Q01-Q00)
            terms.append((
                {(1,1): 1},
                {(0,1): 1, (0,0): -1},
                None
            ))
            # M5 = (P00+P01)Q11
            terms.append((
                {(0,0): 1, (0,1): 1},
                {(1,1): 1},
                None
            ))
            # M6 = (P10-P00)(Q00+Q01)
            terms.append((
                {(1,0): 1, (0,0): -1},
                {(0,0): 1, (0,1): 1},
                None
            ))
            # M7 = (P01-P11)(Q10+Q11)
            terms.append((
                {(0,1): 1, (1,1): -1},
                {(1,0): 1, (1,1): 1},
                None
            ))
            return terms
        
        strassen_terms = get_2x2_strassen_terms()
        
        # Outer Strassen: for each of the 7 outer terms, we have 7 inner terms.
        # Outer term t uses A blocks with coefficients a_coeffs and B blocks with b_coeffs.
        # The result contributes to C blocks with certain coefficients.
        
        # Let me define the outer terms more carefully.
        # Outer term index o (0..6):
        # A_blocks[o]: dict mapping (bi,bj) to coeff
        # B_blocks[o]: dict mapping (bi,bj) to coeff
        # C_blocks[o]: dict mapping (bi,bj) to coeff (how this term contributes to C)
        
        # From Strassen:
        # T1 = (A00+A11)(B00+B11) -> contributes to C00 (+1), C11 (+1)
        # T2 = (A10+A11)B00 -> contributes to C10 (+1), C11 (-1)
        # T3 = A00(B10-B11) -> contributes to C01 (+1), C11 (+1)
        # T4 = A11(B01-B00) -> contributes to C00 (+1), C10 (+1)
        # T5 = (A00+A01)B11 -> contributes to C00 (-1), C01 (+1)
        # T6 = (A10-A00)(B00+B01) -> contributes to C11 (+1)
        # T7 = (A01-A11)(B10+B11) -> contributes to C00 (+1)
        
        outer_A = [
            {(0,0): 1, (1,1): 1},  # T1
            {(1,0): 1, (1,1): 1},  # T2
            {(0,0): 1},             # T3
            {(1,1): 1},             # T4
            {(0,0): 1, (0,1): 1},  # T5
            {(1,0): 1, (0,0): -1}, # T6
            {(0,1): 1, (1,1): -1}, # T7
        ]
        outer_B = [
            {(0,0): 1, (1,1): 1},  # T1
            {(0,0): 1},             # T2
            {(1,0): 1, (1,1): -1}, # T3
            {(0,1): 1, (0,0): -1}, # T4
            {(1,1): 1},             # T5
            {(0,0): 1, (0,1): 1},  # T6
            {(1,0): 1, (1,1): 1},  # T7
        ]
        outer_C = [
            {(0,0): 1, (1,1): 1},  # T1
            {(1,0): 1, (1,1): -1}, # T2
            {(0,1): 1, (1,1): 1},  # T3
            {(0,0): 1, (1,0): 1},  # T4
            {(0,0): -1, (0,1): 1}, # T5
            {(1,1): 1},             # T6
            {(0,0): 1},             # T7
        ]
        
        col = 0
        for o in range(7):
            for s in range(7):
                # Inner Strassen term s within outer term o
                P_coeffs, Q_coeffs, _ = strassen_terms[s]
                
                # P is a 2x2 matrix formed from A blocks with outer_A[o] coefficients
                # P[ii,jj] = sum over (bi,bj) in outer_A[o] of outer_A[o][(bi,bj)] * A[2*bi+ii, 2*bj+jj]
                # Q is similarly formed from B blocks with outer_B[o] coefficients
                # R[ii,jj] = sum over (ci,cj) in outer_C[o] of outer_C[o][(ci,cj)] * C[2*ci+ii, 2*cj+jj]
                
                # For the rank-1 term:
                # U[i,j] coefficient for A[i,j]
                # V[j,k] coefficient for B[j,k]
                # W[k,i] coefficient for C[k,i]
                
                # A[i,j] where i = 2*bi + ii, j = 2*bj + jj
                # B[j,k] where j = 2*bi + ii, k = 2*bj + jj (for the B block)
                # Wait, I need to be more careful.
                
                # In the outer Strassen, A is 4x4, B is 4x4, C is 4x4.
                # A[i,j], B[j,k], C[i,k]
                
                # For outer term o, inner term s:
                # The "P" matrix is 2x2, with P[ii,jj] = sum_{(bi,bj) in outer_A[o]} outer_A[o][(bi,bj)] * A[2*bi+ii, 2*bj+jj]
                # The "Q" matrix is 2x2, with Q[ii,jj] = sum_{(bi,bj) in outer_B[o]} outer_B[o][(bi,bj)] * B[2*bi+ii, 2*bj+jj]
                # Wait, B is indexed by [j,k] where j is row, k is col.
                # In the block structure, B block (bi,bj) is B[2*bi:2*bi+2, 2*bj:2*bj+2]
                # So B[2*bi+ii, 2*bj+jj]
                
                # The product P*Q gives R, where R[ii,jj] = sum_l P[ii,l]*Q[l,jj]
                # And R contributes to C blocks.
                
                # For the rank-1 decomposition:
                # U[i,j] = coefficient of A[i,j] in P
                # V[j,k] = coefficient of B[j,k] in Q
                # W[i,k] = coefficient of C[i,k] in R (but R is computed from P*Q)
                
                # Actually, for a rank-1 term:
                # U[i,j] * V[j,k] * W[i,k] should give the contribution.
                # But P*Q is a matrix product, so it's not directly rank-1.
                
                # I think I'm overcomplicating this. Let me use a different approach.
                
                # For the 4x4x4 case, let me just use schoolbook (rank 64) for now
                # and focus on the 4x4x3 optimization.
                pass
        
        # Let me just use schoolbook for 4x4x4
        r = 64
        U = np.zeros((16, r))
        V = np.zeros((16, r))
        W = np.zeros((16, r))
        c = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U[i * 4 + j, c] = 1
                    V[j * 4 + k, c] = 1
                    W[k * 4 + i, c] = 1
                    c += 1
        return U, V, W
    
    def build_4x4x3_schoolbook():
        """Build schoolbook decomposition for 4x4x3."""
        nn, mm, pp = 4, 4, 3
        r = nn * mm * pp
        U = np.zeros((nn * mm, r))
        V = np.zeros((mm * pp, r))
        W = np.zeros((pp * nn, r))
        c = 0
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    U[i * mm + j, c] = 1
                    V[j * pp + k, c] = 1
                    W[k * nn + i, c] = 1
                    c += 1
        return U, V, W
    
    def try_reduce_4x4x3(U_init, V_init, W_init, max_iter=5000, deadline=None):
        """
        Try to reduce the rank of the 4x4x3 decomposition.
        Start from schoolbook (rank 12) and try to find a lower rank.
        """
        nn, mm, pp = 4, 4, 3
        r_init = U_init.shape[1]
        
        # Try ranks from r_init-1 down to some minimum
        best_U, best_V, best_W = U_init.copy(), V_init.copy(), W_init.copy()
        best_rank = r_init
        
        for target_rank in range(r_init - 1, 7, -1):
            if deadline and time.time() > deadline - 20:
                break
            
            # Try to find a decomposition with target_rank
            success = False
            for attempt in range(10):
                if deadline and time.time() > deadline - 20:
                    break
                
                # Initialize with random perturbations of the schoolbook
                U = U_init[:, :target_rank].copy()
                V = V_init[:, :target_rank].copy()
                W = W_init[:, :target_rank].copy()
                
                # Add small random perturbations
                noise = 0.1
                U += noise * np.random.randn(*U.shape)
                V += noise * np.random.randn(*V.shape)
                W += noise * np.random.randn(*W.shape)
                
                # Iterative refinement
                for it in range(max_iter):
                    if deadline and time.time() > deadline - 20:
                        break
                    
                    # Compute the residual tensor
                    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                    # We want T[i,j,k] = 1 for all i,j,k (schoolbook structure)
                    
                    # Compute current tensor
                    T_current = np.zeros((nn, mm, pp))
                    for r in range(target_rank):
                        for i in range(nn):
                            for j in range(mm):
                                for k in range(pp):
                                    T_current[i,j,k] += U[i*mm+j, r] * V[j*pp+k, r] * W[k*nn+i, r]
                    
                    # Residual
                    T_target = np.ones((nn, mm, pp))
                    residual = T_target - T_current
                    res_norm = np.linalg.norm(residual)
                    
                    if res_norm < 1e-10:
                        success = True
                        break
                    
                    # Gradient-based update
                    # dT/dU[i,j,r] = V[j,k,r] * W[k,i,r] (summed over k)
                    # dT/dV[j,k,r] = U[i,j,r] * W[k,i,r] (summed over i)
                    # dT/dW[k,i,r] = U[i,j,r] * V[j,k,r] (summed over j)
                    
                    lr = 0.01
                    for r in range(target_rank):
                        for i in range(nn):
                            for j in range(mm):
                                grad_U = 0
                                for k in range(pp):
                                    grad_U += residual[i,j,k] * V[j*pp+k, r] * W[k*nn+i, r]
                                U[i*mm+j, r] -= lr * grad_U
                        for j in range(mm):
                            for k in range(pp):
                                grad_V = 0
                                for i in range(nn):
                                    grad_V += residual[i,j,k] * U[i*mm+j, r] * W[k*nn+i, r]
                                V[j*pp+k, r] -= lr * grad_V
                        for k in range(pp):
                            for i in range(nn):
                                grad_W = 0
                                for j in range(mm):
                                    grad_W += residual[i,j,k] * U[i*mm+j, r] * V[j*pp+k, r]
                                W[k*nn+i, r] -= lr * grad_W
                    
                    if it % 100 == 0 and res_norm > 1.0:
                        # Residual is getting worse, break
                        break
                
                if success:
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    best_rank = target_rank
                    break
        
        return best_U, best_V, best_W, best_rank
    
    # Build the 4x4x4 part (schoolbook, rank 64)
    U1, V1, W1 = build_strassen_4x4x4()
    r1 = U1.shape[1]
    
    # Build the 4x4x3 part (schoolbook, rank 12)
    U2, V2, W2 = build_4x4x3_schoolbook()
    r2 = U2.shape[1]
    
    # Try to reduce the 4x4x3 rank
    if time.time() < deadline - 30:
        U2_opt, V2_opt, W2_opt, r2_opt = try_reduce_4x4x3(U2, V2, W2, max_iter=2000, deadline=deadline)
        if r2_opt < r2:
            U2, V2, W2 = U2_opt, V2_opt, W2_opt
            r2 = r2_opt
    
    # Combine
    total_rank = r1 + r2
    
    U = np.zeros((16, total_rank))
    V = np.zeros((28, total_rank))
    W = np.zeros((28, total_rank))
    
    # First part (k=0..3)
    U[:, :r1] = U1
    for j in range(4):
        for k in range(4):
            V[j * 7 + k, :r1] = V1[j * 4 + k, :]
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, :r1] = W1[k * 4 + i, :]
    
    # Second part (k=4..6)
    U[:, r1:r1+r2] = U2
    for j in range(4):
        for kp in range(3):
            k = 4 + kp
            V[j * 7 + k, r1:r1+r2] = V2[j * 3 + kp, :]
    for kp in range(3):
        k = 4 + kp
        for i in range(4):
            W[k * 4 + i, r1:r1+r2] = W2[kp * 4 + i, :]
    
    # Convert to list of lists with integer entries
    U_list = [[int(U[i, j]) for j in range(total_rank)] for i in range(16)]
    V_list = [[int(V[i, j]) for j in range(total_rank)] for i in range(28)]
    W_list = [[int(W[i, j]) for j in range(total_rank)] for i in range(28)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
