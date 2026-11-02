import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS  # n=3, m=3, p=6

    # Strategy: decompose 3x3x6 into three independent 3x3x2 blocks
    # Block b handles B columns {2b, 2b+1} and C columns {2b, 2b+1}
    # Each 3x3x2 block uses 12 terms: 7 (Strassen core) + 5 (third row)
    # Total: 3 * 12 = 36 terms
    
    # For a 3x3x2 block:
    # A is 3x3, B is 3x2, C is 3x2
    # C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0] + A[i,2]*B[2,0]
    # C[i,1] = A[i,0]*B[0,1] + A[i,1]*B[1,1] + A[i,2]*B[2,1]
    
    # Core 2x2x2 (rows 0,1 of A; all cols; output rows 0,1):
    # Strassen's 7-term decomposition:
    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,0])  -> C[0,0]+=M1, C[1,0]-=M1
    # M2 = A[1,0]*(B[0,1]+B[1,1])            -> C[0,1]+=M2, C[1,1]+=M2  (wait, let me recheck)
    
    # Let me use the standard Strassen formulation carefully:
    # For [[a,b],[c,d]] * [[e,f],[g,h]] = [[p,q],[r,s]]
    # p = ae + bg, q = af + bh, r = ce + dg, s = cf + dh
    #
    # M1 = (a+d)(e+h)   -> p+=M1, q+=M1, r-=M1, s-=M1
    # M2 = (c+d)e       -> p+=M2, r+=M2
    # M3 = (a-b)(f+h)   -> q-=M3, s+=M3
    # M4 = (c+d)(g+h)   -> q+=M4, s+=M4
    # M5 = (a-b)f       -> q+=M5
    # M6 = (c+d)g       -> r+=M6
    # M7 = (a-b)(e+h)   -> p+=M7, r-=M7
    #
    # Wait, let me verify:
    # p = M1 + M2 + M7 = (a+d)(e+h) + (c+d)e + (a-b)(e+h)
    #   = ae+ah+de+dh + ce+de + ae+ah-be-bh
    #   = 2ae + 2ah + 2de + dh + ce - be - bh
    # That's not right. Let me use the correct Strassen formulas.
    
    # Correct Strassen (standard):
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(g+h)  -- no wait
    #
    # Let me just use the well-known 7-term:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = (a-b)(f+h)
    # M4 = (c+d)(g+h)
    # M5 = (a-b)f
    # M6 = (c+d)g
    # M7 = (a-b)(e+h)
    #
    # p = M1 + M2 + M7 - M4
    #   = (a+d)(e+h) + (c+d)e + (a-b)(e+h) - (c+d)(g+h)
    #   = ae+ah+de+dh + ce+de + ae+ah-be-bh - cg-ch-dg-dh
    #   = 2ae + 2ah + 2de + dh + ce - be - bh - cg - ch - dg - dh
    #   = 2ae + 2ah + 2de + ce - be - bh - cg - ch - dg
    # That's not ae + bg. I'm making errors.
    
    # Let me just use a verified 7-term Strassen. The standard one is:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(g+h)
    # M3 = (a-b)(f+h)
    # M4 = (c+d)e
    # M5 = (a-b)f
    # M6 = (c+d)g
    # M7 = (a-b)(e+h)
    #
    # p = M1 + M4 + M7 - M2
    # q = M3 + M5
    # r = M6 + M2
    # s = M1 + M3 - M4 - M6
    #
    # Let me verify p:
    # p = (a+d)(e+h) + (c+d)e + (a-b)(e+h) - (c+d)(g+h)
    #   = ae + ah + de + dh + ce + de + ae + ah - be - bh - cg - ch - dg - dh
    #   = 2ae + 2ah + 2de + dh + ce - be - bh - cg - ch - dg - dh
    #   = 2ae + 2ah + 2de + ce - be - bh - cg - ch - dg
    # Still wrong.
    
    # OK let me just use the naive 4-term for 2x2x2 and 5 terms for row 2.
    # 2x2x2 naive: 4 terms (ae, bg, af, bh, ce, dg, cf, dh) -- that's 8 terms!
    # No, each "term" in tensor decomposition is one rank-1 component.
    # For 2x2x2: T[i,j,k] where i,k are output indices, j is the shared index.
    # T = sum of 8 rank-1 terms for naive, or 7 for Strassen.
    
    # Actually, I realize I should just carefully construct a valid decomposition.
    # Let me use a different approach: for 3x3x2, use 12 terms total.
    
    # Terms for rows 0,1 of output (2x3x2 subproblem):
    # This is equivalent to 2 independent 1x3x2 multiplications sharing B.
    # 1x3x2 has rank 3 (it's a 3x2 matrix, rank at most 2, but as a tensor it's 3).
    # Wait, the tensor for 1x3x2 multiplication: C[0,k] = sum_j A[0,j]*B[j,k]
    # The multiplication tensor T[j,k] = 1 for all j,k (it's just the identity structure).
    # As a 3x2 matrix of ones, its rank is 1. So rank(M_{1,3,2}) = 1?
    # No, that's not right either. The tensor is T[i,j,k] where the multiplication
    # C[i,k] = sum_j A[i,j]B[j,k]. For M_{1,3,2}, i is fixed at 0.
    # T[0,j,k] = 1 for all j in {0,1,2}, k in {0,1}.
    # This is a 3x2 matrix of all ones, which has rank 1.
    # So rank(M_{1,3,2}) = 1? That can't be right for a non-trivial multiplication.
    
    # I think I'm confusing the tensor structure. Let me reconsider.
    # The multiplication tensor for M_{m,n,p} is a 3-way tensor T of shape (m*n, n*p, p*m)
    # where T[i,j,k] = 1 if the indices are consistent with the multiplication.
    # Actually, T is defined such that T = sum_c U_c ⊗ V_c ⊗ W_c where
    # (U_c ⊗ V_c ⊗ W_c)[i,j,k] = U_c[i,j] * V_c[j,k] * W_c[k,i]
    # and we need T[i,j,k] = 1 for all valid (i,j,k) triples.
    
    # For M_{3,3,6}: T has shape (9, 18, 18) and T[i*3+j, j*6+k, k*3+i] = 1 for all i,j,k.
    # All other entries are 0.
    
    # The rank is the minimum number of rank-1 tensors to sum to T.
    
    # For the block decomposition approach, let me just carefully construct 12 terms
    # for a 3x3x2 block and verify computationally.
    
    # Let me try a simpler approach: use the known fact that we can achieve rank 36
    # by using 12 terms per 3x3x2 block.
    
    # For 3x3x2 block (local indices: A is 3x3, B is 3x2, C is 3x2):
    # C[i,0] = sum_{j=0}^{2} A[i,j]*B[j,0]
    # C[i,1] = sum_{j=0}^{2} A[i,j]*B[j,1]
    
    # 12-term decomposition:
    # Core (rows 0,1 of A, output rows 0,1): use 7 Strassen terms
    # Row 2 of output: 5 terms
    
    # Let me define the 7 Strassen terms for the 2x2x2 core:
    # Local A indices: a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # Local B indices: e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
    # Output: p=C[0,0], q=C[0,1], r=C[1,0], s=C[1,1]
    
    # Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = (a-b)(f+h)
    # M4 = (c+d)(g+h)
    # M5 = (a-b)f
    # M6 = (c+d)g
    # M7 = (a-b)(e+h)
    #
    # p = M1 + M2 + M7 - M4
    # q = M3 + M5
    # r = M6 + M4
    # s = M1 + M3 - M2 - M6
    
    # Let me verify p = ae + bg:
    # p = (a+d)(e+h) + (c+d)e + (a-b)(e+h) - (c+d)(g+h)
    #   = ae + ah + de + dh + ce + de + ae + ah - be - bh - cg - ch - dg - dh
    #   = 2ae + 2ah + 2de + dh + ce - be - bh - cg - ch - dg - dh
    #   = 2ae + 2ah + 2de + ce - be - bh - cg - ch - dg
    # This is NOT ae + bg. So my Strassen formula is wrong.
    
    # Let me look up the correct Strassen formulas.
    # Correct Strassen for 2x2:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(g+h)
    # M3 = (a-b)(f+h)
    # M4 = (c+d)e
    # M5 = (a-b)f
    # M6 = (c+d)g
    # M7 = (a-b)(e+h)
    #
    # p = M1 + M4 + M7 - M2
    # q = M3 + M5
    # r = M6 + M2
    # s = M1 + M3 - M4 - M6
    
    # Verify p:
    # p = (a+d)(e+h) + (c+d)e + (a-b)(e+h) - (c+d)(g+h)
    # Same as before, still wrong.
    
    # I think the issue is that I'm not using the right Strassen variant.
    # Let me just use a verified decomposition by testing computationally.
    
    # Actually, let me take a completely different approach.
    # I'll construct a rank-36 decomposition by using 12 terms per block,
    # where each term is a rank-1 component of the 3x3x2 tensor.
    
    # For 3x3x2: the tensor T has T[i,j,k] = 1 for i in {0,1,2}, j in {0,1,2}, k in {0,1}
    # (in the local block coordinates).
    
    # A rank-1 component is U ⊗ V ⊗ W where:
    # U[i,j] is a 3x3 matrix, V[j,k] is a 3x2 matrix, W[k,i] is a 2x3 matrix
    # and (U ⊗ V ⊗ W)[i,j,k] = U[i,j]*V[j,k]*W[k,i]
    
    # For the naive decomposition, each (i,j,k) triple gives one term:
    # U has 1 at (i,j), V has 1 at (j,k), W has 1 at (k,i)
    # That's 3*3*2 = 18 terms per block, 54 total.
    
    # To get 12 per block, I need to combine terms.
    
    # Let me try: for each pair of k values (0,1), can I combine the j=0 and j=1 terms?
    # C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0] + A[i,2]*B[2,0]
    # C[i,1] = A[i,0]*B[0,1] + A[i,1]*B[1,1] + A[i,2]*B[2,1]
    
    # If I define M1_i = (A[i,0]+A[i,1])*(B[0,0]+B[1,0]) for k=0
    # and M2_i = (A[i,0]+A[i,1])*(B[0,1]+B[1,1]) for k=1
    # Then C[i,0] += M1_i, C[i,1] += M2_i
    # But I also need to subtract cross terms and add A[i,2]*B[2,k].
    
    # This is getting complicated. Let me just use a computational approach.
    # I'll set up the problem as a linear system and try to find a low-rank solution.
    
    # Given time constraints, let me try rank 36 with a structured approach.
    # I'll use 12 terms per block with the following structure:
    # For each block b (b=0,1,2):
    #   Terms 0-6: Strassen-like for rows 0,1
    #   Terms 7-11: for row 2
    
    # Let me define the terms explicitly and verify.
    
    rank = 36
    U = np.zeros((9, rank), dtype=float)
    V = np.zeros((18, rank), dtype=float)
    W = np.zeros((18, rank), dtype=float)
    
    for b in range(3):
        # Block b: B columns {2b, 2b+1}, C columns {2b, 2b+1}
        base = b * 12
        
        # For this block, local B row j corresponds to global B row j, col 2b or 2b+1
        # Local C row i corresponds to global C row i, col 2b or 2b+1
        
        # Term 0: (A[0,0]+A[1,0]) * (B[0,2b]+B[1,2b]) -> C[0,2b] +=, C[1,2b] -=
        U[0*3+0, base+0] = 1
        U[1*3+0, base+0] = 1
        V[0*6+2*b, base+0] = 1
        V[1*6+2*b, base+0] = 1
        W[2*b*3+0, base+0] = 1
        W[2*b*3+1, base+0] = -1
        
        # Term 1: (A[0,0]+A[1,0]) * (B[0,2b+1]+B[1,2b+1]) -> C[0,2b+1] +=, C[1,2b+1] -=
        U[0*3+0, base+1] = 1
        U[1*3+0, base+1] = 1
        V[0*6+2*b+1, base+1] = 1
        V[1*6+2*b+1, base+1] = 1
        W[(2*b+1)*3+0, base+1] = 1
        W[(2*b+1)*3+1, base+1] = -1
        
        # Term 2: (A[0,1]+A[1,1]) * (B[0,2b]+B[1,2b]) -> C[0,2b] +=, C[1,2b] -=
        U[0*3+1, base+2] = 1
        U[1*3+1, base+2] = 1
        V[0*6+2*b, base+2] = 1
        V[1*6+2*b, base+2] = 1
        W[2*b*3+0, base+2] = 1
        W[2*b*3+1, base+2] = -1
        
        # Term 3: (A[0,1]+A[1,1]) * (B[0,2b+1]+B[1,2b+1]) -> C[0,2b+1] +=, C[1,2b+1] -=
        U[0*3+1, base+3] = 1
        U[1*3+1, base+3] = 1
        V[0*6+2*b+1, base+3] = 1
        V[1*6+2*b+1, base+3] = 1
        W[(2*b+1)*3+0, base+3] = 1
        W[(2*b+1)*3+1, base+3] = -1
        
        # Term 4: A[2,0] * (B[0,2b]+B[1,2b]) -> C[2,2b] +=
        U[2*3+0, base+4] = 1
        V[0*6+2*b, base+4] = 1
        V[1*6+2*b, base+4] = 1
        W[2*b*3+2, base+4] = 1
        
        # Term 5: A[2,0] * (B[0,2b+1]+B[1,2b+1]) -> C[2,2b+1] +=
        U[2*3+0, base+5] = 1
        V[0*6+2*b+1, base+5] = 1
        V[1*6+2*b+1, base+5] = 1
        W[(2*b+1)*3+2, base+5] = 1
        
        # Term 6: A[2,1] * (B[0,2b]+B[1,2b]) -> C[2,2b] +=
        U[2*3+1, base+6] = 1
        V[0*6+2*b, base+6] = 1
        V[1*6+2*b, base+6] = 1
        W[2*b*3+2, base+6] = 1
        
        # Term 7: A[2,1] * (B[0,2b+1]+B[1,2b+1]) -> C[2,2b+1] +=
        U[2*3+1, base+7] = 1
        V[0*6+2*b+1, base+7] = 1
        V[1*6+2*b+1, base+7] = 1
        W[(2*b+1)*3+2, base+7] = 1
        
        # Term 8: A[0,2] * B[2,2b] -> C[0,2b] +=
        U[0*3+2, base+8] = 1
        V[2*6+2*b, base+8] = 1
        W[2*b*3+0, base+8] = 1
        
        # Term 9: A[0,2] * B[2,2b+1] -> C[0,2b+1] +=
        U[0*3+2, base+9] = 1
        V[2*6+2*b+1, base+9] = 1
        W[(2*b+1)*3+0, base+9] = 1
        
        # Term 10: A[1,2] * B[2,2b] -> C[1,2b] +=
        U[1*3+2, base+10] = 1
        V[2*6+2*b, base+10] = 1
        W[2*b*3+1, base+10] = 1
        
        # Term 11: A[1,2] * B[2,2b+1] -> C[1,2b+1] +=
        U[1*3+2, base+11] = 1
        V[2*6+2*b+1, base+11] = 1
        W[(2*b+1)*3+1, base+11] = 1
    
    # Verify
    valid = True
    for i in range(3):
        for j in range(3):
            for k in range(6):
                s = 0.0
                for c in range(rank):
                    s += U[i*3+j, c] * V[j*6+k, c] * W[k*3+i, c]
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fall back to schoolbook rank 54
        rank_sb = n * m * p
        U_sb = np.zeros((n * m, rank_sb), dtype=int)
        V_sb = np.zeros((m * p, rank_sb), dtype=int)
        W_sb = np.zeros((p * n, rank_sb), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
