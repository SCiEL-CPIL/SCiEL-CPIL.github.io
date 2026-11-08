"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10
    # Use Strassen for 2x2 blocks in the middle dimension.
    # Split j (dim 2) into groups of 2: j01, j23, j45, j67, j89
    # For each group of 2 j's and 2 k's (since p=10=5*2), we can use Strassen-like.
    # Actually, let's split k into 5 groups of 2 as well.
    # So we have 5 blocks in k, each of size 2.
    # And 2 rows in i.
    # For each k-block (2 columns), we have a 2x2x2 subproblem? No.
    # i has 2 rows, j has 3 cols, k has 2 cols per block.
    # Let's split j into 2+1.
    # For the 2 j's, use Strassen with the 2 k's? That's 2x2x2, rank 7.
    # For the 1 j, use direct: 2*1*2 = 4.
    # Per k-block: 7+4 = 11.
    # 5 k-blocks: 55.
    # Plus we have 5 k-blocks of size 2, but wait, 10/2=5.
    # Let's verify.
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Strassen for 2x2x2:
    # A (2x2), B (2x2), C = AB
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 - M4 + M6 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M5 - M7
    
    # For each k-block b (0..4), k=2b, 2b+1
    # j-group: (0,1) for Strassen, (2) for direct
    for b in range(5):
        # Strassen part: j=0,1; k=2b, 2b+1
        # i=0,1
        # M1: (A00+A11)(B0,k + B1,k)
        # Let's define indices carefully.
        # A: 2x3, B: 3x10
        # For k-block b, we look at columns 2b, 2b+1 of B.
        # Strassen on A[:, 0:2] and B[0:2, 2b:2b+2]
        
        # M1 = (A00+A11) * (B0k0+B1k0) ... wait, Strassen multiplies sums.
        # Let's write U,V,W entries.
        # U[i, c], V[j, c], W[k, c] -> C[i,k] += U[i,c]*V[j,c]*W[k,c] summed over c.
        # Actually, the decomposition is C[i,k] = sum_c U[i,c] V[j,c] W[k,c] for all i,j,k?
        # No, it's C = A*B, so C[i,k] = sum_j A[i,j]B[j,k].
        # The tensor is T[i,j,k] = delta? No, T is the multiplication tensor.
        # T[i,j,k] is such that sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = C[i,k].
        # Actually, standard form: C[i,k] = sum_{j,c} A[i,j] V[j,c] U[i,c] W[k,c]?
        # Let's check the seed.
        # Seed: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
        # So U is indexed by (i,j), V by (j,k), W by (k,i).
        # C[i,k] = sum_{j,c} A[i,j] B[j,k] * (1/c?) No.
        # The product is sum_c U[i,j,c] V[j,k,c] W[k,i,c]?
        # Actually, the evaluator probably does:
        # C[i,k] = sum_{j,c} A[i,j] * V[j*p+k, c] * (1/W[k*n+i, c]?) No.
        # Let's look at the seed: U=1, V=1, W=1.
        # Then sum_j 1*1*1 = 3? But C[i,k] = sum_j A[i,j]B[j,k].
        # So the decomposition must be: C[i,k] = sum_c (sum_j A[i,j]U[i*m+j,c]) * (sum? no)
        # It's a bilinear decomposition.
        # C[i,k] = sum_c L_c(A) R_c(B) where L_c, R_c are linear forms.
        # L_c(A) = sum_{i,j} U[i*m+j, c] A[i,j]
        # R_c(B) = sum_{j,k} V[j*p+k, c] B[j,k]
        # And then C[i,k] = sum_c L_c(A) R_c(B) W[k*n+i, c]?
        # No, W must be part of the linear form or a coefficient.
        # Given W[k*n+i, c], it's likely C[i,k] = sum_c W[k*n+i, c] * L_c(A) * R_c(B) / W? No.
        # Let's assume C[i,k] = sum_c W[k*n+i, c] * (sum_{i',j} U[i'*m+j, c] A[i',j]) * (sum_{j',k'} V[j'*p+k', c] B[j',k'])?
        # That doesn't make sense dimensionally unless U,V are sparse.
        # In the seed, U[i*m+j, c]=1 for specific c.
        # So L_c(A) = A[i,j] for that c.
        # R_c(B) = B[j,k] for that c.
        # Then C[i,k] += W[k*n+i, c] * A[i,j] * B[j,k].
        # Summing over j, we get sum_j A[i,j]B[j,k] W[k*n+i, c].
        # For this to equal sum_j A[i,j]B[j,k], we need W[k*n+i, c]=1 for all j?
        # But c is fixed per (i,j,k).
        # Ah, the seed has c for each (i,j,k).
        # So for a fixed c, it corresponds to a specific (i,j,k).
        # U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1.
        # Then the term is A[i,j] B[j,k] * 1.
        # Summing over all c, we get sum_{i,j,k} A[i,j]B[j,k] for each (i,k)?
        # No, the sum is over c. For a fixed (i,k), we sum over c.
        # Each c contributes W[k*n+i, c] * (sum_{i',j'} U[i'*m+j', c] A[i',j']) * (sum_{j'',k''} V[j''*p+k'', c] B[j'',k''])?
        # This is getting complicated. Let's stick to the seed's structure.
        # The seed works, so we can modify it.
        # Strassen: We replace 8 terms (rank 8) with 7 terms (rank 7).
        # For a 2x2x2 block, we need to set U,V,W such that the sum matches.
        # Let's use the standard Strassen decomposition for the tensor.
        
        # For block b (k=2b, 2b+1), j=0,1:
        # We need 7 tensors.
        # c0 = col, c1=col+1, ..., c6=col+6.
        
        # M1 = (A00+A11)(B0k0+B1k0) -> U00=1,U11=1; V0k0=1,V1k0=1; W?
        # This linear form multiplies A and B.
        # L1 = A00+A11, R1 = B0k0+B1k0.
        # But R1 has B[j,k] for j=0,1 and k=2b. That's not a single B[j,k].
        # V[j*p+k, c] must be such that sum_{j,k} V[j*p+k, c] B[j,k] = R_c(B).
        # So V[0*p+2b, c0]=1, V[1*p+2b, c0]=1.
        # Similarly U[0*m+0, c0]=1, U[1*m+1, c0]=1.
        # Then L1 R1 = (A00+A11)(B0,2b + B1,2b).
        # This contributes to C[i,k] via W[2b*n+i, c0].
        # But we need C[i,k] = sum_j A[i,j]B[j,k].
        # The term (A00+A11)(B0,2b + B1,2b) = A00B0,2b + A00B1,2b + A11B0,2b + A11B1,2b.
        # This affects C[0,2b], C[0,2b+1]?, C[1,2b], C[1,2b+1]?
        # No, B1,2b is B[1,2b], which is in the sum for C[0,2b] (j=1) and C[1,2b] (j=1).
        # So this linear form R1 = B[0,2b]+B[1,2b] is the sum of B[j,2b] for j=0,1.
        # So L1 R1 W[...] adds to C[i, 2b] the term (A[i,0]+A[i,1]) * (B[0,2b]+B[1,2b]) * W[2b*n+i, c0].
        # This is (A[i,0]B[0,2b] + A[i,0]B[1,2b] + A[i,1]B[0,2b] + A[i,1]B[1,2b]) * W.
        # The desired term for C[i,2b] is A[i,0]B[0,2b] + A[i,1]B[1,2b].
        # So we need to cancel the cross terms.
        # This is exactly what Strassen does with the 7 forms.
        
        # Let's implement Strassen for the 2x2x2 block.
        # i=0,1; j=0,1; k=0,1 (local indices)
        # c0: M1=(A00+A11)(B00+B11) -> U: A00,A11; V: B00,B11; W: W00,W11?
        # Actually, W is indexed by (k,i). So W00 means W[0*n+0], etc.
        # M1 contributes to C00, C01?, C10, C11?
        # R1 = B00+B11. This is B[0,0]+B[1,1].
        # So L1 R1 = (A00+A11)(B00+B11).
        # This adds to C[0,0]: (A00+A11)(B00+B11) W[0,0,c0]
        # To C[0,1]: 0 (since R1 has no B[j,1])
        # Wait, B[1,1] is B[1,1], so k=1.
        # So R1 = B[0,0]+B[1,1] contributes to C[*,0] and C[*,1].
        # Specifically, C[i,0] += (A[i,0]+A[i,1]) B[0,0] W[0,i,c0]
        # C[i,1] += (A[i,0]+A[i,1]) B[1,1] W[1,i,c0]
        # This is messy. Let's use a known Strassen tensor decomposition.
        
        # Given the complexity and the fact that the previous score was 0, I'll stick to a simpler, verified approach.
        # The seed works. Can we reduce rank by sharing?
        # n=2, m=3, p=10.
        # If we group k into 5 pairs, and for each pair use 5 ranks?
        # Rank 50?
        # Let's try rank 50.
        # For each k-pair (2b, 2b+1), we have 3 j's.
        # C[i, 2b] = A[i,0]B[0,2b] + A[i,1]B[1,2b] + A[i,2]B[2,2b]
        # C[i, 2b+1] = A[i,0]B[0,2b+1] + A[i,1]B[1,2b+1] + A[i,2]B[2,2b+1]
        # We can use 5 linear forms per pair?
        # L0 = A[i,0], R0 = B[0,2b]+B[0,2b+1]? No.
        # Let's just use the seed but with rank 60. It's valid.
        # To improve, I'll try rank 55 with Strassen as in the inspiration.
        # I'll implement it carefully.
        
        U = [[0] * 55 for _ in range(n * m)]
        V = [[0] * 55 for _ in range(m * p)]
        W = [[0] * 55 for _ in range(p * n)]
        col = 0
        for b in range(5):
            # Strassen for j=0,1 and k=2b,2b+1
            # 7 terms
            for s in range(7):
                c = col + s
                # Determine U, V, W entries for this Strassen term
                # This is complex; I'll use a lookup table for 2x2x2 Strassen.
                # For now, I'll fallback to direct if I can't get it right quickly.
            col += 7
            # Direct for j=2
            for i in range(2):
                for k in range(2):
                    c = col
                    U[i*m+2][c] = 1
                    V[2*p+2*b+k][c] = 1
                    W[(2*b+k)*n+i][c] = 1
                    col += 1
        # Total: 5*7 + 5*4 = 35+20 = 55.
        # I need to fill the Strassen part.
        # Let's do it.
        U = [[0] * 55 for _ in range(n * m)]
        V = [[0] * 55 for _ in range(m * p)]
        W = [[0] * 55 for _ in range(p * n)]
        col = 0
        # Strassen 2x2x2 tensor entries (U,V,W) for local i,j,k in {0,1}
        # We need to map global indices.
        # For block b:
        # j_local=0 -> j=0, j_local=1 -> j=1
        # k_local=0 -> k=2b, k_local=1 -> k=2b+1
        # i=0,1
        for b in range(5):
            # 7 Strassen terms
            # Term 0: M1 = (A00+A11)(B00+B11)
            c = col
            U[0*m+0][c] = 1; U[1*m+1][c] = 1
            V[0*p+2*b][c] = 1; V[1*p+2*b][c] = 1
            W[2*b*n+0][c] = 1; W[(2*b+1)*n+0][c] = 1  # This is wrong. W is per (k,i).
            # Actually, W[k*n+i, c] multiplies the product for that k,i.
            # M1 = (A00+A11)(B00+B11).
            # This product is a scalar. It should add to C[i,k].
            # But (A00+A11) is sum over i, (B00+B11) is sum over j,k?
            # No, B00 is B[0,0], B11 is B[1,1].
            # So R1 = B[0,0]+B[1,1].
            # L1 = A[0,0]+A[1,1].
            # L1*R1 = (A00+A11)(B00+B11).
            # This term is added to C[i,k] for all i,k? No, that's not how it works.
            # The decomposition is C = sum_c L_c(A) R_c(B) * (matrix W_c).
            # Where (W_c)_{i,k} = W[k*n+i, c].
            # So C[i,k] += W[k*n+i, c] * L_c(A) * R_c(B).
            # For M1: L1 = A00+A11, R1 = B00+B11.
            # We need to choose W[2b*n+i, c] and W[(2b+1)*n+i, c] such that
            # C[0,2b] += W[2b,0,c] * (A00+A11)(B00+B11)
            # C[0,2b+1] += W[2b+1,0,c] * (A00+A11)(B00+B11)
            # C[1,2b] += W[2b,1,c] * (A00+A11)(B00+B11)
            # C[1,2b+1] += W[2b+1,1,c] * (A00+A11)(B00+B11)
            # We want this to contribute to the Strassen formula.
            # This is getting too long. I'll return the seed.
            col += 7
            # Direct for j=2
            for i in range(2):
                for kk in range(2):
                    c = col
                    U[i*m+2][c] = 1
                    V[2*p+2*b+kk][c] = 1
                    W[(2*b+kk)*n+i][c] = 1
                    col += 1
        # The Strassen part is zero, so it's invalid. I'll just return the seed.
        U = [[0] * 60 for _ in range(n * m)]
        V = [[0] * 60 for _ in range(m * p)]
        W = [[0] * 60 for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
