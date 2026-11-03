import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # safety margin
    
    # Trivial rank-60 solution
    rank_trivial = n * m * p  # 60
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1, else 0
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Build the rank-55 solution using Strassen for the 2x2 blocks in the middle
    def build_rank55():
        rank = 55
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        c = 0
        # Part A: j=0 (first column of A times first row of B)
        # This is a 2x1 times 1x10 multiplication
        # Actually, let's be more systematic.
        # A is 2x3, B is 3x10.
        # A = [A0 | A1 | A2] where A0, A1, A2 are 2x1 columns.
        # B = [B0; B1; B2] where B0, B1, B2 are 1x10 rows.
        # C = A0*B0 + A1*B1 + A2*B2.
        
        # A0*B0 is 2x10. We can do this with rank 2*10 = 20? No, rank 2*10 is trivial.
        # But we can do better? For 2x1 * 1x10, the tensor is just the outer product of a 2-vector and a 10-vector.
        # The rank of the outer product tensor (a_i b_k) is 1? No.
        # The tensor T[a, b, c] for A0*B0 has non-zero entries when a corresponds to i in 0..1, b corresponds to k in 0..9, c corresponds to k*n+i.
        # Wait, the indices are:
        # U index: i*m + 0 (since j=0)
        # V index: 0*p + k (since j=0)
        # W index: k*n + i
        # This is a rank-1 tensor? No, it's a sum of 2*10 rank-1 tensors.
        # T[i, k, k*n+i] = 1.
        # This is the matrix multiplication tensor for 2x1 and 1x10.
        # The rank of M_{2,1,10} is 2*10 = 20.
        
        # So Part A costs 20.
        for i in range(n):
            for k in range(p):
                U[i*m+0, c] = 1
                V[0*p+k, c] = 1
                W[k*n+i, c] = 1
                c += 1
        
        # Part B: A1*B1 + A2*B2
        # A1, A2 are 2x1. B1, B2 are 1x10.
        # Let A1 = [a11; a21], A2 = [a12; a22]
        # Let B1 = [b11, ..., b1,10], B2 = [b21, ..., b2,10]
        # We want to compute C12 = A1*B1 + A2*B2, which is a 2x10 matrix.
        # C12[i, k] = a1i*b1k + a2i*b2k (using 0-indexed: a10*b10 + a11*b20 for i=0, k=0)
        # Wait, A1 is column 1 of A (index 1). A2 is column 2 of A (index 2).
        # A = [A0, A1, A2].
        # C = A0*B0 + A1*B1 + A2*B2.
        # A1 is 2x1, B1 is 1x10. A2 is 2x1, B2 is 1x10.
        # So we are adding two 2x10 matrices.
        # We can group the k indices into pairs and use Strassen-like tricks for the 2x2 block?
        # No, the "2" in Strassen is the dimension of the square block.
        # Here we have a 2x1 times 1x10.
        # If we group B1 and B2 into a 2x10 matrix B12, and A1 and A2 into a 2x2 matrix A12?
        # No, A1 and A2 are columns. So A12 is the 2x2 submatrix of A consisting of columns 1 and 2.
        # B12 is the 2x10 submatrix of B consisting of rows 1 and 2.
        # Then A1*B1 + A2*B2 is the (1,1) and (1,2) entries of A12 * B12?
        # Let A12 = [[a11, a12], [a21, a22]]
        # Let B12 = [[b11, ..., b1p], [b21, ..., b2p]]
        # A12 * B12 is a 2x10 matrix.
        # Entry (0, k) = a11*b1k + a12*b2k. This is exactly (A1*B1 + A2*B2)[0, k].
        # Entry (1, k) = a21*b1k + a22*b2k. This is exactly (A1*B1 + A2*B2)[1, k].
        # So we need to compute the product of a 2x2 matrix and a 2x10 matrix.
        # This is M_{2,2,10}.
        # We can compute this by splitting the 10 columns into 5 pairs of 2.
        # For each pair of columns (k, k+1), we compute a 2x2 product.
        # M_{2,2,2} can be done with rank 7 (Strassen).
        # So 5 * 7 = 35.
        # Total rank = 20 (for Part A) + 35 (for Part B) = 55.
        
        # Let's implement this.
        # Part B: 5 blocks of 2x2 * 2x2.
        for b in range(5):
            k0 = 2*b
            k1 = 2*b+1
            
            # Indices in U (A12):
            # A12 is columns 1,2 of A.
            # U index for A12[i, j] is i*m + (1+j) ?
            # A has columns 0, 1, 2.
            # A1 is column 1. A2 is column 2.
            # A12[i, 0] = A[i, 1] -> U index i*m + 1
            # A12[i, 1] = A[i, 2] -> U index i*m + 2
            u11, u12 = 0*m+1, 0*m+2
            u21, u22 = 1*m+1, 1*m+2
            
            # Indices in V (B12):
            # B12 is rows 1,2 of B.
            # B12[0, k] = B[1, k] -> V index 1*p + k
            # B12[1, k] = B[2, k] -> V index 2*p + k
            v11, v12 = 1*p+k0, 1*p+k1
            v21, v22 = 2*p+k0, 2*p+k1
            
            # Indices in W (Result C12):
            # C12[0, k0] = (A1*B1+A2*B2)[0, k0] -> W index k0*n + 0
            # C12[0, k1] = (A1*B1+A2*B2)[0, k1] -> W index k1*n + 0
            # C12[1, k0] = (A1*B1+A2*B2)[1, k0] -> W index k0*n + 1
            # C12[1, k1] = (A1*B1+A2*B2)[1, k1] -> W index k1*n + 1
            w00, w01 = k0*n+0, k1*n+0
            w10, w11 = k0*n+1, k1*n+1
            
            # Strassen for 2x2 * 2x2
            # M1 = (a11+a22)(b11+b22) -> contributes to c11, c22
            U[u11,c]+=1; U[u22,c]+=1; V[v11,c]+=1; V[v22,c]+=1; W[w00,c]+=1; W[w11,c]+=1; c+=1
            # M2 = (a21+a22)b11 -> contributes to c21, c22
            U[u21,c]+=1; U[u22,c]+=1; V[v11,c]+=1; W[w10,c]+=1; W[w11,c]+=-1; c+=1
            # M3 = a11(b12-b22) -> contributes to c12, c22
            U[u11,c]+=1; V[v12,c]+=1; V[v22,c]+=-1; W[w01,c]+=1; W[w11,c]+=1; c+=1
            # M4 = a22(b21-b11) -> contributes to c11, c21
            U[u22,c]+=1; V[v21,c]+=1; V[v11,c]+=-1; W[w00,c]+=1; W[w10,c]+=1; c+=1
            # M5 = (a11+a12)b22 -> contributes to c11, c12
            U[u11,c]+=1; U[u12,c]+=1; V[v22,c]+=1; W[w00,c]+=-1; W[w01,c]+=1; c+=1
            # M6 = (a21-a11)(b11+b12) -> contributes to c22
            U[u21,c]+=1; U[u11,c]+=-1; V[v11,c]+=1; V[v12,c]+=1; W[w11,c]+=1; c+=1
            # M7 = (a12-a22)(b21+b22) -> contributes to c11
            U[u12,c]+=1; U[u22,c]+=-1; V[v21,c]+=1; V[v22,c]+=1; W[w00,c]+=1; c+=1
        
        return U, V, W, rank
    
    U, V, W, rank = build_rank55()
    
    # Verify
    def verify(U, V, W):
        r = U.shape[1]
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    val = np.sum(U[a,:r] * V[b,:r] * W[c,:r])
                    if abs(val - T[a,b,c]) > 1e-6:
                        return False
        return True
    
    if not verify(U, V, W):
        # Fallback to trivial
        rank = n*m*p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1
    
    # Try to improve by finding a lower rank decomposition.
    # We can try to merge some of the terms.
    # For example, in Part A, we have 20 terms.
    # Can we do better for 2x1 * 1x10?
    # The tensor is T[i, k, k*n+i] = 1.
    # This is a "slice" of the matrix multiplication tensor.
    # It is known that the rank of M_{n,1,p} is n*p.
    # So Part A is optimal at 20.
    
    # What about Part B?
    # We used 5 * 7 = 35.
    # Is there a better way to compute M_{2,2,10}?
    # M_{2,2,10} is the product of a 2x2 and a 2x10 matrix.
    # We can view this as 5 separate M_{2,2,2} products.
    # The rank of M_{2,2,10} is at most 5*7 = 35.
    # Is it less?
    # The border rank of M_{2,2,10} is 35?
    # Actually, M_{2,2,10} is isomorphic to M_{10,2,2} by permuting indices?
    # No, the indices are (i, j, k) where i is row of A (2), j is col of A/row of B (2), k is col of B (10).
    # T[i, j, k] = delta_{j, k'?} No.
    # T[i, j, k] = 1 if j == k? No.
    # For M_{2,2,10}, T[i, j, k] = 1 if j == k? No.
    # The standard tensor T_{ijk} = delta_{jk} for M_{n,m,p}?
    # No, T_{ijk} = 1 if i==i, j==j, k==k?
    # The condition is T[i*m+j, j*p+k, k*n+i] = 1.
    # For M_{2,2,10}, n=2, m=2, p=10.
    # T[i*2+j, j*10+k, k*2+i] = 1.
    # This is the standard tensor.
    # The rank of M_{2,2,10} is known to be 35?
    # Actually, M_{2,2,p} can be computed with rank 7*ceil(p/2) if p is even?
    # If p=10, 7*5 = 35.
    # Can we do better?
    # There is a result by Bläser et al. that M_{2,2,p} has rank 7p - something?
    # Actually, for M_{2,2,p}, the rank is 7p - 4 for p >= 2?
    # Let's check p=2: 7*2 - 4 = 10? No, Strassen is 7.
    # So the formula is not 7p-4.
    # It is known that M_{2,2,p} has rank at most 7*ceil(p/2).
    # And it is open whether it can be done with less.
    # So 35 is likely optimal or near-optimal.
    
    # Total rank 55 seems hard to beat.
    # But wait, can we combine Part A and Part B?
    # Part A is A0*B0. Part B is A12*B12.
    # A0 is 2x1, B0 is 1x10.
    # A12 is 2x2, B12 is 2x10.
    # We are computing C = A0*B0 + A12*B12.
    # This is a 2x10 matrix.
    # Can we compute this whole thing with less than 20+35=55?
    # Maybe by using a different decomposition.
    # For example, treat A as a 2x3 matrix and B as a 3x10 matrix.
    # We can use the fact that 3 = 1 + 2.
    # Is there a way to interleave the computations?
    # Probably not easily, because the indices are separate.
    
    # Let's try a different approach: use the "laser method" or "subtensor" method.
    # But that's too complex for this time limit.
    
    # Instead, let's try to find a rank-54 solution by removing one term and adjusting the others.
    # This is a local search.
    
    # For now, return the rank-55 solution.
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
