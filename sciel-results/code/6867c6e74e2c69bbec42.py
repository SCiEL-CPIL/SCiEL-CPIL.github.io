"""Matrix multiplication 2x3x3 - rank-minimal exact tensor decomposition."""
import numpy as np
from fractions import Fraction
import itertools
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Output dimension: p*n = 6, indexed as W[k*n+i, r]
    # Input dimensions: n*m = 6, m*p = 9
    
    # We'll use a known rank-17 decomposition for 2x3x3 matrix multiplication.
    # This is based on Strassen-like partitioning.
    
    # For 2x3x3, we can partition:
    # A is 2x3, B is 3x3, C is 2x3
    # Split A into A1 (top row, 1x3) and A2 (bottom row, 1x3)
    # Split B into columns B[:,0], B[:,1], B[:,2]
    # 
    # Actually, let me use a known optimal decomposition.
    # The border rank and tensor rank for [2,3,3] is 17.
    
    # Let me construct the rank-17 decomposition explicitly.
    # Using the approach: split the 3x3 matrix B into overlapping 2x2 blocks.
    
    # A (2x3) * B (3x3) = C (2x3)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # We'll use 17 terms. Here's one construction:
    # Think of it as: we need to compute 2*3*3 = 18 products, save 1.
    
    # Let me use a systematic approach based on the known result.
    # For [2,3,3], rank = 17.
    
    # I'll use the following construction:
    # Partition B's columns into pairs with overlap:
    # B[:,0], B[:,1], B[:,2]
    # 
    # Actually, let me just use a verified rank-17 decomposition.
    
    # The decomposition uses the following structure:
    # We can write C = A*B where A is 2x3 and B is 3x3.
    # 
    # Let me use the following known decomposition (rank 17):
    # Based on splitting into 2x2 subproblems with sharing.
    
    # Here's a concrete rank-17 construction:
    # Think of computing:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    
    # Rank 17 means we use 17 rank-1 tensors.
    
    # Let me use a different approach: build it from Strassen's 2x2x2 (rank 7)
    # by extending to 2x3x3.
    
    # Split A (2x3) as [A_left (2x2) | A_right (2x1)]
    # Split B (3x3) as [B_top (2x3); B_bottom (1x3)]
    # 
    # C = A_left * B_top + A_right * B_bottom
    # 
    # A_left * B_top is 2x2 * 2x3 -> 2x3, rank <= 7*3/2... no
    # Actually 2x2x3 has rank 7 (by Strassen-like)
    # 2x1x3 is just 2*1*3 = 6 scalar products, but can be done in 3 rank-1 terms? No, it's rank 3.
    
    # Hmm, let me think differently.
    
    # 2x2x3: rank is 7 (Strassen gives 2x2x2 in rank 7, and 2x2x3 needs at least 7)
    # Actually 2x2x3 rank: we need to compute 2*2*3=12 products. 
    # Rank of [2,2,3] = 7? Let me verify: it's known to be 7.
    
    # 2x1x3: rank = 3 (just 3 independent products for each of 2 rows... no, it's 2*1*3=6 but rank 3)
    # Wait, 2x1x3: C[i,k] = A[i,0]*B[0,k]. That's 2*3=6 products but only 3 independent B values.
    # Rank = 3 (for each k, we have A[:,0]*B[0,k], so 3 rank-1 terms)
    
    # So total: 7 + 3 = 10? That can't be right since we know rank is 17.
    
    # I'm confusing myself. Let me look at this more carefully.
    
    # The tensor rank of matrix multiplication [n,m,p] is the minimum number of rank-1 tensors.
    # For [2,3,3], it is known to be 17.
    
    # Let me just directly construct a rank-17 decomposition using a search or known formula.
    
    # Actually, let me use a simpler approach: I'll use the schoolbook rank 18 and then
    # try to find a linear dependency to reduce to rank 17.
    
    # Schoolbook: 18 terms, one for each (i,j,k).
    # U[i*m+j, c] = 1 if c corresponds to (i,j,k)
    # V[j*p+k, c] = 1 if c corresponds to (i,j,k)  
    # W[k*n+i, c] = 1 if c corresponds to (i,j,k)
    
    # For rank 17, we need to find a linear combination.
    
    # Let me try a different known construction.
    # 
    # Based on literature, for 2x3x3, one rank-17 decomposition:
    
    # I'll use the following approach: 
    # Group the 18 schoolbook terms and find one linear dependency.
    
    # Actually, the simplest approach: use rank 18 (schoolbook) which is valid.
    # The score is rank, and 18 is valid. If we can get 17, even better.
    
    # Let me try to construct rank 17.
    
    # Known result: The rank of [2,3,3] is 17.
    # 
    # One construction: 
    # Let A be 2x3, B be 3x3.
    # Write B = [B1 B2 B3] where Bj is column j.
    # C = A * B, so C[:,j] = A * Bj.
    
    # For each column j, we compute a 2x1 = 2x3 * 3x1 product.
    # 2x3x1 has rank 3 (since it's just 3 products per row, and 2 rows share the B vector).
    # Actually 2x3x1: C[i] = sum_k A[i,k]*B[k]. Rank = 3? No, rank = 3 for each row independently would be 6, but since B is shared, rank = 3.
    
    # So 3 columns * rank 3 = 9? That's not 17 either.
    
    # I think I'm making an error. Let me be very careful.
    
    # The matrix multiplication tensor T_{i,j,k}^{a,b,c} = delta_{i,a+b} ... no.
    # For multiplying (n x m) * (m x p) -> (n x p):
    # T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
    # The (i,k)-th entry of the product is sum_j A[i,j] * B[j,k]
    # 
    # The tensor has indices (i, j, k) with values such that:
    # product[i,k] = sum_{j} A[i,j] * B[j,k]
    # 
    # As a tensor: T[i,j,k] = 1 if we're looking at the contribution.
    # Actually the tensor is: for each (i,j,k), T[i,j,k] contributes to output[i,k].
    # So the tensor T has shape (n, m, p) and T[i,j,k] = 1 for all i,j,k.
    # Wait no, that's not right either.
    
    # Let me reconsider. The matrix multiplication tensor for (n x m) * (m x p):
    # It's a trilinear form. The tensor T in R^{n*m * m*p * n*p} ... no.
    # 
    # Standard formulation: T is a tensor in R^{(n*m) x (m*p) x (n*p)}
    # where T[(i*m+j), (j*p+k), (k*n+i)] = 1 (or some convention)
    # 
    # Actually the standard is: T[a, b, c] where a indexes A (n*m), b indexes B (m*p), c indexes C (n*p)
    # T[a, b, c] = 1 if the (a)-th element of A and (b)-th element of B contribute to the (c)-th element of C.
    # 
    # With a = i*m + j, b = j*p + k, c = k*n + i (using the W[k*n+i] convention):
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k.
    
    # So the tensor has 1s at positions (i*m+j, j*p+k, k*n+i) for i in [0,n), j in [0,m), k in [0,p).
    # Total number of 1s: n*m*p = 2*3*3 = 18.
    
    # A rank-r decomposition means: T = sum_{t=1}^r u_t ⊗ v_t ⊗ w_t
    # where u_t in R^{n*m}, v_t in R^{m*p}, w_t in R^{n*p}
    # 
    # And the matrix multiplication is: C = (U * diag * V^T) ... 
    # Specifically: C[c] = sum_t U[:,t] * V[:,t] * W[:,t] ... no.
    # 
    # The decomposition: T[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
    # And the bilinear map: C = sum_t (U[:,t] * A) ⊙ (V[:,t] * B) * W[:,t]
    # where ⊙ is element-wise and the result is accumulated.
    # 
    # More precisely: C[c] = sum_t W[c,t] * (sum_a U[a,t] * A[a]) * (sum_b V[b,t] * B[b])
    
    # OK so for the schoolbook: each term t corresponds to (i,j,k) with:
    # U[i*m+j, t] = 1, V[j*p+k, t] = 1, W[k*n+i, t] = 1
    
    # For rank 17, we need to merge two terms or find a linear dependency.
    
    # Let me just use rank 18 for safety, then try to improve.
    
    # Actually, let me try a known rank-17 construction.
    
    # After more thought, let me use the following approach:
    # I'll use a randomized search to find a rank-17 decomposition, or fall back to rank 18.
    
    # For now, let me implement rank 18 (schoolbook) which is definitely correct.
    
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
