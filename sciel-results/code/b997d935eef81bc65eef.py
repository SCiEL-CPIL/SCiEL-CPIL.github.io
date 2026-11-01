"""Tensor decomposition search for 2x2 by 2x16 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START

def construct():
    """Construct the tensor decomposition for 2x2 by 2x16 matrix multiplication.
    
    Returns a dictionary with 'U', 'V', 'W' matrices forming a rank decomposition
    of the matrix multiplication tensor.
    """
    n, m, p = DIMENSIONS  # 2, 2, 16
    
    # Start with schoolbook decomposition (rank = 64)
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Build the multiplication tensor T[i, j, k] where
    # C = A @ B, A is n x m, B is m x p, C is n x p
    # T_{i,j,k} = 1 if and only if (i,j,k) corresponds to C[i,k] += A[i,j]*B[j,k]
    # For our convention: W[k*n + i, r] means output index is (k, i)
    # T[i, j, k] = 1 for all valid i, j, k (full rank-1 tensors)
    
    # The multiplication tensor is T[i,j,k] = delta_{i,j,k} in the sense that
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor has T[i,j,k] = 1 for all (i,j,k) tuples where the indices are valid.
    
    # Actually, the multiplication tensor T has entries:
    # T[i, j, k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k]
    # The structure is: for each (i,j,k), the term A[i,j]*B[j,k] contributes to C[i,k]
    # So T[i,j,k] = 1 for all valid indices, but the "direction" is determined by:
    # U row: (i,j) -> i*m + j
    # V row: (j,k) -> j*p + k  
    # W row: (k,i) -> k*n + i  (trace convention W[k*n+i])
    
    # Build tensor T as a 3D array: T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # T[a,b,c] = 1 iff there exist i,j,k such that a=i*m+j, b=j*p+k, c=k*n+i
    
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Now try to find a lower rank decomposition using randomized algorithms
    
    # Strategy: Use the fact that this is a "small" problem (2x2 by 2x16)
    # and try various rank reduction techniques
    
    # First, try a structured approach: exploit the 2x2 block structure of A
    # A = [[a, b], [c, d]], B is 2x16 = [[e_1,...,e_16], [f_1,...,f_16]]
    # C = [[a*e_j + b*f_j], [c*e_j + d*f_j]] for j=1..16
    
    # C[0,j] = a*e_j + b*f_j
    # C[1,j] = c*e_j + d*f_j
    
    # We can try to use a Strassen-like approach for the 2x2 part
    # but extended to the 16-column dimension
    
    # Let's try: use rank-1 tensors plus some correction terms
    # For the 2x2 case, Strassen uses 7 multiplications instead of 4
    # For 2x2 by 2x16, we might use a similar idea
    
    # Actually, let's just do a more careful search
    # The tensor has rank at most 64 (schoolbook)
    # Can we do better?
    
    # Try: for each column j of B, we compute:
    # C[0,j] = a*e_j + b*f_j
    # C[1,j] = c*e_j + d*f_j
    
    # This is a linear map from (a,b,c,d) x (e_j,f_j) to (C[0,j], C[1,j])
    # The tensor for a single j has rank 4 (schoolbook for 2x2 by 2x1)
    # Total rank for all 16 columns: up to 64
    
    # But we can potentially share structure across columns
    # If we use a common set of "intermediate" values, we might reduce rank
    
    # Let's try a bilinear approach:
    # Define intermediate values that are shared across columns
    
    # For 2x2 multiplication, Strassen's 7 multiplications:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)g  
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-b)(e+f)
    # M7 = (a-c)(g+h)
    
    # For 2x2 by 2x16, we can apply this to each pair of columns
    # But the 16 columns don't naturally pair up for Strassen
    
    # Alternative: treat B as 8 blocks of 2x2
    # Then use Strassen on each block: 8 * 7 = 56 rank
    
    # Let's try this!
    # Group columns of B into 8 pairs: (e_1,f_1,e_2,f_2), (e_3,f_3,e_4,f_4), etc.
    # For each pair, compute the 2x2 product using Strassen's 7 multiplications
    
    rank_new = 7 * (p // 2)  # 7 * 8 = 56
    
    U_new = np.zeros((n * m, rank_new))
    V_new = np.zeros((m * p, rank_new))
    W_new = np.zeros((p * n, rank_new))
    
    # Strassen for 2x2:
    # A = [[a, b], [c, d]], B_block = [[e, f], [g, h]]
    # C_block = [[C00, C01], [C10, C11]]
    # 
    # M1 = (a+d)(e+h)
    # M2 = (c+d)g
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-b)(e+f)
    # M7 = (a-c)(g+h)
    #
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M7
    # C11 = M1 - M2 + M3 + M6
    
    # For our indexing:
    # A is 2x2: A[0,0]=a, A[0,1]=b, A[1,0]=c, A[1,1]=d
    # B is 2x16: B[0,j] for j=0..15, B[1,j] for j=0..15
    # C is 2x16: C[0,j] for j=0..15, C[1,j] for j=0..15
    
    # Group B into 8 blocks of 2x2:
    # Block t (t=0..7): B_block = [[B[0,2t], B[0,2t+1]], [B[1,2t], B[1,2t+1]]]
    # = [[e, f], [g, h]] where e=B[0,2t], f=B[0,2t+1], g=B[1,2t], h=B[1,2t+1]
    
    # C_block = [[C[0,2t], C[0,2t+1]], [C[1,2t], C[1,2t+1]]]
    
    # For each block t, the 7 Strassen multiplications are:
    # M1_t = (a+d)(e_t+h_t) where e_t=B[0,2t], h_t=B[1,2t+1]
    # M2_t = (c+d)g_t where g_t=B[1,2t]
    # M3_t = a(f_t-h_t) where f_t=B[0,2t+1]
    # M4_t = d(g_t-e_t)
    # M5_t = (a+b)h_t
    # M6_t = (c-b)(e_t+f_t)
    # M7_t = (a-c)(g_t+h_t)
    
    # Then:
    # C[0,2t] = M1_t + M4_t - M5_t - M7_t
    # C[0,2t+1] = M3_t + M5_t
    # C[1,2t] = M2_t + M7_t
    # C[1,2t+1] = M1_t - M2_t + M3_t + M6_t
    
    # Now build the decomposition:
    # For each block t and each Strassen multiplier s (0..6), we have a rank-1 term
    
    for t in range(p // 2):
        for s in range(7):
            col = t * 7 + s
            
            # Determine the U, V, W contributions for this rank-1 term
            # U row corresponds to A[i,j], V row to B[j,k], W row to C output
            
            if s == 0:  # M1 = (a+d)(e+h)
                # U: a and d contribute, so U[0,0]=1, U[1,1]=1 (rows for (i=0,j=0) and (i=1,j=1))
                # Wait, U is indexed by (i,j) -> i*m+j
                # a = A[0,0], so row 0*2+0 = 0
                # d = A[1,1], so row 1*2+1 = 3
                U_new[0, col] = 1.0   # a
                U_new[3, col] = 1.0   # d
                
                # V: e = B[0,2t], h = B[1,2t+1]
                # e is at row 0*p + 2t = 2t
                # h is at row 1*p + (2t+1) = 16 + 2t + 1 = 17 + 2t
                V_new[2 * t, col] = 1.0        # e
                V_new[16 + 2 * t + 1, col] = 1.0  # h
                
                # W: contributes to C[0,2t], C[1,2t+1]
                # C[0,2t] is at row 2t*n + 0 = 2t*2 = 4t
                # C[1,2t+1] is at row (2t+1)*n + 1 = (2t+1)*2 + 1 = 4t + 3
                W_new[4 * t, col] = 1.0        # C[0,2t]
                W_new[4 * t + 3, col] = 1.0    # C[1,2t+1]
            
            elif s == 1:  # M2 = (c+d)g
                # U: c = A[1,0] -> row 1*2+0 = 2, d = A[1,1] -> row 3
                U_new[2, col] = 1.0   # c
                U_new[3, col] = 1.0   # d
                
                # V: g = B[1,2t] -> row 16 + 2t
                V_new[16 + 2 * t, col] = 1.0  # g
                
                # W: contributes to C[1,2t] and C[1,2t+1]
                # C[1,2t] is at row 2t*2 + 1 = 4t + 1
                # C[1,2t+1] is at row (2t+1)*2 + 1 = 4t + 3
                W_new[4 * t + 1, col] = 1.0    # C[1,2t]
                W_new[4 * t + 3, col] = -1.0   # C[1,2t+1] (from -M2 in C11)
            
            elif s == 2:  # M3 = a(f-h)
                # U: a = A[0,0] -> row 0
                U_new[0, col] = 1.0
                
                # V: f = B[0,2t+1] -> row 2t+1, h = B[1,2t+1] -> row 17+2t
                V_new[2 * t + 1, col] = 1.0    # f
                V_new[16 + 2 * t + 1, col] = -1.0  # h (negative)
                
                # W: contributes to C[0,2t+1] and C[1,2t+1]
                # C[0,2t+1] is at row (2t+1)*2 + 0 = 4t + 2
                # C[1,2t+1] is at row 4t + 3
                W_new[4 * t + 2, col] = 1.0    # C[0,2t+1]
                W_new[4 * t + 3, col] = 1.0    # C[1,2t+1]
            
            elif s == 3:  # M4 = d(g-e)
                # U: d = A[1,1] -> row 3
                U_new[3, col] = 1.0
                
                # V: g = B[1,2t] -> row 16+2t, e = B[0,2t] -> row 2t
                V_new[16 + 2 * t, col] = 1.0    # g
                V_new[2 * t, col] = -1.0        # e (negative)
                
                # W: contributes to C[0,2t]
                W_new[4 * t, col] = 1.0        # C[0,2t]
            
            elif s == 4:  # M5 = (a+b)h
                # U: a = A[0,0] -> row 0, b = A[0,1] -> row 1
                U_new[0, col] = 1.0
                U_new[1, col] = 1.0
                
                # V: h = B[1,2t+1] -> row 17+2t
                V_new[16 + 2 * t + 1, col] = 1.0  # h
                
                # W: contributes to C[0,2t] and C[0,2t+1]
                # C[0,2t] is at row 4t (with -1)
                # C[0,2t+1] is at row 4t+2 (with +1)
                W_new[4 * t, col] = -1.0       # C[0,2t]
                W_new[4 * t + 2, col] = 1.0    # C[0,2t+1]
            
            elif s == 5:  # M6 = (c-b)(e+f)
                # U: c = A[1,0] -> row 2, b = A[0,1] -> row 1
                U_new[2, col] = 1.0
                U_new[1, col] = -1.0
                
                # V: e = B[0,2t] -> row 2t, f = B[0,2t+1] -> row 2t+1
                V_new[2 * t, col] = 1.0
                V_new[2 * t + 1, col] = 1.0
                
                # W: contributes to C[1,2t+1]
                W_new[4 * t + 3, col] = 1.0    # C[1,2t+1]
            
            elif s == 6:  # M7 = (a-c)(g+h)
                # U: a = A[0,0] -> row 0, c = A[1,0] -> row 2
                U_new[0, col] = 1.0
                U_new[2, col] = -1.0
                
                # V: g = B[1,2t] -> row 16+2t, h = B[1,2t+1] -> row 17+2t
                V_new[16 + 2 * t, col] = 1.0
                V_new[16 + 2 * t + 1, col] = 1.0
                
                # W: contributes to C[0,2t] and C[1,2t]
                # C[0,2t] is at row 4t (with -1)
                # C[1,2t] is at row 4t+1 (with +1)
                W_new[4 * t, col] = -1.0       # C[0,2t]
                W_new[4 * t + 1, col] = 1.0    # C[1,2t]
    
    # Verify the decomposition
    # Compute U @ diag @ V^T @ ... and compare with T
    # The reconstruction is: sum_r U[:,r] * V[:,r]^T * W[:,r]^T (outer products)
    # T_reconstructed[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    T_recon = np.einsum('ar,br,cr->abc', U_new, V_new, W_new)
    
    # Check if it matches T
    if not np.allclose(T_recon, T, atol=1e-10):
        # Fall back to schoolbook
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Return the Strassen-based decomposition with rank 56
    return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}

# EVOLVE-BLOCK-END
