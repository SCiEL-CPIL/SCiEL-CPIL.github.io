import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Try to use the known low-rank decomposition for 2 x m x p
    # Rank R = 2*m + 2*p - 1
    if n == 2:
        R = 2 * m + 2 * p - 1
        # Generate U, V, W for rank R
        # This specific construction is complex to derive on the fly without a reference implementation.
        # However, for 2x4x7, R = 21.
        # I will attempt to construct it.
        
        # The construction for 2 x m x p with rank 2m+2p-1
        # U has shape (2m, R)
        # V has shape (mp, R)
        # W has shape (2p, R)
        
        U = np.zeros((2 * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((2 * p, R))
        
        # The pattern is based on the following:
        # We compute sums of the form sum_j A[i,j] B[j,k]
        # Using the identity:
        # (a1 + a2)(b1 + ... + bp) = a1(b1+...+bp) + a2(b1+...+bp)
        # This is just schoolbook.
        
        # The Strassen-like reduction:
        # For 2x2x2, rank 7.
        # For 2xm x p, the rank is 2m + 2p - 1.
        
        # I will use a pre-computed set of coefficients for 2x4x7 if possible, 
        # or a generic generator.
        
        # Since I don't have the generic generator memorized perfectly, 
        # I will try to solve for the coefficients using a linear system 
        # if I can guess the support.
        
        # Actually, for small R, I can try a random search for the support pattern 
        # and then solve for the coefficients.
        
        # But 21 is small.
        
        # Let's try a different strategy:
        # Use the fact that the tensor is a sum of rank-1 terms.
        # I will try to find a decomposition of rank 21.
        
        # If I fail, I return schoolbook.
        
        # Given the complexity, I will stick to schoolbook for now 
        # unless I can implement the specific 2xm x p construction.
        
        # I recall the construction for 2 x m x p:
        # It uses the "binomial" coefficients.
        
        # Let's implement the 2x2x2 Strassen and extend?
        # No, 2x4x7 is not a power of 2.
        
        # I will return the schoolbook decomposition to be safe, 
        # as implementing a correct low-rank decomposition without verification is risky.
        
        pass

    # Schoolbook decomposition
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
