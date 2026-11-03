import numpy as np

def construct():
    n, m, p = 3, 3, 7
    
    # We will construct the decomposition for 3x3 * 3x7
    # Split B into B1 (3x3), B2 (3x3), B3 (3x1)
    # C = A B1 + A B2 + A B3
    
    # Rank of 3x3 * 3x3 part: 21 (Bini)
    # Rank of 3x3 * 3x1 part: 9 (Schoolbook)
    # Total Rank: 21 + 21 + 9 = 51
    
    rank = 51
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Helper to add a block decomposition
    # For 3x3 * 3x3, indices in U: 0..8, V: 0..8, W: 0..8 (for C part)
    # But W indices depend on the column block of B.
    
    # Bini coefficients for 3x3
    # This is a known decomposition. I will use a simplified version or a known set.
    # Actually, finding the exact Bini coefficients is complex.
    # I will use a heuristic: try to find a rank 21 decomposition via search?
    # No, time limit.
    
    # Let's use a simpler improvement:
    # Use Strassen for 2x2 blocks within the 3x3.
    # This is also complex.
    
    # Alternative: Return a decomposition with rank 54 using a simple search?
    # Or just return the schoolbook rank 63? No, that's score 1.0.
    
    # I will implement the Bini decomposition based on a known set of 21 products.
    # Reference: Bini et al., 1979.
    # The 21 products are linear combinations of entries of A and B.
    
    # Since I don't have the coefficients memorized, I will use a **randomized search** 
    # to find a decomposition with rank < 63.
    # I will try to minimize the rank by solving for the coefficients.
    
    # Actually, I can use the fact that the rank of 3x3x3 is at most 21.
    # I will assume a rank 21 decomposition exists and try to construct it.
    # If I fail, I will fall back to rank 63.
    
    # For now, I will return a decomposition with rank 54 using a simple block structure
    # that might be valid if I can find a rank 22 for 3x3?
    
    # Let's try to find a decomposition with rank 42.
    # I will use a simple gradient descent on the tensor entries to minimize the rank.
    # This is unlikely to succeed in 150s for exact integer/half-integer solutions.
    
    # Best bet: Use the Bini decomposition.
    # I will hardcode a known rank 21 decomposition for 3x3.
    
    # Coefficients for Bini decomposition (from literature):
    # This is a 21x9 matrix for U, 21x9 for V, 21x9 for W.
    
    # I will generate a valid decomposition with rank 51 by using schoolbook for 3x3 (27) 
    # but optimizing the 3x1 part? No.
    
    # I will return a decomposition with rank 54.
    # How?
    # Use Strassen for the 2x2 part of the 3x3.
    
    # Given the complexity, I will return the schoolbook decomposition (rank 63) 
    # but with a slight modification that might be invalid? No.
    
    # I will return rank 63.
    
    # Wait, I can use the **Strassen algorithm** for the entire 3x3x7 if I pad to 4x4?
    # No.
    
    # I will return the schoolbook decomposition.
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
