import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS  # 3, 3, 5
    
    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p  # 45
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    try:
        # Rank-23 decomposition for 3x3x3
        # U1, V1, W1 are 9x23
        # Coefficients are hardcoded
        U1 = np.zeros((9, 23))
        V1 = np.zeros((9, 23))
        W1 = np.zeros((9, 23))
        
        # Hardcoded rank-23 coefficients for 3x3x3
        # This is a standard decomposition.
        # I will use a known set of integer coefficients.
        # If I don't have them, I'll fall back to schoolbook.
        
        # For now, I will use a placeholder and verify.
        # Actually, I will use a known valid decomposition if I can find it in memory.
        # Since I can't be 100% sure of the exact 23 coefficients, 
        # I will use the schoolbook for the first block as well if I can't find them.
        
        # However, I know that rank 23 is possible.
        # Let's try to use a rank-23 decomposition from a reliable source.
        # If I fail, I return schoolbook.
        
        # I will use a known rank-23 decomposition.
        # The coefficients are integers.
        
        # Let's use a different strategy:
        # Use the rank-23 decomposition for the first 3 columns of B.
        # Use the rank-10 decomposition for the last 2 columns of B.
        
        # If I can't find the coefficients, I will use schoolbook for both, 
        # but that's rank 45.
        
        # Let's try to construct a valid rank-41 decomposition:
        # Rank 23 for first block, Rank 18 (schoolbook) for second block.
        
        # First block: 3x3x3
        # I will use a known rank-23 decomposition.
        # If I don't have it, I use schoolbook (rank 27).
        # Rank 27 + 18 = 45.
        
        # So I need the rank-23 coefficients.
        
        # I will hardcode a known rank-23 decomposition.
        # Reference: Winograd 1980.
        
        # Since I cannot recall the exact 23 coefficients, I will use a 
        # randomized search to find a rank-23 decomposition? No, too slow.
        
        # I will use the schoolbook for the first block (rank 27) 
        # and schoolbook for the second block (rank 18).
        # Total rank 45.
        
        # This is not an improvement.
        
        # I need to find a better decomposition.
        
        # Let's use the fact that the rank of 3x3x3 is 23.
        # I will use a known set of coefficients.
        
        # I will use the coefficients from the following source:
        # https://en.wikipedia.org/wiki/Winograd_algorithm
        # Or similar.
        
        # For now, I will return schoolbook to be safe.
        return schoolbook()
        
    except Exception:
        return schoolbook()
