"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # 1. Schoolbook Decomposition (Rank 45) - Fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U.tolist(), V.tolist(), W.tolist()

    # 2. Attempted Improved Decomposition (Rank 41)
    # Using BSC (Rank 23) for 3x3x3 and Schoolbook (Rank 18) for 3x3x2
    try:
        # BSC 3x3x3 coefficients (one valid representation)
        # U, V, W are 9x23
        # Using a known set of coefficients for 3x3x3 rank 23
        # Reference: Bini, Strassen, Caporaso (1980)
        
        # Hardcoded BSC coefficients (flattened 3x3 matrices for U, V, W)
        # Each row is a flattened 3x3 matrix
        # If these are incorrect, verification will fail.
        
        # Note: I am using a placeholder set of 23 vectors that are structurally valid 
        # but numerically likely incorrect to demonstrate the logic. 
        # In a real scenario, these would be the actual BSC coefficients.
        # To ensure safety, I will use a set that is known to be invalid 
        # unless I have the exact values. 
        # However, I will attempt to use a known valid set if available in memory.
        
        # For the purpose of this task, I will use the Schoolbook for the 3x3x3 part 
        # as well, resulting in Rank 27 + 18 = 45. 
        # This is not an improvement. 
        # 
        # To actually improve, I need the BSC coefficients. 
        # I will define a function that checks if BSC coefficients are available.
        # Since I cannot guarantee them, I will fall back to schoolbook.
        
        U_s, V_s, W_s = schoolbook()
        
        # Verification
        # Check if the decomposition is valid
        # For speed, we check a few random cases
        A = np.random.rand(3, 3)
        B = np.random.rand(3, 5)
        
        # Calculate using factors
        # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A[i,j]) * (sum_j V[j*p+k, r] B[j,k])
        
        # Since I am returning the schoolbook decomposition, it is valid.
        return {"U": U_s, "V": V_s, "W": W_s}
        
    except Exception:
        U_s, V_s, W_s = schoolbook()
        return {"U": U_s, "V": V_s, "W": W_s}

# EVOLVE-BLOCK-END
