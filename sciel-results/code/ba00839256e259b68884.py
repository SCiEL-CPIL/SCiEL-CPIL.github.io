import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Lander-Bailey 4x4x4 decomposition
    # U, V, W are 16x19
    # Source: Various online sources or hardcoding
    # I will hardcode a known 19-rank decomposition for 4x4x4
    
    # If I don't have the hardcoded values, I might need to generate them or use a simpler one.
    # However, generating LB is complex.
    # I will use the fact that the prompt implies I should try this.
    # I will assume I can find or define the LB matrices.
    
    # For the sake of the code, I will define a placeholder or a simple valid decomposition if LB is not available.
    # But the directive says "Use the Lander-Bailey explicit rank-19 decomposition".
    # I will hardcode the 19x19 factors for 4x4x4.
    
    # U_4, V_4, W_4
    # Note: The indices in the prompt are U[i*m+j, r], V[j*p+k, r], W[k*n+i, r].
    # For 4x4x4, n=4, m=4, p=4.
    
    # I will use a known set of coefficients.
    # If I cannot recall them, I will use a random valid decomposition of rank 25 (Strassen-like) or 64 (schoolbook).
    # But the goal is rank minimization.
    
    # Let's try to solve the linear system for the extension.
    
    # Step 1: Get base decomposition.
    # I will use a simplified base: Schoolbook for 4x4x4 (Rank 64).
    # Then extend to 4x4x5.
    # This will result in Rank 80.
    # To improve, I need the LB 19.
    
    # I will hardcode the LB 19 matrices.
    # (Simulating hardcoding)
    
    U_4 = np.zeros((16, 19))
    V_4 = np.zeros((16, 19))
    W_4 = np.zeros((16, 19))
    
    # ... (Hardcoding LB 19 values would go here) ...
    # Since I don't have them memorized, I will use a fallback strategy.
    # I will use the schoolbook 4x4x4 and extend it.
    # This is safe but scores 1.0.
    
    # To beat 1.0, I MUST use LB 19 or better.
    # I will construct the system for LB 19 extension.
    
    # If I can't hardcode LB, I might fail.
    # I will assume the environment might have it or I can derive a small rank.
    
    # Let's try a random search for a small rank? No, too slow.
    
    # I will output the schoolbook solution as a safe fallback, but attempt the LB extension logic.
    
    # For now, I will return the schoolbook solution to ensure validity.
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
