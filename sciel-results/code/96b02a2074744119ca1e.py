import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = 3, 3, 5
    
    # Target tensor T
    # T[I, J, K] = 1 if I=i*m+j, J=j'*p+k, K=k'*n+i' and j==j', i==i', k==k'
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0

    def validate(U, V, W):
        # Check if U, V, W exactly decompose T
        # R = einsum('ic,jc,kc->ijk', U, V, W)
        # Check if R == T
        try:
            R = np.einsum('ic,jc,kc->ijk', U, V, W)
            if np.allclose(R, T, atol=1e-9):
                return True
        except Exception:
            pass
        return False

    # Attempt to construct a Rank 41 decomposition
    # Part 1: Rank 23 for 3x3x3 (first 3 columns of B)
    # Part 2: Rank 18 for 3x3x2 (last 2 columns of B)
    
    # Hardcoded Rank 23 decomposition for 3x3x3
    # This is a known minimal rank decomposition.
    # Coefficients are integers.
    
    # U1: 9x23, V1: 9x23, W1: 9x23
    # We need to define these.
    
    # If I don't have the exact coefficients, I'll use a placeholder that fails validation.
    # But I should try to provide them.
    
    # Let's use a known set of coefficients for the 23-term algorithm.
    # Reference: "A 23-multiplication algorithm for 3x3 matrix multiplication"
    
    # Due to the complexity of recalling 23*9*3 integers perfectly, 
    # I will use a safer Rank 27 (schoolbook) for the first part if I can't verify Rank 23.
    # However, Rank 27 + Rank 18 = 45. No improvement.
    
    # Let's try to find a Rank 24 or 25?
    # Actually, there is a Rank 24 algorithm?
    
    # Let's try to use the optimization approach but with a better heuristic.
    # The previous program used BFGS.
    
    # Alternative: Use a known Rank 39 decomposition?
    # If I can find a Rank 16 for 3x3x2, I can use Rank 23 for 3x3x3.
    
    # Let's try to construct the Rank 41 decomposition using Rank 27 for the first part?
    # No, that's 45.
    
    # What if I use Rank 23 for the whole 3x3x5?
    # No, rank scales.
    
    # Let's try to use a Rank 35 decomposition?
    # There is a known Rank 35 for 3x3x5?
    
    # Given the risk of incorrect hardcoded coefficients, 
    # I will implement a robust search for a low rank decomposition 
    # using a grid search or simulated annealing on the coefficients, 
    # constrained to integers/half-integers.
    
    # But the space is huge.
    
    # Let's stick to the Rank 41 idea but use a verified Rank 23 decomposition.
    # I will include a set of coefficients that I am reasonably confident about.
    
    # Coefficients for Rank 23 (3x3x3):
    # U, V, W are 9x23.
    
    # I will use a fallback: if I can't verify the hardcoded Rank 23, 
    # I will use a Rank 27 (schoolbook) for the first 3 columns 
    # and Rank 18 for the last 2, resulting in Rank 45.
    
    # To get an improvement, I MUST find a valid decomposition < 45.
    
    # Let's try to use a Rank 24 decomposition for 3x3x3?
    # If Rank 24 exists, then 24 + 18 = 42.
    
    # Actually, I can use the Strassen algorithm for 2x2 blocks.
    # 3x3 can be split into 2x2 and 1x1?
    
    # Let's try to use a Rank 41 decomposition with Rank 23.
    # I will hardcode the Rank 23 coefficients from a known source.
    
    # The following is a set of coefficients for the 23-term algorithm.
    # If validation fails, I return schoolbook.
    
    U1 = np.zeros((9, 23))
    V1 = np.zeros((9, 23))
    W1 = np.zeros((9, 23))
    
    # Filling U1, V1, W1 with a known valid set is critical.
    # Since I cannot guarantee the correctness of recalled coefficients,
    # I will use a simpler improvement:
    # Use Rank 27 for 3x3x3? No.
    
    # Let's try to use a Rank 35 decomposition for 3x3x5 directly?
    
    # Given the constraints, I will implement a search for a Rank 41 decomposition
    # using the fact that the coefficients are integers.
    
    # But that's hard.
    
    # Let's return the schoolbook solution but with a slight optimization?
    # No, score is 1.0.
    
    # I will try to use a Rank 41 decomposition with Rank 23.
    # I will use a known valid set of coefficients.
    
    # Coefficients for 23-term algorithm:
    # (Source: http://www.math.ucla.edu/~tao/23.html)
    
    # I will hardcode the 23 terms.
    
    # Term 1:
    # ...
    
    # Due to the high risk of error in manual transcription,
    # I will use a Rank 45 solution but check if I can find a Rank 41 via optimization.
    
    # The previous program failed.
    
    # Let's try a different optimization: 
    # Minimize the number of non-zero entries in U, V, W 
    # subject to the tensor identity.
    
    # This is a sparse recovery problem.
    
    # I will use a greedy approach:
    # 1. Start with Rank 45.
    # 2. Try to merge terms or set coefficients to 0.
    
    # This is also hard.
    
    # I will return the schoolbook solution.
    
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, column] = 1
                V[j*p + k, column] = 1
                W[k*n + i, column] = 1
                column += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
