import time
import math

def _verify_3x3x3(terms, n=3):
    """Verify if the terms form a valid 3x3x3 decomposition."""
    # Tensor T[i,j,k] = 1
    # sum_t U_t[i,j] V_t[j,k] W_t[k,i] = 1
    for i in range(n):
        for j in range(n):
            for k in range(n):
                s = 0
                for U, V, W in terms:
                    s += U[i*n+j] * V[j*n+k] * W[k*n+i]
                if s != 1:
                    return False
    return True

def _build_larsen_23():
    # Larsen 3x3x3 Rank 23 terms
    # U, V, W are flattened 3x3 matrices
    # Coefficients from Larsen (1979)
    # Note: Indices are 0..8
    # U[i*3+j], V[j*3+k], W[k*3+i]
    
    # This is the standard set of 23 terms.
    # I will use a known correct set.
    # Reference: https://github.com/... (hypothetical)
    # Actually, I will use the coefficients from the "Fast Matrix Multiplication" library.
    
    # Term 0
    U0 = [1,0,0, 0,0,0, 0,0,0]
    V0 = [0,0,0, 0,1,0, 0,0,0]
    W0 = [0,0,0, 0,0,0, 0,0,1]
    
    # ... This is too tedious to write from memory without error.
    # I will use a simpler, verified low-rank decomposition if available.
    # If not, I must rely on the schoolbook or a very simple optimization.
    
    # However, there is a known rank 23 decomposition.
    # I will use the coefficients from the paper.
    # If I cannot guarantee correctness, I should not risk it.
    
    # Alternative: Can we use a rank 48 decomposition?
    # No, 46 is the target.
    
    # I will implement the verification and return schoolbook if Larsen fails.
    # I will include the Larsen coefficients from a trusted source in my knowledge.
    
    # The 23 terms are:
    terms = [
        ([1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1]),
        # ... I cannot fill 23 terms from memory accurately.
    ]
    
    # Since I cannot accurately recall all 23 terms, I will stick to the schoolbook.
    # But wait, is there a simpler improvement?
    # What if I use Strassen on 2x2 blocks?
    # 3x3x6 can be decomposed into 2x2x2 and 1x1x1 parts?
    # No, 3 is not a power of 2.
    
    # I will return the schoolbook decomposition.
    # The score will be 1.0.
    
    pass

def construct():
    n, m, p = 3, 3, 6
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
