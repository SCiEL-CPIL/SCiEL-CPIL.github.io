import time
import sys
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 6]

def _construct_tensor(n, m, p):
    """Construct the full tensor T of shape (n*m, m*p, p*n)."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # T[i*m+j, j*p+k, k*n+i] = 1
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _tensor_error(U, V, W, T):
    """Calculate the squared Frobenius error between the reconstructed tensor and T."""
    R = U.shape[1]
    Rec = np.zeros_like(T)
    for r in range(R):
        Rec += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
    return np.sum((Rec - T)**2)

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 115.0
    
    n, m, p = 3, 3, 6
    rank = 46
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Laderman's algorithm for 3x3
    # 23 terms
    # Indices for A (9), B (9), C (9)
    
    # Term definitions: (U_indices, U_vals, V_indices, V_vals, W_indices, W_vals)
    # We will define the 23 terms for the first block (B cols 0-2, C cols 0-2)
    # Then duplicate for second block (B cols 3-5, C cols 3-5)
    
    terms = []
    
    # 1. A00 B00
    terms.append(([0], [1], [0], [1], [0], [1]))
    # 2. A01 B10
    terms.append(([1], [1], [3], [1], [0], [1]))
    # 3. A10 B01
    terms.append(([3], [1], [1], [1], [4], [1]))
    # 4. A02 B20
    terms.append(([2], [1], [6], [1], [0], [1]))
    # 5. A20 B02
    terms.append(([6], [1], [2], [1], [7], [1]))
    # 6. A11 B11
    terms.append(([4], [1], [4], [1], [4], [1]))
    # 7. A12 B21
    terms.append(([5], [1], [7], [1], [4], [1]))
    # 8. A21 B12
    terms.append(([7], [1], [5], [1], [7], [1]))
    # 9. A22 B22
    terms.append(([8], [1], [8], [1], [8], [1]))
    
    # 10. (A00+A11)(B11+B00)
    terms.append(([0,4], [1,1], [4,0], [1,1], [0,4], [1,1])) # W coeffs might be different?
    # Actually, Laderman W coefficients are specific.
    # Let's use the known correct Laderman coefficients.
    # I will assume the simple sum works for the diagonal and adjust off-diagonal?
    # No, I must use the exact algorithm.
    
    # To be safe, I will use the Schoolbook algorithm (Rank 54) if I am not confident.
    # But Rank 54 is the current score.
    # I will try to implement Rank 46 correctly.
    
    # Correct Laderman W coefficients:
    # C00: M1 + M2 + M4 + M10 - M12 - M16 - M22
    # C01: M1 + M3 + M5 + M13 - M15 - M19 - M22
    # ...
    
    # This is too complex to hardcode blindly.
    # I will fall back to Schoolbook Rank 54.
    
    # Schoolbook
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
                
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
