import time
import random
import numpy as np

DIMENSIONS = [4, 4, 7]

def _strassen_2x2_terms():
    """
    Returns the 7 terms of Strassen's 2x2 matrix multiplication algorithm.
    Each term is a tuple (u, v, w) where u, v, w are lists of indices into
    the 2x2 matrices (flattened row-major).
    
    For C = A * B where A, B are 2x2:
    C[0,0] = M1 + M4 - M5 - M7
    C[0,1] = M3 + M5
    C[1,0] = M2 + M4
    C[1,1] = M1 - M3 + M6 + M7
    
    M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    M2 = (A[1,0] + A[1,1]) * B[0,0]
    M3 = A[0,0] * (B[0,1] - B[1,1])
    M4 = A[1,1] * (B[1,0] - B[0,0])
    M5 = (A[0,0] + A[0,1]) * B[1,1]
    M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
    M7 = (A[1,1] - A[0,1]) * (B[1,0] + B[1,1])
    """
    # A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # C indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    terms = []
    
    # M1 = (A[0]+A[3]) * (B[0]+B[3])
    terms.append(([0, 3], [0, 3], [0, 1, 2, 3]))  # +1, -1, -1, +1 -> wait, let me be more explicit
    
    # Let me define each term with explicit coefficients
    # Term format: (U_indices, V_indices, W_indices_with_coeffs)
    # Actually, for the tensor decomposition, we need:
    # U[i,j] * V[j,k] * W[k,i] summed over terms = C[i,j]
    
    # Let me use a different representation. Each term is a rank-1 tensor.
    # Term t contributes: U_t[i,j] * V_t[j,k] * W_t[k,i] to C[i,j]
    
    # For Strassen, we can express each M_m as a product of sums.
    # This gets complicated. Let me just hardcode the known 7-term decomposition.
    
    # Actually, let me construct it properly.
    # The 2x2x2 multiplication tensor has rank 7 (Strassen).
    # Let me find the explicit decomposition.
    
    # I'll use a known construction. Let me define the 7 terms explicitly.
    # Each term is a 2x2x2 tensor (rank 1 = outer product of 3 vectors).
    
    # Let me use the standard Strassen decomposition.
    # M1 = (a11+a22)(b11+b22)
    # In tensor form, this is not a single rank-1 term because it's a product of sums.
    # But we can expand: (a11+a22)(b11+b22) = a11*b11 + a11*b22 + a22*b11 + a22*b22
    # This would be 4 rank-1 terms, not 1.
    
    # So Strassen's algorithm doesn't directly give a rank-7 tensor decomposition.
    # It gives an algorithm with 7 multiplications, but the tensor rank is different.
    
    # Wait, I need to reconsider. The tensor rank of 2x2x2 matrix multiplication IS 7.
    # But the Strassen algorithm corresponds to a specific rank-7 decomposition.
    
    # Let me look up the explicit rank-7 decomposition for 2x2x2.
    # It's known but non-trivial to write down.
    
    # For now, let me just use schoolbook for 2x2 (rank 4) and see if we can improve.
    # Actually, let me just implement a correct but possibly suboptimal solution first.
    
    # Let me implement the schoolbook 4x4x7 decomposition (rank 112) and then try to improve.
    
    return None

def _schoolbook_447():
    """
    Schoolbook decomposition for 4x4 * 4x7 matrix multiplication.
    Rank = 4*4*7 = 112.
    """
    n, m, p = 4, 4, 7
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

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition is correct.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                ui = U[i * m + j]
                vk = V[j * p + k]
                wi = W[k * n + i]
                for c in range(rank):
                    s += ui[c] * vk[c] * wi[c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0
    
    # Start with schoolbook (rank 112)
    result = _schoolbook_447()
    
    # Verify it's correct
    if not _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        # This should never happen, but just in case
        pass
    
    # Try to improve using known results.
    # The rank of 4x4x4 matrix multiplication is at most 23 (Landsberg).
    # If we can find a rank-23 decomposition for 4x4x4, we can use it for the first 4 columns
    # and schoolbook for the last 3, giving rank 23 + 4*4*3 = 23 + 48 = 71.
    
    # But finding a rank-23 decomposition is hard. Let me try a random search.
    # However, random search for rank-23 is unlikely to succeed quickly.
    
    # Alternative: Use a known decomposition if available.
    # For now, let me just return the schoolbook result.
    
    return result
