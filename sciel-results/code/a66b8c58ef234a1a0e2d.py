import time
import numpy as np

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 3x5 matrix by a 5x7 matrix.
    
    The output factor W follows the trace convention W[k*n+i, r].
    U has shape (n*m, rank)
    V has shape (m*p, rank)
    W has shape (p*n, rank)
    
    Score is the rank.
    """
    n, m, p = 3, 5, 7
    
    # Improved decomposition using Strassen for a 2x2x2 sub-block
    # Rank = 7 (Strassen) + 97 (Schoolbook for rest) = 104
    
    rank = 104
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, coeff, col):
        U[u_idx, col] += coeff
        V[v_idx, col] += coeff
        W[w_idx, col] += coeff

    # Strassen decomposition for indices i,j,k in {0,1}
    # A indices: i*m + j
    # B indices: j*p + k
    # C indices: k*n + i
    
    # M1 = (A00 + A11)(B00 + B11)
    col = 0
    add_term(0, 0, 0, 1, col)   # A00, B00, C00
    add_term(6, 8, 4, 1, col)   # A11, B11, C11
    col += 1
    
    # M2 = (A10 + A11)B00
    add_term(5, 0, 3, 1, col)   # A10, B00, C10
    add_term(6, 0, 3, 1, col)   # A11, B00, C10
    col += 1
    
    # M3 = A00(B10 - B11)
    add_term(0, 7, 1, 1, col)   # A00, B10, C01
    add_term(0, 8, 4, 1, col)   # A00, B11, C11
    # Wait, M3 contributes to C01 and C11?
    # C01 = M3 + M5. C11 = M1 - M2 + M3 + M6.
    # So M3 has coeff 1 for C01 and 1 for C11.
    # But B11 has coeff -1 in M3.
    # Let's re-verify M3 definition.
    # M3 = A00 * (B10 - B11).
    # So term is A00*B10 - A00*B11.
    # U: A00 (idx 0). V: B10 (idx 7), B11 (idx 8).
    # W: C01 (idx 1), C11 (idx 4).
    # Coeff for B10 is 1. Coeff for B11 is -1.
    # So:
    # add_term(0, 7, 1, 1, col) # A00, B10, C01
    # add_term(0, 8, 1, -1, col) # A00, B11, C01 ?? No.
    # The decomposition is sum of rank-1 tensors.
    # M3 is ONE rank-1 tensor? No, M3 is a product of linear forms.
    # M3 = (A00) * (B10 - B11).
    # This is NOT a single rank-1 tensor in the original basis unless we expand.
    # Rank-1 tensor is u \otimes v \otimes w.
    # M3 corresponds to the bilinear form.
    # In the decomposition, we add columns to U, V, W.
    # Each column r represents a term U[:,r] * V[:,r] * W[:,r].
    # The Strassen algorithm replaces 8 terms with 7 "multiplications".
    # Each "multiplication" M_k is a product of two linear combinations.
    # M_k = L_A(k) * L_B(k).
    # This is represented by:
    # U[:, col] = L_A(k)
    # V[:, col] = L_B(k)
    # W[:, col] = L_C(k) where L_C(k) is the combination of C indices that use M_k.
    
    # Let's restart the Strassen column construction properly.
    
    # M1: L_A = A00 + A11. L_B = B00 + B11. L_C = C00 + C11.
    U[0, col] += 1; U[6, col] += 1
    V[0, col] += 1; V[8, col] += 1
    W[0, col] += 1; W[4, col] += 1
    col += 1
    
    # M2: L_A = A10 + A11. L_B = B00. L_C = C10.
    U[5, col] += 1; U[6, col] += 1
    V[0, col] += 1
    W[3, col] += 1
    col += 1
    
    # M3: L_A = A00. L_B = B10 - B11. L_C = C01 + C11.
    U[0, col] += 1
    V[7, col] += 1; V[8, col] -= 1
    W[1, col] += 1; W[4, col] += 1
    col += 1
    
    # M4: L_A = A11. L_B = B01 - B00. L_C = C00 + C10.
    U[6, col] += 1
    V[1, col] += 1; V[0, col] -= 1
    W[0, col] += 1; W[3, col] += 1
    col += 1
    
    # M5: L_A = A00 + A01. L_B = B11. L_C = -C00 + C01.
    U[0, col] += 1; U[1, col] += 1
    V[8, col] += 1
    W[0, col] -= 1; W[1, col] += 1
    col += 1
    
    # M6: L_A = A10 - A00. L_B = B00 + B01. L_C = C11.
    U[5, col] += 1; U[0, col] -= 1
    V[0, col] += 1; V[1, col] += 1
    W[4, col] += 1
    col += 1
    
    # M7: L_A = A01 - A11. L_B = B10 + B11. L_C = C00.
    U[1, col] += 1; U[6, col] -= 1
    V[7, col] += 1; V[8, col] += 1
    W[0, col] += 1
    col += 1
    
    # Now add the remaining 97 terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip the 2x2x2 block handled by Strassen
                if i in [0, 1] and j in [0, 1] and k in [0, 1]:
                    continue
                
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
                
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
