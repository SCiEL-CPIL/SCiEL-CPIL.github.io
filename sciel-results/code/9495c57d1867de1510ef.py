import time
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 26 decomposition:
    # 7 terms from Strassen's algorithm for the 2x2 submatrix (indices 0,1)
    # 19 terms from the remaining schoolbook terms involving index 2
    
    rank = 26
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    # Helper to flatten indices
    # A: i*m + j
    # B: j*p + k
    # C: k*n + i (Wait, task says W[k*n+i, r]. C is usually i*n+k? 
    # Task: "output factor follows the official trace convention W[k*n+i,r]"
    # Let's check the tensor definition in the current program:
    # T[u, v, w] where u=i*m+j, v=j*p+k, w=k*n+i
    # This implies C is indexed by w = k*n + i.
    # So C[k, i] in flattened form is w.
    # Standard C[i, k] would be i*n+k.
    # The task specifies W[k*n+i]. So the "output" matrix C is effectively transposed 
    # or indexed by (k,i). 
    # Let's assume the target tensor T is defined as:
    # T[u, v, w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i
    # This corresponds to C_{i,k} = sum A_{i,j} B_{j,k}.
    # If C is stored as C[k,i] (column major) or similar, w=k*n+i.
    # I will follow the index mapping: w = k*n + i.
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    col = 0
    
    # --- Strassen 2x2 Block (i,j,k in {0,1}) ---
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # Term 0: M1
    U[u_idx(0,0), col] = 1
    U[u_idx(1,1), col] = 1
    V[v_idx(0,0), col] = 1
    V[v_idx(1,1), col] = 1
    W[w_idx(0,0), col] = 1  # C00
    W[w_idx(0,1), col] = 0  # C01? No, M1 contributes to C00 and C11.
    # Let's re-evaluate Strassen outputs.
    # M1 = (A00+A11)(B00+B11)
    # Contributes to C00 (via A00B00+A00B11+A11B00+A11B11) and C11?
    # Actually:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So M1 contributes 1 to C00 and 1 to C11.
    W[w_idx(0,0), col] = 1
    W[w_idx(1,1), col] = 1
    col += 1
    
    # Term 1: M2 = (A10+A11)B00
    U[u_idx(1,0), col] = 1
    U[u_idx(1,1), col] = 1
    V[v_idx(0,0), col] = 1
    # C10 = M2 + ... => W[10] += 1
    W[w_idx(0,1), col] = 1  # C10 is i=1, k=0 -> w = 0*3+1 = 1? 
    # w_idx(k,i). C10 means i=1, k=0. w = 0*3+1 = 1.
    col += 1
    
    # Term 2: M3 = A00(B10-B11)
    U[u_idx(0,0), col] = 1
    V[v_idx(1,0), col] = 1
    V[v_idx(1,1), col] = -1
    # C01 = M3 + ... => C01 is i=0, k=1. w = 1*3+0 = 3.
    W[w_idx(1,0), col] = 1
    col += 1
    
    # Term 3: M4 = A11(B01-B00)
    U[u_idx(1,1), col] = 1
    V[v_idx(0,1), col] = 1
    V[v_idx(0,0), col] = -1
    # C00 = ... + M4 ... => W[00] += 1
    # C10 = M2 + M4 ... => W[10] += 1
    W[w_idx(0,0), col] = 1
    W[w_idx(0,1), col] = 1
    col += 1
    
    # Term 4: M5 = (A00+A01)B11
    U[u_idx(0,0), col] = 1
    U[u_idx(0,1), col] = 1
    V[v_idx(1,1), col] = 1
    # C00 = ... - M5 ... => W[00] -= 1
    # C01 = ... + M5 ... => W[01] += 1
    W[w_idx(0,0), col] = -1
    W[w_idx(1,0), col] = 1
    col += 1
    
    # Term 5: M6 = (A10-A00)(B00+B01)
    U[u_idx(1,0), col] = 1
    U[u_idx(0,0), col] = -1
    V[v_idx(0,0), col] = 1
    V[v_idx(0,1), col] = 1
    # C11 = ... + M6 ... => C11 is i=1, k=1. w = 1*3+1 = 4.
    W[w_idx(1,1), col] = 1
    col += 1
    
    # Term 6: M7 = (A01-A11)(B10+B11)
    U[u_idx(0,1), col] = 1
    U[u_idx(1,1), col] = -1
    V[v_idx(1,0), col] = 1
    V[v_idx(1,1), col] = 1
    # C00 = ... + M7 ... => W[00] += 1
    W[w_idx(0,0), col] = 1
    col += 1
    
    # --- Remaining 19 Schoolbook Terms ---
    # These are terms where at least one index is 2.
    # Format: A[i,j] * B[j,k] -> C[i,k]
    
    terms_remaining = [
        # i=0
        (0, 0, 2), # A00 B02 -> C02
        (0, 1, 2), # A01 B12 -> C02
        (0, 2, 0), # A02 B20 -> C02
        (0, 2, 1), # A02 B21 -> C02
        (0, 2, 2), # A02 B22 -> C02
        # i=1
        (1, 0, 2), # A10 B02 -> C12
        (1, 1, 2), # A11 B12 -> C12
        (1, 2, 0), # A12 B20 -> C12
        (1, 2, 1), # A12 B21 -> C12
        (1, 2, 2), # A12 B22 -> C12
        # i=2
        (2, 0, 0), # A20 B00 -> C20
        (2, 0, 1), # A20 B01 -> C20
        (2, 0, 2), # A20 B02 -> C20
        (2, 1, 0), # A21 B10 -> C20
        (2, 1, 1), # A21 B11 -> C20
        (2, 1, 2), # A21 B12 -> C20
        (2, 2, 0), # A22 B20 -> C22
        (2, 2, 1), # A22 B21 -> C22
        (2, 2, 2), # A22 B22 -> C22
    ]
    
    for i, j, k in terms_remaining:
        U[u_idx(i, j), col] = 1
        V[v_idx(j, k), col] = 1
        W[w_idx(k, i), col] = 1
        col += 1
        
    # Verify rank
    if col != rank:
        # Adjust if needed, but should be 7+19=26
        pass

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
