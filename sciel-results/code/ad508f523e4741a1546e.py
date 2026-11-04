import time
import sys

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # We use a decomposition of rank 36
    # Split p=5 into 2, 2, 1
    # For each part, split m=4 into 2, 2
    # For 2x2x2 parts, use Strassen (rank 7)
    # For 2x2x1 parts, use Schoolbook (rank 4)
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term to the decomposition
    def add_term(u_idx, v_idx, w_idx, coeff_u, coeff_v, coeff_w):
        nonlocal col
        U[u_idx][col] += coeff_u
        V[v_idx][col] += coeff_v
        W[w_idx][col] += coeff_w
        col += 1

    # Helper for Strassen 2x2x2
    # A is 2x2 local (rows i0,i1; cols j0,j1)
    # B is 2x2 local (rows j0,j1; cols k0,k1)
    # C is 2x2 local (rows i0,i1; cols k0,k1)
    def add_strassen(i_base, j_base, k_base):
        nonlocal col
        # Local indices 0,1
        # Global A indices: (i_base+di)*m + (j_base+dj)
        # Global B indices: (j_base+dj)*p + (k_base+dk)
        # Global W indices: (k_base+dk)*n + (i_base+di)
        
        # M1 = (A00+A11)(B00+B11)
        # U: A00, A11
        # V: B00, B11
        # W: C00, C11 (with coeffs)
        # We need to map this to the single-term add_term?
        # No, Strassen terms are sums of products.
        # Each Strassen multiplication M_l is a product of two linear combinations.
        # So each M_l adds 1 to the rank.
        # We need to add 7 terms for the block.
        
        # Term 1: M1
        # U parts: A00 (1), A11 (1)
        # V parts: B00 (1), B11 (1)
        # W parts: C00 (1), C11 (1)
        # Wait, W for M1 contributes to C00 and C11.
        # C00 = M1 + ...
        # C11 = M1 - ...
        # So W[C00] += 1, W[C11] += 1.
        
        # Indices:
        a00 = (i_base + 0) * m + (j_base + 0)
        a11 = (i_base + 1) * m + (j_base + 1)
        b00 = (j_base + 0) * p + (k_base + 0)
        b11 = (j_base + 1) * p + (k_base + 1)
        c00 = (k_base + 0) * n + (i_base + 0)
        c11 = (k_base + 1) * n + (i_base + 1)
        
        U[a00][col] = 1
        U[a11][col] = 1
        V[b00][col] = 1
        V[b11][col] = 1
        W[c00][col] = 1
        W[c11][col] = 1
        col += 1
        
        # Term 2: M2 = (A10+A11)B00
        a10 = (i_base + 1) * m + (j_base + 0)
        # U: A10(1), A11(1)
        # V: B00(1)
        # W: C10(1), C11(-1)
        U[a10][col] = 1
        U[a11][col] = 1
        V[b00][col] = 1
        W[(k_base + 0) * n + (i_base + 1)][col] = 1 # C10
        W[(k_base + 1) * n + (i_base + 1)][col] = -1 # C11
        col += 1
        
        # Term 3: M3 = A00(B10-B11)
        b10 = (j_base + 1) * p + (k_base + 0)
        # U: A00(1)
        # V: B10(1), B11(-1)
        # W: C01(1), C11(1)
        U[a00][col] = 1
        V[b10][col] = 1
        V[b11][col] = -1
        W[(k_base + 1) * n + (i_base + 0)][col] = 1 # C01
        W[(k_base + 1) * n + (i_base + 1)][col] = 1 # C11
        col += 1
        
        # Term 4: M4 = A11(B01-B00)
        b01 = (j_base + 0) * p + (k_base + 1)
        # U: A11(1)
        # V: B01(1), B00(-1)
        # W: C00(1), C10(1)
        U[a11][col] = 1
        V[b01][col] = 1
        V[b00][col] = -1
        W[(k_base + 0) * n + (i_base + 0)][col] = 1 # C00
        W[(k_base + 0) * n + (i_base + 1)][col] = 1 # C10
        col += 1
        
        # Term 5: M5 = (A00+A01)B11
        a01 = (i_base + 0) * m + (j_base + 1)
        # U: A00(1), A01(1)
        # V: B11(1)
        # W: C00(-1), C01(1)
        U[a00][col] = 1
        U[a01][col] = 1
        V[b11][col] = 1
        W[(k_base + 0) * n + (i_base + 0)][col] = -1 # C00
        W[(k_base + 1) * n + (i_base + 0)][col] = 1 # C01
        col += 1
        
        # Term 6: M6 = (A10-A00)(B00+B10)
        # U: A10(1), A00(-1)
        # V: B00(1), B10(1)
        # W: C11(1)
        U[a10][col] = 1
        U[a00][col] = -1
        V[b00][col] = 1
        V[b10][col] = 1
        W[(k_base + 1) * n + (i_base + 1)][col] = 1 # C11
        col += 1
        
        # Term 7: M7 = (A01-A11)(B11+B01)
        # U: A01(1), A11(-1)
        # V: B11(1), B01(1)
        # W: C00(1)
        U[a01][col] = 1
        U[a11][col] = -1
        V[b11][col] = 1
        V[b01][col] = 1
        W[(k_base + 0) * n + (i_base + 0)][col] = 1 # C00
        col += 1

    # Helper for Schoolbook 2x2x1
    def add_schoolbook_221(i_base, j_base, k_base):
        nonlocal col
        # A is 2x2, B is 2x1, C is 2x1
        # C0 = A00 B0 + A01 B1
        # C1 = A10 B0 + A11 B1
        
        a00 = (i_base + 0) * m + (j_base + 0)
        a01 = (i_base + 0) * m + (j_base + 1)
        a10 = (i_base + 1) * m + (j_base + 0)
        a11 = (i_base + 1) * m + (j_base + 1)
        
        b0 = (j_base + 0) * p + k_base
        b1 = (j_base + 1) * p + k_base
        
        c0 = k_base * n + (i_base + 0)
        c1 = k_base * n + (i_base + 1)
        
        # Term 1: A00 B0 -> C0
        U[a00][col] = 1
        V[b0][col] = 1
        W[c0][col] = 1
        col += 1
        
        # Term 2: A01 B1 -> C0
        U[a01][col] = 1
        V[b1][col] = 1
        W[c0][col] = 1
        col += 1
        
        # Term 3: A10 B0 -> C1
        U[a10][col] = 1
        V[b0][col] = 1
        W[c1][col] = 1
        col += 1
        
        # Term 4: A11 B1 -> C1
        U[a11][col] = 1
        V[b1][col] = 1
        W[c1][col] = 1
        col += 1

    # Block 1: C[:, 0:2] = A * B[:, 0:2]
    # Split m=4 into 2+2
    # A[:, 0:2] * B[0:2, 0:2] + A[:, 2:4] * B[2:4, 0:2]
    add_strassen(0, 0, 0) # A00 block * B00 block
    add_strassen(0, 2, 0) # A01 block * B10 block
    
    # Block 2: C[:, 2:4] = A * B[:, 2:4]
    # A[:, 0:2] * B[0:2, 2:4] + A[:, 2:4] * B[2:4, 2:4]
    add_strassen(0, 0, 2)
    add_strassen(0, 2, 2)
    
    # Block 3: C[:, 4:5] = A * B[:, 4:5]
    # A[:, 0:2] * B[0:2, 4:5] + A[:, 2:4] * B[2:4, 4:5]
    add_schoolbook_221(0, 0, 4)
    add_schoolbook_221(0, 2, 4)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
