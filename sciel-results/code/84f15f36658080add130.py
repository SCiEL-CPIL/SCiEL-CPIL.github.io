import numpy as np

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use Strassen for pairs of columns (rank 7 per pair)
    # and trivial for the last column (rank 4)
    # 13 columns = 6 pairs + 1 single
    num_pairs = p // 2
    remainder = p % 2
    
    rank = num_pairs * 7 + remainder * 4
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for A (2x2)
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    A_coeffs = [
        [1, 0, 0, 1], # M1: a00 + a11
        [0, 0, 1, 1], # M2: a10 + a11
        [1, 0, 0, 0], # M3: a00
        [0, 0, 0, 1], # M4: a11
        [1, 1, 0, 0], # M5: a00 + a01
        [-1, 0, 1, 0],# M6: a10 - a00
        [0, 1, 0, -1] # M7: a01 - a11
    ]
    
    # Strassen coefficients for B (2x2 block)
    # Local indices: 0:(0,k), 1:(0,k+1), 2:(1,k), 3:(1,k+1)
    # We need to map these to V indices.
    # V index for (j, k) is j*p + k.
    # For pair starting at k:
    # (0, k) -> 0*p + k
    # (0, k+1) -> 0*p + k+1
    # (1, k) -> 1*p + k
    # (1, k+1) -> 1*p + k+1
    
    # Strassen B coeffs for each M_s
    # M1: b00 + b11
    # M2: b00
    # M3: b01 - b11
    # M4: b10 - b00
    # M5: b11
    # M6: b00 + b01
    # M7: b10 + b11
    
    # We will store B coeffs as list of (v_index_offset, coeff)
    # But it's easier to just iterate and add to V.
    
    # Strassen C coeffs
    # C00: M1 + M4 - M5 + M7
    # C01: M3 + M5
    # C10: M2 + M4
    # C11: M1 - M2 + M3 + M6
    
    # C indices in W:
    # (i, k) -> k*n + i
    # For pair k, k+1:
    # C(0, k) -> 2k + 0
    # C(0, k+1) -> 2(k+1) + 0
    # C(1, k) -> 2k + 1
    # C(1, k+1) -> 2(k+1) + 1
    
    C_coeffs = [
        [1, 0, 0, 1], # M1 contributes to C00, C11
        [0, 0, 1, -1],# M2 contributes to C10, C11
        [0, 1, 0, 1], # M3 contributes to C01, C11
        [1, 0, 1, 0], # M4 contributes to C00, C10
        [-1, 1, 0, 0],# M5 contributes to C00, C01
        [0, 0, 0, 1], # M6 contributes to C11
        [1, 0, 0, 0]  # M7 contributes to C00
    ]
    # Note: C_coeffs[s] is [coeff for C00, coeff for C01, coeff for C10, coeff for C11]
    
    for k in range(num_pairs):
        for s in range(7):
            # Fill U
            for j in range(4):
                U[j][col] = A_coeffs[s][j]
                
            # Fill V
            # B indices for this pair:
            # v_00 = 0*p + k
            # v_01 = 0*p + k+1
            # v_10 = 1*p + k
            # v_11 = 1*p + k+1
            
            b_coeffs = [0, 0, 0, 0] # b00, b01, b10, b11
            
            if s == 0: # M1
                b_coeffs = [1, 0, 0, 1]
            elif s == 1: # M2
                b_coeffs = [1, 0, 0, 0]
            elif s == 2: # M3
                b_coeffs = [0, 1, 0, -1]
            elif s == 3: # M4
                b_coeffs = [-1, 0, 1, 0]
            elif s == 4: # M5
                b_coeffs = [0, 0, 0, 1]
            elif s == 5: # M6
                b_coeffs = [1, 1, 0, 0]
            elif s == 6: # M7
                b_coeffs = [0, 0, 1, 1]
                
            V[0*p + k][col] = b_coeffs[0]
            V[0*p + (k+1)][col] = b_coeffs[1]
            V[1*p + k][col] = b_coeffs[2]
            V[1*p + (k+1)][col] = b_coeffs[3]
            
            # Fill W
            # C indices:
            # w_00 = 2*k + 0
            # w_01 = 2*(k+1) + 0
            # w_10 = 2*k + 1
            # w_11 = 2*(k+1) + 1
            
            w_coeffs = C_coeffs[s]
            
            W[2*k + 0][col] = w_coeffs[0]
            W[2*(k+1) + 0][col] = w_coeffs[1]
            W[2*k + 1][col] = w_coeffs[2]
            W[2*(k+1) + 1][col] = w_coeffs[3]
            
            col += 1
            
    # Handle remainder (single column)
    if remainder > 0:
        k = num_pairs * 2
        # Trivial decomposition for 2x2x1
        # C0 = A00 B0 + A01 B1
        # C1 = A10 B0 + A11 B1
        
        # Term 1: A00 B0 -> C0
        U[0][col] = 1
        V[0*p + k][col] = 1
        W[2*k + 0][col] = 1
        col += 1
        
        # Term 2: A01 B1 -> C0
        U[1][col] = 1
        V[1*p + k][col] = 1
        W[2*k + 0][col] = 1
        col += 1
        
        # Term 3: A10 B0 -> C1
        U[2][col] = 1
        V[0*p + k][col] = 1
        W[2*k + 1][col] = 1
        col += 1
        
        # Term 4: A11 B1 -> C1
        U[3][col] = 1
        V[1*p + k][col] = 1
        W[2*k + 1][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
