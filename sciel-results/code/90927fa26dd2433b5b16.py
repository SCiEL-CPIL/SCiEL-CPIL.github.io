import time
import sys

DIMENSIONS = [2, 4, 7]

def construct():
    """
    Constructs a rank-52 tensor decomposition for 2x4 by 4x7 matrix multiplication.
    Uses Strassen's algorithm for 2x2x2 blocks and schoolbook for 2x2x3 blocks.
    """
    n, m, p = 2, 4, 7
    rank = 52
    
    # U: (n*m, rank) = (8, 52)
    # V: (m*p, rank) = (28, 52)
    # W: (p*n, rank) = (14, 52)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(A_indices, B_indices, C_indices):
        nonlocal col
        # A_indices: [(i, j), ...] for 2x2
        # B_indices: [(j, k), ...] for 2x2
        # C_indices: [(i, k), ...] for 2x2
        
        # Map local indices 0,1 to global A, B, C indices
        # A: 0=(i0,j0), 1=(i0,j1), 2=(i1,j0), 3=(i1,j1)
        # B: 0=(j0,k0), 1=(j0,k1), 2=(j1,k0), 3=(j1,k1)
        # C: 0=(i0,k0), 1=(i0,k1), 2=(i1,k0), 3=(i1,k1)
        
        # Strassen coefficients for M1..M7
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b01-b11)
        # M4 = a11(b10-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        
        strassen_defs = [
            # (A_coeffs, B_coeffs, C_adds)
            # A_coeffs: dict {local_idx: coeff}
            # B_coeffs: dict {local_idx: coeff}
            # C_adds: dict {local_idx: coeff}
            
            # M1
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
            # M2
            ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}), # Wait, z10 = M2+M4, z11 = M1-M2+M3+M6
            # Let's re-verify Strassen formula mapping to C
            # z00 = M1 + M4 - M5 + M7
            # z01 = M3 + M5
            # z10 = M2 + M4
            # z11 = M1 - M2 + M3 + M6
            
            # So for each Mk, we need to know which C elements it contributes to.
        ]
        
        # Re-defining properly:
        # Mk = (sum ai * A_i) * (sum bj * B_j)
        # Then C_k += Mk * (sum ck * C_k) ? No.
        # The tensor term is sum_mk u_mk v_mk w_mk
        # If we define Mk as a scalar, then:
        # U[A_idx, col] = coeff_A
        # V[B_idx, col] = coeff_B
        # W[C_idx, col] = coeff_C (which is 1 usually, or part of the combination)
        
        # Actually, the standard way is:
        # For each multiplication Mk:
        # 1. Determine linear form L_A(Mk) on A
        # 2. Determine linear form L_B(Mk) on B
        # 3. Determine how Mk contributes to C (linear form L_C(Mk))
        
        # Mk = L_A * L_B
        # C += Mk * L_C ? No.
        # The product is C = sum_mk Mk * E_mk where E_mk is a matrix of 0/1?
        # No, Strassen works by:
        # C_00 = M1 + M4 - M5 + M7
        # This means:
        # W[C_00, col_M1] += 1
        # W[C_00, col_M4] += 1
        # W[C_00, col_M5] += -1
        # W[C_00, col_M7] += 1
        
        # So we process each Mk separately.
        
        def process_mk(A_coeffs, B_coeffs, C_coeffs):
            nonlocal col
            for local_a, coeff in A_coeffs.items():
                global_a = A_indices[local_a]
                U[global_a[0] * m + global_a[1]][col] += coeff
            for local_b, coeff in B_coeffs.items():
                global_b = B_indices[local_b]
                V[global_b[0] * p + global_b[1]][col] += coeff
            for local_c, coeff in C_coeffs.items():
                global_c = C_indices[local_c]
                W[global_c[1] * n + global_c[0]][col] += coeff
            col += 1

        # M1 = (a00+a11)(b00+b11)
        # C00 += M1, C11 += M1
        process_mk({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1})
        
        # M2 = (a10+a11)b00
        # C10 += M2, C11 -= M2
        process_mk({2:1, 3:1}, {0:1}, {2:1, 3:-1})
        
        # M3 = a00(b01-b11)
        # C01 += M3, C11 += M3
        process_mk({0:1}, {1:1, 3:-1}, {1:1, 3:1})
        
        # M4 = a11(b10-b00)
        # C00 += M4, C10 += M4
        process_mk({3:1}, {2:1, 0:-1}, {0:1, 2:1})
        
        # M5 = (a00+a01)b11
        # C00 -= M5, C01 += M5
        process_mk({0:1, 1:1}, {3:1}, {0:-1, 1:1})
        
        # M6 = (a10-a00)(b00+b01)
        # C11 += M6
        process_mk({2:1, 0:-1}, {0:1, 1:1}, {3:1})
        
        # M7 = (a01-a11)(b10+b11)
        # C00 += M7
        process_mk({1:1, 3:-1}, {2:1, 3:1}, {0:1})

    def add_schoolbook_2x2x3(A_indices, B_indices, C_indices):
        nonlocal col
        # A: 2x2. Indices 0,1,2,3
        # B: 2x3. Indices 0,1,2 (row 0), 3,4,5 (row 1)
        # C: 2x3. Indices 0,1,2 (row 0), 3,4,5 (row 1)
        
        # C00 = A00*B00 + A01*B10
        # C01 = A00*B01 + A01*B11
        # C02 = A00*B02 + A01*B12
        # C10 = A10*B00 + A11*B10
        # C11 = A10*B01 + A11*B11
        # C12 = A10*B02 + A11*B12
        
        # 12 multiplications
        mults = [
            # (A_idx, B_idx, C_idx)
            (0, 0, 0), (0, 1, 1), (0, 2, 2),
            (1, 3, 0), (1, 4, 1), (1, 5, 2),
            (2, 0, 3), (2, 1, 4), (2, 2, 5),
            (3, 3, 3), (3, 4, 4), (3, 5, 5)
        ]
        
        for a_idx, b_idx, c_idx in mults:
            global_a = A_indices[a_idx]
            global_b = B_indices[b_idx]
            global_c = C_indices[c_idx]
            
            U[global_a[0] * m + global_a[1]][col] = 1
            V[global_b[0] * p + global_b[1]][col] = 1
            W[global_c[1] * n + global_c[0]][col] = 1
            col += 1

    # --- Block 1: A_L B_L1 (2x2 * 2x2) ---
    # A_L: i in 0,1; j in 0,1
    A_L = [(0,0), (0,1), (1,0), (1,1)]
    # B_L1: j in 0,1; k in 0,1
    B_L1 = [(0,0), (0,1), (1,0), (1,1)]
    # C_L1: i in 0,1; k in 0,1
    C_L1 = [(0,0), (0,1), (1,0), (1,1)]
    add_strassen_block(A_L, B_L1, C_L1)

    # --- Block 2: A_L B_L2 (2x2 * 2x2) ---
    # B_L2: j in 0,1; k in 2,3
    B_L2 = [(0,2), (0,3), (1,2), (1,3)]
    # C_L2: i in 0,1; k in 2,3
    C_L2 = [(0,2), (0,3), (1,2), (1,3)]
    add_strassen_block(A_L, B_L2, C_L2)

    # --- Block 3: A_L B_L3 (2x2 * 2x3) ---
    # B_L3: j in 0,1; k in 4,5,6
    B_L3 = [(0,4), (0,5), (0,6), (1,4), (1,5), (1,6)]
    # C_L3: i in 0,1; k in 4,5,6
    C_L3 = [(0,4), (0,5), (0,6), (1,4), (1,5), (1,6)]
    add_schoolbook_2x2x3(A_L, B_L3, C_L3)

    # --- Block 4: A_R B_R1 (2x2 * 2x2) ---
    # A_R: i in 0,1; j in 2,3
    A_R = [(0,2), (0,3), (1,2), (1,3)]
    # B_R1: j in 2,3; k in 0,1
    B_R1 = [(2,0), (2,1), (3,0), (3,1)]
    # C_R1: i in 0,1; k in 0,1
    C_R1 = [(0,0), (0,1), (1,0), (1,1)]
    add_strassen_block(A_R, B_R1, C_R1)

    # --- Block 5: A_R B_R2 (2x2 * 2x2) ---
    # B_R2: j in 2,3; k in 2,3
    B_R2 = [(2,2), (2,3), (3,2), (3,3)]
    # C_R2: i in 0,1; k in 2,3
    C_R2 = [(0,2), (0,3), (1,2), (1,3)]
    add_strassen_block(A_R, B_R2, C_R2)

    # --- Block 6: A_R B_R3 (2x2 * 2x3) ---
    # B_R3: j in 2,3; k in 4,5,6
    B_R3 = [(2,4), (2,5), (2,6), (3,4), (3,5), (3,6)]
    # C_R3: i in 0,1; k in 4,5,6
    C_R3 = [(0,4), (0,5), (0,6), (1,4), (1,5), (1,6)]
    add_schoolbook_2x2x3(A_R, B_R3, C_R3)

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
