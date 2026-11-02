import numpy as np

DIMENSIONS = [2, 2, 12]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=12
    # We decompose this into 6 blocks of 2x2 matrix multiplication.
    # Rank of 2x2x2 multiplication is 7 (Strassen).
    # Total rank = 6 * 7 = 42.
    
    rank = 42
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen's 7 linear forms for 2x2 matrix multiplication
    # A is 2x2, B is 2x2, C is 2x2
    # A indices: 0:00, 1:01, 2:10, 3:11
    # B indices: 0:00, 1:01, 2:10, 3:11
    # C indices: 0:00, 1:01, 2:10, 3:11
    
    # U_forms: 7x4 matrix. U_forms[k, i] is coeff of A[i] in form k.
    # V_forms: 7x4 matrix. V_forms[k, j] is coeff of B[j] in form k.
    # W_forms: 7x4 matrix. W_forms[k, l] is coeff of form k in C[l].
    
    # Standard Strassen decomposition (one of the many equivalent ones)
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U_forms = [
        [1, 0, 0, 1], # M1: A00 + A11
        [0, 0, 1, 1], # M2: A10 + A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00 + A01
        [-1, 0, 1, 0], # M6: A10 - A00
        [0, 1, 0, -1]  # M7: A01 - A11
    ]
    
    V_forms = [
        [1, 0, 0, 1], # M1: B00 + B11
        [1, 0, 0, 0], # M2: B00
        [0, 1, 0, -1], # M3: B01 - B11
        [-1, 0, 1, 0], # M4: B10 - B00
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00 + B01
        [0, 0, 1, 1]  # M7: B10 + B11
    ]
    
    W_forms = [
        [1, 0, 0, -1], # M1: C00 +, C11 -
        [0, 0, 1, -1], # M2: C10 +, C11 -
        [0, 1, 0, 1],  # M3: C01 +, C11 +
        [1, 0, 1, 0],  # M4: C00 +, C10 +
        [-1, 1, 0, 0], # M5: C00 -, C01 +
        [0, 0, 0, 1],  # M6: C11 +
        [1, 0, 0, 0]   # M7: C00 +
    ]
    
    # Convert to numpy arrays for easier slicing, though lists work too
    U_forms_np = np.array(U_forms)
    V_forms_np = np.array(V_forms)
    W_forms_np = np.array(W_forms)
    
    num_blocks = p // 2 # 12 // 2 = 6
    
    for j in range(num_blocks):
        for k in range(7):
            r = j * 7 + k
            
            # Fill U: The A part is the same for all blocks.
            # U has shape (4, 42). Row indices 0..3 correspond to A[0,0], A[0,1], A[1,0], A[1,1]
            U[:, r] = U_forms_np[k, :]
            
            # Fill V: The B part is block j.
            # V has shape (24, 42).
            # Rows correspond to B[i, col].
            # Block j uses columns 2*j and 2*j+1.
            # Local indices in B block: 0: (0, 2j), 1: (0, 2j+1), 2: (1, 2j), 3: (1, 2j+1)
            # V_forms_np[k, :] gives coefficients for these 4 local entries.
            
            # Row 0 of V corresponds to B[0, 0..11]
            # Row 1 of V corresponds to B[1, 0..11]
            
            # Local index 0 -> B[0, 2j] -> V row 0, col 2j
            # Local index 1 -> B[0, 2j+1] -> V row 0, col 2j+1
            # Local index 2 -> B[1, 2j] -> V row 1, col 2j
            # Local index 3 -> B[1, 2j+1] -> V row 1, col 2j+1
            
            V[0, 2*j] = V_forms_np[k, 0]
            V[0, 2*j+1] = V_forms_np[k, 1]
            V[1, 2*j] = V_forms_np[k, 2]
            V[1, 2*j+1] = V_forms_np[k, 3]
            
            # Fill W: The C part is block j.
            # W has shape (24, 42).
            # Rows correspond to C[i, col].
            # Block j uses columns 2*j and 2*j+1.
            # Local indices in C block: 0: (0, 2j), 1: (0, 2j+1), 2: (1, 2j), 3: (1, 2j+1)
            # W_forms_np[k, :] gives coefficients for these 4 local entries.
            
            # Row 0 of W corresponds to C[0, 0..11]
            # Row 1 of W corresponds to C[1, 0..11]
            
            # Local index 0 -> C[0, 2j] -> W row 0, col 2j
            # Local index 1 -> C[0, 2j+1] -> W row 0, col 2j+1
            # Local index 2 -> C[1, 2j] -> W row 1, col 2j
            # Local index 3 -> C[1, 2j+1] -> W row 1, col 2j+1
            
            W[0, 2*j] = W_forms_np[k, 0]
            W[0, 2*j+1] = W_forms_np[k, 1]
            W[1, 2*j] = W_forms_np[k, 2]
            W[1, 2*j+1] = W_forms_np[k, 3]
            
    # Convert back to lists of lists as required by interface (implied by "integer/half-integer entries" and previous code style, though numpy arrays are often accepted if converted or if evaluator handles them. The prompt says "factor matrices of shapes...". The previous code returned lists. I will return lists to be safe.)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
