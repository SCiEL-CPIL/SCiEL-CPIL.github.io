import time
import numpy as np

DIMENSIONS = [2, 3, 5]

def _to_list_of_lists(arr):
    return arr.tolist()

def construct():
    # n=2, m=3, p=5
    n, m, p = DIMENSIONS
    # Total rank estimate: 7 (Strassen 2x2x2) + 12 (2x2x3) + 10 (2x1x5) = 29
    # Let's verify the blocks
    # Block 1: i in 0..1, j in 0..1, k in 0..1 -> 2x2x2 (Rank 7)
    # Block 2: i in 0..1, j in 0..1, k in 2..4 -> 2x2x3 (Rank 12)
    # Block 3: i in 0..1, j in 2,     k in 0..4 -> 2x1x5 (Rank 10)
    
    rank = 29
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Helper to add Strassen 2x2x2 block
    # local i, j, k in 0..1
    # Global offsets: I_off=0, J_off=0, K_off=0
    def add_strassen_222(I_off, J_off, K_off):
        nonlocal col
        # M1 = (A00+A11)(B00+B11)
        # U: A00(1), A11(1)
        # V: B00(1), B11(1)
        # W: C00(1), C11(1)
        
        # Term 1
        U[I_off*m + 0 + 0, col] = 1 # A00 local
        U[I_off*m + 1 + 1, col] = 1 # A11 local -> global i=1, j=1? No, local j=1. Global j=J_off+1.
        # Wait, local A_{i,l_j}. Global A_{I_off+i, J_off+l_j}.
        # My indexing in add_strassen needs to map local (i,j) to global.
        
        # Let's define local indices li, lj, lk in 0..1
        # Global i = I_off + li
        # Global j = J_off + lj
        # Global k = K_off + lk
        
        # Strassen Terms for 2x2x2
        # t1: (li0+li1)(lj0+lj1) -- NO, Strassen is on A rows and B rows?
        # Standard Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Local A indices:
        # A00: li=0, lj=0
        # A01: li=0, lj=1
        # A10: li=1, lj=0
        # A11: li=1, lj=1
        
        # Local B indices:
        # B00: lj=0, lk=0
        # B01: lj=0, lk=1
        # B10: lj=1, lk=0
        # B11: lj=1, lk=1
        
        # Local C indices:
        # C00: li=0, lk=0
        # C01: li=0, lk=1
        # C10: li=1, lk=0
        # C11: li=1, lk=1
        
        terms = []
        # Each term: (dict of U coeffs, dict of V coeffs, dict of W coeffs)
        
        # T1
        terms.append(
            (
                {(0,0):1, (1,1):1}, # U
                {(0,0):1, (1,1):1}, # V
                {(0,0):1, (1,1):1}  # W
            )
        )
        # T2
        terms.append(
            (
                {(1,0):1, (1,1):1}, # U
                {(0,0):1},           # V
                {(1,0):1, (1,1):-1}  # W: C10=M2+M4, C11=M1-M2+... -> M2 contributes to C10 (+1) and C11 (-1)
            )
        )
        # T3
        terms.append(
            (
                {(0,0):1},           # U
                {(0,1):1, (1,1):-1}, # V
                {(0,1):1, (1,1):1}   # W: C01=M3+M5, C11=...+M3 -> M3 contributes to C01 (+1) and C11 (+1)
            )
        )
        # T4
        terms.append(
            (
                {(1,1):1},           # U
                {(1,0):1, (0,0):-1}, # V
                {(1,0):1, (0,0):1}   # W: C10=M2+M4, C00=...+M4 -> M4 contributes to C10 (+1) and C00 (+1)
            )
        )
        # T5
        terms.append(
            (
                {(0,0):1, (0,1):1},  # U
                {(1,1):1},           # V
                {(1,1):-1, (0,1):1}  # W: C01=M3+M5, C11=...-M5 -> M5 contributes to C01 (+1) and C11 (-1)
            )
        )
        # T6
        terms.append(
            (
                {(1,0):1, (0,0):-1}, # U
                {(0,0):1, (0,1):1},  # V
                {(1,1):1}            # W: C11=...+M6 -> M6 contributes to C11 (+1)
            )
        )
        # T7
        terms.append(
            (
                {(0,1):1, (1,1):-1}, # U
                {(1,0):1, (1,1):1},  # V
                {(0,0):1, (1,1):-1}  # W: C00=...+M7, C11=...+M7? 
                # C00 = M1+M4-M5+M7. C11 = M1-M2+M3+M6. 
                # Wait, C11 does NOT have M7. 
                # Check formula: C11 = M1 - M2 + M3 + M6. Correct.
                # So M7 only contributes to C00?
                # Let's re-verify Strassen C11 formula.
                # C11 = M1 - M2 + M3 + M6. Yes.
                # So M7 is in C00 only?
                # C00 = M1 + M4 - M5 + M7. Yes.
            )
        )
        # Correction on T7 W coefficients:
        terms[6] = (
            (
                {(0,1):1, (1,1):-1}, # U
                {(1,0):1, (1,1):1},  # V
                {(0,0):1}            # W: Only C00 gets +1
            )
        )

        for u_dict, v_dict, w_dict in terms:
            for (li, lj), val in u_dict.items():
                gi = I_off + li
                gj = J_off + lj
                U[gi*m + gj, col] += val
            for (lj, lk), val in v_dict.items():
                gj = J_off + lj
                gk = K_off + lk
                V[gj*p + gk, col] += val
            for (lk, li), val in w_dict.items():
                gk = K_off + lk
                gi = I_off + li
                W[gk*n + gi, col] += val
            col += 1

    # Add Block 1: Strassen for i=0,1 j=0,1 k=0,1
    add_strassen_222(0, 0, 0)
    
    # Block 2: Schoolbook for i in 0..1, j in 0..1, k in 2..4
    for i in range(2):
        for j in range(2):
            for k in range(2, 5):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
                
    # Block 3: Schoolbook for i in 0..1, j=2, k in 0..4
    for i in range(2):
        for k in range(5):
            U[i*m + 2, col] = 1
            V[2*p + k, col] = 1
            W[k*n + i, col] = 1
            col += 1

    return {
        "U": _to_list_of_lists(U),
        "V": _to_list_of_lists(V),
        "W": _to_list_of_lists(W)
    }
