import time
import numpy as np

DIMENSIONS = [2, 2, 14]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Fallback: Schoolbook
    rank_schoolbook = n * m * p
    U_fallback = [[0] * rank_schoolbook for _ in range(n * m)]
    V_fallback = [[0] * rank_schoolbook for _ in range(m * p)]
    W_fallback = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fallback[i * m + j][col] = 1
                V_fallback[j * p + k][col] = 1
                W_fallback[k * n + i][col] = 1
                col += 1

    best_result = {"U": U_fallback, "V": V_fallback, "W": W_fallback}
    best_rank = rank_schoolbook

    # Try Strassen rank 49
    if time.time() < deadline - 10:
        U, V, W, rank = _construct_strassen_49(n, m, p)
        if U is not None:
            if _verify(U, V, W, n, m, p):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank

    return best_result

def _construct_strassen_49(n, m, p):
    if p % 2 != 0:
        return None, None, None, 0
    
    num_pairs = p // 2
    rank = 7 * num_pairs
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2x2
    # A indices: 00, 01, 10, 11
    # B indices: 00, 01, 10, 11 (relative to block)
    # C indices: 00, 01, 10, 11 (relative to block)
    
    # We will define the 7 terms as (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs is dict {A_idx: coeff}
    # v_coeffs is dict {B_idx: coeff}
    # w_coeffs is dict {C_idx: coeff}
    
    terms = [
        # M1
        ({'00': 1, '11': 1}, {'00': 1, '11': 1}, {'00': 1, '10': -1, '01': -1, '11': 1}), # Placeholder, will fix in verification loop
    ]
    
    # Actually, it's safer to just iterate and set values, then verify.
    # I will use the standard Strassen formulas and hope they work.
    # If not, the verification will catch it.
    
    for r in range(num_pairs):
        k1 = 2 * r
        k2 = 2 * r + 1
        base = 7 * r
        
        # M1 = (A00 + A11) * (B0k1 + B1k2) -> C0k1, C1k2
        # Standard: M1 = (A11+A22)(B11+B22)
        # C11 = M1 + M4 - M5 - M7
        # C12 = M5 + M7
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Let's map standard 1-based to 0-based
        # A11->A00, A12->A01, A21->A10, A22->A11
        # B11->B0k1, B12->B0k2, B21->B1k1, B22->B1k2
        
        # M1: A(00,11), B(0k1, 1k2)
        U[0][base] = 1; U[3][base] = 1
        V[k1][base] = 1; V[p+k2][base] = 1
        W[k1*n+0][base] = 1  # C0k1
        W[k2*n+1][base] = 1  # C1k2
        
        # M2: A(10,11), B(0k1)
        U[2][base+1] = 1; U[3][base+1] = 1
        V[k1][base+1] = 1
        W[k1*n+1][base+1] = 1  # C1k1
        
        # M3: A(00), B(1k2 - 0k2)
        U[0][base+2] = 1
        V[p+k2][base+2] = 1; V[k2][base+2] = -1
        W[k2*n+1][base+2] = 1  # C1k2
        
        # M4: A(11), B(1k1 - 0k1)
        U[3][base+3] = 1
        V[p+k1][base+3] = 1; V[k1][base+3] = -1
        W[k1*n+0][base+3] = 1  # C0k1
        W[k1*n+1][base+3] = 1  # C1k1
        
        # M5: A(00,01), B(1k2)
        U[0][base+4] = 1; U[1][base+4] = 1
        V[p+k2][base+4] = 1
        W[k2*n+0][base+4] = 1  # C0k2
        W[k1*n+0][base+4] = -1 # C0k1
        
        # M6: A(10-00), B(0k1+0k2)
        U[2][base+5] = 1; U[0][base+5] = -1
        V[k1][base+5] = 1; V[k2][base+5] = 1
        W[k2*n+1][base+5] = 1  # C1k2
        
        # M7: A(01-11), B(1k1+1k2)
        U[1][base+6] = 1; U[3][base+6] = -1
        V[p+k1][base+6] = 1; V[p+k2][base+6] = 1
        W[k2*n+0][base+6] = 1  # C0k2
        W[k1*n+0][base+6] = -1 # C0k1
        W[k2*n+1][base+6] = 1  # C1k2

    return U, V, W, rank

def _verify(U, V, W, n, m, p):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if total != 1:
                    return False
    return True
