import numpy as np
import time

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 * 2x6 matrix multiplication.
    Uses Strassen's algorithm on three 2x2 blocks, achieving rank 21.
    
    The tensor T[a,b,c] = 1 for a=i*2+j, b=j*6+k, c=k*2+i.
    Since the three 2x2x2 blocks have disjoint V and W index supports,
    the rank is exactly 3 * 7 = 21 (no cross-block cancellation possible).
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = 2, 2, 6
    nm, mp, pn = n * m, m * p, p * n  # 4, 12, 12
    
    rank = 21
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # Strassen coefficients for 2x2 multiplication
    # A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # B indices: 0=B00, 1=B01, 2=B10, 3=B11
    # C indices: 0=C00, 1=C01, 2=C10, 3=C11
    
    # Each step: (u_coeffs, v_coeffs, w_coeffs) as dicts {local_idx: value}
    strassen_steps = [
        # M1 = (A00+A11)(B00+B11) -> C00 + C11
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (A10+A11)B00 -> C10 - C11
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3 = A00(B01-B11) -> C01 + C11
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4 = A11(B10-B00) -> C00 + C10
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5 = (A00+A01)B11 -> -C00 + C01
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6 = (A10-A00)(B00+B01) -> C11
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (A01-A11)(B10+B11) -> C00
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    # Fill U, V, W for each of the 3 blocks
    for g in range(3):
        k0 = 2 * g
        k1 = 2 * g + 1
        
        # V row indices for this block: B[j, k] -> j*6 + k
        v_map = {
            0: 0 * 6 + k0,   # B00
            1: 0 * 6 + k1,   # B01
            2: 1 * 6 + k0,   # B10
            3: 1 * 6 + k1,   # B11
        }
        
        # W row indices for this block: C[k, i] -> k*2 + i
        w_map = {
            0: k0 * 2 + 0,   # C[k0, 0]
            1: k1 * 2 + 0,   # C[k1, 0]
            2: k0 * 2 + 1,   # C[k0, 1]
            3: k1 * 2 + 1,   # C[k1, 1]
        }
        
        for s in range(7):
            r = 7 * g + s
            u_dict, v_dict, w_dict = strassen_steps[s]
            
            for idx, val in u_dict.items():
                U[idx, r] = val
            for idx, val in v_dict.items():
                V[v_map[idx], r] = val
            for idx, val in w_dict.items():
                W[w_map[idx], r] = val

    # Check time before verification
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Verification: T[a,b,c] should equal sum_r U[a,r]*V[b,r]*W[c,r]
    T_check = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Expected tensor
    T_exp = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_exp[a, b, c] = 1.0
                
    if not np.allclose(T_check, T_exp, atol=1e-10):
        # Fallback to schoolbook rank 24
        rank_school = n * m * p
        U_s = np.zeros((nm, rank_school))
        V_s = np.zeros((mp, rank_school))
        W_s = np.zeros((pn, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U_s[a, col] = 1
                    V_s[b, col] = 1
                    W_s[c, col] = 1
                    col += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
