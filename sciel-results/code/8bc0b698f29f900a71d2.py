import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Helper to add a term
    def add_term(u_coeffs, v_coeffs, w_coeffs, col_idx):
        for idx, coeff in u_coeffs.items():
            U[idx, col_idx] += coeff
        for idx, coeff in v_coeffs.items():
            V[idx, col_idx] += coeff
        for idx, coeff in w_coeffs.items():
            W[idx, col_idx] += coeff

    # Local Strassen terms for 2x2x2
    # Local indices: 0:00, 1:01, 2:10, 3:11
    terms = [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),       # M1
        ({2:1, 3:1}, {0:1}, {2:1}),                 # M2
        ({0:1}, {1:1, 3:-1}, {1:1, 3:-1}),          # M3
        ({3:1}, {2:1, 0:-1}, {0:-1, 2:1}),          # M4
        ({0:1, 1:1}, {3:1}, {0:1, 1:1}),            # M5
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),           # M6
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1, 3:1}),      # M7
    ]

    # Define mappings for the 4 sub-problems
    # Sub-problem 1: A_L (cols 0,1) x B_L0 (rows 0,1, cols 0,1) -> C_L0 (cols 0,1)
    # A_L local->global: 0->0, 1->1, 2->4, 3->5
    # B_L0 local->global: 0->0, 1->1, 2->4, 3->5
    # C_L0 local->global: 0->0, 1->2, 2->1, 3->3
    map_sp1 = {
        'u': {0:0, 1:1, 2:4, 3:5},
        'v': {0:0, 1:1, 2:4, 3:5},
        'w': {0:0, 1:2, 2:1, 3:3}
    }
    
    # Sub-problem 2: A_R (cols 2,3) x B_R0 (rows 2,3, cols 0,1) -> C_L0 (cols 0,1)
    # A_R local->global: 0->2, 1->3, 2->6, 3->7
    # B_R0 local->global: 0->8, 1->9, 2->12, 3->13
    # C_L0 local->global: 0->0, 1->2, 2->1, 3->3
    map_sp2 = {
        'u': {0:2, 1:3, 2:6, 3:7},
        'v': {0:8, 1:9, 2:12, 3:13},
        'w': {0:0, 1:2, 2:1, 3:3}
    }
    
    # Sub-problem 3: A_L (cols 0,1) x B_L1 (rows 0,1, cols 2,3) -> C_L1 (cols 2,3)
    # A_L local->global: 0->0, 1->1, 2->4, 3->5
    # B_L1 local->global: 0->2, 1->3, 2->6, 3->7
    # C_L1 local->global: 0->4, 1->6, 2->5, 3->7
    map_sp3 = {
        'u': {0:0, 1:1, 2:4, 3:5},
        'v': {0:2, 1:3, 2:6, 3:7},
        'w': {0:4, 1:6, 2:5, 3:7}
    }
    
    # Sub-problem 4: A_R (cols 2,3) x B_R1 (rows 2,3, cols 2,3) -> C_L1 (cols 2,3)
    # A_R local->global: 0->2, 1->3, 2->6, 3->7
    # B_R1 local->global: 0->10, 1->11, 2->14, 3->15
    # C_L1 local->global: 0->4, 1->6, 2->5, 3->7
    map_sp4 = {
        'u': {0:2, 1:3, 2:6, 3:7},
        'v': {0:10, 1:11, 2:14, 3:15},
        'w': {0:4, 1:6, 2:5, 3:7}
    }

    mappings = [map_sp1, map_sp2, map_sp3, map_sp4]
    
    for m_idx, mapping in enumerate(mappings):
        base_col = m_idx * 7
        for t_idx, (u_loc, v_loc, w_loc) in enumerate(terms):
            col = base_col + t_idx
            u_glob = {mapping['u'][k]: val for k, val in u_loc.items()}
            v_glob = {mapping['v'][k]: val for k, val in v_loc.items()}
            w_glob = {mapping['w'][k]: val for k, val in w_loc.items()}
            add_term(u_glob, v_glob, w_glob, col)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
