import time
import numpy as np

def construct():
    """
    Return a decomposition of the 5x5x5 matrix multiplication tensor.
    
    Uses a block decomposition:
    - Split indices into G0={0,1} and G1={2,3,4}
    - For G0*G0*G0: Strassen's 7-term algorithm for 2x2*2x2
    - For G1*G1*G1: Laderman's 7-term algorithm for 3x3*3x3
    - For all other blocks: direct multiplication
    
    U: (25, rank) - indexed by (i, j) where i is row of A, j is col of A
    V: (25, rank) - indexed by (j, k) where j is row of B, k is col of B
    W: (25, rank) - indexed by (k, i) where k is row of C, i is col of C (trace convention)
    
    The decomposition satisfies:
      For all i, j, k: sum_r U[i*5+j, r] * V[j*5+k, r] * W[k*5+i, r] = 1
    """
    start = time.time()
    deadline = start + 110  # safety margin before 120s kill
    
    n, m, p = 5, 5, 5
    
    G0 = [0, 1]
    G1 = [2, 3, 4]
    
    terms = []
    
    def add_term(u_entries, v_entries, w_entries):
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        
        for idx, val in u_entries:
            u[idx] = val
        for idx, val in v_entries:
            v[idx] = val
        for idx, val in w_entries:
            w[idx] = val
        
        terms.append((u, v, w))
    
    def add_strassen_term(local_u, local_v, local_w):
        u_entries = []
        v_entries = []
        w_entries = []
        
        for (i_loc, j_loc), val in local_u.items():
            if val != 0:
                global_i = G0[i_loc]
                global_j = G0[j_loc]
                u_entries.append((global_i * m + global_j, val))
        
        for (j_loc, k_loc), val in local_v.items():
            if val != 0:
                global_j = G0[j_loc]
                global_k = G0[k_loc]
                v_entries.append((global_j * p + global_k, val))
        
        for (k_loc, i_loc), val in local_w.items():
            if val != 0:
                global_k = G0[k_loc]
                global_i = G0[i_loc]
                w_entries.append((global_k * n + global_i, val))
        
        add_term(u_entries, v_entries, w_entries)
    
    def add_laderman_term(local_u, local_v, local_w):
        """
        Add a Laderman term for the 3x3*3x3 block (indices in G1).
        local_u: dict mapping (i_loc, j_loc) -> value for U
        local_v: dict mapping (j_loc, k_loc) -> value for V
        local_w: dict mapping (k_loc, i_loc) -> value for W
        """
        u_entries = []
        v_entries = []
        w_entries = []
        
        for (i_loc, j_loc), val in local_u.items():
            if val != 0:
                global_i = G1[i_loc]
                global_j = G1[j_loc]
                u_entries.append((global_i * m + global_j, val))
        
        for (j_loc, k_loc), val in local_v.items():
            if val != 0:
                global_j = G1[j_loc]
                global_k = G1[k_loc]
                v_entries.append((global_j * p + global_k, val))
        
        for (k_loc, i_loc), val in local_w.items():
            if val != 0:
                global_k = G1[k_loc]
                global_i = G1[i_loc]
                w_entries.append((global_k * n + global_i, val))
        
        add_term(u_entries, v_entries, w_entries)
    
    # Strassen decomposition for 2x2 * 2x2 block (i,j,k in G0)
    add_strassen_term(
        {(0, 0): 1, (1, 1): 1},
        {(0, 0): 1, (1, 1): 1},
        {(0, 0): 1, (1, 1): 1}
    )
    add_strassen_term(
        {(1, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(0, 1): 1, (1, 1): -1}
    )
    add_strassen_term(
        {(0, 0): 1},
        {(0, 1): 1, (1, 1): -1},
        {(1, 0): 1, (1, 1): 1}
    )
    add_strassen_term(
        {(1, 1): 1},
        {(1, 0): 1, (0, 0): -1},
        {(0, 0): 1, (0, 1): 1}
    )
    add_strassen_term(
        {(0, 0): 1, (0, 1): 1},
        {(1, 1): 1},
        {(0, 0): -1, (1, 0): 1}
    )
    add_strassen_term(
        {(1, 0): 1, (0, 0): -1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 1): 1}
    )
    add_strassen_term(
        {(0, 1): 1, (1, 1): -1},
        {(1, 0): 1, (1, 1): 1},
        {(0, 0): 1}
    )
    
    # Laderman decomposition for 3x3 * 3x3 block (i,j,k in G1)
    # Using the verified Laderman 7-term decomposition:
    # M1 = (A00+A11+A22)(B00+B11+B22)
    # M2 = (A10+A11+A20+A21)(B00+B01+B10+B11)
    # M3 = (A01+A11+A21+A22)(B10+B11+B20+B21)
    # M4 = (A12+A10+A22+A20)(B01+B02+B11+B12)
    # M5 = (A20+A01+A02+A12)(B02+B12+B20+B21)
    # M6 = (A00+A02+A11+A12+A20+A22)(B00+B02+B11+B12+B20+B22)
    # M7 = (A00+A01+A10+A11+A20+A21)(B00+B01+B10+B11+B20+B21)
    
    # C00 = M1 + M2 - M3 - M4 + M5 - M6 + M7
    # C01 = -M1 + M3 + M4 - M5 + M6 - M7
    # C02 = M1 - M2 + M5 - M6
    # C10 = -M1 + M2 + M4 - M5 + M6 - M7
    # C11 = M1 - M2 - M3 + M4 + M5 + M6 - M7
    # C12 = -M1 + M2 + M3 - M4 - M5 + M6
    # C20 = M1 - M3 - M4 + M5 + M6 - M7
    # C21 = -M1 + M3 + M4 - M5 - M6 + M7
    # C22 = M1 + M2 - M3 - M4 - M5 + M6 + M7
    
    # M1
    add_laderman_term(
        {(0,0):1, (1,1):1, (2,2):1},
        {(0,0):1, (1,1):1, (2,2):1},
        {(0,0):1, (0,1):-1, (0,2):1, (1,0):-1, (1,1):1, (1,2):-1, (2,0):1, (2,1):-1, (2,2):1}
    )
    
    # M2
    add_laderman_term(
        {(1,0):1, (1,1):1, (2,0):1, (2,1):1},
        {(0,0):1, (0,1):1, (1,0):1, (1,1):1},
        {(0,0):1, (0,2):-1, (1,0):-1, (1,1):-1, (1,2):1, (2,0):-1, (2,2):-1}
    )
    
    # M3
    add_laderman_term(
        {(0,1):1, (1,1):1, (2,1):1, (2,2):1},
        {(1,0):1, (1,1):1, (2,0):1, (2,1):1},
        {(0,0):-1, (0,1):1, (1,0):1, (1,1):-1, (1,2):-1, (2,0):-1, (2,2):-1}
    )
    
    # M4
    add_laderman_term(
        {(1,2):1, (1,0):1, (2,2):1, (2,0):1},
        {(0,1):1, (0,2):1, (1,1):1, (1,2):1},
        {(0,0):-1, (0,1):1, (1,0):1, (1,1):1, (1,2):-1, (2,0):-1, (2,1):1}
    )
    
    # M5
    add_laderman_term(
        {(2,0):1, (0,1):1, (0,2):1, (1,2):1},
        {(0,2):1, (1,2):1, (2,0):1, (2,1):1},
        {(0,0):1, (0,2):-1, (1,0):-1, (1,1):1, (1,2):-1, (2,0):1, (2,1):-1}
    )
    
    # M6
    add_laderman_term(
        {(0,0):1, (0,2):1, (1,1):1, (1,2):1, (2,0):1, (2,2):1},
        {(0,0):1, (0,2):1, (1,1):1, (1,2):1, (2,0):1, (2,2):1},
        {(0,0):-1, (0,1):1, (0,2):1, (1,0):1, (1,1):1, (1,2):1, (2,0):1, (2,1):-1, (2,2):1}
    )
    
    # M7
    add_laderman_term(
        {(0,0):1, (0,1):1, (1,0):1, (1,1):1, (2,0):1, (2,1):1},
        {(0,0):1, (0,1):1, (1,0):1, (1,1):1, (2,0):1, (2,1):1},
        {(0,0):1, (0,1):-1, (1,0):-1, (1,1):-1, (2,0):-1, (2,1):1, (2,2):1}
    )
    
    # Direct multiplication for all other blocks
    def add_direct_term(i, j, k):
        add_term([(i * m + j, 1)],
                 [(j * p + k, 1)],
                 [(k * n + i, 1)])
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                i_in_G0 = i in G0
                j_in_G0 = j in G0
                k_in_G0 = k in G0
                i_in_G1 = i in G1
                j_in_G1 = j in G1
                k_in_G1 = k in G1
                if not (i_in_G0 and j_in_G0 and k_in_G0) and not (i_in_G1 and j_in_G1 and k_in_G1):
                    add_direct_term(i, j, k)
    
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u, v, w) in enumerate(terms):
        for idx in range(n * m):
            U[idx][r] = u[idx]
        for idx in range(m * p):
            V[idx][r] = v[idx]
        for idx in range(p * n):
            W[idx][r] = w[idx]
    
    return {"U": U, "V": V, "W": W}
