"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We implement a hybrid decomposition:
    # A 2x2x2 Strassen core for the sub-block (i<2, j<2, k<2)
    # And schoolbook for all other triples (i,j,k)
    
    # Global flattened indices
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    rank_total = 0
    U = []
    V = []
    W = []
    
    def add_factor(u_val, v_val, w_val):
        # u_val: length n*m
        # v_val: length m*p
        # w_val: length p*n
        U.append(u_val)
        V.append(v_val)
        W.append(w_val)
        return 1

    # 1. Define Strassen factors for the 2x2x2 core
    # Local indices for A (2x2): 0:00, 1:01, 2:10, 3:11
    # Local indices for B (2x2): 0:00, 1:01, 2:10, 3:11
    # Local indices for C (2x2): 0:00 (k0i0), 1:10 (k0i1), 2:01 (k1i0), 3:11 (k1i1)
    # Note: W index is k*n + i. For local n=2, k=1, i=0 -> 2. k=0, i=1 -> 1.
    
    # Strassen Linear Forms
    # z1 = (a11+a22)(b11+b22)
    # z2 = (a21+a22)b11
    # z3 = a11(b12-b22)
    # z4 = a22(b21-b11)
    # z5 = (a11+a12)b22
    # z6 = (a21-a11)(b11+b12)
    # z7 = (a12-a22)(b21+b22)
    
    # c11 = z1 + z4 - z5 + z7  -> Local C[0]
    # c12 = z3 + z5            -> Local C[2]
    # c21 = z2 + z4            -> Local C[1]
    # c22 = z1 - z2 + z3 + z6  -> Local C[3]
    
    strassen_factors = [
        # u (local A), v (local B), w (local C)
        # z1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # z2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0]),
        # z3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 0]),
        # z4
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 1, 0, 0]),
        # z5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        # z6
        ([1, 0, -1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # z7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, -1, -1]), # c11 has +z7, c22 has 0? 
        # Let's re-verify c22: z1 - z2 + z3 + z6. So z7 is NOT in c22.
        # My list for z7 in C:
        # c11: +z7
        # c12: 0
        # c21: 0
        # c22: 0
        # So W for z7 should be [1, 0, 0, 0]
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    # Correct W for z7:
    # c11 = ... + z7. c12 = ... c21 = ... c22 = ...
    # Yes, z7 only appears in c11.
    # So W_local_z7 = [1, 0, 0, 0]
    strassen_factors[6] = ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    
    # Map local factors to global
    for uf, vf, wf in strassen_factors:
        u_global = [0] * (n * m)
        v_global = [0] * (m * p)
        w_global = [0] * (p * n)
        
        # A: i in 0..1, j in 0..1. Global idx = i*m + j
        for loc_i in range(2):
            for loc_j in range(2):
                loc_idx = loc_i * 2 + loc_j
                glob_idx = loc_i * m + loc_j
                u_global[glob_idx] = uf[loc_idx]
                
        # B: j in 0..1, k in 0..1. Global idx = j*p + k
        for loc_j in range(2):
            for loc_k in range(2):
                loc_idx = loc_j * 2 + loc_k
                glob_idx = loc_j * p + loc_k
                v_global[glob_idx] = vf[loc_idx]
                
        # C: k in 0..1, i in 0..1. Global idx = k*n + i
        # Local C indices:
        # 0: k=0, i=0
        # 1: k=0, i=1
        # 2: k=1, i=0
        # 3: k=1, i=1
        for loc_k in range(2):
            for loc_i in range(2):
                loc_idx = loc_k * 2 + loc_i
                glob_idx = loc_k * n + loc_i
                w_global[glob_idx] = wf[loc_idx]
                
        add_factor(u_global, v_global, w_global)
        
    rank_total += 7
    
    # 2. Add schoolbook factors for all other triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip the core block (i<2, j<2, k<2)
                if i < 2 and j < 2 and k < 2:
                    continue
                
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                
                u[i * m + j] = 1
                v[j * p + k] = 1
                w[k * n + i] = 1
                
                add_factor(u, v, w)
                rank_total += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
