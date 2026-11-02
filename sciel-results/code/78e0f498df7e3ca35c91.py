import time
import itertools
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START

def _add_tensor(U, V, W, offset_u, offset_v, offset_w, u_local, v_local, w_local):
    """
    Add local tensors u_local, v_local, w_local to global U, V, W
    at the specified offsets.
    """
    for a in range(len(u_local)):
        for r in range(len(u_local[0])):
            val = u_local[a][r]
            if val != 0:
                U[offset_u + a][offset_w + r] += val
                
    for b in range(len(v_local)):
        for r in range(len(v_local[0])):
            val = v_local[b][r]
            if val != 0:
                V[offset_v + b][offset_w + r] += val
                
    for c in range(len(w_local)):
        for r in range(len(w_local[0])):
            val = w_local[c][r]
            if val != 0:
                W[offset_w + c][offset_w + r] += val # Wait, W index is c, r.
                # W is (p*n, rank).
                # offset_w is for the c index? No, W rows are c.
                # So W[offset_w + c][r] += val.
                # But the rank index r is global.
                # We need to map local rank r to global rank R.
                # The function signature implies u_local, v_local, w_local have the same rank.
                # And we add them at global rank offset?
                # No, the rank is the total number of terms.
                # We should append to the rank dimension.
                # So we need to know the current rank size.
                # Let's change the approach: build lists of terms.

def _build_strassen_222():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    Rank 7.
    Indices:
    U: 2x2 (4 rows)
    V: 2x2 (4 rows)
    W: 2x2 (4 rows)
    """
    # Strassen's 7 terms
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # Term 1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    W[1][0] = -1; W[2][0] = -1 # Wait, C11 gets M1, C22 gets M1.
    # C11 (0) = M1 ...
    # C22 (3) = M1 ...
    # So W[0][0]=1, W[3][0]=1.
    # What about the other signs?
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Reset W
    W = [[0]*7 for _ in range(4)]
    
    # T1
    W[0][0] = 1
    W[3][0] = 1
    
    # T2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    W[3][1] = -1
    
    # T3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    W[3][2] = 1
    
    # T4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1
    W[2][3] = 1
    
    # T5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1
    W[1][4] = 1
    
    # T6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # T7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _build_schoolbook_221():
    """
    Returns U, V, W for 2x2x1 matrix multiplication.
    Rank 4.
    """
    U = [[0]*4 for _ in range(4)]
    V = [[0]*4 for _ in range(2)] # 2x1 B matrix has 2 entries
    W = [[0]*4 for _ in range(2)] # 2x1 C matrix has 2 entries
    
    # C1 = A11 B11 + A12 B21
    # C2 = A21 B11 + A22 B21
    
    # T1: A11 B11 -> C1
    U[0][0] = 1
    V[0][0] = 1
    W[0][0] = 1
    
    # T2: A12 B21 -> C1
    U[1][1] = 1
    V[1][1] = 1
    W[0][1] = 1
    
    # T3: A21 B11 -> C2
    U[2][2] = 1
    V[0][2] = 1
    W[1][2] = 1
    
    # T4: A22 B21 -> C2
    U[3][3] = 1
    V[1][3] = 1
    W[1][3] = 1
    
    return U, V, W

def _build_223():
    """
    Builds 2x2x3 decomposition from 222 (Strassen) and 221 (Schoolbook).
    Rank 11.
    """
    U1, V1, W1 = _build_strassen_222()
    U2, V2, W2 = _build_schoolbook_221()
    
    U = [[0]*11 for _ in range(4)]
    V = [[0]*11 for _ in range(6)] # 2x3 B has 6 entries
    W = [[0]*11 for _ in range(6)] # 2x3 C has 6 entries
    
    # Map 222 to k=0,1
    # B indices for k=0: 0, 2 (B10, B20) -> 0, 2?
    # B is 2x3. Indices:
    # 0: (0,0), 1: (0,1), 2: (0,2)
    # 3: (1,0), 4: (1,1), 5: (1,2)
    # 222 uses B10, B11, B20, B21.
    # B10 -> 0, B11 -> 1, B20 -> 3, B21 -> 4.
    # C indices for k=0,1:
    # C is 2x3.
    # 0: (0,0), 1: (0,1), 2: (0,2)
    # 3: (1,0), 4: (1,1), 5: (1,2)
    # 222 uses C10, C11, C20, C21.
    # C10 -> 0, C11 -> 1, C20 -> 3, C21 -> 4.
    
    for a in range(4):
        for r in range(7):
            U[a][r] = U1[a][r]
            
    # V mapping: 0->0, 1->1, 2->3, 3->4
    v_map = [0, 1, 3, 4]
    for b in range(4):
        for r in range(7):
            V[v_map[b]][r] = V1[b][r]
            
    # W mapping: 0->0, 1->1, 2->3, 3->4
    w_map = [0, 1, 3, 4]
    for c in range(4):
        for r in range(7):
            W[w_map[c]][r] = W1[c][r]
            
    # Map 221 to k=2
    # 221 uses B10, B20 (from 2x1 perspective, but here it's the 3rd column).
    # In 221 local indices:
    # B: 0 (B10), 1 (B20).
    # In global 2x3:
    # B12 -> 2, B22 -> 5.
    # C: 0 (C10), 1 (C20).
    # In global 2x3:
    # C12 -> 2, C22 -> 5.
    
    for a in range(4):
        for r in range(4):
            U[a][r + 7] = U2[a][r]
            
    v_map2 = [2, 5]
    for b in range(2):
        for r in range(4):
            V[v_map2[b]][r + 7] = V2[b][r]
            
    w_map2 = [2, 5]
    for c in range(2):
        for r in range(4):
            W[w_map2[c]][r + 7] = W2[c][r]
            
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 115

    U_dim = n * m
    V_dim = m * p
    W_dim = p * n
    
    # We will build the tensor term by term.
    # List of (u_vec, v_vec, w_vec)
    terms = []
    
    def add_term(u, v, w):
        terms.append((u, v, w))
        
    # Helper to create sparse vector
    def make_vec(dim, indices_vals):
        vec = [0] * dim
        for idx, val in indices_vals:
            vec[idx] = val
        return vec

    # 1. M_1,4,6 (i=2)
    # Rank 24
    # T = sum_{j,k} e_{8+j} (U) * e_{j*6+k} (V) * e_{k*3+2} (W)
    for j in range(4):
        for k in range(6):
            u = make_vec(U_dim, [(8+j, 1)])
            v = make_vec(V_dim, [(j*6+k, 1)])
            w = make_vec(W_dim, [(k*3+2, 1)])
            add_term(u, v, w)
            
    # 2. M_2,4,6 (i=0,1)
    # Split into 2 M_2,4,3
    for k_block in range(2): # k=0..2, k=3..5
        k_offset = k_block * 3
        # Split M_2,4,3 into 2 M_2,2,3
        for j_block in range(2): # j=0..1, j=2..3
            j_offset = j_block * 2
            # Get 223 decomposition
            U223, V223, W223 = _build_223()
            
            # Map indices
            # U: i in {0,1}, j in {j_offset, j_offset+1}
            # Local U indices: 0,1,2,3 correspond to (0,0), (0,1), (1,0), (1,1) in 2x2
            # Global U indices:
            # i=0: 0*4 + j
            # i=1: 1*4 + j
            
            # Local V indices: 0..5 correspond to (0,0)..(1,2) in 2x3
            # Global V indices: j*6 + (k_offset + k_local)
            
            # Local W indices: 0..5 correspond to (0,0)..(1,2) in 2x3
            # Global W indices: (k_offset + k_local)*3 + i
            
            for r in range(11):
                u_vec = make_vec(U_dim, [])
                v_vec = make_vec(V_dim, [])
                w_vec = make_vec(W_dim, [])
                
                # U mapping
                # 0 -> (0, j_offset) -> 0*4 + j_offset
                # 1 -> (0, j_offset+1) -> 0*4 + j_offset + 1
                # 2 -> (1, j_offset) -> 1*4 + j_offset
                # 3 -> (1, j_offset+1) -> 1*4 + j_offset + 1
                u_map = [
                    0*4 + j_offset,
                    0*4 + j_offset + 1,
                    1*4 + j_offset,
                    1*4 + j_offset + 1
                ]
                for a in range(4):
                    if U223[a][r] != 0:
                        u_vec[u_map[a]] = U223[a][r]
                        
                # V mapping
                # Local 0 (0,0) -> j_offset * 6 + (k_offset + 0)
                # Local 1 (0,1) -> j_offset * 6 + (k_offset + 1)
                # Local 2 (0,2) -> j_offset * 6 + (k_offset + 2)
                # Local 3 (1,0) -> (j_offset+1) * 6 + (k_offset + 0)
                # Local 4 (1,1) -> (j_offset+1) * 6 + (k_offset + 1)
                # Local 5 (1,2) -> (j_offset+1) * 6 + (k_offset + 2)
                v_map = [
                    j_offset * 6 + k_offset + 0,
                    j_offset * 6 + k_offset + 1,
                    j_offset * 6 + k_offset + 2,
                    (j_offset+1) * 6 + k_offset + 0,
                    (j_offset+1) * 6 + k_offset + 1,
                    (j_offset+1) * 6 + k_offset + 2
                ]
                for b in range(6):
                    if V223[b][r] != 0:
                        v_vec[v_map[b]] = V223[b][r]
                        
                # W mapping
                # Local 0 (0,0) -> (k_offset+0)*3 + 0
                # Local 1 (0,1) -> (k_offset+1)*3 + 0
                # Local 2 (0,2) -> (k_offset+2)*3 + 0
                # Local 3 (1,0) -> (k_offset+0)*3 + 1
                # Local 4 (1,1) -> (k_offset+1)*3 + 1
                # Local 5 (1,2) -> (k_offset+2)*3 + 1
                w_map = [
                    (k_offset+0)*3 + 0,
                    (k_offset+1)*3 + 0,
                    (k_offset+2)*3 + 0,
                    (k_offset+0)*3 + 1,
                    (k_offset+1)*3 + 1,
                    (k_offset+2)*3 + 1
                ]
                for c in range(6):
                    if W223[c][r] != 0:
                        w_vec[w_map[c]] = W223[c][r]
                        
                add_term(u_vec, v_vec, w_vec)
                
    # Convert terms to U, V, W matrices
    rank = len(terms)
    U = [[0] * rank for _ in range(U_dim)]
    V = [[0] * rank for _ in range(V_dim)]
    W = [[0] * rank for _ in range(W_dim)]
    
    for r, (u, v, w) in enumerate(terms):
        for a in range(U_dim):
            if u[a] != 0:
                U[a][r] = u[a]
        for b in range(V_dim):
            if v[b] != 0:
                V[b][r] = v[b]
        for c in range(W_dim):
            if w[c] != 0:
                W[c][r] = w[c]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
