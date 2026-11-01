import time
import numpy as np

DIMENSIONS = [2, 4, 7]

def _decomp_221():
    """Returns U, V, W for 2x2x1 multiplication, rank 4."""
    U = [[1,0,0,0],
         [0,1,0,0],
         [0,0,1,0],
         [0,0,0,1]]
    V = [[1,0,1,0],
         [0,1,0,1]]
    W = [[1,1,0,0],
         [0,0,1,1]]
    return U, V, W

def _decomp_222():
    """Returns U, V, W for 2x2x2 multiplication, rank 7 (Strassen)."""
    # M1: (a+d)(e+h)
    # M2: (c+d)e
    # M3: a(f-h)
    # M4: d(g-e)
    # M5: (a+b)h
    # M6: (c-a)(e+f)
    # M7: (b-d)(g+h)
    
    # U: 4x7
    U = [[0]*7 for _ in range(4)]
    # M1: a(0), d(3)
    U[0][0] = 1; U[3][0] = 1
    # M2: c(2), d(3)
    U[2][1] = 1; U[3][1] = 1
    # M3: a(0)
    U[0][2] = 1
    # M4: d(3)
    U[3][3] = 1
    # M5: a(0), b(1)
    U[0][4] = 1; U[1][4] = 1
    # M6: c(2), a(0)
    U[2][5] = 1; U[0][5] = -1
    # M7: b(1), d(3)
    U[1][6] = 1; U[3][6] = -1
    
    # V: 4x7 (indices: e0, f1, g2, h3)
    V = [[0]*7 for _ in range(4)]
    # M1: e(0), h(3)
    V[0][0] = 1; V[3][0] = 1
    # M2: e(0)
    V[0][1] = 1
    # M3: f(1), h(3)
    V[1][2] = 1; V[3][2] = -1
    # M4: g(2), e(0)
    V[2][3] = 1; V[0][3] = -1
    # M5: h(3)
    V[3][4] = 1
    # M6: e(0), f(1)
    V[0][5] = 1; V[1][5] = 1
    # M7: g(2), h(3)
    V[2][6] = 1; V[3][6] = 1
    
    # W: 4x7 (indices: C00=0, C01=1, C10=2, C11=3)
    W = [[0]*7 for _ in range(4)]
    # C00: M1(1), M4(1), M5(-1), M7(1)
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    # C01: M3(1), M5(1)
    W[1][2] = 1; W[1][4] = 1
    # C10: M2(1), M4(1)
    W[2][1] = 1; W[2][3] = 1
    # C11: M1(1), M2(-1), M3(1), M6(-1)
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = -1
    
    return U, V, W

def _decomp_22p(p):
    """
    Returns U, V, W for 2x2xp multiplication.
    Uses concatenation of base cases.
    """
    if p == 1:
        return _decomp_221()
    elif p == 2:
        return _decomp_222()
    else:
        # Split p into parts of size 2 and 1
        # Greedy: use as many 2s as possible, then 1 if needed
        parts = []
        while p > 0:
            if p >= 2:
                parts.append(2)
                p -= 2
            else:
                parts.append(1)
                p -= 1
        
        U_list = []
        V_list = []
        W_list = []
        current_col_offset = 0
        
        for part_p in parts:
            if part_p == 1:
                U_part, V_part, W_part = _decomp_221()
            else:
                U_part, V_part, W_part = _decomp_222()
            
            rank_part = len(U_part[0])
            
            # Shift column indices for V and W? No, V and W are just appended.
            # But V and W rows correspond to specific columns of B and C.
            # For 2x2xp, B is 2xp, C is 2xp.
            # The parts cover disjoint columns of B and C.
            
            U_list.append(U_part)
            V_list.append(V_part)
            W_list.append(W_part)
            
        # Concatenate
        # U is 4 x TotalRank
        total_rank = sum(len(u[0]) for u in U_list)
        U = [[0]*total_rank for _ in range(4)]
        V = [[0]*total_rank for _ in range(2*p)]
        W = [[0]*total_rank for _ in range(2*p)]
        
        col_start = 0
        # We need to map the local column indices of the parts to the global p columns.
        # The parts are sequential.
        # Part 0 covers B cols 0..part_p0-1, C cols 0..part_p0-1.
        # Part 1 covers B cols part_p0..part_p0+part_p1-1, etc.
        
        current_p_offset = 0
        for i, part_p in enumerate(parts):
            U_part, V_part, W_part = U_list[i], V_list[i], W_list[i]
            rank_part = len(U_part[0])
            
            # U is same for all parts (indices 0..3)
            for r in range(4):
                for c in range(rank_part):
                    U[r][col_start + c] = U_part[r][c]
            
            # V: local row r (0..1), local col k (0..part_p-1)
            # Global row r, global col current_p_offset + k
            for r in range(2):
                for k in range(part_p):
                    local_v_idx = r * part_p + k
                    # V_part is 2 x rank_part? No.
                    # In _decomp_22p, V is (2*part_p) x rank_part.
                    # Wait, my base cases return V of size (2*part_p) x rank.
                    # For p=1, V is 2x4.
                    # For p=2, V is 4x7.
                    # So V_part[r * part_p + k] is the row.
                    global_v_idx = r * p + (current_p_offset + k)
                    for c in range(rank_part):
                        V[global_v_idx][col_start + c] = V_part[r * part_p + k][c]
            
            # W: local row r (0..1), local col k (0..part_p-1)
            # W index is k*n + i = k*2 + r.
            # Global W index is (current_p_offset + k)*2 + r.
            for r in range(2):
                for k in range(part_p):
                    local_w_idx = k * 2 + r
                    global_w_idx = (current_p_offset + k) * 2 + r
                    for c in range(rank_part):
                        W[global_w_idx][col_start + c] = W_part[local_w_idx][c]
            
            col_start += rank_part
            current_p_offset += part_p
            
        return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=7
    
    # Split into two 2x2x7 blocks
    # Block 1: A cols 0,1 ; B rows 0,1
    # Block 2: A cols 2,3 ; B rows 2,3
    
    U1, V1, W1 = _decomp_22p(7)
    U2, V2, W2 = _decomp_22p(7)
    
    rank1 = len(U1[0])
    rank2 = len(U2[0])
    total_rank = rank1 + rank2
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    # Map Block 1
    # U: local 4xrank1 -> global 8xrank1
    # Local A index i*2+j -> Global A index i*4+j
    for i in range(2):
        for j in range(2):
            local_idx = i * 2 + j
            global_idx = i * 4 + j
            for c in range(rank1):
                U[global_idx][c] = U1[local_idx][c]
                
    # V: local 2*7 x rank1 -> global 4*7 x rank1
    # Local B index i*7+k -> Global B index i*7+k (rows 0,1)
    for i in range(2):
        for k in range(7):
            local_idx = i * 7 + k
            global_idx = i * 7 + k
            for c in range(rank1):
                V[global_idx][c] = V1[local_idx][c]
                
    # W: local 2*7 x rank1 -> global 2*7 x rank1
    # Same indices
    for idx in range(14):
        for c in range(rank1):
            W[idx][c] = W1[idx][c]
            
    # Map Block 2
    # U: local 4xrank2 -> global 8xrank2
    # Local A index i*2+j -> Global A index i*4 + (j+2)
    for i in range(2):
        for j in range(2):
            local_idx = i * 2 + j
            global_idx = i * 4 + (j + 2)
            for c in range(rank2):
                U[global_idx][rank1 + c] = U2[local_idx][c]
                
    # V: local 2*7 x rank2 -> global 4*7 x rank2
    # Local B index i*7+k -> Global B index (i+2)*7+k
    for i in range(2):
        for k in range(7):
            local_idx = i * 7 + k
            global_idx = (i + 2) * 7 + k
            for c in range(rank2):
                V[global_idx][rank1 + c] = V2[local_idx][c]
                
    # W: local 2*7 x rank2 -> global 2*7 x rank2
    # Same indices
    for idx in range(14):
        for c in range(rank2):
            W[idx][rank1 + c] = W2[idx][c]
            
    return {"U": U, "V": V, "W": W}
