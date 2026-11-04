import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s limit
    
    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if total != 1:
                        return False
        return True
    
    # Build rank-49 decomposition for 4x4x4 using double Strassen
    # Then schoolbook for 4x4x3, total rank = 49 + 48 = 97
    
    def build_4x4x4_strassen():
        """Build rank-49 decomposition for 4x4x4 matrix multiplication."""
        # Each entry is (U_idx, V_idx, W_idx, coeff)
        # U_idx = i*m + j where i,j in 0..3 (row,col of A)
        # V_idx = j*p + k where j,k in 0..3 (row,col of B) -- but p=4 here
        # W_idx = k*n + i where k,i in 0..3 (row,col of C)
        
        # Strassen level 1: 4x4 -> 7 block operations on 2x2 blocks
        # Strassen level 2: each 2x2 -> 7 scalar operations
        
        # The 7 Strassen multiplications at level 1:
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B01-B11)
        # M4 = A11*(B10-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)
        
        # Where A00 = rows 0-1, cols 0-1 of A, etc.
        
        # For each level-1 multiplication, we do a 2x2x2 Strassen
        # 2x2x2 Strassen:
        # M1' = (a00+a11)*(b00+b11)
        # M2' = (a10+a11)*b00
        # M3' = a00*(b01-b11)
        # M4' = a11*(b10-b00)
        # M5' = (a00+a01)*b11
        # M6' = (a10-a00)*(b00+b01)
        # M7' = (a01-a11)*(b10+b11)
        # c00 = M1' - M2' + M4' - M5'
        # c01 = M3' + M5'
        # c10 = M2' + M4'
        # c11 = M1' - M3' + M6' + M7'
        
        terms = []  # list of (u_idx, v_idx, w_idx, coeff)
        
        # Level 1 Strassen: 7 combinations
        # Each has: U_linear_combo, V_linear_combo
        # U combos (for 2x2 blocks of A):
        # Block (bi, bj) where bi,bj in {0,1}
        # A block (bi,bj) has entries at A[2*bi + di, 2*bj + dj] for di,dj in {0,1}
        
        # Level 1 U linear combinations (7 of them):
        # M1: A00+A11 -> blocks (0,0) and (1,1)
        # M2: A10+A11 -> blocks (1,0) and (1,1)
        # M3: A00 -> block (0,0)
        # M4: A11 -> block (1,1)
        # M5: A00+A01 -> blocks (0,0) and (0,1)
        # M6: A10-A00 -> blocks (1,0) and (0,0) with sign
        # M7: A01-A11 -> blocks (0,1) and (1,1) with sign
        
        # Level 1 V linear combinations:
        # M1: B00+B11
        # M2: B00
        # M3: B01-B11
        # M4: B10-B00
        # M5: B11
        # M6: B00+B01
        # M7: B10+B11
        
        # For each level-1 mult s (0..6), we have U_combo_s and V_combo_s
        # Each is a linear combination of 2x2 blocks
        # Then we do 2x2 Strassen on the resulting 2x2 matrices
        
        # Let's enumerate all terms explicitly
        
        # Level 1 mult s, with U blocks [(u_bi, u_bj, u_sign)] and V blocks [(v_bi, v_bj, v_sign)]
        # For each pair of blocks, we do 2x2x2 Strassen (7 scalar terms)
        
        l1_mults = [
            # (U_blocks, V_blocks) where each block is (bi, bj, sign)
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),  # M1
            ([(1,0,1), (1,1,1)], [(0,0,1)]),            # M2
            ([(0,0,1)], [(0,1,1), (1,1,-1)]),           # M3
            ([(1,1,1)], [(1,0,1), (0,0,-1)]),           # M4
            ([(0,0,1), (0,1,1)], [(1,1,1)]),            # M5
            ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]),  # M6
            ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)]),  # M7
        ]
        
        # For each level-1 mult, for each pair of U-block and V-block,
        # do 2x2x2 Strassen
        for s, (u_blocks, v_blocks) in enumerate(l1_mults):
            for (ubi, ubj, usign) in u_blocks:
                for (vbi, vbj, vsign) in v_blocks:
                    coeff = usign * vsign
                    # Do 2x2x2 Strassen for blocks at positions (ubi,ubj) and (vbi,vbj)
                    # This contributes to C blocks
                    # The 2x2x2 Strassen has 7 scalar multiplications
                    # Each scalar mult gives contributions to c00,c01,c10,c11 of the 2x2 result
                    # The 2x2 result goes into C block (ubi, vbj)
                    
                    # 2x2 Strassen terms:
                    # M1' = (a00+a11)*(b00+b11) -> c00 += M1', c11 -= M1'
                    # M2' = (a10+a11)*b00 -> c00 -= M2', c10 += M2'
                    # M3' = a00*(b01-b11) -> c01 += M3', c11 -= M3'
                    # M4' = a11*(b10-b00) -> c00 += M4', c10 += M4'
                    # M5' = (a00+a01)*b11 -> c00 -= M5', c01 += M5'
                    # M6' = (a10-a00)*(b00+b01) -> c11 += M6'
                    # M7' = (a01-a11)*(b10+b11) -> c11 += M7'
                    
                    # 2x2 Strassen U,V combos:
                    l2_mults = [
                        # (U_entries, V_entries) where entry is (di, dj, sign)
                        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),  # M1'
                        ([(1,0,1), (1,1,1)], [(0,0,1)]),            # M2'
                        ([(0,0,1)], [(0,1,1), (1,1,-1)]),           # M3'
                        ([(1,1,1)], [(1,0,1), (0,0,-1)]),           # M4'
                        ([(0,0,1), (0,1,1)], [(1,1,1)]),            # M5'
                        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]),  # M6'
                        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)]),  # M7'
                    ]
                    
                    # W contributions for each 2x2 Strassen mult:
                    # M1': c00 += 1, c11 -= 1
                    # M2': c00 -= 1, c10 += 1
                    # M3': c01 += 1, c11 -= 1
                    # M4': c00 += 1, c10 += 1
                    # M5': c00 -= 1, c01 += 1
                    # M6': c11 += 1
                    # M7': c11 += 1
                    l2_w = [
                        [(0,0,1), (1,1,-1)],   # M1'
                        [(0,0,-1), (1,0,1)],   # M2'
                        [(0,1,1), (1,1,-1)],   # M3'
                        [(0,0,1), (1,0,1)],    # M4'
                        [(0,0,-1), (0,1,1)],   # M5'
                        [(1,1,1)],             # M6'
                        [(1,1,1)],             # M7'
                    ]
                    
                    for t, (u_entries, v_entries) in enumerate(l2_mults):
                        for (udi, udj, ues) in u_entries:
                            for (vdi, vdj, ves) in v_entries:
                                # U index: A[2*ubi + udi, 2*ubj + udj]
                                u_idx = (2*ubi + udi) * 4 + (2*ubj + udj)
                                # V index: B[2*vbi + vdi, 2*vbj + vdj]
                                v_idx = (2*vbi + vdi) * 4 + (2*vbj + vdj)
                                c_coeff = coeff * ues * ves
                                for (wci, wcj, wcs) in l2_w[t]:
                                    # W index: C[2*ubi + wci, 2*vbj + wcj]
                                    w_idx = (2*ubi + wci) * 4 + (2*vbj + wcj)
                                    total_coeff = c_coeff * wcs
                                    terms.append((u_idx, v_idx, w_idx, total_coeff))
        
        # Now combine terms with same (u_idx, v_idx, w_idx)
        from collections import defaultdict
        term_map = defaultdict(float)
        for u_idx, v_idx, w_idx, coeff in terms:
            term_map[(u_idx, v_idx, w_idx)] += coeff
        
        # Build U, V, W matrices
        # Rank is the number of distinct (u_idx, v_idx, w_idx) with nonzero coeff
        # But we can have multiple terms with same u,v,w that cancel or add
        # Actually, each term is a rank-1 tensor, so we need to keep them separate
        # unless we can merge them
        
        # For now, just use all terms
        # But first, let's see how many unique (u,v,w) tuples we have
        unique_tuples = set()
        for u_idx, v_idx, w_idx, coeff in terms:
            if abs(coeff) > 1e-10:
                unique_tuples.add((u_idx, v_idx, w_idx))
        
        # The rank is the number of terms, but we can potentially merge
        # For a proper tensor decomposition, each rank-1 term is separate
        # So rank = len(terms) after combining same (u,v,w)
        
        # Let's just build it with all terms
        # Actually, we should combine terms with same (u,v,w) since they contribute to the same tensor entry
        # But in tensor decomposition, each rank-1 term is U[:,c] ⊗ V[:,c] ⊗ W[:,c]
        # So if two terms have the same (u,v,w) but different coeffs, we can merge them
        
        # Build the decomposition
        rank = len(term_map)
        U = [[0.0] * rank for _ in range(16)]
        V = [[0.0] * rank for _ in range(16)]
        W = [[0.0] * rank for _ in range(16)]
        
        col = 0
        for (u_idx, v_idx, w_idx), coeff in term_map.items():
            if abs(coeff) < 1e-10:
                continue
            U[u_idx][col] = 1.0
            V[v_idx][col] = 1.0
            W[w_idx][col] = coeff
            col += 1
        
        # Trim if needed
        if col < rank:
            U = [row[:col] for row in U]
            V = [row[:col] for row in V]
            W = [row[:col] for row in W]
            rank = col
        
        return U, V, W, rank
    
    def build_4x4x3_schoolbook():
        """Build schoolbook decomposition for 4x4x3."""
        rank = 4 * 4 * 3  # 48
        U = [[0.0] * rank for _ in range(16)]
        V = [[0.0] * rank for _ in range(12)]  # 4*3
        W = [[0.0] * rank for _ in range(12)]  # 3*4
        
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    U[i*4+j][col] = 1.0
                    V[j*3+k][col] = 1.0
                    W[k*4+i][col] = 1.0
                    col += 1
        
        return U, V, W, rank
    
    # Try the Strassen approach
    try:
        U1, V1, W1, r1 = build_4x4x4_strassen()
        U2, V2, W2, r2 = build_4x4x3_schoolbook()
        
        # Verify each part
        def verify_part4x4x4(U, V, W):
            rank = len(U[0])
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        total = 0
                        for c in range(rank):
                            total += U[i*4+j][c] * V[j*4+k][c] * W[k*4+i][c]
                        if abs(total - 1) > 1e-9:
                            return False
            return True
        
        if not verify_part4x4x4(U1, V1, W1):
            # Strassen decomposition is wrong, fall back to schoolbook
            pass
        else:
            # Combine: for 4x4x7, U has 16 rows, V has 28 rows, W has 28 rows
            # Part 1: 4x4x4 (cols 0-3 of B)
            # Part 2: 4x4x3 (cols 4-6 of B)
            
            total_rank = r1 + r2
            U_full = [[0.0] * total_rank for _ in range(16)]
            V_full = [[0.0] * total_rank for _ in range(28)]  # 4*7
            W_full = [[0.0] * total_rank for _ in range(28)]  # 7*4
            
            # Part 1
            for i in range(16):
                for c in range(r1):
                    U_full[i][c] = U1[i][c]
            for j in range(4):
                for k in range(4):
                    for c in range(r1):
                        V_full[j*7+k][c] = V1[j*4+k][c]
            for k in range(4):
                for i in range(4):
                    for c in range(r1):
                        W_full[k*4+i][c] = W1[k*4+i][c]
            
            # Part 2 (offset by r1)
            for i in range(16):
                for c in range(r2):
                    U_full[i][r1+c] = U2[i][c]
            for j in range(4):
                for k in range(3):
                    for c in range(r2):
                        V_full[j*7+4+k][r1+c] = V2[j*3+k][c]
            for k in range(3):
                for i in range(4):
                    for c in range(r2):
                        W_full[(4+k)*4+i][r1+c] = W2[k*4+i][c]
            
            # Verify full decomposition
            if verify(U_full, V_full, W_full):
                # Convert to integers if possible
                for mat in [U_full, V_full, W_full]:
                    for row in mat:
                        for idx, val in enumerate(row):
                            if abs(val - round(val)) < 1e-9:
                                row[idx] = int(round(val))
                            elif abs(2*val - round(2*val)) < 1e-9:
                                row[idx] = round(2*val) / 2
                return {"U": U_full, "V": V_full, "W": W_full}
    
    except Exception as e:
        pass
    
    # Fallback: schoolbook
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    if verify(U_sb, V_sb, W_sb):
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}

# EVOLVE-BLOCK-END
