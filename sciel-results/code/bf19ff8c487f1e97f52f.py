"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 110
    
    # Strategy: Structured Kronecker decomposition
    # We view the multiplication (n x m) * (m x p) -> (n x p)
    # m = 4 can be viewed as 2 x 2.
    # p = 8 can be viewed as 2 x 4.
    # So we are effectively doing:
    # A (3 x 2 x 2) * B (2 x 2 x 2 x 4) -> C (3 x 2 x 4)
    
    # Let's decompose the indices:
    # A[i, (u, v)] where i in 0..2, u in 0..1, v in 0..1
    # B[(u, v), (w, x)] where u,v in 0..1, w in 0..1, x in 0..3
    # C[i, (w, x)] where i in 0..2, w in 0..1, x in 0..3
    
    # The sum is over u, v.
    # C[i, w, x] = sum_{u,v} A[i, u, v] * B[u, v, w, x]
    
    # We can group by the "outer" dimensions i, w, x and sum over u, v.
    # This is equivalent to matrix multiplication where the "inner" dimension is 4 (u,v).
    # But we can use Strassen for the 2x2x2 part?
    # Let's fix x. Then we have a 3x2 (i,u) * 2x2 (v,w)?? No.
    
    # Let's use the property:
    # Rank of (n1 x m1) * (m1 x p1) combined with (n2 x m2) * (m2 x p2) etc.
    
    # A better structural approach:
    # Split m=4 into two 2s: m = m1 + m2, m1=2, m2=2.
    # Split p=8 into two 4s: p = p1 + p2, p1=4, p2=4.
    # n=3 stays as is.
    
    # This gives 4 blocks of matrix multiplication:
    # 1. (3x2) * (2x4) -> (3x4). Rank 24.
    # 2. (3x2) * (2x4) -> (3x4). Rank 24.
    # 3. (3x2) * (2x4) -> (3x4). Rank 24.
    # 4. (3x2) * (2x4) -> (3x4). Rank 24.
    # Total Rank 96. No gain.
    
    # To gain, we need to exploit Strassen.
    # Strassen works for 2x2 * 2x2.
    # Can we map (3x4) * (4x8) to multiple 2x2 * 2x2 problems?
    
    # Let's try:
    # n=3: split into 2+1.
    # m=4: split into 2+2.
    # p=8: split into 4+4.
    
    # Blocks:
    # n-blocks: [0,2), [2,3)
    # m-blocks: [0,2), [2,4)
    # p-blocks: [0,4), [4,8)
    
    # Block (i,j,k) involves:
    # A[i, j] (size ni x mj)
    # B[j, k] (size mj x pk)
    # C[i, k] (size ni x pk)
    
    # 1. i=[0,2), j=[0,2), k=[0,4): (2x2)*(2x4) -> (2x4). Rank 16 (Schoolbook).
    #    Can we do better? (2x2)*(2x4) is two (2x2)*(2x2) multiplications.
    #    Rank of (2x2)*(2x2) is 7 (Strassen).
    #    So two of them is 14.
    #    Wait, (2x4) output can be seen as two (2x2) outputs.
    #    So yes, Rank 14.
    
    # 2. i=[0,2), j=[0,2), k=[4,8): (2x2)*(2x4) -> (2x4). Rank 14.
    
    # 3. i=[0,2), j=[2,4), k=[0,4): (2x2)*(2x4) -> (2x4). Rank 14.
    
    # 4. i=[0,2), j=[2,4), k=[4,8): (2x2)*(2x4) -> (2x4). Rank 14.
    
    # 5. i=[2,3), j=[0,2), k=[0,4): (1x2)*(2x4) -> (1x4). Rank 8 (Schoolbook).
    #    (1x2)*(2x4) is two (1x2)*(2x2) -> (1x2).
    #    Rank of (1x2)*(2x2) is 4? 
    #    C = A B. A is 1x2 [a,b]. B is 2x2.
    #    C = [a B0 + b B1].
    #    This is just linear combination. Rank is 4 (schoolbook for 1x2x2 is 4).
    #    Is there a better way? No, bilinear rank of 1x2x2 is 4.
    
    # 6. i=[2,3), j=[0,2), k=[4,8): (1x2)*(2x4) -> (1x4). Rank 8.
    
    # 7. i=[2,3), j=[2,4), k=[0,4): (1x2)*(2x4) -> (1x4). Rank 8.
    
    # 8. i=[2,3), j=[2,4), k=[4,8): (1x2)*(2x4) -> (1x4). Rank 8.
    
    # Total Rank = 14 + 14 + 14 + 14 + 8 + 8 + 8 + 8 = 88.
    # This is an improvement over 96!
    
    # Let's implement this.
    
    # Helper to build Strassen decomposition for 2x2 * 2x2
    def strassen_2x2(U, V, W, i0, j0, k0, col_start):
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # Indices:
        # A indices: i in [i0, i0+2), j in [j0, j0+2)
        # B indices: j in [j0, j0+2), k in [k0, k0+2)
        # C indices: i in [i0, i0+2), k in [k0, k0+2)
        
        # Strassen terms:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B10)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        
        # C00 = M1 - M5 + M6 - M7
        # C01 = M4 + M5
        # C10 = M2 + M3
        # C11 = M1 - M3 + M4 + M7
        
        # Coefficients for A (2x2), B (2x2), C (2x2)
        # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        
        terms = [
            # M1
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            # M2
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
            # M3
            ([1, 0, 0, 0], [0, -1, 1, 0], [0, 1, 0, 1]),
            # M4
            ([0, 0, 0, 1], [0, 1, -1, 0], [0, 1, 1, 0]),
            # M5
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
            # M6
            ([1, 0, -1, 0], [1, 0, 1, 0], [0, 0, 1, 0]),
            # M7
            ([0, 1, 0, -1], [0, 1, 0, 1], [0, 0, 0, 1]),
        ]
        
        col = col_start
        for a_coeffs, b_coeffs, c_coeffs in terms:
            for ai, ac in enumerate(a_coeffs):
                if ac != 0:
                    i_idx = i0 + ai // 2
                    j_idx = j0 + ai % 2
                    U[i_idx * m + j_idx, col] = ac
            for bi, bc in enumerate(b_coeffs):
                if bc != 0:
                    j_idx = j0 + bi // 2
                    k_idx = k0 + bi % 2
                    V[j_idx * p + k_idx, col] = bc
            for ci, cc in enumerate(c_coeffs):
                if cc != 0:
                    i_idx = i0 + ci // 2
                    k_idx = k0 + ci % 2
                    W[k_idx * n + i_idx, col] = cc
            col += 1
        return col
    
    # Helper to build schoolbook decomposition for ni x mj * mj x pk -> ni x pk
    def schoolbook_block(U, V, W, i0, j0, k0, ni, mj, pk, col_start):
        col = col_start
        for i in range(ni):
            for j in range(mj):
                for k in range(pk):
                    U[(i0+i)*m + (j0+j), col] = 1
                    V[(j0+j)*p + (k0+k), col] = 1
                    W[(k0+k)*n + (i0+i), col] = 1
                    col += 1
        return col
    
    # Build the full decomposition
    # i_blocks: [(0, 2), (2, 3)]
    # j_blocks: [(0, 2), (2, 4)]
    # k_blocks: [(0, 4), (4, 8)]
    
    i_blocks = [(0, 2), (2, 3)]
    j_blocks = [(0, 2), (2, 4)]
    k_blocks = [(0, 4), (4, 8)]
    
    # Calculate total rank
    total_rank = 0
    blocks_info = []
    for i0, i1 in i_blocks:
        for j0, j1 in j_blocks:
            for k0, k1 in k_blocks:
                ni = i1 - i0
                mj = j1 - j0
                pk = k1 - k0
                if ni == 2 and mj == 2 and pk == 4:
                    # This is (2x2)*(2x4).
                    # Split pk into two 2s.
                    # So two Strassen blocks.
                    total_rank += 14
                    blocks_info.append((i0, j0, k0, ni, mj, pk, 'strassen_split'))
                elif ni == 1 and mj == 2 and pk == 4:
                    # This is (1x2)*(2x4).
                    # Split pk into two 2s.
                    # Two schoolbook (1x2)*(2x2). Rank 4 each. Total 8.
                    total_rank += 8
                    blocks_info.append((i0, j0, k0, ni, mj, pk, 'schoolbook_split'))
                else:
                    # Fallback schoolbook
                    total_rank += ni * mj * pk
                    blocks_info.append((i0, j0, k0, ni, mj, pk, 'schoolbook'))
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for (i0, j0, k0, ni, mj, pk, method) in blocks_info:
        if method == 'strassen_split':
            # Split pk into two 2s: k0..k0+2, k0+2..k0+4
            col = strassen_2x2(U, V, W, i0, j0, k0, col)
            col = strassen_2x2(U, V, W, i0, j0, k0 + 2, col)
        elif method == 'schoolbook_split':
            # Split pk into two 2s
            col = schoolbook_block(U, V, W, i0, j0, k0, ni, mj, 2, col)
            col = schoolbook_block(U, V, W, i0, j0, k0 + 2, ni, mj, 2, col)
        else:
            col = schoolbook_block(U, V, W, i0, j0, k0, ni, mj, pk, col)
    
    if col != total_rank:
        # Fallback to full schoolbook if something went wrong
        U, V, W, total_rank = [], [], 0, n*m*p
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
    
    # Verify
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    is_valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(total_rank):
                    s += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    is_valid = False
                    break
            if not is_valid: break
        if not is_valid: break
    
    if not is_valid:
        # Fallback to schoolbook
        total_rank = n*m*p
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
