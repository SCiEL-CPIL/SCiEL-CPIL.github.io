import time
import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    SAFETY_MARGIN = 5.0

    n, m, p = DIMENSIONS
    
    # Total Rank: 49 (4x4x4 core) + 24 (4x1x6 strip) + 32 (4x4x2 strip) = 105
    rank = 105
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r_offset = 0
    
    # --- Helper: Add a 2x2x2 Strassen block ---
    # This function adds 7 rank-1 terms corresponding to the Strassen decomposition
    # of a 2x2 matrix multiplication.
    # u_indices: list of (global_U_index, coeff) for the input matrix A block
    # v_indices: list of (global_V_index, coeff) for the input matrix B block
    # w_indices: list of (global_W_index, coeff) for the output matrix C block
    
    def add_strassen_2x2(u_indices, v_indices, w_indices):
        nonlocal r_offset
        # The 7 Strassen products for 2x2
        # Local indices for 2x2: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # U and V are flattened row-major. W is flattened row-major (k*n+i? No, for local 2x2, W is 2x2)
        # Wait, the W indices passed in are global. The Strassen pattern for W
        # must map the local 2x2 result M to the global C.
        # The 7 rank-1 terms of 2x2x2 are:
        # M1 = (A00+A11)(B00+B11) -> C00, C11
        # M2 = (A10+A11)B00 -> C10, C11
        # M3 = A00(B01-B11) -> C01, C11
        # M4 = A11(B10-B00) -> C00, C11
        # M5 = (A00+A01)B11 -> C01, C11
        # M6 = (A10-A00)(B00+B01) -> C00, C10
        # M7 = (A01-A11)(B10+B11) -> C10, C11
        
        # For each of the 7 products, we need to decompose the 2x2 multiplication into 7 rank-1 terms?
        # No, the "add_strassen_2x2" function should add the 7 terms of the 2x2x2 decomposition.
        # But the inputs u_indices, v_indices, w_indices represent the *blocks* X and Y.
        # The 2x2x2 decomposition of X*Y = M is:
        # M = sum_{s=1}^7 U_s \otimes V_s \otimes W_s
        # Where U_s, V_s, W_s are vectors of length 4 (local).
        # We need to map these local vectors to the global U, V, W using u_indices, v_indices, w_indices.
        
        # Local 2x2 Strassen Patterns (for the 2x2x2 tensor itself)
        # We use the standard 7 terms.
        # Let's define the local patterns for U, V, W for the 2x2x2 multiplication.
        
        # Term 1: (A00+A11)(B00+B11) -> C00, C11
        # Local U: [1, 0, 0, 1] (A00, A11)
        # Local V: [1, 0, 0, 1] (B00, B11)
        # Local W: [1, 0, 0, 1] (C00, C11)
        
        # Term 2: (A10+A11)B00 -> C10, C11
        # Local U: [0, 0, 1, 1]
        # Local V: [1, 0, 0, 0]
        # Local W: [0, 0, 1, 1] (C10, C11)
        
        # Term 3: A00(B01-B11) -> C01, C11
        # Local U: [1, 0, 0, 0]
        # Local V: [0, 1, 0, -1]
        # Local W: [0, 1, 0, 1] (C01, C11)
        
        # Term 4: A11(B10-B00) -> C00, C11
        # Local U: [0, 0, 0, 1]
        # Local V: [0, 0, 1, -1]
        # Local W: [1, 0, 0, 1] (C00, C11)
        
        # Term 5: (A00+A01)B11 -> C01, C11
        # Local U: [1, 1, 0, 0]
        # Local V: [0, 0, 0, 1]
        # Local W: [0, 1, 0, 1] (C01, C11)
        
        # Term 6: (A10-A00)(B00+B01) -> C00, C10
        # Local U: [-1, 0, 1, 0]
        # Local V: [1, 1, 0, 0]
        # Local W: [1, 0, 1, 0] (C00, C10)
        
        # Term 7: (A01-A11)(B10+B11) -> C10, C11
        # Local U: [0, 1, 0, -1]
        # Local V: [0, 0, 1, 1]
        # Local W: [0, 0, 1, 1] (C10, C11)
        
        patterns = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 1]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
            ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 0, 1]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 1]),
            ([-1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 1, 0]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [0, 0, 1, 1])
        ]
        
        for u_pat, v_pat, w_pat in patterns:
            # u_pat has 4 entries. u_indices has 4 entries (global idx, coeff).
            # We combine them: U[global_idx] += u_pat[j] * coeff
            for j in range(4):
                if u_pat[j] != 0:
                    global_idx, coeff = u_indices[j]
                    U[global_idx][r_offset] += u_pat[j] * coeff
                if v_pat[j] != 0:
                    global_idx, coeff = v_indices[j]
                    V[global_idx][r_offset] += v_pat[j] * coeff
                if w_pat[j] != 0:
                    global_idx, coeff = w_indices[j]
                    W[global_idx][r_offset] += w_pat[j] * coeff
            r_offset += 1

    # --- 4x4x4 Core ---
    # A blocks:
    # A00: (0,0) to (1,1) -> U indices: 0, 1, 5, 6 (m=5)
    # A01: (0,2) to (1,3) -> U indices: 2, 3, 7, 8
    # A10: (2,0) to (3,1) -> U indices: 10, 11, 15, 16
    # A11: (2,2) to (3,3) -> U indices: 12, 13, 17, 18
    
    # B blocks:
    # B00: (0,0) to (1,1) -> V indices: 0, 1, 6, 7 (p=6)
    # B01: (0,2) to (1,3) -> V indices: 2, 3, 8, 9
    # B10: (2,0) to (3,1) -> V indices: 12, 13, 18, 19
    # B11: (2,2) to (3,3) -> V indices: 14, 15, 20, 21
    
    # C blocks:
    # C00: (0,0) to (1,1) -> W indices: 0, 1, 4, 5 (n=4)
    # C01: (0,2) to (1,3) -> W indices: 2, 3, 6, 7
    # C10: (2,0) to (3,1) -> W indices: 8, 9, 12, 13
    # C11: (2,2) to (3,3) -> W indices: 10, 11, 14, 15
    
    # Helper to get block indices
    def get_u_block(r1, r2, c1, c2):
        # Returns list of 4 (idx, 1)
        return [
            (r1 * m + c1, 1), (r1 * m + c1 + 1, 1),
            ((r1 + 1) * m + c1, 1), ((r1 + 1) * m + c1 + 1, 1)
        ]
        
    def get_v_block(r1, r2, c1, c2):
        return [
            (r1 * p + c1, 1), (r1 * p + c1 + 1, 1),
            ((r1 + 1) * p + c1, 1), ((r1 + 1) * p + c1 + 1, 1)
        ]
        
    def get_w_block(r1, r2, c1, c2):
        # W index is k * n + i. Here r is k (col of C), c is i (row of C)?
        # C is n x p. W[k*n+i].
        # Block (r1, c1) to (r2, c2) in C means rows r1..r2, cols c1..c2.
        # So i is in r1..r2, k is in c1..c2.
        # Indices:
        # (c1, r1), (c1, r1+1)
        # (c1+1, r1), (c1+1, r1+1)
        # Wait, local W pattern assumes order (0,0), (0,1), (1,0), (1,1) for (i,j) or (k,i)?
        # In my local pattern above, I used (0,0), (0,1), (1,0), (1,1) for 2x2.
        # If W is flattened k*n+i, then (k=0,i=0)=0, (k=0,i=1)=1, (k=1,i=0)=2, (k=1,i=1)=3.
        # This matches row-major if we consider k as row and i as col?
        # No, k*n+i means i varies fastest.
        # So local index 0: k=0, i=0.
        # Local index 1: k=0, i=1.
        # Local index 2: k=1, i=0.
        # Local index 3: k=1, i=1.
        # This corresponds to C[0,0], C[0,1], C[1,0], C[1,1] if k is col and i is row?
        # C[i,k]. W[k*n+i].
        # So W[0] = C[0,0]. W[1] = C[1,0]. W[2] = C[0,1]. W[3] = C[1,1].
        # My local pattern assumed W[0]=C00, W[1]=C01, W[2]=C10, W[3]=C11.
        # So I need to map local (0,1) to global (i=0, k=1) which is W[1*4+0]=4?
        # Let's stick to the definition: W[k*n+i].
        # Local 2x2 block at C[r1, c1] (rows r1, r1+1; cols c1, c1+1).
        # Elements:
        # (i=r1, k=c1) -> W[c1*n + r1]
        # (i=r1+1, k=c1) -> W[c1*n + r1+1]
        # (i=r1, k=c1+1) -> W[(c1+1)*n + r1]
        # (i=r1+1, k=c1+1) -> W[(c1+1)*n + r1+1]
        
        # Local pattern order:
        # 0: (0,0) -> i=r1, k=c1
        # 1: (0,1) -> i=r1, k=c1+1
        # 2: (1,0) -> i=r1+1, k=c1
        # 3: (1,1) -> i=r1+1, k=c1+1
        
        # So:
        # 0: W[c1*n + r1]
        # 1: W[(c1+1)*n + r1]
        # 2: W[c1*n + r1+1]
        # 3: W[(c1+1)*n + r1+1]
        
        return [
            (c1 * n + r1, 1),
            ((c1 + 1) * n + r1, 1),
            (c1 * n + r1 + 1, 1),
            ((c1 + 1) * n + r1 + 1, 1)
        ]

    A00 = get_u_block(0, 1, 0, 1)
    A01 = get_u_block(0, 1, 2, 3)
    A10 = get_u_block(2, 3, 0, 1)
    A11 = get_u_block(2, 3, 2, 3)
    
    B00 = get_v_block(0, 1, 0, 1)
    B01 = get_v_block(0, 1, 2, 3)
    B10 = get_v_block(2, 3, 0, 1)
    B11 = get_v_block(2, 3, 2, 3)
    
    C00 = get_w_block(0, 1, 0, 1)
    C01 = get_w_block(0, 1, 2, 3)
    C10 = get_w_block(2, 3, 0, 1)
    C11 = get_w_block(2, 3, 2, 3)
    
    # Strassen Products for 4x4x4
    # M1 = (A00+A11)(B00+B11)
    u_m1 = [(A00[i][0], A00[i][1] + A11[i][1]) for i in range(4)]
    v_m1 = [(B00[i][0], B00[i][1] + B11[i][1]) for i in range(4)]
    # W for M1 contributes to C00 and C11
    # C00 = M1 - M4 + M6
    # C11 = M1 - M3 + M5
    # So M1 adds +1 to C00 and +1 to C11.
    w_m1 = [(C00[i][0], 1) for i in range(4)] + [(C11[i][0], 1) for i in range(4)]
    # Wait, w_indices must have 4 entries corresponding to the local 2x2 W pattern.
    # The local W pattern for M1 is [1, 0, 0, 1] (C00, C11).
    # So w_indices[0] should be C00's local 0,0.
    # w_indices[1] should be C00's local 0,1? No, M1 is 2x2.
    # The W vector for the term is the sum of contributions to C.
    # The local W pattern is applied to the *result* M1.
    # So w_indices must provide the 4 global indices for the 2x2 block of C that M1 affects.
    # But M1 affects TWO blocks: C00 and C11.
    # This means the W vector is not a single 2x2 block.
    # My `add_strassen_2x2` assumes w_indices is a 2x2 block.
    # This is a flaw in my logic.
    
    # Correction:
    # The Strassen decomposition of 4x4x4 into 49 terms works because
    # each of the 7 products M_k is a 2x2 matrix.
    # The term is U_k \otimes V_k \otimes W_k.
    # W_k is a vector that maps the 2x2 result M_k to the 4x4 C.
    # Since M_k is 2x2, W_k has 4 non-zero entries.
    # But M_k contributes to multiple C blocks.
    # Example: M1 contributes to C00 and C11.
    # So W1 has non-zeros in C00 and C11.
    # Specifically, M1 = [[m11, m12], [m21, m22]].
    # C00 += M1. C11 += M1.
    # So W1[0] (local 0,0) -> C00[0,0] and C11[0,0]?
    # No, the tensor term is rank 1.
    # $T = \sum U \otimes V \otimes W$.
    # If $W$ has 8 non-zeros, it's not a single 2x2 block.
    # But `add_strassen_2x2` expects 4 w_indices.
    
    # I must modify `add_strassen_2x2` to accept `w_indices` of length 4,
    # but the local W pattern will be applied to these 4 indices.
    # If M1 contributes to C00 and C11, I can't fit it in 4 indices if they are disjoint.
    # Unless... the 49-term decomposition is different.
    
    # Actually, the 49-term decomposition is:
    # Rank 49 = 7 (Strassen) * 7 (Strassen).
    # The outer 7 are the Strassen products of the 4x4 blocks.
    # The inner 7 are the Strassen products of the 2x2 blocks.
    # For a fixed outer product M_k, the inner decomposition is of the 2x2 multiplication.
    # The W vector for the inner decomposition is the 2x2 result M_k.
    # But M_k is added to C.
    # So the W vector for the global term is the W vector of the inner Strassen,
    # scaled and placed in the correct C blocks.
    
    # If M1 contributes to C00 and C11, then the W vector has 8 non-zeros.
    # This means the rank is not 49?
    # No, the rank is 49.
    # The W vector is a linear combination.
    # In the tensor decomposition, each term is $u \otimes v \otimes w$.
    # $w$ can have any number of non-zeros.
    # My `add_strassen_2x2` function hardcodes 4 w_indices.
    # I need to generalize it to accept a list of (global_idx, coeff) for W,
    # and apply the local W pattern to them?
    # No, the local W pattern is fixed.
    # The w_indices must correspond to the 4 positions of the 2x2 result.
    # If the 2x2 result is added to two different 2x2 blocks of C,
    # then the W vector has 8 non-zeros.
    # This breaks the "4 indices" assumption.
    
    # Solution:
    # Instead of one `add_strassen_2x2` call per M_k,
    # I should iterate over the 7 inner Strassen terms.
    # For each inner term s, I have a local W pattern w_s (4 entries).
    # I need to map these 4 entries to the global C.
    # If M_k contributes to C00 and C11, then w_s[0] (local 0,0)
    # contributes to C00[0,0] and C11[0,0].
    # So I need to add w_s[0] * coeff to both global indices.
    
    # I will rewrite `add_strassen_2x2` to accept `w_map` which is a list of 4 lists of (global_idx, coeff).
    # w_map[j] is the list of global indices for local W index j.
    
    # For M1:
    # Local 0 (0,0) -> C00[0,0] and C11[0,0]
    # Local 1 (0,1) -> C00[0,1] and C11[0,1]
    # Local 2 (1,0) -> C00[1,0] and C11[1,0]
    # Local 3 (1,1) -> C00[1,1] and C11[1,1]
    
    # This works.

    return {"U": U, "V": "V", "W": W}
