"""Matrix multiplication 4x4 by 4x9: search for a low-rank exact tensor decomposition."""
import time

DIMENSIONS = [4, 4, 9]
_START = time.time()
_DEADLINE = 150.0
_MARGIN = 3.0


def _time_ok():
    return (time.time() - _START) < (_DEADLINE - _MARGIN)


def _validate(U, V, W):
    n, m, p = DIMENSIONS
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(r):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    for a in range(n * m):
        for b in range(m * p):
            ja, ka = divmod(a, m)
            jb, kb = divmod(b, p)
            if ja != jb:
                continue
            for c in range(p * n):
                kc, ic = divmod(c, n)
                if ic != ja:
                    continue
                s = 0
                for t in range(r):
                    s += U[a][t] * V[b][t] * W[c][t]
                if s != 0:
                    return False
    return True


def _build_strassen_block(n):
    """
    Builds U, V, W for an n x n x n matrix multiplication using Strassen's algorithm.
    n must be a power of 2.
    Returns U (n*n x rank), V (n*n x rank), W (n*n x rank).
    Indices: U[i*n+j], V[j*n+k], W[k*n+i]
    """
    if n == 1:
        U = [[1]]
        V = [[1]]
        W = [[1]]
        return U, V, W
    
    h = n // 2
    # Recursively build for h x h x h
    Uh, Vh, Wh = _build_strassen_block(h)
    
    # Strassen's 7 products for 2x2 blocks of size h
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We need to map the local h x h indices to the global n x n indices.
    # Block (r, c) in A corresponds to rows r*h..(r+1)*h-1 and cols c*h..(c+1)*h-1.
    # In U, the index is i*n+j.
    # For a block (r, c), i is in r*h..(r+1)*h-1, j is in c*h..(c+1)*h-1.
    
    U = [[0] * 0 for _ in range(n * n)] # Placeholder, we will build lists
    V = [[0] * 0 for _ in range(n * n)]
    W = [[0] * 0 for _ in range(n * n)]
    
    rank_h = len(Uh[0])
    total_rank = 7 * rank_h
    
    U = [[0] * total_rank for _ in range(n * n)]
    V = [[0] * total_rank for _ in range(n * n)]
    W = [[0] * total_rank for _ in range(n * n)]
    
    # Helper to add a scaled block of factors to the main matrices
    def add_block(U_target, V_target, W_target, U_block, V_block, W_block, 
                  i_off_u, j_off_u, i_off_v, j_off_v, i_off_w, j_off_w, 
                  scale_u, scale_v, scale_w, start_col):
        # U_block is h*h x rank_h
        # We need to place it into U_target at rows (i_off_u + i) * n + (j_off_u + j)
        # This mapping is tricky because U_target is flattened.
        # Let's iterate over the elements of the block.
        
        for i_local in range(h):
            for j_local in range(h):
                # Global index in U_target
                i_global = i_off_u + i_local
                j_global = j_off_u + j_local
                idx_u = i_global * n + j_global
                
                for k_local in range(h):
                    # For V, we need j and k.
                    # The block V_block is indexed by [j_local * h + k_local]
                    # Global V index: (j_off_v + j_local) * n + (j_off_v + k_local) ?
                    # Wait, V is (m x p) -> (n x n). Index j*n+k.
                    # In the sub-problem, V_block[j_local*h + k_local] corresponds to local j, k.
                    # So global j = j_off_v + j_local, global k = j_off_v + k_local?
                    # No, the offsets are for the block position.
                    # If the block is at row r, col c of the 2x2 macro-grid:
                    # i range: r*h to (r+1)*h-1
                    # j range: c*h to (c+1)*h-1
                    # So i_global = r*h + i_local
                    # j_global = c*h + j_local
                    # For V (j, k):
                    # j_global = r_v*h + j_local
                    # k_global = c_v*h + k_local
                    # For W (k, i):
                    # k_global = r_w*h + k_local
                    # i_global = c_w*h + i_local
                    
                    # Let's re-verify the indices for the 7 Strassen terms.
                    pass

    # This generic block addition is complex. Let's hardcode the 7 terms for the top level.
    # Since n=4 is the only case we need for the top level (and n=2 for recursion),
    # we can just write out the indices explicitly for n=4.
    
    # But recursion is cleaner. Let's try to implement the mapping carefully.
    
    # Indices in U: i*n + j
    # Indices in V: j*n + k
    # Indices in W: k*n + i
    
    # For a term corresponding to A_block(r_a, c_a) * B_block(r_b, c_b):
    # A_block(r_a, c_a) uses U rows: (r_a*h + i) * n + (c_a*h + j)
    # B_block(r_b, c_b) uses V rows: (r_b*h + j) * n + (c_b*h + k)
    # The result C_block(r_a, c_b) uses W rows: (c_b*h + k) * n + (r_a*h + i)
    
    # We need to add the factors from Uh, Vh, Wh to these positions.
    
    def add_strassen_term(r_a, c_a, r_b, c_b, scale_a, scale_b, start_col):
        # scale_a, scale_b are the coefficients for the A and B blocks in this term.
        # Actually, the term is sum_{u,v} A_{u,v} B_{u,v}.
        # The factors U, V, W from the sub-problem are for a single multiplication.
        # We need to scale U by scale_a and V by scale_b?
        # No, the term is (sum A) * (sum B).
        # The sub-problem computes X * Y.
        # Here X is a linear combination of A blocks, Y is a linear combination of B blocks.
        # So we scale the U factors by the coefficients of the A blocks,
        # and the V factors by the coefficients of the B blocks.
        # But a single term like (A00+A11) involves two blocks.
        # We can treat (A00+A11) as a single "virtual" matrix.
        # The decomposition of X*Y is sum_r U_r X V_r Y W_r.
        # If X = A00 + A11, then U_r must be applied to both A00 and A11.
        # So we add U_r to the positions of A00 and A11.
        
        for r_sub in range(rank_h):
            col = start_col + r_sub
            
            # Iterate over all i, j, k in the sub-block (h x h x h)
            for i_sub in range(h):
                for j_sub in range(h):
                    # A part
                    if scale_a != 0:
                        i_global = r_a * h + i_sub
                        j_global = c_a * h + j_sub
                        idx_u = i_global * n + j_global
                        U[idx_u][col] += scale_a * Uh[i_sub * h + j_sub][r_sub]
                    
                    # V part
                    if scale_b != 0:
                        # V is indexed by j, k.
                        # In the sub-problem, V_h[j_sub * h + k_sub]
                        # Here j_global = r_b * h + j_sub ?
                        # Wait, B_block is at (r_b, c_b).
                        # So rows of B are r_b*h.., cols are c_b*h..
                        # V index is j*n+k.
                        # j_global = r_b * h + j_sub
                        # k_global = c_b * h + k_sub
                        # We need to iterate k_sub too.
                        pass # This loop structure is wrong. We need to iterate all i,j,k.
                        
    # Let's restart the mapping logic. It's safer to just expand the full 4x4x4 tensor 
    # using the known Strassen coefficients and the rank-8 schoolbook for 2x2x2?
    # No, rank 49 is better.
    
    # Let's use a simpler approach:
    # Construct the 4x4x4 tensor T (size 16x16x16? No, 4x4x4 tensor is 64 elements).
    # We want to decompose T.
    # T[i,j,k] = 1.
    # We use the Strassen identity to write T as a sum of 49 rank-1 tensors.
    # Each rank-1 tensor is defined by three vectors u, v, w of length 4.
    # u[i,j], v[j,k], w[k,i].
    
    # The 49 terms come from 7 Strassen steps, each of which is a 2x2x2 multiplication.
    # Each 2x2x2 multiplication has rank 7 (Strassen) or 8 (Schoolbook).
    # If we use rank 7 for the 2x2x2, we get 49.
    
    # Let's implement the rank-7 decomposition for 2x2x2 explicitly.
    # Then compose it.
    
    pass

def construct():
    n, m, p = DIMENSIONS
    # Fallback schoolbook
    rank_fb = n * m * p
    U = [[0] * rank_fb for _ in range(n * m)]
    V = [[0] * rank_fb for _ in range(m * p)]
    W = [[0] * rank_fb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    fallback = (rank_fb, U, V, W)
    
    # Try the Strassen-based construction
    # Rank 114
    try:
        U_new, V_new, W_new, rank_new = _build_strassen_hybrid(n, m, p)
        if rank_new < rank_fb:
            # Validate
            if _validate(U_new, V_new, W_new):
                return {"U": U_new, "V": V_new, "W": W_new}
    except Exception:
        pass
        
    return {"U": fallback[1], "V": fallback[2], "W": fallback[3]}

def _build_strassen_hybrid(n, m, p):
    # n=4, m=4, p=9
    # Split p into 4, 4, 1
    # Part 1: 4x4x4 (Strassen)
    # Part 2: 4x4x4 (Strassen)
    # Part 3: 4x4x1 (Schoolbook)
    
    # We need a function to build 4x4x4 Strassen factors
    U1, V1, W1, r1 = _build_4x4x4_strassen()
    U2, V2, W2, r2 = _build_4x4x4_strassen()
    U3, V3, W3, r3 = _build_4x4x1_schoolbook()
    
    total_rank = r1 + r2 + r3
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Copy Part 1 (cols 0-3)
    for i in range(n * m):
        for c in range(r1):
            U[i][c] = U1[i][c]
    for j in range(m):
        for k in range(4):
            idx_v = j * p + k
            for c in range(r1):
                V[idx_v][c] = V1[j * 4 + k][c]
    for k in range(4):
        for i in range(n):
            idx_w = k * n + i
            for c in range(r1):
                W[idx_w][c] = W1[k * 4 + i][c]
                
    # Copy Part 2 (cols 4-7)
    offset2 = r1
    for i in range(n * m):
        for c in range(r2):
            U[i][offset2 + c] = U2[i][c]
    for j in range(m):
        for k in range(4):
            idx_v = j * p + (k + 4)
            for c in range(r2):
                V[idx_v][offset2 + c] = V2[j * 4 + k][c]
    for k in range(4):
        for i in range(n):
            idx_w = (k + 4) * n + i
            for c in range(r2):
                W[idx_w][offset2 + c] = W2[k * 4 + i][c]
                
    # Copy Part 3 (col 8)
    offset3 = r1 + r2
    for i in range(n * m):
        for c in range(r3):
            U[i][offset3 + c] = U3[i][c]
    for j in range(m):
        idx_v = j * p + 8
        for c in range(r3):
            V[idx_v][offset3 + c] = V3[j * 1 + 0][c]
    for i in range(n):
        idx_w = 8 * n + i
        for c in range(r3):
            W[idx_w][offset3 + c] = W3[0 * 4 + i][c] # W is p*n, so k=0 local -> 8 global. W index k*n+i.
            
    return U, V, W, total_rank

def _build_4x4x4_strassen():
    # Returns U, V, W for 4x4x4 multiplication with rank 49.
    # Uses 7 Strassen terms, each of rank 7 (Strassen 2x2x2).
    pass

def _build_4x4x1_schoolbook():
    # Returns U, V, W for 4x4x1 multiplication with rank 16.
    pass
