import numpy as np
import time

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """Return the 7 Strassen terms for 2x2 matrix multiplication.
    
    Each term is (A_coeffs, B_coeffs, C_coeffs) where each is a dict
    mapping (row, col) -> coefficient for the 2x2 matrices.
    """
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    terms = [
        # P1 = (a+d)(e+h)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1},
         {(0,0): 1, (1,1): 1}),
        # P2 = (c+d)e
        ({(1,0): 1, (1,1): 1}, {(0,0): 1},
         {(1,0): 1, (1,1): -1}),
        # P3 = a(f-h)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1},
         {(0,1): 1, (1,1): 1}),
        # P4 = d(g-e)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1},
         {(0,0): 1, (1,0): 1}),
        # P5 = (a+b)h
        ({(0,0): 1, (0,1): 1}, {(1,1): 1},
         {(0,0): -1, (0,1): 1}),
        # P6 = (c-a)(e+f)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1},
         {(1,1): 1}),
        # P7 = (b-d)(g+h)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1},
         {(0,0): 1}),
    ]
    return terms

def _generate_strassen_decomposition(n=4):
    """Recursively generate the Strassen-based rank decomposition for n x n matrix multiplication.
    
    For n=2: use the 7-term Strassen decomposition directly.
    For n=4: split into 2x2 blocks, apply Strassen at block level,
             then recursively decompose each 2x2 block product.
    
    Returns (U, V, W) as lists of lists with integer entries.
    U: shape (n*n, rank), V: shape (n*n, rank), W: shape (n*n, rank)
    Convention: U[i*n+k, r], V[k*n+j, r], W[j*n+i, r] for C[i,j]
    """
    if n == 2:
        terms = _strassen_2x2()
        rank = 7
        U = [[0] * rank for _ in range(4)]
        V = [[0] * rank for _ in range(4)]
        W = [[0] * rank for _ in range(4)]
        
        for t, (a_coeffs, b_coeffs, c_coeffs) in enumerate(terms):
            for (r, c), coeff in a_coeffs.items():
                U[r * 2 + c][t] += coeff
            for (r, c), coeff in b_coeffs.items():
                V[r * 2 + c][t] += coeff
            for (r, c), coeff in c_coeffs.items():
                W[c * 2 + r][t] += coeff  # W[j*2+i, t] for C[i,j]
        
        return U, V, W, rank
    
    # For n=4: use Strassen at the 2x2 block level
    half = n // 2  # 2
    terms = _strassen_2x2()
    outer_rank = 7
    inner_U, inner_V, inner_W, inner_rank = _generate_strassen_decomposition(half)
    total_rank = outer_rank * inner_rank
    
    U = [[0] * total_rank for _ in range(n * n)]
    V = [[0] * total_rank for _ in range(n * n)]
    W = [[0] * total_rank for _ in range(n * n)]
    
    for t, (a_blocks, b_blocks, c_blocks) in enumerate(terms):
        for s in range(inner_rank):
            rank_idx = t * inner_rank + s
            
            # For each block in A
            for (br, bc), block_coeff in a_blocks.items():
                for (er, ec) in [(0,0),(0,1),(1,0),(1,1)]:
                    elem_coeff = inner_U[er * half + ec][s]
                    if elem_coeff != 0:
                        row = 2 * br + er
                        col = 2 * bc + ec
                        U[row * n + col][rank_idx] += block_coeff * elem_coeff
            
            # For each block in B
            for (br, bc), block_coeff in b_blocks.items():
                for (er, ec) in [(0,0),(0,1),(1,0),(1,1)]:
                    elem_coeff = inner_V[er * half + ec][s]
                    if elem_coeff != 0:
                        row = 2 * br + er
                        col = 2 * bc + ec
                        V[row * n + col][rank_idx] += block_coeff * elem_coeff
            
            # For each block in C
            for (br, bc), block_coeff in c_blocks.items():
                for (er, ec) in [(0,0),(0,1),(1,0),(1,1)]:
                    # inner_W is indexed [j*2+i, s] for C[i,j]
                    # Here (br, bc) corresponds to block position (i_block, j_block) in C?
                    # In _strassen_2x2, c_coeffs maps (r,c) -> coeff for C[r,c].
                    # So (br, bc) is (row_block, col_block) of C.
                    # inner_W index for local (er, ec) is ec * half + er?
                    # Let's check the base case:
                    # W[c * 2 + r][t] += coeff. Here c is row of C, r is col of C?
                    # No, W[j*2+i, t] for C[i,j].
                    # So W index is j*2 + i.
                    # In base case: for (r, c) in c_coeffs: W[c * 2 + r][t] += coeff.
                    # So c is j, r is i.
                    # So W[j*2+i] is correct.
                    
                    # In recursive step:
                    # We are filling W for the 4x4 matrix.
                    # Global indices: I, J.
                    # Block indices: br, bc.
                    # Local indices: er, ec.
                    # I = 2*br + er, J = 2*bc + ec.
                    # W index should be J*4 + I = (2*bc + ec)*4 + (2*br + er).
                    
                    # The coefficient from inner_W:
                    # inner_W corresponds to the 2x2 decomposition.
                    # inner_W[local_j * 2 + local_i][s].
                    # local_j = ec, local_i = er.
                    elem_coeff = inner_W[ec * half + er][s]
                    if elem_coeff != 0:
                        row = 2 * br + er  # I
                        col = 2 * bc + ec  # J
                        W[col * n + row][rank_idx] += block_coeff * elem_coeff
    
    return U, V, W, total_rank

def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication via recursive Strassen.
    """
    start_time = time.time()
    DEADLINE = 140
    
    n = 4
    
    # Generate decomposition programmatically
    U, V, W, rank = _generate_strassen_decomposition(n)
    
    # Verify the decomposition
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    np.random.seed(42)
    A_test = np.random.randn(4, 4)
    B_test = np.random.randn(4, 4)
    C_expected = A_test @ B_test
    
    U4 = U_np.reshape(4, 4, rank)   # [i, k, r]
    V4 = V_np.reshape(4, 4, rank)   # [k, j, r]
    W4 = W_np.reshape(4, 4, rank)   # [j, i, r]
    
    u_vals = np.einsum('ikr,ik->ir', U4, A_test)
    v_vals = np.einsum('kjr,kj->jr', V4, B_test)
    C_computed = np.einsum('jir,ir,jr->ij', W4, u_vals, v_vals)
    
    max_err = np.max(np.abs(C_computed - C_expected))
    
    if max_err > 1e-9:
        # Fallback: trivial rank-64 decomposition
        rank_fallback = 64
        U_f = [[0] * rank_fallback for _ in range(16)]
        V_f = [[0] * rank_fallback for _ in range(16)]
        W_f = [[0] * rank_fallback for _ in range(16)]
        
        for i in range(4):
            for k in range(4):
                for j in range(4):
                    r = i * 16 + k * 4 + j
                    U_f[i*4+k][r] = 1
                    V_f[k*4+j][r] = 1
                    W_f[j*4+i][r] = 1
        
        return {"U": U_f, "V": V_f, "W": W_f}
    
    # If we have time, we could try to reduce rank, but for 4x4, 49 is likely minimal.
    # The previous _reduce_rank was very slow. We will skip it to ensure we finish fast.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
