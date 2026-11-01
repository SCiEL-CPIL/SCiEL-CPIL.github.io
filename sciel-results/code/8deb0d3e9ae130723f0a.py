<<<<<<< SEARCH
def construct():
    import time
    start_time = time.time()
    DEADLINE = 140  # seconds, leave margin
    
    # Try rank-37 Strassen construction
    result = _build_strassen_2x2_2x11()
    if _verify(result['U'], result['V'], result['W']):
        return result
    
    # Fall back to schoolbook rank 44
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {'U': U, 'V': V, 'W': W}
=======
def _try_random_search(max_rank, time_budget=30):
    """Try to find a decomposition of rank <= max_rank using randomized search."""
    import numpy as np
    import time
    
    start = time.time()
    n, m, p = 2, 2, 11
    dim_U, dim_V, dim_W = n * m, m * p, p * n  # 4, 22, 22
    
    # Build target tensor
    T = np.zeros((dim_U, dim_V, dim_W), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Non-zero entries of T
    nz = [(a, b, c) for a in range(dim_U) for b in range(dim_V) for c in range(dim_W) if T[a,b,c] != 0]
    
    # Candidate coefficient values
    coeffs = [-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2]
    
    for rank in range(max_rank, 29, -1):
        if time.time() - start > time_budget:
            break
        max_iters = 5000
        for it in range(max_iters):
            if time.time() - start > time_budget:
                break
            # Random initialization with sparse structure
            U = np.zeros((dim_U, rank), dtype=float)
            V = np.zeros((dim_V, rank), dtype=float)
            W = np.zeros((dim_W, rank), dtype=float)
            
            for r in range(rank):
                # Each term should have some structure
                # U: pick 1-3 random A-indices
                u_supp = np.random.choice(dim_U, size=np.random.randint(1, 4), replace=False)
                for idx in u_supp:
                    U[idx, r] = np.random.choice(coeffs)
                
                # V: pick 1-3 random B-indices
                v_supp = np.random.choice(dim_V, size=np.random.randint(1, 4), replace=False)
                for idx in v_supp:
                    V[idx, r] = np.random.choice(coeffs)
                
                # W: pick 1-4 random C-indices
                w_supp = np.random.choice(dim_W, size=np.random.randint(1, 5), replace=False)
                for idx in w_supp:
                    W[idx, r] = np.random.choice(coeffs)
            
            # Compute decomposition tensor
            T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
            if np.allclose(T, T_decomp, atol=1e-9):
                # Convert to lists
                return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
            
            # Try gradient-free refinement: adjust one random entry
            if it % 10 == 0:
                mat = np.random.choice([U, V, W])
                row, col = np.random.randint(dim_U if mat is U else dim_V if mat is V else dim_W, 1)[0], np.random.randint(rank, 1)[0]
                mat[row, col] = np.random.choice(coeffs)
                
                T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
                if np.allclose(T, T_decomp, atol=1e-9):
                    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    return None

def construct():
    import time
    start_time = time.time()
    DEADLINE = 140  # seconds, leave margin
    
    # Try rank-37 Strassen construction first (fast, guaranteed)
    result = _build_strassen_2x2_2x11()
    if _verify(result['U'], result['V'], result['W']):
        best_result = result
        best_rank = 37
    else:
        best_result = None
        best_rank = 44
    
    # Try to find a lower rank using randomized search
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 10:
        search_result = _try_random_search(best_rank - 1, time_budget=min(remaining - 10, 60))
        if search_result is not None:
            if _verify(search_result['U'], search_result['V'], search_result['W']):
                best_result = search_result
                best_rank = len(search_result['U'][0])
    
    if best_result is not None:
        return best_result
    
    # Fall back to schoolbook rank 44
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {'U': U, 'V': V, 'W': W}
>>>>>>> REPLACE

Wait, the randomized search as written is unlikely to find a valid decomposition in 5000 iterations because the search space is enormous. Let me think of a better structural approach.

A better idea: **use the algebraic structure more carefully**. The tensor for 2×2 × 2×11 can be decomposed using the fact that it's a "block" tensor. Let me try a **structured search** where I fix the U matrix to have a specific pattern (like Strassen) and then search for V and W.

Actually, let me try a more principled approach: **use the border rank / secant variety approach**. The key insight is that for the 2×2 × 2×p matrix multiplication tensor, there might be a decomposition with rank less than 37 that uses a different algebraic structure.

Let me try: **use a "Strassen-like" approach but with a different grouping**. Instead of 5 blocks of 2 + 1 column, try 4 blocks of 3 columns. For 2×2 × 2×3, I'll try to find a rank-5 or rank-6 decomposition.

For 2×2 × 2×3, let me try to find a rank-5 decomposition manually. The tensor has 12 non-zero entries. Let me set up the equations...

Actually, this is getting complex. Let me try a different structural change: **use an integer linear programming / constraint satisfaction approach** to find the minimal rank. I'll encode the problem as: for each rank-r candidate, check if there exist integer/half-integer U, V, W satisfying the tensor equations.

Given the time constraints, let me implement a more targeted search:

1. Try rank 36 with a structured approach: take the rank-37 construction and try to merge two terms into one (if possible).
2. Use a local search around the rank-37 solution.

Let me implement a local search: start from the rank-37 solution, try to reduce rank by 1 by finding a linear dependency or by modifying terms.

Actually, the simplest structural change that might work: **try all possible "Strassen-like" templates with different parameterizations** and see if any give rank < 37.

Let me implement a more systematic search over structured templates:

