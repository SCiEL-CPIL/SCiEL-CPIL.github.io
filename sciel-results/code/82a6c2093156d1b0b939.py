"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _build_factors(n, m, p, offset, U, V, W, deadline):
    """
    Build factor matrices for M_{n,m,p} and append to U, V, W.
    Returns the rank used (number of columns appended).
    U, V, W are numpy arrays of shape (n*m, rank_total), (m*p, rank_total), (p*n, rank_total)
    with pre-allocated columns. offset is the starting column index.
    """
    if time.time() > deadline:
        return 0
    
    # Base cases: schoolbook
    if n == 1 or m == 1 or p == 1:
        r = n * m * p
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, offset] = 1
                    V[j*p+k, offset] = 1
                    W[k*n+i, offset] = 1
                    offset += 1
        return r
    
    # Strassen for 2x2x2
    if n == 2 and m == 2 and p == 2:
        # 7 Strassen products
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Index mapping:
        # A indices: A[i,j] -> row i*m+j in U
        # B indices: B[j,k] -> row j*p+k in V
        # C indices: C[k,i] -> row k*n+i in W (trace convention W[k*n+i, r])
        
        # For M1 = (A11 + A22)(B11 + B22):
        # U: A11 at (0,0) -> U[0], A22 at (1,1) -> U[3]
        # V: B11 at (0,0) -> V[0], B22 at (1,1) -> V[3]
        # W: C11 at (0,0) -> W[0], C22 at (1,1) -> W[3]
        # Wait, C11 contributes to C[0,0] which is W[0*2+0] = W[0]
        # C22 contributes to C[1,1] which is W[1*2+1] = W[3]
        
        terms = [
            # (U_indices, V_indices, W_indices, signs)
            # Each term is a single scalar multiplication
            # We need to represent linear combinations
            
            # M1 = (A11 + A22)(B11 + B22)
            # This expands to A11*B11 + A11*B22 + A22*B11 + A22*B22
            # But in tensor decomposition, each "term" is a rank-1 tensor.
            # Actually, the Strassen algorithm uses 7 scalar multiplications.
            # Each multiplication corresponds to a rank-1 tensor in the decomposition.
            
            # Let's define the 7 Strassen rank-1 tensors directly.
        ]
        
        # Strassen rank-1 tensors for 2x2x2:
        # T1: (A11+A22) * (B11+B22) -> contributes to C11 (+1), C22 (+1)
        #   U: A11 (1), A22 (1)
        #   V: B11 (1), B22 (1)
        #   W: C11 (1), C22 (1)
        # T2: (A21+A22) * B11 -> contributes to C21 (+1)
        #   U: A21 (1), A22 (1)
        #   V: B11 (1)
        #   W: C21 (1)
        # T3: A11 * (B12-B22) -> contributes to C12 (+1)
        #   U: A11 (1)
        #   V: B12 (1), B22 (-1)
        #   W: C12 (1)
        # T4: A22 * (B21-B11) -> contributes to C21 (+1)
        #   U: A22 (1)
        #   V: B21 (1), B11 (-1)
        #   W: C21 (1)
        # T5: (A11+A12) * B22 -> contributes to C12 (+1), C11 (-1)
        #   U: A11 (1), A12 (1)
        #   V: B22 (1)
        #   W: C12 (1), C11 (-1)
        # T6: (A21-A11) * (B11+B12) -> contributes to C22 (+1)
        #   U: A21 (1), A11 (-1)
        #   V: B11 (1), B12 (1)
        #   W: C22 (1)
        # T7: (A12-A22) * (B21+B22) -> contributes to C11 (+1)
        #   U: A12 (1), A22 (-1)
        #   V: B21 (1), B22 (1)
        #   W: C11 (1)
        
        # Let's build these 7 rank-1 tensors
        strassen_terms = [
            # (u_coeffs: {row: val}, v_coeffs: {row: val}, w_coeffs: {row: val})
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),  # T1
            ({1: 1, 3: 1}, {0: 1}, {2: 1}),              # T2: C21 is W[1*2+0] = W[2]
            ({0: 1}, {1: 1, 3: -1}, {1: 1}),             # T3: C12 is W[0*2+1] = W[1]
            ({3: 1}, {2: 1, 0: -1}, {2: 1}),             # T4: C21 is W[2]
            ({0: 1, 2: 1}, {3: 1}, {1: 1, 0: -1}),       # T5
            ({1: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),       # T6: C22 is W[3]
            ({2: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),       # T7
        ]
        
        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            if time.time() > deadline:
                break
            for row, val in u_coeffs.items():
                U[row, offset] += val
            for row, val in v_coeffs.items():
                V[row, offset] += val
            for row, val in w_coeffs.items():
                W[row, offset] += val
            offset += 1
        
        return 7
    
    # General case: split into blocks
    # Split n into n1 + n2, m into m1 + m2, p into p1 + p2
    # To minimize rank, we should use the best known decompositions for sub-blocks.
    # For 5x5x5, we can split 5 = 2 + 3
    # The 8 sub-blocks are:
    # (2,2,2), (2,2,3), (2,3,2), (2,3,3)
    # (3,2,2), (3,2,3), (3,3,2), (3,3,3)
    
    # We need ranks for these sub-problems.
    # For simplicity and correctness, we can use schoolbook for all except 2x2x2.
    # But that gives 2*2*2 + 2*2*3 + 2*3*2 + 2*3*3 + 3*2*2 + 3*2*3 + 3*3*2 + 3*3*3
    # = 8 + 12 + 12 + 18 + 12 + 18 + 18 + 27 = 125. Same as schoolbook!
    
    # The benefit comes from using better decompositions for sub-blocks.
    # For example, if we had a rank 7 decomposition for 2x2x2, and we used it,
    # the total would be 7 + 12 + 12 + 18 + 12 + 18 + 18 + 27 = 124.
    
    # But the sub-blocks are not independent in the same way. The Strassen algorithm
    # works on the whole 2x2x2 block, not on sub-blocks of larger blocks.
    
    # Actually, the correct approach is to use the Strassen algorithm recursively.
    # For 5x5x5, we can't directly apply Strassen. We need to pad to 8x8x8 or use
    # a different decomposition.
    
    # A better approach: use the known fact that R(M_5,5,5) <= 117.
    # One way to achieve 117 is to use a specific decomposition.
    
    # Let's try a different strategy: use schoolbook for most parts, but use
    # Strassen where possible.
    
    # For now, let's return the schoolbook rank for non-2x2x2 cases.
    # This will give us 125, which is the same as the current best.
    
    # To do better, we need to implement a proper recursive Strassen-like algorithm.
    
    # Let's try splitting 5 = 2 + 3 and using:
    # - 2x2x2: Strassen (7)
    # - 2x2x3: schoolbook (12)
    # - 2x3x2: schoolbook (12)
    # - 2x3x3: schoolbook (18)
    # - 3x2x2: schoolbook (12)
    # - 3x2x3: schoolbook (18)
    # - 3x3x2: schoolbook (18)
    # - 3x3x3: schoolbook (27)
    # Total: 7 + 12 + 12 + 18 + 12 + 18 + 18 + 27 = 124
    
    # This is better than 125!
    
    # Let's implement this.
    
    n1, n2 = 2, 3
    m1, m2 = 2, 3
    p1, p2 = 2, 3
    
    total_rank = 0
    
    # (n1, m1, p1) = (2, 2, 2)
    r = _build_factors(2, 2, 2, offset, U, V, W, deadline)
    total_rank += r
    offset += r
    if time.time() > deadline:
        return total_rank
    
    # (n1, m1, p2) = (2, 2, 3)
    r = _build_factors(2, 2, 3, offset, U, V, W, deadline)
    total_rank += r
    offset += r
    if time.time() > deadline:
        return total_rank
    
    # (n1, m2, p1) = (2, 3, 2)
    r = _build_factors(2, 3, 2, offset, U, V, W, deadline)
    total_rank += r
    offset += r
    if time.time() > deadline:
        return total_rank
    
    # (n1, m2, p2) = (2, 3, 3)
    r = _build_factors(2, 3, 3, offset, U, V, W, deadline)
    total_rank += r
    offset += r
    if time.time() > deadline:
        return total_rank
    
    # (n2, m1, p1) = (3, 2, 2)
    r = _build_factors(3, 2, 2, offset, U, V, W, deadline)
    total_rank += r
    offset += r
    if time.time() > deadline:
        return total_rank
    
    # (n2, m1, p2) = (3, 2, 3)
    r = _build_factors(3, 2, 3, offset, U, V, W, deadline)
    total_rank += r
    offset += r
    if time.time() > deadline:
        return total_rank
    
    # (n2, m2, p1) = (3, 3, 2)
    r = _build_factors(3, 3, 2, offset, U, V, W, deadline)
    total_rank += r
    offset += r
    if time.time() > deadline:
        return total_rank
    
    # (n2, m2, p2) = (3, 3, 3)
    r = _build_factors(3, 3, 3, offset, U, V, W, deadline)
    total_rank += r
    offset += r
    
    return total_rank

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10 seconds safety margin
    
    n, m, p = DIMENSIONS
    # Maximum possible rank is n*m*p = 125
    # We'll try to use the block decomposition to get 124
    # If that fails, fall back to schoolbook
    
    # First, try the block decomposition
    try:
        # Pre-allocate with max rank
        max_rank = n * m * p
        U = np.zeros((n * m, max_rank), dtype=float)
        V = np.zeros((m * p, max_rank), dtype=float)
        W = np.zeros((p * n, max_rank), dtype=float)
        
        rank_used = _build_factors(n, m, p, 0, U, V, W, deadline)
        
        if rank_used > 0 and rank_used < max_rank:
            # Trim to actual rank used
            U = U[:, :rank_used]
            V = V[:, :rank_used]
            W = W[:, :rank_used]
            
            # Convert to lists for the required output format
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            
            # Round to nearest integer if close
            def round_if_close(lst):
                result = []
                for row in lst:
                    new_row = []
                    for val in row:
                        nearest = round(val)
                        if abs(val - nearest) < 1e-9:
                            new_row.append(int(nearest))
                        else:
                            new_row.append(val)
                    result.append(new_row)
                return result
            
            U_list = round_if_close(U_list)
            V_list = round_if_close(V_list)
            W_list = round_if_close(W_list)
            
            return {"U": U_list, "V": V_list, "W": W_list}
        
        # Fall back to schoolbook
    except Exception as e:
        pass
    
    # Schoolbook fallback
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
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
