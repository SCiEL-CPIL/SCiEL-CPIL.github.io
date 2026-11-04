"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 by 3x6 matrix multiplication.
    Strategy: Partition the 3x6 B matrix into two 3x3 blocks, reducing the problem
    to two independent 2x3x3 multiplications. The rank is the sum of the ranks
    of the two sub-problems.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for safety

    n, m, p = DIMENSIONS
    dim_a = n * m  # 6
    dim_b = m * p  # 18
    dim_c = p * n  # 12

    # Build the target tensor T[a, b, c]
    T = np.zeros((dim_a, dim_b, dim_c), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    def verify_decomposition(U, V, W):
        """Verify that U, V, W exactly decompose the tensor T."""
        rank = U.shape[1]
        result = np.zeros((dim_a, dim_b, dim_c), dtype=np.int64)
        for c in range(rank):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            result += np.einsum('i,j,k->ijk', uc, vc, wc)
        return np.array_equal(result, T)

    def schoolbook():
        """Return the schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = np.zeros((dim_a, rank), dtype=np.int64)
        V = np.zeros((dim_b, rank), dtype=np.int64)
        W = np.zeros((dim_c, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        return U, V, W

    # Start with schoolbook
    best_U, best_V, best_W = schoolbook()
    best_rank = n * m * p

    # Compute lower bounds from matricizations
    T1 = T.reshape(dim_a, dim_b * dim_c)
    rank1 = np.linalg.matrix_rank(T1)
    T2 = T.reshape(dim_b, dim_a * dim_c)
    rank2 = np.linalg.matrix_rank(T2)
    T3 = T.reshape(dim_c, dim_a * dim_b)
    rank3 = np.linalg.matrix_rank(T3)
    lower_bound = max(rank1, rank2, rank3)

    # Strategy: Partition B into two 3x3 blocks
    # B is 3x6, split into B1 (3x3, cols 0-2) and B2 (3x3, cols 3-5)
    # C is 2x6, split into C1 (2x3, cols 0-2) and C2 (2x3, cols 3-5)
    # C1 = A * B1, C2 = A * B2
    # This is two independent 2x3x3 problems.
    
    # For each 2x3x3 problem, the schoolbook rank is 2*3*3 = 18.
    # Total rank for two such problems: 18 + 18 = 36.
    # Search for lower-rank decompositions of each 2x3x3 subproblem.
    
    def build_subproblem_tensor(n_sub, m_sub, p_sub):
        """Build tensor for n_sub x m_sub by m_sub x p_sub multiplication."""
        da = n_sub * m_sub
        db = m_sub * p_sub
        dc = p_sub * n_sub
        T_sub = np.zeros((da, db, dc), dtype=np.int64)
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    a = i * m_sub + j
                    b = j * p_sub + k
                    c = k * n_sub + i
                    T_sub[a, b, c] = 1
        return T_sub, da, db, dc
    
    def search_subproblem(T_sub, da, db, dc, max_rank, time_limit):
        """Search for low-rank decomposition of a subproblem tensor."""
        best = None
        best_r = max_rank
        
        # Compute lower bound
        T1 = T_sub.reshape(da, db * dc)
        T2 = T_sub.reshape(db, da * dc)
        T3 = T_sub.reshape(dc, da * db)
        lb = max(np.linalg.matrix_rank(T1), np.linalg.matrix_rank(T2), np.linalg.matrix_rank(T3))
        
        for r in range(lb, max_rank):
            if time.time() > time_limit:
                break
            
            max_attempts = 2000
            for _ in range(max_attempts):
                if time.time() > time_limit:
                    break
                
                # Try integer entries in {-1, 0, 1} and {-2, -1, 0, 1, 2}
                scale = np.random.choice([1, 2])
                U = np.random.randint(-scale, scale + 1, size=(da, r))
                V = np.random.randint(-scale, scale + 1, size=(db, r))
                W = np.random.randint(-scale, scale + 1, size=(dc, r))
                
                # Verify
                result = np.zeros((da, db, dc), dtype=np.int64)
                for c in range(r):
                    result += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
                
                if np.array_equal(result, T_sub):
                    return U, V, W, r
        
        return None
    
    # Search for 2x3x3 subproblem decomposition
    T_sub, da_sub, db_sub, dc_sub = build_subproblem_tensor(2, 3, 3)
    sub_time_limit = deadline - 30  # Leave time for combining and verification
    
    sub_decomp = search_subproblem(T_sub, da_sub, db_sub, dc_sub, 18, sub_time_limit)
    
    if sub_decomp is not None:
        U_sub, V_sub, W_sub, sub_rank = sub_decomp
        # Combine two identical subproblems
        best_rank = 2 * sub_rank
        best_U = np.hstack([U_sub, U_sub])
        best_V = np.hstack([V_sub, V_sub])
        best_W = np.hstack([W_sub, W_sub])
        
        # Verify the combined decomposition
        if verify_decomposition(best_U, best_V, best_W):
            pass  # Good
        else:
            # Fallback to schoolbook
            best_U, best_V, best_W = schoolbook()
            best_rank = n * m * p
    else:
        # No improvement found, keep schoolbook
        pass

    # If no improvement found, return the schoolbook decomposition
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
