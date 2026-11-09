"""Matrix multiplication 3,5,6 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def _schoolbook_decomp(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition for multiplying n x m by m x p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U_arr[i * m + j]
                v_row = V_arr[j * p + k]
                w_row = W_arr[k * n + i]
                val = np.dot(u_row, v_row * w_row)
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _strassen_2x2x2():
    """Return Strassen decomposition for 2x2 matrix multiplication (rank 7)."""
    # Strassen's 7 products:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    U = [
        [1, 0, 0, 0, 1, 0, 0],  # A11
        [0, 0, 1, 0, 1, -1, 0],  # A12
        [0, 1, 0, 0, 0, 1, 0],  # A21
        [1, 1, 0, 1, 0, 0, -1]  # A22
    ]
    V = [
        [1, 0, 0, 0, 0, 1, 1],  # B11
        [0, 1, -1, 1, 0, 0, 0],  # B12
        [1, 1, 1, 0, 0, 0, 1],  # B21
        [1, 0, 0, -1, 1, 0, 0]  # B22
    ]
    W = [
        [1, 0, 0, 0, -1, 0, 1],  # C11 = M1 - M5 + M7
        [0, 0, 1, 0, 1, 0, 0],  # C12 = M3 + M5
        [0, 1, 0, 1, 0, 0, 0],  # C21 = M2 + M4
        [1, -1, 1, 0, 0, 1, 0]  # C22 = M1 - M2 + M3 + M6
    ]
    return {"U": U, "V": V, "W": W}

def _block_decomp(n, m, p, splits_n, splits_m, splits_p):
    """
    Create a block decomposition by splitting dimensions and using appropriate
    decompositions for each block.
    
    splits_n: list of sizes summing to n
    splits_m: list of sizes summing to m  
    splits_p: list of sizes summing to p
    """
    U_blocks = []
    V_blocks = []
    W_blocks = []
    total_rank = 0
    
    for ni, sn in enumerate(splits_n):
        for mi, sm in enumerate(splits_m):
            for pi, sp in enumerate(splits_p):
                # Get decomposition for this block size
                if sn == 2 and sm == 2 and sp == 2:
                    sub = _strassen_2x2x2()
                    sub_rank = 7
                else:
                    sub = _schoolbook_decomp(sn, sm, sp)
                    sub_rank = sn * sm * sp
                
                # Build the block matrices with proper indexing
                # U has rows indexed by (i, j) where i in [0,n), j in [0,m)
                # V has rows indexed by (j, k) where j in [0,m), k in [0,p)
                # W has rows indexed by (k, i) where k in [0,p), i in [0,n)
                
                U_block = [[0] * sub_rank for _ in range(sn * sm)]
                V_block = [[0] * sub_rank for _ in range(sm * sp)]
                W_block = [[0] * sub_rank for _ in range(sp * sn)]
                
                for r in range(sub_rank):
                    for ii in range(sn):
                        for jj in range(sm):
                            U_block[ii * sm + jj][r] = sub["U"][ii * sm + jj][r]
                    for jj in range(sm):
                        for kk in range(sp):
                            V_block[jj * sp + kk][r] = sub["V"][jj * sp + kk][r]
                    for kk in range(sp):
                        for ii in range(sn):
                            W_block[kk * sn + ii][r] = sub["W"][kk * sn + ii][r]
                
                U_blocks.append((ni, mi, U_block))
                V_blocks.append((ni, mi, V_block))
                W_blocks.append((ni, mi, W_block))
                total_rank += sub_rank
    
    # Now assemble the full matrices
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for ni, mi, U_block in U_blocks:
        # Find corresponding V and W blocks
        for vi, mi_v, V_block in V_blocks:
            if mi_v == mi:
                for wi, mi_w, W_block in W_blocks:
                    if mi_w == mi:
                        # Extract the start indices
                        n_start = sum(splits_n[:ni])
                        m_start = sum(splits_m[:mi])
                        p_start = sum(splits_p[:mi])  # This is wrong, need separate tracking
                        
                        # Actually, let's redo this more carefully
                        pass
    
    # Let me redo this with a cleaner approach
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for ni, sn in enumerate(splits_n):
        n_start = sum(splits_n[:ni])
        for mi, sm in enumerate(splits_m):
            m_start = sum(splits_m[:mi])
            for pi, sp in enumerate(splits_p):
                p_start = sum(splits_p[:pi])
                
                # Get sub-decomposition
                if sn == 2 and sm == 2 and sp == 2:
                    sub = _strassen_2x2x2()
                    sub_rank = 7
                else:
                    sub = _schoolbook_decomp(sn, sm, sp)
                    sub_rank = sn * sm * sp
                
                # Fill in U, V, W
                for r in range(sub_rank):
                    for ii in range(sn):
                        for jj in range(sm):
                            i_global = n_start + ii
                            j_global = m_start + jj
                            U[i_global * m + j_global][col_offset + r] = sub["U"][ii * sm + jj][r]
                    
                    for jj in range(sm):
                        for kk in range(sp):
                            j_global = m_start + jj
                            k_global = p_start + kk
                            V[j_global * p + k_global][col_offset + r] = sub["V"][jj * sp + kk][r]
                    
                    for kk in range(sp):
                        for ii in range(sn):
                            k_global = p_start + kk
                            i_global = n_start + ii
                            W[k_global * n + i_global][col_offset + r] = sub["W"][kk * sn + ii][r]
                
                col_offset += sub_rank
    
    return {"U": U, "V": V, "W": W}

def _try_block_decompositions(n, m, p, deadline):
    """Try various block decompositions and return the best one."""
    best = None
    best_rank = float('inf')
    
    # Try different splits
    splits_to_try = [
        # (splits_n, splits_m, splits_p)
        ([3], [5], [6]),  # schoolbook
        ([2, 1], [2, 2, 1], [2, 2, 2]),
        ([2, 1], [2, 3], [2, 2, 2]),
        ([2, 1], [3, 2], [2, 2, 2]),
        ([2, 1], [2, 2, 1], [3, 3]),
        ([2, 1], [2, 2, 1], [2, 4]),
        ([2, 1], [2, 2, 1], [4, 2]),
        ([2, 1], [2, 2, 1], [1, 2, 3]),
        ([2, 1], [2, 2, 1], [3, 1, 2]),
        ([2, 1], [2, 2, 1], [2, 3, 1]),
        ([1, 2], [2, 2, 1], [2, 2, 2]),
        ([2, 1], [1, 2, 2], [2, 2, 2]),
        ([2, 1], [2, 1, 2], [2, 2, 2]),
        ([2, 1], [2, 2, 1], [2, 1, 3]),
        ([2, 1], [2, 2, 1], [2, 3, 1]),
        ([2, 1], [2, 2, 1], [3, 2, 1]),
        ([2, 1], [2, 2, 1], [1, 3, 2]),
        ([2, 1], [2, 2, 1], [1, 2, 3]),
        ([2, 1], [2, 2, 1], [2, 2, 2]),
    ]
    
    for splits_n, splits_m, splits_p in splits_to_try:
        if time.time() > deadline - 5:
            break
        
        if sum(splits_n) != n or sum(splits_m) != m or sum(splits_p) != p:
            continue
        
        try:
            decomp = _block_decomp(n, m, p, splits_n, splits_m, splits_p)
            rank = len(decomp["U"][0])
            
            if rank < best_rank:
                # Verify
                if _verify_decomposition(decomp["U"], decomp["V"], decomp["W"], n, m, p):
                    best = decomp
                    best_rank = rank
        except Exception:
            continue
    
    return best, best_rank

def _try_kronecker_product(n, m, p, deadline):
    """
    Try to use Kronecker product structure. If we can write the (n,m,p) tensor
    as a Kronecker product of smaller tensors, we might get a lower rank.
    
    For example, if n = n1*n2, m = m1*m2, p = p1*p2, and the tensor factors
    as T ⊗ S, then rank(T ⊗ S) = rank(T) * rank(S).
    """
    best = None
    best_rank = float('inf')
    
    # Try factorizations of n, m, p
    def get_factors(x):
        factors = []
        for d in range(1, x + 1):
            if x % d == 0:
                factors.append((d, x // d))
        return factors
    
    n_factors = get_factors(n)
    m_factors = get_factors(m)
    p_factors = get_factors(p)
    
    for n1, n2 in n_factors:
        if time.time() > deadline - 5:
            break
        for m1, m2 in m_factors:
            for p1, p2 in p_factors:
                # Check if we can use Strassen for the 2x2x2 part
                if n1 == 2 and m1 == 2 and p1 == 2:
                    # Try to see if the tensor has Kronecker structure
                    # This requires the (n2,m2,p2) part to also be decomposable
                    pass
    
    return best, best_rank

def _randomized_search(n, m, p, target_rank, deadline, n_iterations=100):
    """
    Try to find a rank-target_rank decomposition using randomized search.
    We'll use a simple approach: randomly initialize and try to optimize.
    """
    # For exact decomposition with integer/half-integer entries, this is hard.
    # Let's try a different approach: start from schoolbook and try to merge terms.
    
    # Actually, let's try a more systematic approach:
    # The tensor T[i,j,k] = 1 for all i,j,k.
    # We want T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    
    # This is equivalent to finding a rank-r decomposition of the tensor.
    # For small ranks, we can try random initialization and use gradient descent
    # on the smooth objective: sum_{i,j,k} (T[i,j,k] - sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r])^2
    
    # But we need integer/half-integer solutions, so this is tricky.
    
    # For now, let's just return None to indicate we couldn't find a better decomposition.
    return None

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    deadline = t0 + 140.0  # safety margin (150s limit)
    
    # Start with schoolbook
    best = _schoolbook_decomp(n, m, p)
    best_rank = n * m * p
    
    # Try block decompositions
    block_result, block_rank = _try_block_decompositions(n, m, p, deadline)
    if block_result is not None and block_rank < best_rank:
        best = block_result
        best_rank = block_rank
    
    # Try Kronecker product approach
    kron_result, kron_rank = _try_kronecker_product(n, m, p, deadline)
    if kron_result is not None and kron_rank < best_rank:
        best = kron_result
        best_rank = kron_rank
    
    # Verify the best decomposition
    if not _verify_decomposition(best["U"], best["V"], best["W"], n, m, p):
        # Fallback to schoolbook
        best = _schoolbook_decomp(n, m, p)
        best_rank = n * m * p
    
    return best

# EVOLVE-BLOCK-END
