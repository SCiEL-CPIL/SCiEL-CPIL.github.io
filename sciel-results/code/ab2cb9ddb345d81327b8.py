"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
import time
import itertools
_start_time = time.time()
_DEADLINE = _start_time + 110  # 110s safety margin before 120s limit

def _build_strassen_21():
    """Build the rank-21 Strassen decomposition for 2x2 * 2x6."""
    n, m, p = 2, 2, 6
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_coeffs = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1}
    ]
    
    V_coeffs = [
        [(0, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(0, 1, 1), (1, 1, -1)],
        [(1, 0, 1), (0, 0, -1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (1, 1, 1)]
    ]
    
    W_map = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]
    
    for r in range(3):
        for k in range(7):
            col = r * 7 + k
            for idx, val in U_coeffs[k].items():
                U[idx][col] = val
            for j_local, k_local, val in V_coeffs[k]:
                global_k = 2 * r + k_local
                v_idx = j_local * p + global_k
                V[v_idx][col] = val
            for i, j_local, val in W_map[k]:
                global_k = 2 * r + j_local
                w_idx = global_k * n + i
                W[w_idx][col] = val
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication.
    For each (i, j, k) in C[i,k] = sum_j A[i,j]*B[j,k],
    check that sum_t U[i,j,t] * V[j,k,t] * W[k,n+i,t] == 1 if the product term exists, else 0.
    Actually: C[i,k] = sum_j A[i,j] B[j,k]
    Decomposition: C[i,k] = sum_t W[k*n+i, t] * (sum_j U[i*m+j, t] * V[j*p+k, t])
    We need: for each (i,j,k), sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] = 1
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for t in range(rank):
                    s += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                if s != 1:
                    return False
    return True

def _try_remove_product(U, V, W, remove_col, n, m, p):
    """Try to remove column `remove_col` by redistributing its contribution
    to other columns. Returns (U', V', W') if successful, else None.
    
    The idea: the removed product contributes to C entries via W.
    We need to find corrections to U, V, W on the remaining columns such that
    the tensor identity still holds.
    
    For the removed column t0, we have contributions:
    sum_t U[i,j,t]*V[j,k,t]*W[k*n+i,t] must equal 1 for each (i,j,k).
    
    If we remove column t0, we need the remaining columns to still sum to 1.
    The deficit from removing t0 is: U[i,j,t0]*V[j,k,t0]*W[k*n+i,t0]
    We need to add this back using the other columns.
    
    This is generally not possible with simple redistribution since the
    product structure makes it a nonlinear constraint. Instead, we try
    a different approach: modify the remaining U, V, W entries.
    
    Actually, let's try a simpler algebraic approach:
    For each remaining column t != t0, we can add a scalar multiple of
    the removed column's U, V, W patterns... but that doesn't work because
    the product is trilinear.
    
    Let me try: for the removed column, its contribution to the tensor is
    T_removed[i,j,k] = U[i,j,t0] * V[j,k,t0] * W[k*n+i,t0]
    
    We need: sum_{t != t0} (U'[i,j,t] * V'[j,k,t] * W'[k*n+i,t]) = 1
    where U' = U + delta_U, etc.
    
    This is a hard nonlinear problem. Let's try a different strategy:
    Try to find if there's a linear combination that works.
    
    Alternative: Try all pairs of columns to see if one can be expressed
    as a combination that maintains the identity.
    """
    rank = len(U[0])
    new_rank = rank - 1
    
    # Create new matrices without the removed column
    U_new = [[U[i][t] for t in range(rank) if t != remove_col] for i in range(len(U))]
    V_new = [[V[i][t] for t in range(rank) if t != remove_col] for i in range(len(V))]
    W_new = [[W[i][t] for t in range(rank) if t != remove_col] for i in range(len(W))]
    
    # The deficit tensor from removing column t0:
    # deficit[i,j,k] = U[i,j,t0] * V[j,k,t0] * W[k*n+i,t0]
    # We need to see if we can adjust U_new, V_new, W_new to compensate.
    
    # For a 2x2*2x6 problem, each column has at most 2 nonzero U entries,
    # 2 nonzero V entries, and 2 nonzero W entries.
    # The deficit is a rank-1 tensor (outer product of three vectors).
    # We need to express this rank-1 tensor as a sum of modifications to
    # the remaining rank-1 tensors.
    
    # This is generally infeasible with simple adjustments. Let's try a
    # brute-force approach: for each remaining column, try adding the
    # deficit scaled appropriately.
    
    # Actually, let's just verify if the decomposition without the column
    # still works (it won't, but let's be thorough):
    if _verify_decomposition(U_new, V_new, W_new, n, m, p):
        return U_new, V_new, W_new
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Start with rank-21 Strassen decomposition
    U, V, W = _build_strassen_21()
    
    # Try to find a rank-20 decomposition by removing one product
    # and trying to fix the remaining columns
    for remove_col in range(21):
        if time.time() > _DEADLINE - 10:
            break
        
        result = _try_remove_product(U, V, W, remove_col, n, m, p)
        if result is not None:
            U_new, V_new, W_new = result
            if _verify_decomposition(U_new, V_new, W_new, n, m, p):
                return {"U": U_new, "V": V_new, "W": W_new}
    
    # Try a different approach: search for a rank-20 decomposition by
    # modifying the Strassen structure. Try merging products across blocks.
    # For example, combine M1 of block 0 with M1 of block 1 using a shared
    # multiplication if possible.
    
    # Try: can we share a product between two blocks?
    # M1_r computes (A00+A11) * (B_{r,00}+B_{r,11})
    # If we could compute (A00+A11) * (B_{r1,00}+B_{r1,11}+B_{r2,00}+B_{r2,11})
    # and then split, we'd save one multiplication. But splitting requires
    # additional multiplications, so this doesn't help directly.
    
    # Try: algebraic geometry / linear algebra approach
    # The multiplication tensor for M_{2,2,6} has border rank at most 21.
    # The true rank might be 20 or 21.
    
    # Let's try a randomized search: perturb the Strassen decomposition
    # and see if we can find a rank-20 solution.
    
    # Actually, let's try a more systematic approach:
    # The key insight is that for 2x2 * 2x6, we might be able to use
    # a modified Strassen where one of the 7 products per block is shared
    # or eliminated.
    
    # Try: use 7 products for blocks 0 and 1, but only 6 for block 2
    # by absorbing one product into the others.
    
    # For block 2 (columns 14-20), try to express one Strassen product
    # as a combination of the others.
    
    # M7 for block 2: (A01-A11) * (B_{2,01}-B_{2,11})
    # Can this be expressed using M1-M6 of block 2?
    # M1: (A00+A11)*(B_{2,00}+B_{2,11})
    # M2: (A10+A11)*B_{2,00}
    # M3: A00*(B_{2,01}-B_{2,11})
    # M4: A11*(B_{2,10}-B_{2,00})
    # M5: (A00+A01)*B_{2,11}
    # M6: (A10-A00)*(B_{2,00}+B_{2,01})
    
    # M7 = (A01-A11)*(B_{2,01}-B_{2,11})
    # = A01*B_{2,01} - A01*B_{2,11} - A11*B_{2,01} + A11*B_{2,11}
    
    # From M3: A00*B_{2,01} - A00*B_{2,11}
    # From M5: A00*B_{2,11} + A01*B_{2,11}
    # From M6: A10*B_{2,00} + A10*B_{2,01} - A00*B_{2,00} - A00*B_{2,01}
    
    # This doesn't directly give M7. The products are bilinear and
    # you can't generally express one bilinear form as a linear combination
    # of others unless they're linearly dependent as tensors.
    
    # Let's check linear independence of the 21 rank-1 tensors.
    # Each rank-1 tensor T_t is defined by (u_t, v_t, w_t) where
    # T_t[i,j,k] = u_t[i,j] * v_t[j,k] * w_t[k,i]
    
    # If any tensor is a linear combination of the others, we could
    # potentially reduce the rank. But for Strassen, they should be
    # linearly independent.
    
    # Given the difficulty, return the rank-21 solution.
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
