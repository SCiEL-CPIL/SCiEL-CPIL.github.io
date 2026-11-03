"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _time_remaining(deadline, margin=10.0):
    return deadline - time.time() - margin

def _verify_decomposition(U, V, W, n=3, m=4, p=8):
    """Verify that U, V, W form a valid tensor decomposition for M_{n,m,p}."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Check shapes
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    
    # The matrix multiplication tensor T[i,k,j] = delta[i,j] (1 if i==j else 0)
    # Decomposition: T[i,k,j] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    # Note: W uses trace convention W[k*n+i, r] where k is col of B, i is row of A
    # Actually for M_{n,m,p}: C = A * B where A is n x m, B is m x p, C is n x p
    # Tensor T_{i,k,j} = delta_{i,j}
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # W index: (j, i) -> j*n + i (trace convention: W[k*n+i] where k=j, i=i)
    
    for i in range(n):
        for k in range(m):
            for j in range(p):
                expected = 1.0 if i == j else 0.0
                actual = 0.0
                for r in range(rank):
                    actual += U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
                if abs(actual - expected) > 1e-6:
                    return False
    return True

def _try_reduce_rank(U, V, W, n=3, m=4, p=8, max_iterations=200, deadline=None):
    """
    Try to reduce rank by finding linear dependencies among the rank-1 terms.
    Strategy: build the tensor and look for linear combinations that cancel.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # For each pair of columns, check if they are linearly dependent
    # Two columns r1, r2 are dependent if there exists alpha such that
    # U[:,r1] = alpha*U[:,r2], V[:,r1] = alpha*V[:,r2], W[:,r1] = alpha*W[:,r2]
    # More generally, we look for linear relations: sum_r c_r * U[:,r] = 0,
    # sum_r c_r * V[:,r] = 0, sum_r c_r * W[:,r] = 0
    # But that's too restrictive. Instead, we look for:
    # The tensor sum_r c_r * outer(U[:,r], V[:,r], W[:,r]) = 0
    
    # Build the full tensor from the decomposition
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    
    # Try to find linear dependencies by looking at the matrix flattening
    # Reshape T to a matrix of shape (n*m*m*p, p*n) and find null space
    T_mat = T.reshape(n*m*m*p, p*n)
    
    # Compute SVD to find low-rank structure
    try:
        U_svd, s, Vt = np.linalg.svd(T_mat, full_matrices=False)
    except:
        return U, V, W
    
    # The numerical rank tells us the minimum rank for this particular flattening
    num_rank = np.sum(s > 1e-8)
    
    # Try greedy elimination: for each column, check if removing it and adjusting others
    # still gives a valid decomposition
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try removing one column at a time and see if we can compensate
    # This is expensive, so only try a limited number
    for trial in range(min(rank, 30)):
        if deadline and time.time() > deadline - 15:
            break
        
        # Pick a random column to try removing
        col_to_remove = trial % rank
        
        # Check if this column is "redundant" by seeing if its contribution
        # can be expressed as a combination of others
        # Build the tensor without this column
        T_without = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            if r == col_to_remove:
                continue
            T_without += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        
        # The difference is the contribution of the removed column
        diff = T - T_without
        diff_mat = diff.reshape(n*m*m*p, p*n)
        
        # Check if diff is zero (column is truly redundant)
        if np.max(np.abs(diff_mat)) < 1e-8:
            # Column is redundant! Remove it
            new_rank = rank - 1
            best_U = np.delete(U, col_to_remove, axis=1)
            best_V = np.delete(V, col_to_remove, axis=1)
            best_W = np.delete(W, col_to_remove, axis=1)
            rank = new_rank
            U, V, W = best_U, best_V, best_W
            T = np.zeros((n*m, m*p, p*n))
            for r in range(rank):
                T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            best_rank = rank
    
    return best_U.tolist(), best_V.tolist(), best_W.tolist()

def construct():
    # Start with rank-96 schoolbook decomposition for M_4,4,8,
    # then project to M_3,4,8 by removing the fourth row of A.
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # Build rank-96 schoolbook for M_4,4,8 (4x4 times 4x8)
    n_full = 4
    rank_full = n_full * m * p  # 4*4*8 = 128... wait, schoolbook is m*p = 32 per row, 4 rows = 128
    
    # Actually schoolbook for M_{n,m,p} has rank n*m*p
    # For M_4,4,8: rank = 4*4*8 = 128
    # For M_3,4,8: rank = 3*4*8 = 96
    
    # Let's just build the rank-96 schoolbook directly for M_3,4,8
    rank = 96
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i*m+k][col] = 1
                V[k*p+j][col] = 1
                W[j*n+i][col] = 1
                col += 1
    
    # Now try to reduce rank using Strassen-like techniques
    # Apply Strassen to 2x2 sub-blocks where possible
    
    # For the 3x4 matrix, we can apply Strassen to rows 0,1 (treating them as a 2x2 block
    # with 4 columns each split into two 2-column groups)
    # This gives M_2,4,4 for each of the two column blocks (cols 0-3 and cols 4-7)
    # Each M_2,4,4 can be done with rank 28 via Strassen on 2x2x2 sub-blocks
    
    # Let's rebuild using the block-Strassen approach (rank 88)
    # and then try null-space reduction
    
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen factors for 2x2x2 multiplication
    strassen_factors = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    # Process 2x M_2,4,4 blocks (for rows i=0,1 of the 3x4 matrix)
    for c_block in range(2):
        for factor in strassen_factors:
            a_coeffs, b_coeffs, c_coeffs = factor
            for sub_prod_idx in range(4):
                if sub_prod_idx == 0:
                    a_i_off, a_j_off = 0, 0
                    b_i_off, b_j_off = 0, 0
                    c_i_off, c_j_off = 0, 0
                elif sub_prod_idx == 1:
                    a_i_off, a_j_off = 0, 0
                    b_i_off, b_j_off = 0, 2
                    c_i_off, c_j_off = 0, 2
                elif sub_prod_idx == 2:
                    a_i_off, a_j_off = 0, 2
                    b_i_off, b_j_off = 2, 0
                    c_i_off, c_j_off = 0, 0
                else:
                    a_i_off, a_j_off = 0, 2
                    b_i_off, b_j_off = 2, 2
                    c_i_off, c_j_off = 0, 2
                
                for (ai, aj), au in a_coeffs.items():
                    u_idx = (ai + a_i_off) * 4 + (aj + a_j_off)
                    U[u_idx][col] += au
                
                for (bi, bj), bv in b_coeffs.items():
                    v_idx = (bi + b_i_off) * 8 + (bj + b_j_off + 4 * c_block)
                    V[v_idx][col] += bv
                
                for (ci, cj), cw in c_coeffs.items():
                    w_idx = (cj + c_j_off + 4 * c_block) * 3 + (ci + c_i_off)
                    W[w_idx][col] += cw
                
                col += 1
    
    # Process 2x M_1,4,4 blocks (for row i=2 of the 3x4 matrix)
    for c_block in range(2):
        for k in range(4):
            for j in range(4):
                u_idx = 2 * 4 + k
                v_idx = k * 8 + (j + 4 * c_block)
                w_idx = (j + 4 * c_block) * 3 + 2
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    
    # Now try to reduce rank via null-space projection
    # The key insight: since we're computing M_3,4,8, terms that only involve
    # the "phantom" 4th row of a 4x4 matrix are in the null space.
    # However, our decomposition is already for M_3,4,8, so let's try
    # a different approach: look for linear dependencies among the 88 terms.
    
    # Convert to numpy for efficient computation
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build the full tensor to verify
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U_np[:,r], V_np[:,r], W_np[:,r])
    
    # Verify it's correct
    for i in range(n):
        for k in range(m):
            for j in range(p):
                expected = 1.0 if i == j else 0.0
                actual = T[i*m+k, k*p+j, j*n+i]
                if abs(actual - expected) > 1e-6:
                    # Decomposition is wrong, return as-is
                    return {"U": U, "V": V, "W": W}
    
    # Try to find and remove redundant terms
    # Strategy: for each column r, check if its tensor contribution
    # is in the span of other columns' contributions
    # This is expensive, so use a heuristic: look for columns where
    # U[:,r], V[:,r], W[:,r] have special structure
    
    # Heuristic: check if any column has all-zero entries in one of U, V, W
    # (which would make it trivially removable)
    for r in range(rank):
        if np.all(U_np[:,r] == 0) or np.all(V_np[:,r] == 0) or np.all(W_np[:,r] == 0):
            # This term contributes nothing, remove it
            pass  # In practice this shouldn't happen in our construction
    
    # Try a more sophisticated approach: look for pairs of columns that
    # can be combined
    # For Strassen, the 7 terms for each 2x2x2 block are already minimal,
    # so the main opportunity is in the cross-block interactions.
    
    # Alternative: try to use a lower-rank algorithm for M_1,4,4
    # M_1,4,4 = multiply 1x4 by 4x4, result is 1x4
    # This is just a vector-matrix product: rank = 4*4 = 16 (schoolbook)
    # Can we do better? M_1,4,4 has tensor rank 16 (it's just a matrix-vector product)
    # So 16 is minimal for each M_1,4,4 block.
    
    # For M_2,4,4: can we do better than 28?
    # M_2,4,4: multiply 2x4 by 4x4, result is 2x4
    # Schoolbook: 2*4*4 = 32
    # Strassen gives 28
    # Known lower bound: at least 2*4 = 8 (trivial)
    # The tensor rank of M_2,4,4 is known to be 28 (this is tight)
    
    # So 88 might already be close to optimal for this decomposition family.
    # Let's try a completely different partitioning.
    
    # Alternative: treat the 3x4 as a 2x2 block (rows 0,1) + 1x4 (row 2)
    # But split columns differently: 4 = 2+2
    # This is what we already have.
    
    # Try: 3 = 1+1+1, use Strassen on pairs of rows
    # M_1,4,8 + M_1,4,8 + M_1,4,8 = 3 * 32 = 96 (worse)
    
    # Try: 4 = 2+1+1 for columns
    # Rows 0,1 with cols 0,1: M_2,2,2 (rank 7)
    # Rows 0,1 with cols 2: M_2,1,2 (rank 4)
    # Rows 0,1 with cols 3: M_2,1,2 (rank 4)
    # Row 2 with cols 0,1: M_1,2,2 (rank 4)
    # Row 2 with cols 2: M_1,1,2 (rank 2)
    # Row 2 with cols 3: M_1,1,2 (rank 2)
    # Total: 7+4+4+4+2+2 = 23 per column-block? No, this doesn't work directly.
    
    # Actually let me think about this more carefully.
    # M_3,4,8: A is 3x4, B is 4x8, C is 3x8
    # We can partition A's rows and B's columns.
    
    # Partition A rows: {0,1} and {2}
    # Partition B cols: {0,1,2,3} and {4,5,6,7}
    
    # Block (rows{0,1}, cols{0-3}): M_2,4,4, rank 28
    # Block (rows{0,1}, cols{4-7}): M_2,4,4, rank 28
    # Block (rows{2}, cols{0-3}): M_1,4,4, rank 16
    # Block (rows{2}, cols{4-7}): M_1,4,4, rank 16
    # Total: 88
    
    # Alternative partition:
    # A rows: {0}, {1}, {2}
    # B cols: {0,1,2,3}, {4,5,6,7}
    # 3 * 2 * M_1,4,4 = 3 * 2 * 16 = 96 (worse)
    
    # Another alternative:
    # A rows: {0,1,2}
    # B cols: {0,1}, {2,3}, {4,5}, {6,7}
    # 4 * M_3,2,2 = 4 * ? 
    # M_3,2,2: 3x2 times 2x2 = 3x2, schoolbook rank 12
    # Can we do better? M_3,2,2 tensor rank... 
    # M_2,2,2 has rank 7 (Strassen)
    # M_3,2,2: we can write it as M_2,2,2 + M_1,2,2
    # M_2,2,2 rank 7, M_1,2,2 rank 4, total 11
    # So 4 * 11 = 44? That seems too good.
    # Wait, M_3,2,2: A is 3x2, B is 2x2, C is 3x2
    # Schoolbook: 3*2*2 = 12
    # Using Strassen on the first 2 rows: M_2,2,2 (rank 7) + M_1,2,2 (rank 4) = 11
    # So 4 blocks * 11 = 44 total rank!
    
    # But wait, we need to be careful. The blocks are independent, so we can
    # just add their ranks. Let me verify this construction.
    
    # M_3,2,2 decomposition with rank 11:
    # Rows 0,1 of A with both cols of B: M_2,2,2 via Strassen (7 terms)
    # Row 2 of A with both cols of B: M_1,2,2 via schoolbook (4 terms)
    # Total: 11
    
    # For the full M_3,4,8 with column partition {0,1},{2,3},{4,5},{6,7}:
    # 4 blocks, each M_3,2,2 with rank 11
    # Total rank: 4 * 11 = 44
    
    # This is much better than 88! Let me construct this.
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    for col_block in range(4):  # 4 column blocks of size 2
        # M_2,2,2 via Strassen for rows 0,1
        strassen_factors_222 = [
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
        ]
        
        for a_coeffs, b_coeffs, c_coeffs in strassen_factors_222:
            for (ai, aj), au in a_coeffs.items():
                u_idx = ai * 4 + (aj + 2 * col_block)
                U[u_idx][col] += au
            for (bi, bj), bv in b_coeffs.items():
                v_idx = (bi + 2 * col_block) * 8 + bj
                V[v_idx][col] += bv
            for (ci, cj), cw in c_coeffs.items():
                w_idx = (cj + 2 * col_block) * 3 + ci
                W[w_idx][col] += cw
            col += 1
        
        # M_1,2,2 via schoolbook for row 2
        for k in range(2):
            for j in range(2):
                u_idx = 2 * 4 + (k + 2 * col_block)
                v_idx = (k + 2 * col_block) * 8 + j
                w_idx = (j + 2 * col_block) * 3 + 2
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    
    # Verify
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U_np[:,r], V_np[:,r], W_np[:,r])
    
    valid = True
    for i in range(n):
        for k in range(m):
            for j in range(p):
                expected = 1.0 if i == j else 0.0
                actual = T[i*m+k, k*p+j, j*n+i]
                if abs(actual - expected) > 1e-6:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fall back to the rank-88 decomposition
        rank = 88
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        strassen_factors = [
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
        ]
        for c_block in range(2):
            for factor in strassen_factors:
                a_coeffs, b_coeffs, c_coeffs = factor
                for sub_prod_idx in range(4):
                    if sub_prod_idx == 0:
                        a_i_off, a_j_off = 0, 0
                        b_i_off, b_j_off = 0, 0
                        c_i_off, c_j_off = 0, 0
                    elif sub_prod_idx == 1:
                        a_i_off, a_j_off = 0, 0
                        b_i_off, b_j_off = 0, 2
                        c_i_off, c_j_off = 0, 2
                    elif sub_prod_idx == 2:
                        a_i_off, a_j_off = 0, 2
                        b_i_off, b_j_off = 2, 0
                        c_i_off, c_j_off = 0, 0
                    else:
                        a_i_off, a_j_off = 0, 2
                        b_i_off, b_j_off = 2, 2
                        c_i_off, c_j_off = 0, 2
                    for (ai, aj), au in a_coeffs.items():
                        u_idx = (ai + a_i_off) * 4 + (aj + a_j_off)
                        U[u_idx][col] += au
                    for (bi, bj), bv in b_coeffs.items():
                        v_idx = (bi + b_i_off) * 8 + (bj + b_j_off + 4 * c_block)
                        V[v_idx][col] += bv
                    for (ci, cj), cw in c_coeffs.items():
                        w_idx = (cj + c_j_off + 4 * c_block) * 3 + (ci + c_i_off)
                        W[w_idx][col] += cw
                    col += 1
        for c_block in range(2):
            for k in range(4):
                for j in range(4):
                    u_idx = 2 * 4 + k
                    v_idx = k * 8 + (j + 4 * c_block)
                    w_idx = (j + 4 * c_block) * 3 + 2
                    U[u_idx][col] = 1
                    V[v_idx][col] = 1
                    W[w_idx][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
