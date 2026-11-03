import numpy as np
import time
import itertools
import random

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def _make_strassen_222():
    """Return U, V, W for 2x2x2 Strassen decomposition, each shape (4, 7).
    Index mapping: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    # M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    # M3 = A11(B21-B22)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    # M4 = A22(B12-B11)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    # M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    # M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    # M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    return U, V, W

def _make_schoolbook_block(ri, mi, pi):
    """Schoolbook decomposition for a ri x mi x pi block."""
    rank = ri * mi * pi
    U = np.zeros((ri * mi, rank))
    V = np.zeros((mi * pi, rank))
    W = np.zeros((pi * ri, rank))
    col = 0
    for i in range(ri):
        for j in range(mi):
            for k in range(pi):
                U[i * mi + j, col] = 1
                V[j * pi + k, col] = 1
                W[k * ri + i, col] = 1
                col += 1
    return U, V, W

def _build_block_decomposition(n, m, p):
    """Build the block-partitioned Strassen decomposition.
    Returns U (n*m, rank), V (m*p, rank), W (p*n, rank), and metadata.
    """
    row_blocks = [2, 1]
    inner_blocks = [2, 2, 1]
    outer_blocks = [2, 2, 2]
    
    row_offsets = [0]
    for s in row_blocks[:-1]:
        row_offsets.append(row_offsets[-1] + s)
    inner_offsets = [0]
    for s in inner_blocks[:-1]:
        inner_offsets.append(inner_offsets[-1] + s)
    outer_offsets = [0]
    for s in outer_blocks[:-1]:
        outer_offsets.append(outer_offsets[-1] + s)
    
    total_rank = 0
    U_parts = []
    V_parts = []
    W_parts = []
    block_info = []  # list of (U_indices, V_indices, W_indices, local_rank, block_key)
    
    for ri, rs in enumerate(row_blocks):
        for mi, ms in enumerate(inner_blocks):
            for pi, ps in enumerate(outer_blocks):
                key = (rs, ms, ps)
                if key == (2, 2, 2):
                    U_b, V_b, W_b = _make_strassen_222()
                else:
                    U_b, V_b, W_b = _make_schoolbook_block(rs, ms, ps)
                rank_b = U_b.shape[1]
                total_rank += rank_b
                
                U_full = np.zeros((n * m, rank_b))
                V_full = np.zeros((m * p, rank_b))
                W_full = np.zeros((p * n, rank_b))
                
                for li in range(rs):
                    for lj in range(ms):
                        gi = row_offsets[ri] + li
                        gj = inner_offsets[mi] + lj
                        U_full[gi * m + gj, :] = U_b[li * ms + lj, :]
                
                for lj in range(ms):
                    for lk in range(ps):
                        gj = inner_offsets[mi] + lj
                        gk = outer_offsets[pi] + lk
                        V_full[gj * p + gk, :] = V_b[lj * ps + lk, :]
                
                for lk in range(ps):
                    for li in range(rs):
                        gk = outer_offsets[pi] + lk
                        gi = row_offsets[ri] + li
                        W_full[gk * n + gi, :] = W_b[lk * rs + li, :]
                
                U_parts.append(U_full)
                V_parts.append(V_full)
                W_parts.append(W_full)
                
                # Track which global indices this block uses
                u_idx = []
                for li in range(rs):
                    for lj in range(ms):
                        u_idx.append((row_offsets[ri] + li) * m + (inner_offsets[mi] + lj))
                v_idx = []
                for lj in range(ms):
                    for lk in range(ps):
                        v_idx.append((inner_offsets[mi] + lj) * p + (outer_offsets[pi] + lk))
                w_idx = []
                for lk in range(ps):
                    for li in range(rs):
                        w_idx.append((outer_offsets[pi] + lk) * n + (row_offsets[ri] + li))
                        
                block_info.append((u_idx, v_idx, w_idx, rank_b, key))
    
    U = np.hstack(U_parts)
    V = np.hstack(V_parts)
    W = np.hstack(W_parts)
    
    return U, V, W, block_info, total_rank

def _verify_decomp(n, m, p, U, V, W, seed=12345):
    """Verify exact decomposition using a fixed random test."""
    rng = np.random.RandomState(seed)
    A = rng.randint(-5, 5, (n, m)).astype(float)
    B = rng.randint(-5, 5, (m, p)).astype(float)
    C_expected = A @ B
    
    rank = U.shape[1]
    C_computed = np.zeros((n, p))
    
    for c in range(rank):
        u_c = U[:, c].reshape(n, m)
        v_c = V[:, c].reshape(m, p)
        w_c = W[:, c].reshape(p, n).T
        
        X = np.sum(A * u_c, axis=1)
        Y = np.sum(B * v_c, axis=0)
        C_computed += w_c * np.outer(X, Y)
    
    if not np.allclose(C_computed, C_expected, atol=1e-9):
        return False
    return True

def _verify_decomp_multi(n, m, p, U, V, W, n_tests=5):
    """Verify with multiple random tests for robustness."""
    for t in range(n_tests):
        if not _verify_decomp(n, m, p, U, V, W, seed=1000 + t * 7919):
            return False
    return True

def _greedy_sparsify(n, m, p, U, V, W, deadline):
    """Try to reduce rank by zeroing out entries and checking if rank can be reduced.
    Strategy: try to find columns that are linearly dependent or can be merged.
    """
    start_time = time.time()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    # First: try to identify and remove linearly dependent columns
    # Check rank of U, V, W matrices
    def try_remove_column(U, V, W, col):
        """Remove column 'col' and check if remaining still decomposes."""
        # This is expensive; only try if we have time
        pass
    
    # Alternative: try to find a lower-rank decomposition by optimization
    # Use the fact that the tensor is bilinear - try random linear combinations
    
    # Strategy: for each pair of columns, try to see if one can be absorbed
    # into the other by adjusting W entries
    
    rank = U.shape[1]
    for c in range(rank - 1, -1, -1):
        if time.time() > deadline:
            break
        
        # Try zeroing out column c and see if we can compensate
        # by adjusting other columns (this is the key structural idea)
        # Skip for now - too expensive. Instead, try a different approach:
        
        # Try to find if column c is a linear combination of others
        # in the tensor sense
        pass
    
    # Better approach: use the structure to find explicit savings
    # The 2x2x2 Strassen blocks are already rank-7 (optimal for 2x2x2)
    # The savings must come from cross-block interactions
    
    # Let's try: for the three 2x2x2 blocks, can we share Strassen products?
    # Block (0,0,0): rows 0-1, cols 0-1, outer 0-1
    # Block (0,0,1): rows 0-1, cols 0-1, outer 2-3
    # Block (0,1,0): rows 0-1, cols 2-3, outer 0-1
    # etc.
    
    # Actually, let's try a completely different approach:
    # Use the known rank upper bound from literature
    # R(3,5,6) <= min of various bounds
    
    # Strassen bound: R(3,5,6) <= 3^log2(7) * (5/2)^log2(7) * (6/2)^log2(7) / 2^log2(7)
    # That's not quite right. Let me use the simple bound:
    # R(3,5,6) <= R(4,8,8) / R(2,2,2) ... no
    
    # Simplest: R(3,5,6) <= 3*5*6 = 90 (schoolbook)
    # With Strassen on 2x2x2: we got 84
    # Can we do better?
    
    # Try: use rank-7 for all 2x2x2 blocks (already done)
    # The remaining blocks are 1x2x2, 2x1x2, 2x2x1, 1x1x2, 2x1x1, 1x2x1, 1x1x1
    # These are already at schoolbook rank (which is optimal for these small sizes)
    
    # So the only way to improve is cross-block sharing.
    # Let's try a random search for a lower-rank decomposition.
    
    return best_U, best_V, best_W, best_rank

def _try_random_rank_reduction(n, m, p, U0, V0, W0, deadline):
    """Try to find a lower-rank decomposition by random perturbation and sparsification."""
    start_time = time.time()
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    best_rank = U0.shape[1]
    
    # Try to reduce rank by 1, 2, 3...
    for target_rank in range(best_rank - 1, max(best_rank - 10, 70), -1):
        if time.time() > deadline:
            break
        
        # Try random initialization near the current solution
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            # Start from current and try to zero out some entries
            U = U0.copy()
            V = V0.copy()
            W = W0.copy()
            
            # Try to find a subset of columns that spans the same tensor
            # by checking if we can drop some columns
            rng = np.random.RandomState(42 + attempt)
            
            # Try dropping random columns and re-solving
            cols_to_try = list(range(best_rank))
            rng.shuffle(cols_to_try)
            
            kept = []
            for c in cols_to_try:
                if len(kept) >= target_rank:
                    break
                kept.append(c)
            
            # Check if the kept columns still decompose the tensor
            U_sub = U[:, kept]
            V_sub = V[:, kept]
            W_sub = W[:, kept]
            
            if _verify_decomp(n, m, p, U_sub, V_sub, W_sub):
                best_U, best_V, best_W = U_sub, V_sub, W_sub
                best_rank = target_rank
                break
    
    return best_U, best_V, best_W, best_rank

def _try_lagrangian_rank_search(n, m, p, deadline):
    """Try to find a lower-rank decomposition using a structured approach.
    
    Key insight: The matrix multiplication tensor T_{ijk} = delta_{j,k} (in the 
    appropriate indices) has a known structure. For 3x5x6, we can try to use
    the fact that the tensor is a sum of rank-1 terms.
    
    Alternative: use the Laderman-like approach for rectangular matrices.
    """
    start_time = time.time()
    
    # Known results:
    # R(2,2,2) = 7 (Strassen)
    # R(2,2,3) = ? (known to be 11 or 12)
    # R(3,3,3) = 23 (Laderman)
    
    # For 3x5x6, we can embed in 4x8x8 and use Strassen recursively
    # R(3,5,6) <= R(4,8,8)
    # R(4,8,8) = R(2,2,2)^3 * R(2,4,4) ... this gets complicated
    
    # Let's try: 3x5x6 can be seen as 3x(4+1)x(4+2)
    # Or: (2+1)x(4+1)x(4+2)
    
    # Use block decomposition with Strassen where possible:
    # Split 3 = 2+1, 5 = 2+2+1, 6 = 2+2+2
    
    # This is what we already have. The rank is 84.
    
    # Can we do better by using a different split?
    # Try: 3 = 2+1, 5 = 3+2, 6 = 3+3
    # Then blocks: (2,3,3), (2,2,3), (1,3,3), (1,2,3)
    # R(2,3,3) <= ? , R(2,2,3) <= 11 or 12, R(1,3,3) = 9, R(1,2,3) = 6
    
    # Actually, let's just try to verify that 84 is correct and return it.
    # The main improvement opportunity is cross-block sharing.
    
    # For now, return the block decomposition as-is.
    return None

def construct():
    n, m, p = 3, 5, 6
    start_time = time.time()
    deadline = start_time + 115
    
    # Build the block-partitioned Strassen decomposition
    U, V, W, block_info, total_rank = _build_block_decomposition(n, m, p)
    
    # Verify
    if not _verify_decomp(n, m, p, U, V, W):
        # Fallback to schoolbook
        rank_schoolbook = n * m * p
        U_school = np.zeros((n * m, rank_schoolbook))
        V_school = np.zeros((m * p, rank_schoolbook))
        W_school = np.zeros((p * n, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j, col] = 1
                    V_school[j * p + k, col] = 1
                    W_school[k * n + i, col] = 1
                    col += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    # Try to improve via greedy sparsification
    if time.time() < deadline - 5:
        U2, V2, W2, rank2 = _greedy_sparsify(n, m, p, U, V, W, deadline - 5)
        if rank2 < total_rank and _verify_decomp_multi(n, m, p, U2, V2, W2):
            U, V, W = U2, V2, W2
            total_rank = rank2
    
    # Try random rank reduction
    if time.time() < deadline - 5:
        U3, V3, W3, rank3 = _try_random_rank_reduction(n, m, p, U, V, W, deadline - 5)
        if rank3 < total_rank and _verify_decomp_multi(n, m, p, U3, V3, W3):
            U, V, W = U3, V3, W3
            total_rank = rank3
    
    # Final verification
    if not _verify_decomp_multi(n, m, p, U, V, W, n_tests=3):
        # Fallback to schoolbook
        rank_schoolbook = n * m * p
        U_school = np.zeros((n * m, rank_schoolbook))
        V_school = np.zeros((m * p, rank_schoolbook))
        W_school = np.zeros((p * n, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j, col] = 1
                    V_school[j * p + k, col] = 1
                    W_school[k * n + i, col] = 1
                    col += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
