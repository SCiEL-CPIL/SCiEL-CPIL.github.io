"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = U.shape[1]
    # Build the tensor T[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Check against the true multiplication tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # True value: delta[i][j] * delta[j][k] which is 1 if this is a "product" term
                # Actually, the matrix multiplication tensor has T[i,j,k] = 1 if the (i,j) entry of A
                # times (j,k) entry of B contributes to (i,k) of C
                # T[i*m+j, j*p+k, k*n+i] should equal 1 for the "diagonal" terms
                # More precisely: T[a,b,c] = 1 iff a = i*m+j, b = j*p+k, c = k*n+i for some valid i,j,k
                # And the value is 1 (since it's the identity in the tensor sense)
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # The true tensor value: this is 1 if the indices form a valid "edge" in the bipartite graph
                # For matrix mult, T[i*m+j, j*p+k, k*n+i] = 1 always (it's the structural tensor)
                # Wait, no. The tensor T satisfies: C[i,k] = sum_j A[i,j]*B[j,k]
                # So T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
                expected = 1.0
                if abs(val - expected) > 1e-9:
                    return False, val, expected
    return True, 0.0, 1.0

def _try_block_decomposition(n, m, p, block_sizes, block_decomps, deadline):
    """
    Try a block decomposition with given block sizes and precomputed decompositions.
    block_sizes: list of (n_b, m_b, p_b) but here all n_b=n, m_b=m, p_b varies
    block_decomps: list of (U, V, W) for each block
    Returns (rank, U, V, W) or None if invalid
    """
    total_rank = sum(U.shape[1] for U, V, W in block_decomps)
    U_full = np.zeros((n * m, total_rank))
    V_full = np.zeros((m * p, total_rank))
    W_full = np.zeros((p * n, total_rank))
    
    col_offset = 0
    k_offset = 0
    for idx, (U_b, V_b, W_b) in enumerate(block_decomps):
        p_b = block_sizes[idx]
        rank_b = U_b.shape[1]
        
        U_full[:, col_offset:col_offset+rank_b] = U_b
        
        for j in range(m):
            for k_local in range(p_b):
                k_global = k_offset + k_local
                V_full[j * p + k_global, col_offset:col_offset+rank_b] = V_b[j * p_b + k_local, :]
        
        for k_local in range(p_b):
            k_global = k_offset + k_local
            for i in range(n):
                W_full[k_global * n + i, col_offset:col_offset+rank_b] = W_b[k_local * n + i, :]
        
        col_offset += rank_b
        k_offset += p_b
    
    # Verify
    valid, _, _ = _verify_decomposition(U_full, V_full, W_full, n, m, p)
    if not valid:
        return None
    
    return total_rank, U_full, V_full, W_full

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140s deadline with 10s safety margin
    
    # Pre-compute Strassen matrices for 2x2x2
    U_strassen = np.array([
        [1, 0, 1, 0, 1, -1, 0], 
        [0, 0, 0, 0, 1, 0, 1],  
        [0, 1, 0, 0, 0, 1, 0],  
        [1, 1, 0, 1, 0, 0, -1]  
    ])
    
    V_strassen = np.array([
        [1, 1, 0, -1, 0, 1, 0], 
        [0, 0, 1, 0, 0, 1, 0],  
        [0, 0, 0, 1, 0, 0, 1],  
        [1, 0, -1, 0, 1, 0, 1]  
    ])
    
    W_strassen = np.array([
        [1, 0, 0, 1, -1, 0, 1], 
        [0, 1, 0, 1, 0, 0, 0],  
        [0, 0, 1, 0, 1, 0, 0],  
        [1, -1, 1, 0, 0, 1, 0]  
    ])
    
    # Pre-compute Schoolbook matrices for 2x2x1
    U_school = np.array([
        [1, 0, 0, 0], 
        [0, 1, 0, 0], 
        [0, 0, 1, 0], 
        [0, 0, 0, 1]  
    ])
    
    V_school = np.array([
        [1, 0, 1, 0], 
        [0, 1, 0, 1]  
    ])
    
    W_school = np.array([
        [1, 1, 0, 0], 
        [0, 0, 1, 1]  
    ])
    
    # Helper: create a block decomposition for 2x2xp using given p
    def make_block(p_b):
        """Create a 2x2xp_b block decomposition."""
        if p_b == 1:
            return U_school.copy(), V_school.copy(), W_school.copy()
        elif p_b == 2:
            return U_strassen.copy(), V_strassen.copy(), W_strassen.copy()
        else:
            # For p_b >= 3, use a combination of Strassen and schoolbook
            # This is a fallback; we'll try to find better decompositions
            blocks = []
            remaining = p_b
            while remaining > 2:
                blocks.append((U_strassen.copy(), V_strassen.copy(), W_strassen.copy()))
                remaining -= 2
            if remaining == 1:
                blocks.append((U_school.copy(), V_school.copy(), W_school.copy()))
            elif remaining == 2:
                blocks.append((U_strassen.copy(), V_strassen.copy(), W_strassen.copy()))
            
            # Concatenate blocks
            total_rank = sum(U.shape[1] for U, V, W in blocks)
            U_block = np.zeros((n * m, total_rank))
            V_block = np.zeros((m * p_b, total_rank))
            W_block = np.zeros((p_b * n, total_rank))
            
            col_off = 0
            k_off = 0
            for U_b, V_b, W_b in blocks:
                p_bb = V_b.shape[0] // m
                r_b = U_b.shape[1]
                U_block[:, col_off:col_off+r_b] = U_b
                for j in range(m):
                    for k_l in range(p_bb):
                        V_block[j * p_b + k_off + k_l, col_off:col_off+r_b] = V_b[j * p_bb + k_l, :]
                for k_l in range(p_bb):
                    for i in range(n):
                        W_block[(k_off + k_l) * n + i, col_off:col_off+r_b] = W_b[k_l * n + i, :]
                col_off += r_b
                k_off += p_bb
            
            return U_block, V_block, W_block
    
    best_rank = 53
    best_result = None
    
    # Strategy 1: Original block decomposition (7x2 + 1x1)
    block_decomps = []
    block_sizes = []
    for _ in range(7):
        block_decomps.append((U_strassen.copy(), V_strassen.copy(), W_strassen.copy()))
        block_sizes.append(2)
    block_decomps.append((U_school.copy(), V_school.copy(), W_school.copy()))
    block_sizes.append(1)
    
    result = _try_block_decomposition(n, m, p, block_sizes, block_decomps, deadline)
    if result:
        rank, U, V, W = result
        if rank < best_rank:
            best_rank = rank
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if time.time() > deadline:
        if best_result:
            return best_result
        # Fallback to original
        total_rank = 7 * 7 + 4
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        col_offset = 0
        for b in range(7):
            k_start = 2 * b
            U[:, col_offset:col_offset+7] = U_strassen
            for j in range(m):
                for k_local in range(2):
                    k_global = k_start + k_local
                    V[j * p + k_global, col_offset:col_offset+7] = V_strassen[j * 2 + k_local, :]
            for k_local in range(2):
                k_global = k_start + k_local
                for i in range(n):
                    W[k_global * n + i, col_offset:col_offset+7] = W_strassen[k_local * 2 + i, :]
            col_offset += 7
        k_start = 14
        U[:, col_offset:col_offset+4] = U_school
        for j in range(m):
            V[j * p + k_start, col_offset:col_offset+4] = V_school[j, :]
        for i in range(n):
            W[k_start * n + i, col_offset:col_offset+4] = W_school[i, :]
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Strategy 2: Try 3x(2x2x4) + 1x(2x2x3) = 3*13 + 10 = 49
    # We need rank-13 2x2x4 and rank-10 2x2x3 decompositions
    # For now, use the fallback make_block which may not be optimal
    # Let's try a few different block size combinations
    
    # Try: 4x(2x2x3) + 1x(2x2x3) = 5x3 = 15, each 2x2x3
    # Or: 3x(2x2x4) + 1x(2x2x3)
    # Or: 2x(2x2x4) + 1x(2x2x7)
    
    # For now, just return the best we have
    if best_result:
        return best_result
    
    # Fallback to original
    total_rank = 7 * 7 + 4
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    col_offset = 0
    for b in range(7):
        k_start = 2 * b
        U[:, col_offset:col_offset+7] = U_strassen
        for j in range(m):
            for k_local in range(2):
                k_global = k_start + k_local
                V[j * p + k_global, col_offset:col_offset+7] = V_strassen[j * 2 + k_local, :]
        for k_local in range(2):
            k_global = k_start + k_local
            for i in range(n):
                W[k_global * n + i, col_offset:col_offset+7] = W_strassen[k_local * 2 + i, :]
        col_offset += 7
    k_start = 14
    U[:, col_offset:col_offset+4] = U_school
    for j in range(m):
        V[j * p + k_start, col_offset:col_offset+4] = V_school[j, :]
    for i in range(n):
        W[k_start * n + i, col_offset:col_offset+4] = W_school[i, :]
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
