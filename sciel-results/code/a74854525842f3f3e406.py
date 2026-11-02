"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

import numpy as np
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def _verify_rank(n, m, p, rank, U, V, W):
    """Verify that the given factors decompose the matmul tensor correctly."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                if abs(val - 1.0) > 1e-6:
                    return False, (i, j, k, val)
    return True, None

def _update_W(U, V, W, n, m, p, r):
    """Update W rows given U and V."""
    for i in range(n):
        for k in range(p):
            A = np.zeros((m, r))
            for j in range(m):
                A[j, :] = U[i*m + j, :] * V[j*p + k, :]
            sol, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
            W[k*n + i, :] = sol

def _update_U(U, V, W, n, m, p, r):
    """Update U rows given V and W."""
    for i in range(n):
        for j in range(m):
            B = np.zeros((p, r))
            for k in range(p):
                B[k, :] = V[j*p + k, :] * W[k*n + i, :]
            sol, _, _, _ = np.linalg.lstsq(B, np.ones(p), rcond=None)
            U[i*m + j, :] = sol

def _update_V(U, V, W, n, m, p, r):
    """Update V rows given U and W."""
    for j in range(m):
        for k in range(p):
            C = np.zeros((n, r))
            for i in range(n):
                C[i, :] = U[i*m + j, :] * W[k*n + i, :]
            sol, _, _, _ = np.linalg.lstsq(C, np.ones(n), rcond=None)
            V[j*p + k, :] = sol

def _is_half_integer(mat, tol=1e-5):
    """Check if all entries are close to multiples of 0.5."""
    rounded = np.round(mat * 2) / 2
    return np.allclose(mat, rounded, atol=tol)

def _round_to_half_integer(mat):
    return np.round(mat * 2) / 2

def _compute_error(U, V, W, n, m, p, r):
    """Compute the sum of squared errors for the decomposition."""
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                err += (val - 1.0)**2
    return err

def _als_optimize(U, V, W, n, m, p, r, max_iters=200, tol=1e-8, deadline=None):
    """Run ALS optimization with proper loop structure."""
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        err = _compute_error(U, V, W, n, m, p, r)
        if err < tol:
            break
        
        # Update W
        _update_W(U, V, W, n, m, p, r)
        # Update U
        _update_U(U, V, W, n, m, p, r)
        # Update V
        _update_V(U, V, W, n, m, p, r)
    
    return U, V, W

def _polish_solution(U, V, W, n, m, p, r, deadline=None):
    """Final high-precision polish pass."""
    for it in range(50):
        if deadline is not None and time.time() > deadline:
            break
        
        err = _compute_error(U, V, W, n, m, p, r)
        if err < 1e-10:
            break
        
        _update_W(U, V, W, n, m, p, r)
        _update_U(U, V, W, n, m, p, r)
        _update_V(U, V, W, n, m, p, r)
    
    return U, V, W

def _structured_init(n, m, p, target_rank, seed=42):
    """Create a structured initialization based on schoolbook with perturbations."""
    rng = np.random.RandomState(seed)
    U = np.zeros((n * m, target_rank))
    V = np.zeros((m * p, target_rank))
    W = np.zeros((p * n, target_rank))
    
    # Start with schoolbook-like structure
    for c in range(min(target_rank, n * m * p)):
        i = c // (m * p)
        j = (c // p) % m
        k = c % p
        
        U[i * m + j, c] = 1.0
        V[j * p + k, c] = 1.0
        W[k * n + i, c] = 1.0
    
    if target_rank < n * m * p:
        for c in range(target_rank):
            U[:, c] += rng.randn(n * m) * 0.1
            V[:, c] += rng.randn(m * p) * 0.1
            W[:, c] += rng.randn(p * n) * 0.1
    
    return U, V, W

def _try_merge_schoolbook(n, m, p, target_rank, deadline=None):
    """Try to find a solution by starting from schoolbook and merging terms."""
    if target_rank >= n * m * p:
        U = np.zeros((n * m, target_rank))
        V = np.zeros((m * p, target_rank))
        W = np.zeros((p * n, target_rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = i * m * p + j * p + k
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
        return U, V, W
    
    U, V, W = _structured_init(n, m, p, target_rank, seed=42)
    U, V, W = _als_optimize(U, V, W, n, m, p, target_rank, max_iters=300, tol=1e-9, deadline=deadline)
    
    err = _compute_error(U, V, W, n, m, p, target_rank)
    
    if err < 1e-6:
        if _is_half_integer(U) and _is_half_integer(V) and _is_half_integer(W):
            U_r = _round_to_half_integer(U)
            V_r = _round_to_half_integer(V)
            W_r = _round_to_half_integer(W)
            valid, _ = _verify_rank(n, m, p, target_rank, U_r, V_r, W_r)
            if valid:
                return U_r, V_r, W_r
        else:
            U, V, W = _polish_solution(U, V, W, n, m, p, target_rank, deadline=deadline)
            err = _compute_error(U, V, W, n, m, p, target_rank)
            if err < 1e-8:
                if _is_half_integer(U) and _is_half_integer(V) and _is_half_integer(W):
                    U_r = _round_to_half_integer(U)
                    V_r = _round_to_half_integer(V)
                    W_r = _round_to_half_integer(W)
                    valid, _ = _verify_rank(n, m, p, target_rank, U_r, V_r, W_r)
                    if valid:
                        return U_r, V_r, W_r
    
    return None

def _try_block_embedding(n, m, p, deadline=None):
    """
    Try block-embedding strategy: partition into 2x2x3 subproblems.
    For 4x4x6, we can partition into blocks of size 2x2x3.
    Each 2x2x3 matmul has schoolbook rank 12.
    If we can find a rank-5 or lower decomposition for 2x2x3, we get total rank 4*5=20.
    
    Actually, let's think more carefully. The tensor for 2x2x3 multiplication:
    T[i,j,k] = sum over the multiplication.
    
    The standard approach: try to find a low-rank decomposition for the 2x2x3 sub-tensor,
    then embed it into the full 4x4x6 by placing it in appropriate blocks.
    """
    if deadline is not None and time.time() > deadline:
        return None
    
    # For 4x4x6, partition into 2x2x3 blocks
    # Block (bi, bj, bk) covers rows 2*bi:2*bi+2, cols 2*bj:2*bj+2, and "depth" 3*bk:3*bk+3
    # bi in {0,1}, bj in {0,1}, bk in {0,1}
    # So 2*2*2 = 8 blocks, each of size 2x2x3
    
    # The schoolbook rank for 2x2x3 is 12.
    # Let's try to find a rank-r decomposition for 2x2x3, then total rank = 8*r
    
    # First, let's try to find a good decomposition for 2x2x3
    # The 2x2x3 matmul tensor: C[i,j] = sum_k A[i,k] * B[k,j] where A is 2x3, B is 3x2
    # Wait, no. For multiplying a 2x2 matrix by a 2x3 matrix:
    # Actually, the subproblem structure depends on how we partition.
    
    # Let me reconsider. For 4x4 * 4x6 = 4x6:
    # A is 4x4, B is 4x6, C = A*B is 4x6
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Partition: split i into 2 groups of 2, j into 2 groups of 2, k into 2 groups of 3
    # Block (bi, bj, bk): C[2*bi:2*bi+2, 3*bk:3*bk+3] gets contribution from A[2*bi:2*bi+2, 2*bj:2*bj+2] * B[2*bj:2*bj+2, 3*bk:3*bk+3]
    
    # So each block is a 2x2 * 2x3 = 2x3 multiplication
    # The sub-tensor for 2x2 * 2x3 has indices (i', j', k') where i' in {0,1}, j' in {0,1}, k' in {0,1,2}
    # T[i',j',k'] = A[i',j'] * B[j',k']
    # This is a 2x2x3 tensor with 12 entries.
    
    # Schoolbook rank is 12 (one term per (i',j',k') triple).
    # Can we do better? The rank of this tensor...
    
    # Let's try to find a low-rank decomposition for the 2x2x3 sub-tensor
    # and then embed it.
    
    sub_n, sub_m, sub_p = 2, 2, 3
    sub_schoolbook = sub_n * sub_m * sub_p  # 12
    
    # Try ranks from 5 to 11 for the sub-problem
    best_sub_rank = None
    best_sub_U = None
    best_sub_V = None
    best_sub_W = None
    
    for sub_r in range(5, sub_schoolbook + 1):
        if deadline is not None and time.time() > deadline:
            break
        
        found = False
        for attempt in range(5):
            if deadline is not None and time.time() > deadline:
                break
            
            rng = np.random.RandomState(42 + attempt)
            sU = rng.randn(sub_n * sub_m, sub_r) * 0.5
            sV = rng.randn(sub_m * sub_p, sub_r) * 0.5
            sW = rng.randn(sub_p * sub_n, sub_r) * 0.5
            
            # Normalize
            for c in range(sub_r):
                norm = np.sqrt(np.sum(sU[:, c]**2) * np.sum(sV[:, c]**2) * np.sum(sW[:, c]**2))
                if norm > 1e-10:
                    scale_c = 1.0 / (norm ** (1.0/3.0))
                    sU[:, c] *= scale_c
                    sV[:, c] *= scale_c
                    sW[:, c] *= scale_c
            
            sU, sV, sW = _als_optimize(sU, sV, sW, sub_n, sub_m, sub_p, sub_r, max_iters=100, deadline=deadline)
            err = _compute_error(sU, sV, sW, sub_n, sub_m, sub_p, sub_r)
            
            if err < 1e-8:
                if _is_half_integer(sU) and _is_half_integer(sV) and _is_half_integer(sW):
                    sU_r = _round_to_half_integer(sU)
                    sV_r = _round_to_half_integer(sV)
                    sW_r = _round_to_half_integer(sW)
                    valid, _ = _verify_rank(sub_n, sub_m, sub_p, sub_r, sU_r, sV_r, sW_r)
                    if valid:
                        best_sub_rank = sub_r
                        best_sub_U = sU_r
                        best_sub_V = sV_r
                        best_sub_W = sW_r
                        found = True
                        break
                else:
                    sU, sV, sW = _polish_solution(sU, sV, sW, sub_n, sub_m, sub_p, sub_r, deadline=deadline)
                    err = _compute_error(sU, sV, sW, sub_n, sub_m, sub_p, sub_r)
                    if err < 1e-9:
                        if _is_half_integer(sU) and _is_half_integer(sV) and _is_half_integer(sW):
                            sU_r = _round_to_half_integer(sU)
                            sV_r = _round_to_half_integer(sV)
                            sW_r = _round_to_half_integer(sW)
                            valid, _ = _verify_rank(sub_n, sub_m, sub_p, sub_r, sU_r, sV_r, sW_r)
                            if valid:
                                best_sub_rank = sub_r
                                best_sub_U = sU_r
                                best_sub_V = sV_r
                                best_sub_W = sW_r
                                found = True
                                break
        
        if found:
            break
    
    if best_sub_rank is None:
        return None
    
    # Now embed the sub-decomposition into the full 4x4x6
    total_rank = 8 * best_sub_rank
    U = np.zeros((4 * 4, total_rank))
    V = np.zeros((4 * 6, total_rank))
    W = np.zeros((6 * 4, total_rank))
    
    for bi in range(2):
        for bj in range(2):
            for bk in range(2):
                base = (bi * 2 + bj) * 2 + bk  # block index 0..7
                offset = base * best_sub_rank
                
                for si in range(2):
                    for sj in range(2):
                        i = 2 * bi + si
                        j = 2 * bj + sj
                        for c in range(best_sub_rank):
                            U[i * 4 + j, offset + c] = best_sub_U[si * 2 + sj, c]
                
                for sj in range(2):
                    for sk in range(3):
                        j = 2 * bj + sj
                        k = 3 * bk + sk
                        for c in range(best_sub_rank):
                            V[j * 6 + k, offset + c] = best_sub_V[sj * 3 + sk, c]
                
                for sk in range(3):
                    for si in range(2):
                        k = 3 * bk + sk
                        i = 2 * bi + si
                        for c in range(best_sub_rank):
                            W[k * 4 + i, offset + c] = best_sub_W[sk * 2 + si, c]
    
    # Verify
    valid, _ = _verify_rank(4, 4, 6, total_rank, U, V, W)
    if valid:
        return U, V, W
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s margin out of 150s
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 96
    
    # Initialize best with schoolbook solution
    best_rank = schoolbook_rank
    best_U = np.zeros((n * m, schoolbook_rank))
    best_V = np.zeros((m * p, schoolbook_rank))
    best_W = np.zeros((p * n, schoolbook_rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                best_U[i * m + j, c] = 1
                best_V[j * p + k, c] = 1
                best_W[k * n + i, c] = 1
                
    random.seed(42)
    np.random.seed(42)
    
    # First, try block embedding strategy
    result = _try_block_embedding(n, m, p, deadline=deadline)
    if result is not None:
        U, V, W = result
        rank = U.shape[1]
        if rank < best_rank:
            best_rank = rank
            best_U, best_V, best_W = U, V, W
    
    # Then try searching for lower ranks
    for target_rank in range(min(best_rank - 1, 95), max(min(best_rank - 1, 95) - 30, 0), -1):
        if time.time() > deadline:
            break
        
        found = False
        
        # Try structured initialization first
        result = _try_merge_schoolbook(n, m, p, target_rank, deadline=deadline)
        if result is not None:
            best_rank = target_rank
            best_U, best_V, best_W = result
            found = True
        
        if not found:
            for attempt in range(5):
                if time.time() > deadline:
                    break
                
                scale = 0.5
                U = np.random.randn(n * m, target_rank) * scale
                V = np.random.randn(m * p, target_rank) * scale
                W = np.random.randn(p * n, target_rank) * scale
                
                for c in range(target_rank):
                    norm = np.sqrt(np.sum(U[:, c]**2) * np.sum(V[:, c]**2) * np.sum(W[:, c]**2))
                    if norm > 1e-10:
                        scale_c = 1.0 / (norm ** (1.0/3.0))
                        U[:, c] *= scale_c
                        V[:, c] *= scale_c
                        W[:, c] *= scale_c
                
                U, V, W = _als_optimize(U, V, W, n, m, p, target_rank, max_iters=150, deadline=deadline)
                err = _compute_error(U, V, W, n, m, p, target_rank)
                
                if err < 1e-6:
                    if _is_half_integer(U) and _is_half_integer(V) and _is_half_integer(W):
                        U_r = _round_to_half_integer(U)
                        V_r = _round_to_half_integer(V)
                        W_r = _round_to_half_integer(W)
                        valid, _ = _verify_rank(n, m, p, target_rank, U_r, V_r, W_r)
                        if valid:
                            best_rank = target_rank
                            best_U = U_r
                            best_V = V_r
                            best_W = W_r
                            found = True
                            break
                    else:
                        U, V, W = _polish_solution(U, V, W, n, m, p, target_rank, deadline=deadline)
                        err = _compute_error(U, V, W, n, m, p, target_rank)
                        if err < 1e-8:
                            if _is_half_integer(U) and _is_half_integer(V) and _is_half_integer(W):
                                U_r = _round_to_half_integer(U)
                                V_r = _round_to_half_integer(V)
                                W_r = _round_to_half_integer(W)
                                valid, _ = _verify_rank(n, m, p, target_rank, U_r, V_r, W_r)
                                if valid:
                                    best_rank = target_rank
                                    best_U = U_r
                                    best_V = V_r
                                    best_W = W_r
                                    found = True
                                    break
        
        if found:
            continue
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
