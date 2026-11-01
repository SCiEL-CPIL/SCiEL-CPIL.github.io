"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    Vectorized for speed.
    """
    try:
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        if U_arr.ndim != 2 or V_arr.ndim != 2 or W_arr.ndim != 2:
            return False
        
        rank = U_arr.shape[1]
        if V_arr.shape[1] != rank or W_arr.shape[1] != rank:
            return False
        
        # Build valid triples
        valid_a, valid_b, valid_c = [], [], []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    valid_a.append(i * m + j)
                    valid_b.append(j * p + k)
                    valid_c.append(k * n + i)
        
        valid_a = np.array(valid_a)
        valid_b = np.array(valid_b)
        valid_c = np.array(valid_c)
        
        # Check valid entries: sum_r U[a,r]*V[b,r]*W[c,r] should be 1
        vals = np.sum(U_arr[valid_a] * V_arr[valid_b] * W_arr[valid_c], axis=1)
        if np.max(np.abs(vals - 1.0)) > tol:
            return False
        
        # Build invalid triples
        all_a = np.arange(n * m)
        all_b = np.arange(m * p)
        all_c = np.arange(p * n)
        
        # Create meshgrid
        A_grid, B_grid, C_grid = np.meshgrid(all_a, all_b, all_c, indexing='ij')
        A_flat = A_grid.ravel()
        B_flat = B_grid.ravel()
        C_flat = C_grid.ravel()
        
        # Determine validity
        i_a = A_flat // m
        j_a = A_flat % m
        j_b = B_flat // p
        k_b = B_flat % p
        k_c = C_flat // n
        i_c = C_flat % n
        
        valid_mask = (j_a == j_b) & (k_b == k_c) & (i_c == i_a)
        invalid_mask = ~valid_mask
        
        if np.any(invalid_mask):
            inv_a = A_flat[invalid_mask]
            inv_b = B_flat[invalid_mask]
            inv_c = C_flat[invalid_mask]
            
            # Sample if too many
            max_check = 10000
            if len(inv_a) > max_check:
                rng = np.random.RandomState(42)
                idx = rng.choice(len(inv_a), max_check, replace=False)
                inv_a = inv_a[idx]
                inv_b = inv_b[idx]
                inv_c = inv_c[idx]
            
            inv_vals = np.sum(U_arr[inv_a] * V_arr[inv_b] * W_arr[inv_c], axis=1)
            if np.max(np.abs(inv_vals)) > tol:
                return False
        
        return True
    except Exception:
        return False

def _strassen_2x2x2():
    """
    Return rank-7 decomposition for 2x2x2 matrix multiplication.
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    U[2, 5] = 1; U[3, 5] = -1
    V[1, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[2, 6] = -1
    V[0, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _replicate_strassen(n=2, m=2, p=16):
    """
    Replicate the 2x2x2 Strassen decomposition 8 times for 2x2x16.
    Rank = 7 * 8 = 56
    """
    U2, V2, W2 = _strassen_2x2x2()
    
    num_blocks = p // 2
    rank = 7 * num_blocks
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for block in range(num_blocks):
        for r in range(7):
            col = block * 7 + r
            U[:, col] = U2[:, r]
            for j in range(m):
                for k in range(2):
                    b_idx = j * p + (block * 2 + k)
                    v_idx = j * 2 + k
                    V[b_idx, col] = V2[v_idx, r]
            for i in range(n):
                for k_local in range(2):
                    k = block * 2 + k_local
                    w_idx = k * n + i
                    w2_idx = k_local * n + i
                    W[w_idx, col] = W2[w2_idx, r]
    
    return U, V, W

def _round_to_half_int(arr, tol=1e-3):
    """Round array to nearest half-integer if close."""
    result = arr.copy()
    # Try rounding to nearest 0.5
    rounded = np.round(result * 2) / 2
    if np.max(np.abs(result - rounded)) < tol:
        return rounded
    # Try rounding to nearest integer
    rounded_int = np.round(result)
    if np.max(np.abs(result - rounded_int)) < tol:
        return rounded_int
    return result

def _try_find_decomposition(n, m, p, rank, seed, time_budget, init=None):
    """
    Try to find a rank-rank decomposition using gradient-based optimization
    with vectorized gradient computation and integer rounding.
    """
    start_time = time.time()
    rng = np.random.RandomState(seed)
    
    # Collect all valid and invalid triples
    valid_triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid_triples.append((a, b, c))
    valid_triples = np.array(valid_triples)
    
    invalid_triples = []
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                i_a = a // m
                j_a = a % m
                j_b = b // p
                k_b = b % p
                k_c = c // n
                i_c = c % n
                if not (j_a == j_b and k_b == k_c and i_c == i_a):
                    invalid_triples.append((a, b, c))
    invalid_triples = np.array(invalid_triples) if invalid_triples else np.empty((0, 3), dtype=int)
    
    max_invalid = 5000
    if len(invalid_triples) > max_invalid:
        idx = rng.choice(len(invalid_triples), max_invalid, replace=False)
        invalid_triples = invalid_triples[idx]
    
    all_triples = np.vstack([valid_triples, invalid_triples])
    targets = np.concatenate([np.ones(len(valid_triples)), np.zeros(len(invalid_triples))])
    a_idx = all_triples[:, 0]
    b_idx = all_triples[:, 1]
    c_idx = all_triples[:, 2]
    
    # Initialize
    if init is not None:
        U, V, W = init
        U = U.astype(np.float64).copy()
        V = V.astype(np.float64).copy()
        W = W.astype(np.float64).copy()
    else:
        U = rng.randn(n * m, rank) * 0.3
        V = rng.randn(m * p, rank) * 0.3
        W = rng.randn(p * n, rank) * 0.3
    
    # Adam optimizer
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    m_U = np.zeros_like(U)
    v_U = np.zeros_like(U)
    m_V = np.zeros_like(V)
    v_V = np.zeros_like(V)
    m_W = np.zeros_like(W)
    v_W = np.zeros_like(W)
    t = 0
    lr = 0.01
    
    best_loss = float('inf')
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    num_triples = len(all_triples)
    
    for step in range(5000):
        if time.time() - start_time > time_budget:
            break
        t += 1
        
        # Compute predictions
        U_sub = U[a_idx, :]
        V_sub = V[b_idx, :]
        W_sub = W[c_idx, :]
        preds = np.sum(U_sub * V_sub * W_sub, axis=1)
        residuals = preds - targets
        
        # Compute gradients vectorized
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        UVW_U = residuals[:, None] * V_sub * W_sub
        UVW_V = residuals[:, None] * U_sub * W_sub
        UVW_W = residuals[:, None] * U_sub * V_sub
        
        np.add.at(grad_U, a_idx, UVW_U)
        np.add.at(grad_V, b_idx, UVW_V)
        np.add.at(grad_W, c_idx, UVW_W)
        
        # Adam update
        lr_t = lr * np.sqrt(1 - beta2**t) / (1 - beta1**t)
        
        m_U = beta1 * m_U + (1 - beta1) * grad_U
        v_U = beta2 * v_U + (1 - beta2) * grad_U**2
        U -= lr_t * m_U / (np.sqrt(v_U) + eps)
        
        m_V = beta1 * m_V + (1 - beta1) * grad_V
        v_V = beta2 * v_V + (1 - beta2) * grad_V**2
        V -= lr_t * m_V / (np.sqrt(v_V) + eps)
        
        m_W = beta1 * m_W + (1 - beta1) * grad_W
        v_W = beta2 * v_W + (1 - beta2) * grad_W**2
        W -= lr_t * m_W / (np.sqrt(v_W) + eps)
        
        # Compute loss
        U_sub = U[a_idx, :]
        V_sub = V[b_idx, :]
        W_sub = W[c_idx, :]
        preds = np.sum(U_sub * V_sub * W_sub, axis=1)
        loss = np.sum((preds - targets) ** 2)
        
        if loss < best_loss:
            best_loss = loss
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
        
        if best_loss < 1e-10:
            break
    
    # Try rounding to half-integers
    U_round = _round_to_half_int(best_U)
    V_round = _round_to_half_int(best_V)
    W_round = _round_to_half_int(best_W)
    
    if _verify_decomposition(U_round.tolist(), V_round.tolist(), W_round.tolist(), n, m, p, tol=1e-4):
        return U_round.tolist(), V_round.tolist(), W_round.tolist()
    
    if _verify_decomposition(best_U.tolist(), best_V.tolist(), W_round.tolist(), n, m, p, tol=1e-4):
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    
    return None

def _try_circulant_decomposition(n, m, p, target_rank, seed, time_budget):
    """
    Try a block-circulant structured decomposition using DFT across blocks.
    Treat the p=16 dimension as 8 blocks of size 2.
    Use circulant structure: each "frequency" gets a rank-r decomposition.
    """
    start_time = time.time()
    rng = np.random.RandomState(seed)
    
    U2, V2, W2 = _strassen_2x2x2()
    num_blocks = p // 2  # 8 blocks
    
    best_result = None
    
    # Try different rank allocations per frequency
    for r_per_freq in range(8, 3, -1):
        if time.time() - start_time > time_budget:
            break
        
        total_rank = r_per_freq * num_blocks
        if total_rank > target_rank:
            continue
        
        # Initialize using DFT structure
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        
        # For each frequency f (0 to 7), assign r_per_freq columns
        # The DFT coupling means block b gets contribution from all frequencies
        # with phase exp(2πi*f*b/8)
        
        for f in range(num_blocks):
            for r in range(r_per_freq):
                col = f * r_per_freq + r
                
                if r < 7:
                    U[:, col] = U2[:, r]
                    
                    for j in range(m):
                        for k_local in range(2):
                            for block in range(num_blocks):
                                b_idx = j * p + (block * 2 + k_local)
                                v_idx = j * 2 + k_local
                                # DFT phase
                                phase = np.exp(2j * np.pi * f * block / num_blocks)
                                V[b_idx, col] = V2[v_idx, r] * phase.real
                                # Store imaginary part in a companion column if needed
                                # For now, just use real part
                                if r + 1 < r_per_freq:
                                    # Use next column for imaginary part
                                    V[b_idx, col + 1] = V2[v_idx, r] * phase.imag
                            
                    for i in range(n):
                        for k_local in range(2):
                            for block in range(num_blocks):
                                k = block * 2 + k_local
                                w_idx = k * n + i
                                w2_idx = k_local * n + i
                                phase = np.exp(2j * np.pi * f * block / num_blocks)
                                W[w_idx, col] = W2[w2_idx, r] * phase.real
                                if r + 1 < r_per_freq:
                                    W[w_idx, col + 1] = W2[w2_idx, r] * phase.imag
                else:
                    pass
        
        # Optimize
        result = _try_find_decomposition(n, m, p, total_rank, seed, time_budget / 2, init=(U, V, W))
        if result is not None:
            best_result = result
            break
    
    return best_result

def _try_block_shared(n, m, p, target_rank, seed, time_budget):
    """
    Try a block-shared structure where all 8 blocks share the same rank-r
    decomposition, with the coupling handled by linear combinations.
    """
    start_time = time.time()
    rng = np.random.RandomState(seed)
    
    U2, V2, W2 = _strassen_2x2x2()
    num_blocks = p // 2  # 8 blocks
    
    best_result = None
    
    # Try: use rank r for the shared structure, total rank = r * num_blocks
    # but with the constraint that U is the same for all blocks
    for r in range(8, 3, -1):
        if time.time() - start_time > time_budget:
            break
        
        total_rank = r * num_blocks
        if total_rank > target_rank:
            continue
        
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        
        for block in range(num_blocks):
            for r_idx in range(r):
                col = block * r + r_idx
                if r_idx < 7:
                    U[:, col] = U2[:, r_idx]
                    for j in range(m):
                        for k_local in range(2):
                            b_idx = j * p + (block * 2 + k_local)
                            v_idx = j * 2 + k_local
                            V[b_idx, col] = V2[v_idx, r_idx]
                    for i in range(n):
                        for k_local in range(2):
                            k = block * 2 + k_local
                            w_idx = k * n + i
                            w2_idx = k_local * n + i
                            W[w_idx, col] = W2[w2_idx, r_idx]
        
        # Optimize with shared U constraint (relaxed by optimization)
        result = _try_find_decomposition(n, m, p, total_rank, seed, time_budget / 3, init=(U, V, W))
        if result is not None:
            best_result = result
            break
    
    return best_result

def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for safety
    
    def _schoolbook():
        rank_school = n * m * p
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    try:
        # Start with replicated Strassen decomposition (rank 56)
        U_str, V_str, W_str = _replicate_strassen(n, m, p)
        best_result = {"U": U_str.tolist(), "V": V_str.tolist(), "W": W_str.tolist()}
        best_rank = 56
        
        if not _verify_decomposition(best_result["U"], best_result["V"], best_result["W"], n, m, p, tol=1e-6):
            best_result = _schoolbook()
            best_rank = n * m * p
        
        # Try to find lower rank decompositions
        # Focus on ranks that are plausible: 55, 54, ..., down to 32
        seed = 42
        
        for rank in range(55, 31, -1):
            if time.time() > deadline:
                break
            
            remaining_time = deadline - time.time()
            if remaining_time < 5:
                break
            
            # Try circulant structured approach
            per_attempt_budget = min(remaining_time * 0.25, 12)
            if per_attempt_budget > 3:
                result = _try_circulant_decomposition(n, m, p, rank, seed, per_attempt_budget)
                seed += 1
                
                if result is not None:
                    U, V, W = result
                    if _verify_decomposition(U, V, W, n, m, p, tol=1e-4):
                        best_result = {"U": U, "V": V, "W": W}
                        best_rank = rank
                        continue
            
            # Try block-shared approach
            per_attempt_budget = min(remaining_time * 0.25, 12)
            if per_attempt_budget > 3:
                result = _try_block_shared(n, m, p, rank, seed, per_attempt_budget)
                seed += 1
                
                if result is not None:
                    U, V, W = result
                    if _verify_decomposition(U, V, W, n, m, p, tol=1e-4):
                        best_result = {"U": U, "V": V, "W": W}
                        best_rank = rank
                        continue
            
            # Try fresh optimization with random init
            if time.time() < deadline - 5:
                result = _try_find_decomposition(n, m, p, rank, seed, remaining_time * 0.25)
                seed += 1
                
                if result is not None:
                    U, V, W = result
                    if _verify_decomposition(U, V, W, n, m, p, tol=1e-4):
                        best_result = {"U": U, "V": V, "W": W}
                        best_rank = rank
                        continue
        
        return best_result
    except Exception:
        return _schoolbook()

# EVOLVE-BLOCK-END
