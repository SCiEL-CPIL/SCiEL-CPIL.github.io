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
    U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    W indexed by k*n + i (output factor follows W[k*n+i, r])
    """
    try:
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        # Ensure 2D
        if U_arr.ndim != 2 or V_arr.ndim != 2 or W_arr.ndim != 2:
            return False
            
        # Check valid entries: for each (i, j, k), a=i*m+j, b=j*p+k, c=k*n+i should give 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(U_arr[a, :], np.dot(V_arr[b, :], W_arr[c, :]))
                    if np.abs(val - 1.0) > tol:
                        return False
        
        # Check invalid entries should be 0
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
                        val = np.dot(U_arr[a, :], np.dot(V_arr[b, :], W_arr[c, :]))
                        if np.abs(val) > tol:
                            return False
        
        return True
    except Exception:
        return False

def _strassen_2x2x2():
    """
    Return rank-7 decomposition for 2x2x2 matrix multiplication.
    U shape (4, 7), V shape (4, 7), W shape (4, 7)
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (c-d)(f+g)
    U[2, 5] = 1; U[3, 5] = -1
    V[1, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    # M7 = (b-c)(e+h)
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
    
    num_blocks = p // 2  # 8 blocks
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

def _try_structured_optimization(n, m, p, rank, seed, time_budget):
    """
    Try to find a rank-rank decomposition using structured optimization.
    Exploits the block structure of the problem.
    """
    start_time = time.time()
    rng = np.random.RandomState(seed)
    
    # Initialize with random values
    U = rng.randn(n * m, rank) * 0.5
    V = rng.randn(m * p, rank) * 0.5
    W = rng.randn(p * n, rank) * 0.5
    
    # Collect valid triples
    valid_triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid_triples.append((a, b, c))
    valid_triples = np.array(valid_triples)
    
    # Collect invalid triples (sample if too many)
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
    
    # Subsample invalid if too many
    max_invalid = 500
    if len(invalid_triples) > max_invalid:
        idx = rng.choice(len(invalid_triples), max_invalid, replace=False)
        invalid_triples = invalid_triples[idx]
    
    all_triples = np.vstack([valid_triples, invalid_triples])
    targets = np.concatenate([np.ones(len(valid_triples)), np.zeros(len(invalid_triples))])
    
    a_idx = all_triples[:, 0]
    b_idx = all_triples[:, 1]
    c_idx = all_triples[:, 2]
    
    # Use L-BFGS-like approach with careful initialization
    lr = 0.05
    momentum = 0.9
    
    v_U = np.zeros_like(U)
    v_V = np.zeros_like(V)
    v_W = np.zeros_like(W)
    
    best_loss = float('inf')
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    for step in range(3000):
        if time.time() - start_time > time_budget:
            break
        
        # Compute predictions
        U_sub = U[a_idx, :]
        V_sub = V[b_idx, :]
        W_sub = W[c_idx, :]
        
        preds = np.sum(U_sub * V_sub * W_sub, axis=1)
        residuals = preds - targets
        
        # Compute gradients
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        # Process in batches for efficiency
        batch_size = min(200, len(all_triples))
        for start in range(0, len(all_triples), batch_size):
            end = min(start + batch_size, len(all_triples))
            a_b = a_idx[start:end]
            b_b = b_idx[start:end]
            c_b = c_idx[start:end]
            res_b = residuals[start:end]
            
            U_b = U[a_b, :]
            V_b = V[b_b, :]
            W_b = W[c_b, :]
            
            grad_U[a_b, :] += res_b[:, None] * V_b * W_b
            grad_V[b_b, :] += res_b[:, None] * U_b * W_b
            grad_W[c_b, :] += res_b[:, None] * U_b * V_b
        
        # Momentum update
        v_U = momentum * v_U + (1 - momentum) * grad_U
        v_V = momentum * v_V + (1 - momentum) * grad_V
        v_W = momentum * v_W + (1 - momentum) * grad_W
        
        U -= lr * v_U
        V -= lr * v_V
        W -= lr * v_W
        
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
        
        # Early stopping if loss is very small
        if best_loss < 1e-10:
            break
    
    # Check if the best found is valid
    if _verify_decomposition(best_U.tolist(), best_V.tolist(), best_W.tolist(), n, m, p, tol=1e-4):
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    
    return None

def _try_block_shared(n, m, p, rank, seed, time_budget):
    """
    Try a decomposition where we share structure across p-blocks.
    For 2x2x16, we have 8 blocks of 2x2x2.
    Try to find a shared U and block-specific V, W.
    """
    start_time = time.time()
    rng = np.random.RandomState(seed)
    
    num_blocks = p // 2  # 8 blocks
    block_rank = rank // num_blocks + 1  # approximate rank per block
    
    # Try different rank allocations
    best_result = None
    best_loss = float('inf')
    
    for attempt in range(5):
        if time.time() - start_time > time_budget:
            break
        
        # Random rank allocation
        remaining_rank = rank
        block_ranks = []
        for b in range(num_blocks - 1):
            r_b = max(1, min(remaining_rank - (num_blocks - 1 - b), rng.randint(1, 8)))
            block_ranks.append(r_b)
            remaining_rank -= r_b
        block_ranks.append(max(1, remaining_rank))
        
        # Initialize
        U = rng.randn(n * m, rank) * 0.3
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col_offset = 0
        for b in range(num_blocks):
            r_b = block_ranks[b]
            V[:, col_offset:col_offset+r_b] = rng.randn(m * p, r_b) * 0.3
            W[:, col_offset:col_offset+r_b] = rng.randn(p * n, r_b) * 0.3
            col_offset += r_b
        
        # Optimize
        lr = 0.03
        momentum = 0.9
        v_U = np.zeros_like(U)
        v_V = np.zeros_like(V)
        v_W = np.zeros_like(W)
        
        # Collect triples
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
        
        max_invalid = 300
        if len(invalid_triples) > max_invalid:
            idx = rng.choice(len(invalid_triples), max_invalid, replace=False)
            invalid_triples = invalid_triples[idx]
        
        all_triples = np.vstack([valid_triples, invalid_triples])
        targets = np.concatenate([np.ones(len(valid_triples)), np.zeros(len(invalid_triples))])
        a_idx = all_triples[:, 0]
        b_idx = all_triples[:, 1]
        c_idx = all_triples[:, 2]
        
        for step in range(1500):
            if time.time() - start_time > time_budget:
                break
            
            U_sub = U[a_idx, :]
            V_sub = V[b_idx, :]
            W_sub = W[c_idx, :]
            preds = np.sum(U_sub * V_sub * W_sub, axis=1)
            residuals = preds - targets
            
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            batch_size = min(150, len(all_triples))
            for start in range(0, len(all_triples), batch_size):
                end = min(start + batch_size, len(all_triples))
                a_b = a_idx[start:end]
                b_b = b_idx[start:end]
                c_b = c_idx[start:end]
                res_b = residuals[start:end]
                
                U_b = U[a_b, :]
                V_b = V[b_b, :]
                W_b = W[c_b, :]
                
                grad_U[a_b, :] += res_b[:, None] * V_b * W_b
                grad_V[b_b, :] += res_b[:, None] * U_b * W_b
                grad_W[c_b, :] += res_b[:, None] * U_b * V_b
            
            v_U = momentum * v_U + (1 - momentum) * grad_U
            v_V = momentum * v_V + (1 - momentum) * grad_V
            v_W = momentum * v_W + (1 - momentum) * grad_W
            
            U -= lr * v_U
            V -= lr * v_V
            W -= lr * v_W
            
            U_sub = U[a_idx, :]
            V_sub = V[b_idx, :]
            W_sub = W[c_idx, :]
            preds = np.sum(U_sub * V_sub * W_sub, axis=1)
            loss = np.sum((preds - targets) ** 2)
            
            if loss < best_loss:
                best_loss = loss
                best_result = (U.copy(), V.copy(), W.copy())
            
            if best_loss < 1e-10:
                break
    
    if best_result is not None:
        U, V, W = best_result
        if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e-4):
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin
    
    # Fallback schoolbook decomposition
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
        
        # Verify it works
        if not _verify_decomposition(best_result["U"], best_result["V"], best_result["W"], n, m, p, tol=1e-6):
            best_result = _schoolbook()
            best_rank = n * m * p
        
        # Try to find lower rank decompositions
        min_rank_to_try = 20
        max_rank_to_try = best_rank - 1
        
        seed = 42
        methods = [_try_structured_optimization, _try_block_shared]
        
        for rank in range(min_rank_to_try, max_rank_to_try + 1):
            if time.time() > deadline:
                break
            
            for method in methods:
                if time.time() > deadline:
                    break
                
                remaining_time = deadline - time.time()
                per_attempt_budget = remaining_time / 2
                per_attempt_budget = max(3, per_attempt_budget)
                
                result = method(n, m, p, rank, seed, per_attempt_budget)
                seed += 1
                
                if result is not None:
                    U, V, W = result
                    if _verify_decomposition(U, V, W, n, m, p, tol=1e-4):
                        best_result = {"U": U, "V": V, "W": W}
                        best_rank = rank
                        break  # Found this rank, try lower
        
        return best_result
    except Exception:
        # If anything goes wrong, return schoolbook
        return _schoolbook()

# EVOLVE-BLOCK-END
