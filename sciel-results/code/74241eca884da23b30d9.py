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
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    rank = U_arr.shape[1]
    
    # Check valid entries: for each (i, j, k), a=i*m+j, b=j*p+k, c=k*n+i should give 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U_arr[a, :], np.dot(V_arr[b, :], W_arr[c, :]))
                if abs(val - 1.0) > tol:
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
                    if abs(val) > tol:
                        return False
    
    return True

def _strassen_2x2x2():
    """
    Return rank-7 decomposition for 2x2x2 matrix multiplication.
    U shape (4, 7), V shape (4, 7), W shape (4, 7)
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    # Strassen's 7-multiplication decomposition
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    #
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-d)(f+g)
    # M7 = (b-c)(e+h)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # U: A indices [a,b,c,d] = [0,1,2,3]
    # V: B indices [e,f,g,h] = [0,1,2,3]
    # W: C indices [C11,C12,C21,C22] = [0,1,2,3] where C[k*2+i]
    #   C11: i=0,k=0 -> idx 0
    #   C12: i=0,k=1 -> idx 1
    #   C21: i=1,k=0 -> idx 2
    #   C22: i=1,k=1 -> idx 3
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (a+d)(e+h): U: a=1,d=1; V: e=1,h=1; W: C11=1, C22=1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (c+d)e: U: c=1,d=1; V: e=1; W: C21=1, C22=-1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = a(f-h): U: a=1; V: f=1,h=-1; W: C12=1, C22=1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = d(g-e): U: d=1; V: g=1,e=-1; W: C11=1, C21=1
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (a+b)h: U: a=1,b=1; V: h=1; W: C11=-1, C12=1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (c-d)(f+g): U: c=1,d=-1; V: f=1,g=1; W: C22=1
    U[2, 5] = 1; U[3, 5] = -1
    V[1, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    # M7 = (b-c)(e+h): U: b=1,c=-1; V: e=1,h=1; W: C11=1
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
            # U: same for all blocks (A is 2x2)
            U[:, col] = U2[:, r]
            # V: B[j, block*2 + k] -> index j*p + (block*2 + k)
            for j in range(m):
                for k in range(2):
                    b_idx = j * p + (block * 2 + k)
                    v_idx = j * 2 + k
                    V[b_idx, col] = V2[v_idx, r]
            # W: C[k, i] where k = block*2 + k_local, i as usual
            # W index: k*n + i
            for i in range(n):
                for k_local in range(2):
                    k = block * 2 + k_local
                    w_idx = k * n + i
                    w2_idx = k_local * n + i
                    W[w_idx, col] = W2[w2_idx, r]
    
    return U, V, W

def _try_optimize_rank(n, m, p, rank, seed, time_budget):
    """
    Try to find a rank-rank decomposition using gradient-based optimization.
    Uses the structure that valid entries must be 1 and invalid must be 0.
    """
    start_time = time.time()
    rng = np.random.RandomState(seed)
    
    # Initialize with random values
    U = rng.randn(n * m, rank) * 0.1
    V = rng.randn(m * p, rank) * 0.1
    W = rng.randn(p * n, rank) * 0.1
    
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
    max_invalid = 200
    if len(invalid_triples) > max_invalid:
        idx = rng.choice(len(invalid_triples), max_invalid, replace=False)
        invalid_triples = invalid_triples[idx]
    
    all_triples = np.vstack([valid_triples, invalid_triples])
    targets = np.concatenate([np.ones(len(valid_triples)), np.zeros(len(invalid_triples))])
    
    a_idx = all_triples[:, 0]
    b_idx = all_triples[:, 1]
    c_idx = all_triples[:, 2]
    
    # Gradient descent with Adam-like updates
    lr = 0.01
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    m_U = np.zeros_like(U)
    m_V = np.zeros_like(V)
    m_W = np.zeros_like(W)
    v_U = np.zeros_like(U)
    v_V = np.zeros_like(V)
    v_W = np.zeros_like(W)
    t = 0
    
    best_loss = float('inf')
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
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
        
        # Compute gradients
        # dL/dU[a,r] = sum over triples t where a_idx[t]=a of residuals[t] * V[b_idx[t],r] * W[c_idx[t],r]
        # Vectorized: for each triple, contribution to gradient
        
        # grad_U[a, r] = sum_t (residuals[t] * V[b_idx[t], r] * W[c_idx[t], r]) * (a_idx[t] == a)
        # We can compute this as: for each r, grad_U[:, r] = (V_sub * W_sub * residuals[:, None]).T @ one_hot(a_idx)
        # But that's expensive. Let's use a simpler approach.
        
        # Compute gradients by accumulating
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        # For efficiency, process in batches
        batch_size = min(100, len(all_triples))
        for start in range(0, len(all_triples), batch_size):
            end = min(start + batch_size, len(all_triples))
            a_b = a_idx[start:end]
            b_b = b_idx[start:end]
            c_b = c_idx[start:end]
            res_b = residuals[start:end]
            
            U_b = U[a_b, :]  # (batch, rank)
            V_b = V[b_b, :]
            W_b = W[c_b, :]
            
            # grad_U[a_b, r] += res_b * V_b * W_b
            grad_U[a_b, :] += res_b[:, None] * V_b * W_b
            grad_V[b_b, :] += res_b[:, None] * U_b * W_b
            grad_W[c_b, :] += res_b[:, None] * U_b * V_b
        
        # Adam update
        m_U = beta1 * m_U + (1 - beta1) * grad_U
        v_U = beta2 * v_U + (1 - beta2) * grad_U ** 2
        m_U_hat = m_U / (1 - beta1 ** t)
        v_U_hat = v_U / (1 - beta2 ** t)
        U -= lr * m_U_hat / (np.sqrt(v_U_hat) + eps)
        
        m_V = beta1 * m_V + (1 - beta1) * grad_V
        v_V = beta2 * v_V + (1 - beta2) * grad_V ** 2
        m_V_hat = m_V / (1 - beta1 ** t)
        v_V_hat = v_V / (1 - beta2 ** t)
        V -= lr * m_V_hat / (np.sqrt(v_V_hat) + eps)
        
        m_W = beta1 * m_W + (1 - beta1) * grad_W
        v_W = beta2 * v_W + (1 - beta2) * grad_W ** 2
        m_W_hat = m_W / (1 - beta1 ** t)
        v_W_hat = v_W / (1 - beta2 ** t)
        W -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
        
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
    
    # Check if the best found is valid
    if _verify_decomposition(best_U.tolist(), best_V.tolist(), best_W.tolist(), n, m, p, tol=1e-3):
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Start with replicated Strassen decomposition (rank 56)
    U_str, V_str, W_str = _replicate_strassen(n, m, p)
    best_result = {"U": U_str.tolist(), "V": V_str.tolist(), "W": W_str.tolist()}
    best_rank = 56
    
    # Verify it works
    if _verify_decomposition(best_result["U"], best_result["V"], best_result["W"], n, m, p, tol=1e-6):
        pass  # Good
    else:
        # Fallback to schoolbook
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
        best_result = {"U": U, "V": V, "W": W}
        best_rank = rank_school
    
    # Try to find lower rank decompositions
    # Lower bound: at least max(n*m, m*p, p*n) / something
    # For 2x2x16, lower bound is at least 16 (since output is 2x16=32 entries)
    # But more tightly, rank >= ceil(n*p / m) = ceil(32/2) = 16 for this case? No, that's not right.
    # The rank must be at least the max of the dimensions in some sense.
    # Let's try ranks from 24 up to best_rank - 1
    
    min_rank_to_try = 24
    max_rank_to_try = best_rank - 1
    
    seed = 42
    attempts_per_rank = 2
    
    for rank in range(min_rank_to_try, max_rank_to_try + 1):
        if time.time() > deadline:
            break
        
        for attempt in range(attempts_per_rank):
            if time.time() > deadline:
                break
            
            remaining_time = deadline - time.time()
            per_attempt_budget = remaining_time / max(1, attempts_per_rank - attempt)
            per_attempt_budget = max(5, per_attempt_budget)  # At least 5 seconds
            
            result = _try_optimize_rank(n, m, p, rank, seed, per_attempt_budget)
            seed += 1
            
            if result is not None:
                U, V, W = result
                if _verify_decomposition(U, V, W, n, m, p, tol=1e-4):
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = rank
                    break  # Found this rank, try lower
    
    return best_result

# EVOLVE-BLOCK-END

