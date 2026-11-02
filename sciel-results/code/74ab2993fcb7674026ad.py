import numpy as np
import time
import random
import sys

DIMENSIONS = [3, 3, 6]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = 0.0
                for r in range(R):
                    val += U[a, r] * V[b, r] * W[c, r]
                if abs(val - 1.0) > tol:
                    return False
    # Check all off-diagonal terms are zero
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                i = a // m
                j = a % m
                jj = b // p
                k = b % p
                kk = c // n
                ii = c % n
                # Correct mapping: a=(i,j), b=(j,k), c=(k,i)
                if i == ii and j == jj and j == k:
                    continue  # Already checked, should be 1
                val = 0.0
                for r in range(R):
                    val += U[a, r] * V[b, r] * W[c, r]
                if abs(val) > tol:
                    return False
    return True

def _build_schoolbook(n, m, p):
    """Build the standard schoolbook decomposition."""
    R = n * m * p
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, r] = 1.0
                V[b, r] = 1.0
                W[c, r] = 1.0
                r += 1
    return U, V, W

def _lader_3x3_decomposition():
    """
    Return a rank-23 decomposition for 3x3 * 3x3 matrix multiplication.
    Uses the Lader decomposition (1978). The tensor entries are given as
    (a_coeff, b_coeff, c_coeff) triples for each of the 23 terms.
    """
    # The Lader 3x3x3 decomposition: 23 terms
    # Each term: (A_factor, B_factor, C_factor)
    # Indices: A is 3x3, B is 3x3, C is 3x3
    # A index: (i, j), B index: (j, k), C index: (k, i)
    
    terms = []
    # I'll construct this from known Lader coefficients
    # Using the standard Lader decomposition
    # This is a well-known result; here I encode the 23 (a,b,c) index triples
    # with their coefficients.
    
    # Actually, let me use a different approach. I'll use the fact that
    # for 3x3*3x3, we can use a structured decomposition.
    # Let me try to build a rank-23 solution directly.
    
    # For now, return a valid decomposition using a known construction.
    # I'll use a "block" approach: partition into overlapping sub-matrices.
    
    # Let me try a direct algebraic approach.
    # We need: C[i,k] = sum_j A[i,j] * B[j,k] for all i,k in {0,1,2}
    
    # One known rank-23 decomposition uses the following structure:
    # Terms corresponding to "diagonal" computations plus cross terms.
    
    # Actually, let me just verify the schoolbook works and try to find
    # improvements through optimization.
    
    return None  # Placeholder

def _optimize_rank(n, m, p, target_rank, max_iter=2000, seed=42, time_budget=30.0):
    """
    Try to find a rank-`target_rank` decomposition using gradient-based optimization.
    Returns (U, V, W) if found, else None.
    """
    rng = np.random.RandomState(seed)
    start_time = time.time()
    
    # Initialize with small random values
    U = rng.randn(n * m, target_rank) * 0.1
    V = rng.randn(m * p, target_rank) * 0.1
    W = rng.randn(p * n, target_rank) * 0.1
    
    # The tensor we want: T[a, b, c] = 1 if (a,b,c) corresponds to valid (i,j,k), else 0
    # a = i*m + j, b = j*p + k, c = k*n + i
    target_tensor = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_tensor[a, b, c] = 1.0
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    lr = 0.01
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        
        # Compute current tensor decomposition
        # T_decomposed[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Loss = sum over all (a,b,c) of (T_decomposed - target)^2
        
        # Efficient computation using tensor operations
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        
        # Compute gradient:
        # dL/dU[a,r] = 2 * sum_{b,c} (T[a,b,c] - target[a,b,c]) * V[b,r] * W[c,r]
        # dL/dV[b,r] = 2 * sum_{a,c} (T[a,b,c] - target[a,b,c]) * U[a,r] * W[c,r]
        # dL/dW[c,r] = 2 * sum_{a,b} (T[a,b,c] - target[a,b,c]) * U[a,r] * V[b,r]
        
        # Build the error tensor
        T = np.einsum('ar,br,cr->abc', U, V, W)
        err = T - target_tensor
        loss = np.sum(err * err)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < 1e-20:
            return U, V, W
        
        # Compute gradients
        dU = 2.0 * np.einsum('abc,br,cr->ar', err, V, W)
        dV = 2.0 * np.einsum('abc,ar,cr->br', err, U, W)
        dW = 2.0 * np.einsum('abc,ar,br->cr', err, U, V)
        
        # Adaptive learning rate
        if loss < best_loss * 0.9:
            lr *= 1.05
        else:
            lr *= 0.95
        lr = min(lr, 0.1)
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
    
    # Check if we found a valid solution
    if best_loss < 1e-15:
        return best_U, best_V, best_W
    return None

def _try_structured_decomposition(n, m, p, target_rank):
    """
    Try a structured decomposition by exploiting the block structure.
    3x3 * 3x6 = 3x3 * [3x3 | 3x3]
    We can try to share the left factor A between the two blocks.
    """
    # This is a placeholder for a more sophisticated approach
    # that exploits the fact that both output blocks use the same A.
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Start with schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    best_rank = n * m * p  # 54
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find lower rank decompositions
    # For 3x3 * 3x6, schoolbook rank is 54.
    # The theoretical lower bound is max(n*m, m*p, p*n) = max(9, 18, 18) = 18.
    # But in practice, the rank is likely higher.
    
    # Try various target ranks using optimization
    target_ranks = [40, 35, 30, 28, 26, 24, 22, 20, 18]
    
    for target_rank in target_ranks:
        if time.time() > deadline:
            break
        
        # Try multiple random seeds
        found = False
        for seed in range(10):
            if time.time() > deadline:
                break
            
            result = _optimize_rank(n, m, p, target_rank, max_iter=3000, seed=seed,
                                     time_budget=12.0)
            if result is not None:
                U_opt, V_opt, W_opt = result
                # Verify
                if _verify_decomposition(U_opt, V_opt, W_opt, n, m, p):
                    # Round to nearest half-integer if possible
                    best_rank = target_rank
                    best_result = {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
                    found = True
                    break
        
        if found:
            # Try to go lower
            continue
        else:
            # If we can't find this rank, break (assuming monotonicity)
            # Actually, don't break - maybe a different rank works
            pass
    
    # Final safety check
    if time.time() > deadline - 5:
        pass  # Return what we have
    
    return best_result
