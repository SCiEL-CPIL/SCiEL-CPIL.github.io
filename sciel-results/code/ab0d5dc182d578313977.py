import time
import sys
import numpy as np
from itertools import product

N, M, P = 2, 3, 7
DEADLINE = 130.0

def _get_tensor_entry(a, b, c):
    """
    Get the tensor entry T[a,b,c] where:
    a = i*M + j (index into A)
    b = j*P + k (index into B)  
    c = k*N + i (index into C, using W[k*n+i] convention)
    
    T[a,b,c] = 1 if there exist i,j,k such that:
      a = i*M + j, b = j*P + k, c = k*N + i
    else T[a,b,c] = 0
    """
    n, m, p = N, M, P
    # a = i*m + j => i = a // m, j = a % m
    i_from_a = a // m
    j_from_a = a % m
    
    # b = j*p + k => j = b // p, k = b % p
    j_from_b = b // p
    k_from_b = b % p
    
    # c = k*n + i => k = c // n, i = c % n
    k_from_c = c // n
    i_from_c = c % n
    
    # Check consistency
    if i_from_a == i_from_c and j_from_a == j_from_b and k_from_b == k_from_c:
        return 1
    else:
        return 0

def _verify(U, V, W):
    """Verify exact tensor decomposition."""
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    rank = U.shape[1]
    n, m, p = N, M, P
    
    # Compute tensor T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # and check against expected values
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                expected = _get_tensor_entry(a, b, c)
                val = 0
                for r in range(rank):
                    val += U[a, r] * V[b, r] * W[c, r]
                if val != expected:
                    return False
    return True

def _build_trivial():
    """Build the trivial rank-42 decomposition."""
    n, m, p = N, M, P
    trivial_rank = n * m * p
    
    U = np.zeros((n*m, trivial_rank), dtype=object)
    V = np.zeros((m*p, trivial_rank), dtype=object)
    W = np.zeros((p*n, trivial_rank), dtype=object)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
    
    return U.tolist(), V.tolist(), W.tolist()

def _compute_tensor(U, V, W, rank):
    """Compute the full tensor from factor matrices."""
    n, m, p = N, M, P
    T = {}
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                val = 0
                for r in range(rank):
                    val += U[a, r] * V[b, r] * W[c, r]
                T[(a, b, c)] = val
    return T

def _greedy_decomposition(target_rank, max_iterations=3000, seed=42):
    """
    Try to find a decomposition with target_rank terms.
    """
    n, m, p = N, M, P
    np.random.seed(seed)
    
    # Initialize with random values
    U = np.zeros((n*m, target_rank), dtype=object)
    V = np.zeros((m*p, target_rank), dtype=object)
    W = np.zeros((p*n, target_rank), dtype=object)
    
    for r in range(target_rank):
        for a in range(n*m):
            U[a, r] = np.random.choice([-1, 0, 1])
        for b in range(m*p):
            V[b, r] = np.random.choice([-1, 0, 1])
        for c in range(p*n):
            W[c, r] = np.random.choice([-1, 0, 1])
    
    # Precompute expected tensor
    expected = {}
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                expected[(a,b,c)] = _get_tensor_entry(a, b, c)
    
    for iteration in range(max_iterations):
        # Compute current tensor
        T = _compute_tensor(U, V, W, target_rank)
        
        # Check if we've achieved the target
        success = True
        for key, exp_val in expected.items():
            if T[key] != exp_val:
                success = False
                break
        
        if success:
            return U.tolist(), V.tolist(), W.tolist()
        
        # Find entries with errors
        errors = []
        for key, exp_val in expected.items():
            err = T[key] - exp_val
            if err != 0:
                errors.append((abs(err), key))
        
        if not errors:
            break
        
        errors.sort(reverse=True)
        
        # Try to fix the worst error
        worst_err, (a, b, c) = errors[0]
        
        # Try adjusting random columns
        improved = False
        for _ in range(20):
            r = np.random.randint(target_rank)
            
            # Try changing U[a, r]
            old_u = U[a, r]
            for new_u in [-2, -1, 0, 1, 2]:
                if new_u == old_u:
                    continue
                U[a, r] = new_u
                # Recompute error at this position
                new_val = 0
                for rr in range(target_rank):
                    new_val += U[a, rr] * V[b, rr] * W[c, rr]
                new_err = abs(new_val - expected[(a,b,c)])
                if new_err < worst_err:
                    improved = True
                    break
                U[a, r] = old_u
            
            if improved:
                break
            
            # Try changing V[b, r]
            old_v = V[b, r]
            for new_v in [-2, -1, 0, 1, 2]:
                if new_v == old_v:
                    continue
                V[b, r] = new_v
                new_val = 0
                for rr in range(target_rank):
                    new_val += U[a, rr] * V[b, rr] * W[c, rr]
                new_err = abs(new_val - expected[(a,b,c)])
                if new_err < worst_err:
                    improved = True
                    break
                V[b, r] = old_v
            
            if improved:
                break
            
            # Try changing W[c, r]
            old_w = W[c, r]
            for new_w in [-2, -1, 0, 1, 2]:
                if new_w == old_w:
                    continue
                W[c, r] = new_w
                new_val = 0
                for rr in range(target_rank):
                    new_val += U[a, rr] * V[b, rr] * W[c, rr]
                new_err = abs(new_val - expected[(a,b,c)])
                if new_err < worst_err:
                    improved = True
                    break
                W[c, r] = old_w
        
        if not improved:
            # Randomize one column
            r = np.random.randint(target_rank)
            for a_idx in range(n*m):
                U[a_idx, r] = np.random.choice([-1, 0, 1])
            for b_idx in range(m*p):
                V[b_idx, r] = np.random.choice([-1, 0, 1])
            for c_idx in range(p*n):
                W[c_idx, r] = np.random.choice([-1, 0, 1])
    
    # Final check
    T = _compute_tensor(U, V, W, target_rank)
    for key, exp_val in expected.items():
        if T[key] != exp_val:
            return None
    
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    start_time = time.time()
    
    # Start with trivial solution
    U_trivial, V_trivial, W_trivial = _build_trivial()
    best_solution = {"U": U_trivial, "V": V_trivial, "W": W_trivial}
    best_rank = N * M * P
    
    # Try to find lower rank decompositions
    for target_rank in range(best_rank - 1, 15, -1):
        if time.time() > DEADLINE - 10:
            break
        
        for seed in range(3):
            if time.time() > DEADLINE - 5:
                break
            
            result = _greedy_decomposition(target_rank, max_iterations=2000, seed=seed)
            if result is not None:
                U, V, W = result
                if _verify(U, V, W):
                    best_solution = {"U": U, "V": V, "W": W}
                    best_rank = target_rank
                    break
    
    return best_solution
