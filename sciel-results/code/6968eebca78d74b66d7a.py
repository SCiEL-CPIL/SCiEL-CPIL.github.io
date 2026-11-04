"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _build_tensor(n, m, p):
    """Build the 6x9x6 tensor T where T[a,b,c]=1 iff indices are consistent."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that sum_r U[:,r] * V[:,r] * W[:,r] == T."""
    rank = U.shape[1]
    result = np.zeros(T.shape)
    for r in range(rank):
        result += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    return np.allclose(result, T, atol=tol)

def _make_rank17():
    """Return the known rank-17 decomposition."""
    n, m, p = DIMENSIONS
    n_a, n_b, n_c = n*m, m*p, p*n
    rank = 17
    U = np.zeros((n_a, rank))
    V = np.zeros((n_b, rank))
    W = np.zeros((n_c, rank))
    
    terms = [
        ([0, 4], [1, 1], [0, 4], [1, 1], [0, 3], [1, 1]),
        ([3, 4], [1, 1], [0], [1], [1, 3], [1, -1]),
        ([0], [1], [1, 4], [1, -1], [2, 3], [1, 1]),
        ([4], [1], [3, 0], [1, -1], [0, 1], [1, 1]),
        ([0, 1], [1, 1], [4], [1], [0, 2], [-1, 1]),
        ([3, 0], [1, -1], [0, 1], [1, 1], [3], [1]),
        ([1, 4], [1, -1], [3, 4], [1, 1], [0], [1]),
        ([0], [1], [2], [1], [4], [1]),
        ([1], [1], [5], [1], [4], [1]),
        ([3], [1], [2], [1], [5], [1]),
        ([4], [1], [5], [1], [5], [1]),
        ([2], [1], [6], [1], [0], [1]),
        ([2], [1], [7], [1], [2], [1]),
        ([2], [1], [8], [1], [4], [1]),
        ([5], [1], [6], [1], [1], [1]),
        ([5], [1], [7], [1], [3], [1]),
        ([5], [1], [8], [1], [5], [1]),
    ]
    
    for r, (u_idx, u_val, v_idx, v_val, w_idx, w_val) in enumerate(terms):
        for a, val in zip(u_idx, u_val):
            U[a, r] = val
        for b, val in zip(v_idx, v_val):
            V[b, r] = val
        for c, val in zip(w_idx, w_val):
            W[c, r] = val
    return U, V, W

def _try_sat_rank16(deadline):
    """
    Try to find a rank-16 decomposition with entries in {-1, 0, 1}
    using a local search / SAT-like approach.
    Returns (U, V, W) or None.
    """
    n, m, p = DIMENSIONS
    n_a, n_b, n_c = n*m, m*p, p*n
    T = _build_tensor(n, m, p)
    rank = 16
    
    # Start from rank-17 and try to merge two terms or find a better assignment
    # Strategy: random restart with local search on {-1,0,1}^entries
    
    best_solution = None
    
    for restart in range(50):
        if time.time() > deadline - 5:
            break
        
        # Initialize: take rank-17 and zero out one column, then perturb
        U, V, W = _make_rank17()
        
        # Remove one column (randomly choose which)
        drop_col = random.randint(0, 16)
        U = np.delete(U, drop_col, axis=1)
        V = np.delete(V, drop_col, axis=1)
        W = np.delete(W, drop_col, axis=1)
        
        # Now we have 16 columns. Try to fix them by local search.
        # Objective: minimize sum of squared differences from T
        # Each entry can be -1, 0, or 1
        
        # Flatten for easier manipulation
        U_flat = U.flatten()
        V_flat = V.flatten()
        W_flat = W.flatten()
        
        # Local search: try changing individual entries
        max_iters = 2000
        for it in range(max_iters):
            if time.time() > deadline - 3:
                break
            
            # Compute current error
            result = np.zeros((n_a, n_b, n_c))
            for r in range(rank):
                result += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            err = np.sum((result - T) ** 2)
            
            if err < 1e-6:
                return U, V, W
            
            # Pick a random entry to flip
            which = random.randint(0, 2)  # 0=U, 1=V, 2=W
            if which == 0:
                a, r = random.randint(0, n_a-1), random.randint(0, rank-1)
                old_val = U[a, r]
            elif which == 1:
                b, r = random.randint(0, n_b-1), random.randint(0, rank-1)
                old_val = V[b, r]
            else:
                c, r = random.randint(0, n_c-1), random.randint(0, rank-1)
                old_val = W[c, r]
            
            # Try a different value from {-1, 0, 1}
            new_val = random.choice([-1, 0, 1])
            if new_val == old_val:
                new_val = (new_val + 1) % 3 - 1  # -1->0->1->-1
            
            # Apply change
            if which == 0:
                U[a, r] = new_val
            elif which == 1:
                V[b, r] = new_val
            else:
                W[c, r] = new_val
            
            # Check if error decreased
            result_new = np.zeros((n_a, n_b, n_c))
            for r in range(rank):
                result_new += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            err_new = np.sum((result_new - T) ** 2)
            
            if err_new < err:
                pass  # keep the change
            else:
                # Revert
                if which == 0:
                    U[a, r] = old_val
                elif which == 1:
                    V[b, r] = old_val
                else:
                    W[c, r] = old_val
        
        # Check final
        result = np.zeros((n_a, n_b, n_c))
        for r in range(rank):
            result += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        if np.allclose(result, T, atol=1e-9):
            return U, V, W
    
    return None

def _try_structural_rank16(deadline):
    """
    Try structural modifications to the rank-17 solution to get rank 16.
    Key insight: In the rank-17 solution, terms 8-11 handle A1(2x2)*B_right(2x1)
    and terms 12-17 handle A2(2x1)*B2(1x3). Some of these might be mergeable
    with the Strassen terms.
    
    Specifically, term 14 (A[0,2]*B[2,2] -> C[0,2]) and term 17 (A[1,2]*B[2,2] -> C[1,2])
    both contribute to column 2 of C. Similarly terms 8-11 also contribute to column 2.
    Maybe we can combine the column-2 contributions more efficiently.
    
    Let's try: instead of 4 separate terms for A1*B_right, use 2 terms that also
    help with other parts.
    """
    n, m, p = DIMENSIONS
    n_a, n_b, n_c = n*m, m*p, p*n
    T = _build_tensor(n, m, p)
    rank = 16
    
    # Alternative structure: Use 7 Strassen-like terms for the 2x2x2 core,
    # then 9 terms for the remaining contributions, but try to share.
    
    # Let me try a different block decomposition:
    # C = A1*B1 + A2*B2 where A1 is 2x2, A2 is 2x1, B1 is 2x3, B2 is 1x3
    # C[:,0:2] = A1*B1[:,0:2] + A2*B2[:,0:2]
    # C[:,2] = A1*B1[:,2] + A2*B2[:,2]
    
    # For C[:,2]: C[i,2] = A[i,0]*B[0,2] + A[i,1]*B[1,2] + A[i,2]*B[2,2]
    # That's 6 terms naively. But maybe we can use 3 terms with shared structure.
    
    # Try: use the Strassen 7 terms for the 2x2x2 part (giving C[0:2,0:2] contributions)
    # Then handle C[:,2] and A2*B2[:,0:2] with 9 more terms = 16 total
    
    # But that's the same as before (7 + 4 + 6 = 17).
    # The issue is C[:,2] needs 4 terms from A1*B1[:,2] and A2*B2[:,2] adds 2 more.
    
    # What if we fold A2*B2 into the Strassen computation?
    # A is 2x3, B is 3x3.
    # Let me try: partition B as B = [B_left(3x2), B_right(3x1)]
    # C[:,0:2] = A*B_left, C[:,2] = A*B_right
    
    # A*B_left: A is 2x3, B_left is 3x2 -> 2x2 result
    # This is a 2x3x2 multiplication. Rank of this tensor?
    # T2[a,b,c] where a in [0,5] (A entries), b in [0,5] (B_left entries), c in [0,3] (C entries)
    # a = i*3+j, b = j*2+k, c = k*2+i for i in 0,1; j in 0,1,2; k in 0,1
    # This has 12 nonzero entries.
    
    # Hmm, let me just try a direct SAT-style search for rank 16.
    
    # Encode as: for each rank r, each factor entry is in {-1,0,1}
    # Total variables: 6*16 + 9*16 + 6*16 = 96 + 144 + 96 = 336
    # Constraints: 6*9*6 = 324 equations (but only 18 are nonzero in T)
    
    # This is too large for exhaustive search. Let me try a smarter approach.
    
    # Use the known rank-17 and try to eliminate one term by adjusting others.
    # Pick term 11 (A[1,1]*B[1,2] -> C[1,2]) as the one to eliminate.
    # Its contribution: U[4,11]=1, V[5,11]=1, W[5,11]=1
    # We need to redistribute this contribution among the other 15 terms.
    
    # This is getting complex. Let me try a different approach:
    # Use bilinear complexity results. The rank of 2x3 by 3x3 matrix multiplication.
    # Known: R(2,3,3) <= 17 (from Strassen + direct).
    # Is R(2,3,3) = 16 possible?
    
    # Lower bound: omega(2,3,3) >= 2*3*3 / 2^(log2(3)-1) ... not helpful.
    # Simple lower bound: max(mn, mp, np) = max(6,9,6) = 9. Not tight.
    # Another: the tensor has 18 nonzero entries, so rank >= 18/... no.
    
    # Let me just try a more aggressive local search.
    
    random.seed(42)
    
    for restart in range(100):
        if time.time() > deadline - 5:
            break
        
        # Start from rank-17, drop a random column, then do more aggressive search
        U, V, W = _make_rank17()
        drop_col = random.randint(0, 16)
        U = np.delete(U, drop_col, axis=1)
        V = np.delete(V, drop_col, axis=1)
        W = np.delete(W, drop_col, axis=1)
        
        # Aggressive local search with simulated annealing-like acceptance
        T_local = T
        current_err = None
        
        for it in range(5000):
            if time.time() > deadline - 3:
                break
            
            # Compute error efficiently
            result = np.zeros((n_a, n_b, n_c))
            for r in range(rank):
                result += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            err = np.sum((result - T_local) ** 2)
            
            if err < 1e-6:
                return U, V, W
            
            if current_err is None:
                current_err = err
            
            # Pick random entry
            which = random.randint(0, 2)
            if which == 0:
                a, r = random.randint(0, n_a-1), random.randint(0, rank-1)
                old_val = U[a, r]
            elif which == 1:
                b, r = random.randint(0, n_b-1), random.randint(0, rank-1)
                old_val = V[b, r]
            else:
                c, r = random.randint(0, n_c-1), random.randint(0, rank-1)
                old_val = W[c, r]
            
            new_val = random.choice([-1, 0, 1])
            if new_val == old_val:
                new_val = (new_val + 1) % 3 - 1
            
            if which == 0:
                U[a, r] = new_val
            elif which == 1:
                V[b, r] = new_val
            else:
                W[c, r] = new_val
            
            result_new = np.zeros((n_a, n_b, n_c))
            for r in range(rank):
                result_new += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            err_new = np.sum((result_new - T_local) ** 2)
            
            # Accept if better, or with small probability if worse (annealing)
            if err_new <= current_err or (it > 1000 and random.random() < 0.01):
                current_err = err_new
            else:
                if which == 0:
                    U[a, r] = old_val
                elif which == 1:
                    V[b, r] = old_val
                else:
                    W[c, r] = old_val
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    n_a, n_b, n_c = n*m, m*p, p*n
    
    # First, try to find rank 16
    T = _build_tensor(n, m, p)
    
    # Try SAT-style search for rank 16
    result_16 = _try_sat_rank16(deadline)
    if result_16 is not None:
        U, V, W = result_16
        if _verify_decomposition(U, V, W, T):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try structural approach
    result_16b = _try_structural_rank16(deadline)
    if result_16b is not None:
        U, V, W = result_16b
        if _verify_decomposition(U, V, W, T):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to rank 17
    U, V, W = _make_rank17()
    if _verify_decomposition(U, V, W, T):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Final fallback: schoolbook
    rank_fb = n * m * p
    U_fb = np.zeros((n_a, rank_fb))
    V_fb = np.zeros((n_b, rank_fb))
    W_fb = np.zeros((n_c, rank_fb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
# EVOLVE-BLOCK-END
