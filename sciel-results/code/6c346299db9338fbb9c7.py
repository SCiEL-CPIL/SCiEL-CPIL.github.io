import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition exactly reconstructs the matrix multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check non-zero entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(val - 1.0) > tol:
                    return False
    
    # Check zero entries (sample a few to save time, or check all if fast enough)
    # Total entries in T are (n*m)*(m*p)*(p*n) = 48*16*16 = 12288
    # Rank is likely small, so sum is fast.
    # We need to ensure T[a,b,c] = 0 for invalid (a,b,c).
    # Invalid means a,b,c do not correspond to a valid i,j,k.
    # a = i*m + j, b = j*p + k, c = k*n + i.
    # We can construct a set of valid (a,b,c) tuples.
    valid_tups = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid_tups.add((i*m+j, j*p+k, k*n+i))
                
    # Iterate over all possible a,b,c? That is 12288 iterations.
    # For each, sum over rank.
    # If rank is 20, 12288 * 20 = 245k ops. Very fast.
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                if (a, b, c) in valid_tups:
                    continue
                val = 0.0
                for r in range(rank):
                    val += U_arr[a, r] * V_arr[b, r] * W_arr[c, r]
                if abs(val) > tol:
                    return False
    return True

def _try_find_decomposition(n, m, p, target_rank, max_iter=2000, lr=0.01, seed=42):
    """
    Try to find a decomposition of exactly target_rank.
    """
    np.random.seed(seed)
    
    U = np.random.randn(n * m, target_rank) * 0.1
    V = np.random.randn(m * p, target_rank) * 0.1
    W = np.random.randn(p * n, target_rank) * 0.1
    
    best_error = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        # Compute gradients
        # Error = sum_{i,j,k} (sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] - 1)^2
        #       + sum_{a,b,c not valid} (sum_r U[a,r] V[b,r] W[c,r])^2
        
        # This is expensive to compute naively.
        # We can use the tensor structure.
        # T_target is sparse.
        
        # Let's just compute the error and gradients for the non-zero part first,
        # and assume the zero part is small initially or handle it separately.
        # Actually, the zero part is crucial.
        
        # Efficient computation:
        # Let A = U, B = V, C = W.
        # We want to minimize F(A,B,C) = || T_target - sum_r A_r \otimes B_r \otimes C_r ||^2
        
        # We can approximate the gradient by sampling.
        # Or just compute the full tensor if it's small enough.
        # 48 x 16 x 16 = 12288.
        # Rank R.
        # Sum over R of outer products.
        
        # Let's compute the current tensor T_current
        T_current = np.zeros((n*m, m*p, p*n))
        for r in range(target_rank):
            T_current += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            
        # T_target
        T_target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i*m+j, j*p+k, k*n+i] = 1.0
        
        error = np.sum((T_current - T_target)**2)
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if error < 1e-10:
            break
        
        # Gradients
        # dF/dU[a, r] = 2 * sum_{b,c} (T_current[a,b,c] - T_target[a,b,c]) * V[b,r] * W[c,r]
        # We can compute this efficiently.
        diff = T_current - T_target
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        # grad_U[a, r] = 2 * sum_{b,c} diff[a,b,c] V[b,r] W[c,r]
        # = 2 * (diff * (V_r \otimes W_r))_a
        # Let M_r = V[:, r] \otimes W[:, r] (shape mp x pn)
        # Then grad_U[:, r] = 2 * diff @ M_r (where diff is flattened? No)
        # diff is (nm, mp, pn).
        # V[:, r] is (mp). W[:, r] is (pn).
        # sum_{b,c} diff[a,b,c] V[b,r] W[c,r] = (diff @ (V_r \otimes W_r))[a]
        
        for r in range(target_rank):
            # V_r \otimes W_r is (mp, pn)
            M_r = np.outer(V[:, r], W[:, r])
            # diff @ M_r: (nm, mp, pn) @ (mp, pn) -> (nm)
            # np.einsum('abc, bc -> a', diff, M_r)
            grad_U[:, r] = 2 * np.einsum('abc,bc->a', diff, M_r)
            
            M_r2 = np.outer(U[:, r], W[:, r]) # (nm, pn)
            # sum_{a,c} diff[a,b,c] U[a,r] W[c,r]
            grad_V[:, r] = 2 * np.einsum('abc,ac->b', diff, M_r2)
            
            M_r3 = np.outer(U[:, r], V[:, r]) # (nm, mp)
            # sum_{a,b} diff[a,b,c] U[a,r] V[b,r]
            grad_W[:, r] = 2 * np.einsum('abc,ab->c', diff, M_r3)
            
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
    if best_error < 1e-8:
        return best_U, best_V, best_W
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n, m, p = DIMENSIONS
    
    # Try ranks from lower bound up
    # Lower bound 16
    # Schoolbook 48
    
    # We will try a few candidate ranks
    # It is likely the rank is close to the lower bound or slightly higher.
    # For 2x2x2, rank 7 (LB 4).
    # For 3x3x3, rank 23 (LB 9).
    # For 3x4x4, LB 16.
    
    # Let's try 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30
    
    # To save time, we can try in parallel or sequentially with time checks.
    # Sequential is safer.
    
    # Start with a known valid decomposition (schoolbook) as fallback
    U_fb, V_fb, W_fb = _schoolbook_decomposition(n, m, p)
    
    best_rank = n * m * p
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    
    # Try to improve
    for target_rank in range(16, 30):
        if time.time() > deadline - 10.0:
            break
            
        # Try multiple seeds for each rank
        for seed in range(5):
            if time.time() > deadline - 5.0:
                break
                
            res = _try_find_decomposition(n, m, p, target_rank, max_iter=1000, lr=0.01, seed=seed)
            if res is not None:
                U_new, V_new, W_new = res
                # Verify
                if _verify_decomposition(U_new, V_new, W_new, n, m, p):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_new, V_new, W_new
                    # Found a valid one, we can break inner loop
                    # But maybe we can find a lower one? No, we are iterating up.
                    # So this is the minimal rank found so far.
                    # We can break both loops?
                    # Yes, since we want rank-minimal.
                    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

def _schoolbook_decomposition(n, m, p):
    """Return the standard schoolbook rank-n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return U, V, W
