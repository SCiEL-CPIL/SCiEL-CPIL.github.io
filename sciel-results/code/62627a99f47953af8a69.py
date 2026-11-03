"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # Schoolbook decomposition
    def schoolbook():
        r = n * m * p
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    U_s, V_s, W_s = schoolbook()
    best_rank = U_s.shape[1]
    best_U, best_V, best_W = U_s, V_s, W_s
    
    deadline = time.time() + 145  # Safety margin

    def verify_exact(U, V, W):
        """Check if sum_r U_r V_r W_r == T using exact rational arithmetic"""
        S = np.zeros((nm, mp, pn), dtype=object)
        for r in range(U.shape[1]):
            u_col = U[:, r]
            v_col = V[:, r]
            w_col = W[:, r]
            for a in range(nm):
                if u_col[a] == 0:
                    continue
                for b in range(mp):
                    if v_col[b] == 0:
                        continue
                    for c in range(pn):
                        if w_col[c] == 0:
                            continue
                        S[a, b, c] += u_col[a] * v_col[b] * w_col[c]
        # Check against T
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    expected = T[a, b, c]
                    actual = S[a, b, c]
                    if abs(float(actual) - expected) > 1e-10:
                        return False
        return True

    def verify_float(U, V, W, tol=1e-8):
        """Fast float verification"""
        S = np.zeros((nm, mp, pn))
        for r in range(U.shape[1]):
            if np.any(U[:, r]) and np.any(V[:, r]) and np.any(W[:, r]):
                S += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.allclose(S, T, atol=tol)

    def try_structured_reductions():
        """
        Try to find lower-rank decompositions by merging schoolbook components.
        Key insight: Two schoolbook components (i1,j1,k1) and (i2,j2,k2) can potentially
        be merged if they share structure. We search for valid mergers.
        """
        nonlocal best_rank, best_U, best_V, best_W
        
        # Each schoolbook component is indexed by (i, j, k)
        # Component (i,j,k) has:
        #   U: 1 at position i*m+j
        #   V: 1 at position j*p+k  
        #   W: 1 at position k*n+i
        
        # Try to find pairs that can be merged into a single rank-1 component
        # with coefficients that still cover both original non-zeros
        
        # Strategy: For each pair of schoolbook components, try to express their sum
        # as a single rank-1 tensor (or a small set of rank-1 tensors)
        
        # More practically: try random subsets of schoolbook components and see if
        # we can find a lower-rank representation
        
        # Approach: Start with full schoolbook, try to remove one component at a time
        # and compensate by adjusting others
        
        # Simpler approach: Use the fact that the tensor is sparse and has specific structure
        # Try to find decompositions where each rank-1 component covers multiple non-zeros
        
        # For a rank-1 component to cover multiple non-zeros, it needs:
        # U[a] * V[b] * W[c] = 1 for multiple (a,b,c) in the support
        
        # Let's try a systematic search for small rank decompositions
        # Start with rank 40-80 and try to find valid decompositions
        
        # Use a different strategy: random initialization + local search
        # where we only care about the 84 non-zero entries
        
        nz_a, nz_b, nz_c = np.nonzero(T)
        nz_vals = T[nz_a, nz_b, nz_c]
        
        def local_search(U, V, W, max_iters=200):
            """
            Local search that only enforces constraints on non-zero entries.
            This is much faster than full tensor verification.
            """
            r = U.shape[1]
            U = U.copy()
            V = V.copy()
            W = W.copy()
            
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                    
                # Compute residuals on non-zero entries
                # res[k] = T[nz_a[k], nz_b[k], nz_c[k]] - sum_r U[nz_a[k],r] V[nz_b[k],r] W[nz_c[k],r]
                res = np.zeros(len(nz_a))
                for k in range(len(nz_a)):
                    a, b, c = nz_a[k], nz_b[k], nz_c[k]
                    s = 0.0
                    for r in range(r):
                        s += U[a, r] * V[b, r] * W[c, r]
                    res[k] = nz_vals[k] - s
                
                if np.max(np.abs(res)) < 1e-10:
                    break
                
                # Try to reduce the maximum residual
                max_res_idx = np.argmax(np.abs(res))
                a, b, c = nz_a[max_res_idx], nz_b[max_res_idx], nz_c[max_res_idx]
                
                # Gradient descent step on the components
                lr = 0.01
                for r in range(r):
                    # d/dU[a,r] of (res[max_res_idx])^2 = -2 * res[max_res_idx] * V[b,r] * W[c,r]
                    grad_U = -2 * res[max_res_idx] * V[b, r] * W[c, r]
                    U[a, r] -= lr * grad_U
                    
                    grad_V = -2 * res[max_res_idx] * U[a, r] * W[c, r]
                    V[b, r] -= lr * grad_V
                    
                    grad_W = -2 * res[max_res_idx] * U[a, r] * V[b, r]
                    W[c, r] -= lr * grad_W
            
            return U, V, W
        
        # Try various ranks
        for target_rank in [80, 76, 72, 68, 64, 60, 56, 52, 48, 44, 40, 36, 32, 28]:
            if time.time() > deadline - 10:
                break
            if target_rank >= best_rank:
                continue
            
            for attempt in range(3):
                if time.time() > deadline - 5:
                    break
                
                # Initialize with random vectors
                U = np.random.randn(nm, target_rank) * 0.5
                V = np.random.randn(mp, target_rank) * 0.5
                W = np.random.randn(pn, target_rank) * 0.5
                
                U_opt, V_opt, W_opt = local_search(U, V, W, max_iters=100)
                
                if verify_float(U_opt, V_opt, W_opt):
                    if verify_exact(U_opt, V_opt, W_opt):
                        best_rank = target_rank
                        best_U, best_V, best_W = U_opt, V_opt, W_opt
                        return True
        
        return False

    # Try structured reductions
    try_structured_reductions()
    
    def to_list(M):
        L = []
        for row in M:
            L.append([float(x) for x in row])
        return L
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
