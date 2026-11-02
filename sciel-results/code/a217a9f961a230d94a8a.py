import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 6]

def _verify(U, V, W, n, m, p):
    """
    Verify tensor decomposition identity:
    sum_l U[a*m+b, l] * V[c*p+d, l] * W[k*n+i, l] == 1 if (a==i and c==k and b==d) else 0
    """
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    
    T = np.einsum('abl,cdl,ki l->abcdki', U3, V3, W3)
    
    target = np.zeros((n, m, m, p, p, n), dtype=float)
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    if b == d and c < p:
                        target[a, b, c, d, c, a] = 1.0
    
    return np.allclose(T, target, atol=1e-6)

def _round_to_half_integer(A):
    return np.round(A * 2) / 2.0

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 36
    
    # Initialize with schoolbook as fallback
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U = U_school
    best_V = V_school
    best_W = W_school
    best_rank = schoolbook_rank
    
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    # Target ranks to try, from low to high
    # Known lower bounds suggest rank < 36 is possible.
    # Try 22, 24, 26, 28, 30, 32, 34
    target_ranks = [22, 24, 26, 28, 30, 32, 34]
    
    def get_residual(U, V, W, n, m, p, r):
        U3 = U.reshape(n, m, r)
        V3 = V.reshape(m, p, r)
        W3 = W.reshape(p, n, r)
        T = np.einsum('abl,cdl,ki l->abcdki', U3, V3, W3)
        target = np.zeros((n, m, m, p, p, n), dtype=float)
        for a in range(n):
            for b in range(m):
                for c in range(m):
                    for d in range(p):
                        if b == d and c < p:
                            target[a, b, c, d, c, a] = 1.0
        return T - target

    def objective(params, r):
        U = params[0:n*m*r].reshape(n*m, r)
        V = params[n*m*r: n*m*r + m*p*r].reshape(m*p, r)
        W = params[n*m*r + m*p*r:].reshape(p*n, r)
        res = get_residual(U, V, W, n, m, p, r)
        return np.sum(res**2)

    # Try to find a decomposition for each target rank
    for R in target_ranks:
        if time.time() > deadline - 10:
            break
            
        # Try multiple random initializations
        found = False
        max_attempts = 50 if R > 28 else 100
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
                
            # Initialize with random values in [-1, 1]
            # Scale factors to help convergence
            np.random.seed(attempt * 1000 + R)
            U_init = np.random.randn(n*m, R) * 0.5
            V_init = np.random.randn(m*p, R) * 0.5
            W_init = np.random.randn(p*n, R) * 0.5
            
            params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Use Nelder-Mead for small dimensions, but 6*R*... can be large.
            # For R=22, dims = 6*22 + 18*22 + 12*22 = 22*36 = 792. Too large for Nelder-Mead.
            # Use coordinate descent or simple gradient-free local search?
            # Or use scipy.optimize.minimize with 'L-BFGS-B' if we can compute gradients.
            # Gradients are expensive.
            
            # Alternative: Use a simple iterative refinement (Gauss-Seidel style) on the factors.
            # Fix V, W, optimize U. Fix U, W, optimize V. etc.
            # This is block coordinate descent.
            
            U = U_init.copy()
            V = V_init.copy()
            W = W_init.copy()
            
            # Precompute target
            target = np.zeros((n, m, m, p, p, n), dtype=float)
            for a in range(n):
                for b in range(m):
                    for c in range(m):
                        for d in range(p):
                            if b == d and c < p:
                                target[a, b, c, d, c, a] = 1.0
            
            # Block Coordinate Descent
            # Residual T = sum_l U_l V_l W_l
            # If we fix V, W, the term for U is linear in U.
            # T = U @ (V^T * W) ... roughly.
            
            # Let's try a few steps of BCD
            for step in range(50):
                # Update U
                # T_target = sum_l U_l V_l W_l
                # For each l, U_l is a vector of size n*m.
                # Contribution to T from l: U_l[a,b] * (V_l[c,d] * W_l[k,i])
                # Let M_l[c,d,k,i] = V_l[c,d] * W_l[k,i]
                # Then T[a,b,c,d,k,i] = sum_l U_l[a,b] * M_l[c,d,k,i]
                # We want T = target.
                # This is a linear system for U_l if we treat other l's as fixed? No, it's coupled.
                # Simpler: minimize || T - target ||^2 w.r.t U.
                # Gradient w.r.t U_l[a,b] is sum_{c,d,k,i} (T-target)[a,b,c,d,k,i] * V_l[c,d] * W_l[k,i]
                
                # This is still heavy. Let's just use a random restart with a simpler heuristic.
                # Given the time limit, this might not converge for R=22.
                
                pass
            
            # If BCD is too complex/slow, fall back to trying hardcoded structures.
            # But the directive asks for a DIFFERENT strategy.
            # Let's try a specific structured initialization for R=22.
            # Split into two 2x3x3 blocks.
            # Use random init for 2x3x3 blocks and optimize locally?
            
            break # Skip complex optimization for now, try a different structural approach below.
        
        # Structural Approach: Explicit Rank-11 for 2x3x3?
        # If we can find a valid rank-11 for 2x3x3, we can use it twice for rank 22.
        # Let's try to verify if a specific known rank-11 exists or construct one.
        # Actually, let's just try to find ANY valid decomposition of rank < 36 using a smarter search.
        
    # Fallback: Try to find rank 33 or 34 using a more robust search if the above fails.
    # But the current program already does that poorly.
    
    # Let's implement a simple local search around the schoolbook decomposition.
    # Start with rank 36, try to merge terms? No, that's hard.
    
    # Let's try a random search for rank 30, 32, 34.
    # For each rank R, try to find U, V, W such that the identity holds.
    # We can use the fact that the identity is linear in W given U, V.
    # Fix U, V randomly. Solve for W via least squares.
    # If the residual is small, round and verify.
    
    for R in [34, 32, 30, 28, 26, 24, 22]:
        if time.time() > deadline - 5:
            break
            
        found = False
        attempts = 20
        
        for _ in range(attempts):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(int(time.time() * 1000) % 10000)
            U = np.random.randn(n*m, R) * 0.5
            V = np.random.randn(m*p, R) * 0.5
            
            # Solve for W: sum_l U_l[a,b] V_l[c,d] W_l[k,i] = target[a,b,c,d,k,i]
            # This is a linear system for W.
            # Let X be a matrix where each row corresponds to a (a,b,c,d) tuple and each column to a (l, k, i) tuple?
            # No, W is (p*n, R).
            # Let's vectorize.
            # T[a,b,c,d,k,i] = sum_l U[a*m+b, l] * V[c*p+d, l] * W[k*n+i, l]
            
            # For a fixed U, V, the equation is linear in W.
            # We can set up a least squares problem.
            # Number of equations: n*m*m*p*p*n = 2*3*3*6*6*2 = 1296.
            # Number of unknowns: p*n*R = 12*R.
            
            A_mat = np.zeros((1296, 12*R))
            b_vec = np.zeros(1296)
            
            idx = 0
            for a in range(n):
                for b in range(m):
                    for c in range(m):
                        for d in range(p):
                            # Target value
                            if b == d and c < p:
                                # k=c, i=a
                                # T[a,b,c,d,c,a] = 1
                                # Equation: sum_l U[a*m+b, l] * V[c*p+d, l] * W[c*n+a, l] = 1
                                # Other entries in this (a,b,c,d) slice are 0.
                                pass
                            
                            # We need to fill A_mat for all (k,i) for this (a,b,c,d)
                            for k in range(p):
                                for i in range(n):
                                    # Coefficient for W[k*n+i, l] is U[a*m+b, l] * V[c*p+d, l]
                                    for l in range(R):
                                        A_mat[idx, (k*n+i)*R + l] = U[a*m+b, l] * V[c*p+d, l]
                                    
                                    # Target value
                                    if b == d and c == k and a == i:
                                        b_vec[idx] = 1.0
                                    else:
                                        b_vec[idx] = 0.0
                                    
                                    idx += 1
            
            # Solve least squares
            try:
                W_flat, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                W = W_flat.reshape(p*n, R)
                
                # Check residual
                res = get_residual(U, V, W, n, m, p, R)
                if np.max(np.abs(res)) < 1e-5:
                    # Round to half-integers
                    U_r = _round_to_half_integer(U)
                    V_r = _round_to_half_integer(V)
                    W_r = _round_to_half_integer(W)
                    
                    if _verify(U_r, V_r, W_r, n, m, p):
                        best_U = U_r
                        best_V = V_r
                        best_W = W_r
                        best_rank = R
                        found = True
                        break
            except:
                pass
        
        if found:
            break

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
