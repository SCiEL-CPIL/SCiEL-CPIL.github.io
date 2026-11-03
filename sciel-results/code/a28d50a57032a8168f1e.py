"""Program implementing a rank-minimal exact tensor decomposition for 3x4 * 4x6 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Tensor T[i, j, k] = 1 for all i in [0, n), j in [0, m), k in [0, p)
    # Decomposition: T = sum_r U[:,r] o V[:,r] o W[:,r]
    # U shape: (n*m, R)
    # V shape: (m*p, R)
    # W shape: (p*n, R)
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition is exact using Fractions."""
        # Convert to Fractions for exact check
        U_frac = [[Fraction(U[i][r]) for r in range(rank)] for i in range(len(U))]
        V_frac = [[Fraction(V[i][r]) for r in range(rank)] for i in range(len(V))]
        W_frac = [[Fraction(W[i][r]) for r in range(rank)] for i in range(len(W))]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = Fraction(0)
                    for r in range(rank):
                        val += U_frac[i * m + j][r] * V_frac[j * p + k][r] * W_frac[k * n + i][r]
                    if val != 1:
                        return False
        return True

    def hals_decompose(T, R, max_iters=200, tol=1e-12):
        """
        Hierarchical Alternating Least Squares (HALS) for rank-R decomposition.
        T is a 3D array of shape (n*m, m*p, p*n) flattened or structured.
        We represent T as a 3D array T[i, j, k] where indices are flattened.
        """
        # T is implicitly 1.0 everywhere.
        # We maintain U, V, W as (dim, R) matrices.
        
        # Initialize
        np.random.seed(42)
        U = np.random.randn(n*m, R)
        V = np.random.randn(m*p, R)
        W = np.random.randn(p*n, R)
        
        # Precompute indices for speed
        indices = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    indices.append((i*m+j, j*p+k, k*n+i))
        
        best_err = float('inf')
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            # HALS Step: For each component r, update U[:,r], V[:,r], W[:,r]
            # to minimize || T - sum_{r'} U_{r'} o V_{r'} o W_{r'} ||^2
            
            # Compute residual tensor R_T = T - sum_{r != current} ...
            # To be efficient, we maintain the current sum S
            S = np.zeros((n, m, p))
            for r in range(R):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            S[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            
            # This naive S computation is O(R*n*m*p). 
            # For R~60, n*m*p=72, this is ~4320 ops per update step, times 3 matrices.
            # It's feasible but might be slow in pure Python loops. 
            # Let's vectorize the S computation.
            
            # Actually, let's just do the standard ALS but with a "hierarchical" flavor:
            # Update one rank-1 component at a time, keeping others fixed.
            
            prev_err = best_err
            
            for r in range(R):
                if time.time() > deadline - 5:
                    break
                
                # Compute residual excluding component r
                # Res = T - Sum_{r' != r} U_{r'} o V_{r'} o W_{r'}
                # T is all 1s.
                # Sum_{r' != r} = Sum_{all} - Component_r
                # We can compute Sum_{all} efficiently?
                # Or just recompute the sum for the residual.
                
                # Efficient Residual Computation:
                # Res[i,j,k] = 1 - sum_{r' != r} U[i*m+j, r'] * V[j*p+k, r'] * W[k*n+i, r']
                
                # Let's compute the full sum S_total first? No, that's expensive.
                # Let's compute the contribution of r to subtract from the "current sum" 
                # if we maintained it. But we didn't.
                
                # Alternative: Compute the gradient / optimal update for r given others.
                # Minimize || Res - U_r o V_r o W_r ||^2
                # This is equivalent to finding the best rank-1 approximation to Res.
                # Res[i,j,k] = 1 - sum_{r' != r} ...
                
                # Let's construct Res explicitly.
                Res = np.ones((n, m, p))
                for r_prime in range(R):
                    if r_prime == r:
                        continue
                    # Add contribution of r_prime
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                Res[i, j, k] -= U[i*m+j, r_prime] * V[j*p+k, r_prime] * W[k*n+i, r_prime]
                
                # Now find best rank-1 approximation to Res.
                # Reshape Res to (n*m, m*p, p*n) is tricky for SVD.
                # We can use a simple ALS on the Residual for this single component.
                # Or use a greedy approach:
                # 1. Fix V_r, W_r to current values, solve for U_r.
                # 2. Fix U_r, W_r, solve for V_r.
                # 3. Fix U_r, V_r, solve for W_r.
                # Repeat a few times.
                
                # Initialize U_r, V_r, W_r with current values
                u_r = U[:, r].copy()
                v_r = V[:, r].copy()
                w_r = W[:, r].copy()
                
                for _ in range(10): # Inner iterations for this component
                    # Update U_r
                    # Res_i = sum_{j,k} Res[i,j,k] V_r[j,k] W_r[k,i] ... wait, indices.
                    # U[i*m+j, r] should be updated.
                    # For fixed j, U[i*m+j, r] appears in terms Res[i,j,k] * V[j*p+k, r] * W[k*n+i, r]
                    # Sum over i,k: Res[i,j,k] * V[j*p+k, r] * W[k*n+i, r] = 1
                    # This is a linear system for U[i*m+j, r]? No, U is the variable.
                    # Wait, the term is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
                    # If we fix V_r and W_r, then for a specific (i,j), U[i*m+j, r] multiplies 
                    # sum_k Res[i,j,k] * V[j*p+k, r] * W[k*n+i, r].
                    # We want this product to be 1.
                    # So U[i*m+j, r] = 1 / sum_k Res[i,j,k] * V[j*p+k, r] * W[k*n+i, r]
                    
                    for i in range(n):
                        for j in range(m):
                            denom = 0.0
                            for k in range(p):
                                denom += Res[i, j, k] * v_r[j*p+k] * w_r[k*n+i]
                            if abs(denom) < 1e-12:
                                u_r[i*m+j] = 0.0
                            else:
                                u_r[i*m+j] = 1.0 / denom
                    
                    # Update V_r
                    for j in range(m):
                        for k in range(p):
                            denom = 0.0
                            for i in range(n):
                                denom += Res[i, j, k] * u_r[i*m+j] * w_r[k*n+i]
                            if abs(denom) < 1e-12:
                                v_r[j*p+k] = 0.0
                            else:
                                v_r[j*p+k] = 1.0 / denom
                    
                    # Update W_r
                    for k in range(p):
                        for i in range(n):
                            denom = 0.0
                            for j in range(m):
                                denom += Res[i, j, k] * u_r[i*m+j] * v_r[j*p+k]
                            if abs(denom) < 1e-12:
                                w_r[k*n+i] = 0.0
                            else:
                                w_r[k*n+i] = 1.0 / denom
                    
                    U[:, r] = u_r
                    V[:, r] = v_r
                    W[:, r] = w_r
                    
                    # Recompute Res for this component? 
                    # The Residual for the next inner iteration should be 
                    # Res_new = Res_old - (u_r o v_r o w_r) + (old_u_r o old_v_r o old_w_r)
                    # But we changed u,v,w.
                    # Let's just recompute Res from scratch for simplicity in this inner loop?
                    # That's O(n*m*p) per inner iter. 10 iters * R * n*m*p.
                    # 10 * 60 * 72 = 43200 ops. Very fast.
                    
                    Res = np.ones((n, m, p))
                    for r_prime in range(R):
                        if r_prime == r:
                            continue
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    Res[i, j, k] -= U[i*m+j, r_prime] * V[j*p+k, r_prime] * W[k*n+i, r_prime]
                    # Note: Res here does NOT include the current r, so we are approximating T - Sum_{r' != r}.
                    # The error of the whole tensor is || T - Sum_{all} || = || Res - (U_r o V_r o W_r) ||
                    # But we just optimized U_r, V_r, W_r to make U_r o V_r o W_r close to Res.
                    # So the new residual for the whole tensor is Res - (U_r o V_r o W_r).
                    
            # Check convergence of the whole tensor
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(R):
                            val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        err += (val - 1.0)**2
            
            if err < tol:
                break
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            else:
                # If error increases significantly, revert? 
                # HALS usually monotonically decreases error if done correctly.
                # But our inner solver is approximate.
                pass

        return best_U, best_V, best_W, best_err

    def exact_greedy_peel(U_num, V_num, W_num, R):
        """
        Attempt to convert the numerical solution to an exact integer/half-integer solution.
        Strategy: 
        1. Round U, V, W to nearest half-integers.
        2. Verify. If fail, try to "peel" off exact rank-1 components from the residual.
        """
        # First, try simple rounding
        U_round = np.round(U_num * 2) / 2
        V_round = np.round(V_num * 2) / 2
        W_round = np.round(W_num * 2) / 2
        
        if verify_decomposition(U_round.tolist(), V_round.tolist(), W_round.tolist(), R):
            return U_round.tolist(), V_round.tolist(), W_round.tolist()
        
        # If rounding fails, try a greedy exact peeling approach.
        # This is complex to implement robustly in the time limit.
        # Instead, we will return the rounded version as a "best effort" if it's close,
        # but the evaluator requires EXACT.
        
        # So if rounding fails, we must fail this candidate or find another way.
        # Let's try integer rounding as well.
        U_int = np.round(U_num)
        V_int = np.round(V_num)
        W_int = np.round(W_num)
        
        if verify_decomposition(U_int.tolist(), V_int.tolist(), W_int.tolist(), R):
            return U_int.tolist(), V_int.tolist(), W_int.tolist()
            
        return None

    # Fallback: Schoolbook
    U_fb = [[0] * 72 for _ in range(12)]
    V_fb = [[0] * 72 for _ in range(24)]
    W_fb = [[0] * 72 for _ in range(18)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    best_rank = 72

    # Try to find a lower rank using HALS
    # Known lower bounds: 
    # Rank(3x4x6) <= 3*4*6 = 72 (schoolbook)
    # Can we do better?
    # Strassen for 2x2x2 is 7.
    # 3x4x6 can be decomposed into blocks.
    # Let's try ranks from 63 downwards.
    
    for target_rank in range(63, 40, -1):
        if time.time() > deadline - 10:
            break
            
        # Try multiple random seeds for HALS
        found = False
        for seed in range(20):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(seed)
            U_h, V_h, W_h, err = hals_decompose(None, target_rank, max_iters=50, tol=1e-10)
            
            if err < 1e-9:
                # Try to convert to exact
                exact_res = exact_greedy_peel(U_h, V_h, W_h, target_rank)
                if exact_res is not None:
                    U_exact, V_exact, W_exact = exact_res
                    best_result = {"U": U_exact, "V": V_exact, "W": W_exact}
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            break

    return best_result
# EVOLVE-BLOCK-END
