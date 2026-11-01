"""Deterministic block-decomposition search for 2x5x6 matrix multiplication tensor."""
DIMENSIONS = [2, 5, 6]

import time
import random
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    
    # Helper to validate a decomposition (vectorized)
    def is_valid_decomposition(U, V, W, rank):
        U_mat = np.array(U, dtype=float)
        V_mat = np.array(V, dtype=float)
        W_mat = np.array(W, dtype=float)
        
        # Build tensor T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
        # T should have T[i,j,k] = 1 if consistent, 0 otherwise
        # For matrix mult: T[i,j,k] = 1 iff this is a valid (i,j,k) triple
        # Actually the tensor for matmul is: T[i,j,k] = 1 for all (i,j,k)
        # Wait, let me re-read the problem.
        
        # The matrix multiplication tensor has entries:
        # T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
        # T[i,j,k] = 1 for all valid (i,j,k)
        # A rank-1 decomposition is U[i,j]*V[j,k]*W[k,i]
        
        # So we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
        
        # Let's verify this
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = np.sum(U_mat[i*m+j, :] * V_mat[j*p+k, :] * W_mat[k*n+i, :])
                    if not np.isclose(total, 1.0, atol=1e-9):
                        return False
        return True
    
    # Schoolbook baseline: rank = n*m*p = 60
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    best_rank = rank_sb
    
    # Strategy: Generate candidate rank-1 tensors and use greedy set cover
    # A rank-1 tensor is defined by (U_row, V_row, W_row) where:
    # U_row has shape (n*m,), V_row has shape (m*p,), W_row has shape (p*n,)
    # The tensor entry at (i,j,k) is U[i*m+j] * V[j*p+k] * W[k*n+i]
    
    # For a decomposition of rank r, we need r such rank-1 tensors whose sum gives the matmul tensor
    # i.e., for all (i,j,k): sum_{c=0}^{r-1} U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i] = 1
    
    # Approach: 
    # 1. Generate a pool of candidate rank-1 tensors with small integer/half-integer entries
    # 2. Use greedy or exact methods to find a minimal subset that covers all (i,j,k) triples
    
    # First, let's try a structured approach based on Strassen-like ideas
    # For 2x2x2, Strassen uses 7 multiplications. For larger matrices, we can block.
    
    # Let's try: partition the 2x5x6 into blocks and apply Strassen to sub-blocks
    # 2x5: can't easily block into 2x2
    # 5x6: can't easily block into 2x2
    
    # Alternative: Use the fact that rank(A*B) <= rank(A)*rank(B) doesn't help directly
    
    # Let's try a more direct approach: 
    # Generate random sparse rank-1 candidates and solve a linear system
    
    # For rank r, we have r variables for each of U, V, W
    # Total variables: r*(n*m + m*p + p*n) = r*(10 + 30 + 12) = 52r
    # Constraints: n*m*p = 60 equations (one for each (i,j,k))
    
    # But the constraints are nonlinear in U,V,W. However, if we fix U and V, 
    # the constraints become linear in W.
    
    # Strategy: 
    # 1. Fix U and V to random sparse patterns
    # 2. Solve for W via least squares
    # 3. Check if W has small rational entries
    
    # Let's vectorize the constraint building
    def build_constraint_matrix(U_mat, V_mat, r):
        """
        Build the linear system for W given U and V.
        For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        Variables: W[k*n+i, c] for k in [0,p), i in [0,n), c in [0,r)
        """
        num_constraints = n * m * p
        num_vars = p * n * r
        
        A = np.zeros((num_constraints, num_vars))
        b = np.ones(num_constraints)
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Coefficients for W[k*n+i, :]
                    coeffs = U_mat[i*m+j, :] * V_mat[j*p+k, :]
                    var_start = (k*n + i) * r
                    A[idx, var_start:var_start+r] = coeffs
                    idx += 1
        
        return A, b
    
    # Try different ranks with random U, V
    random.seed(42)
    
    # Try ranks from 50 down to 58 (since we know 58 is achievable)
    # Actually, let's try to find the minimum rank
    # Lower bound: rank >= max(n, m, p) = 6, but that's very loose
    # Better lower bound: rank >= n*m*p / (some factor)
    
    # Let's try ranks from 50 to 59
    for r in range(50, 60):
        if time.time() - start_time > time_limit:
            break
        
        max_attempts = 200
        found = False
        
        for attempt in range(max_attempts):
            if time.time() - start_time > time_limit:
                break
            
            # Generate sparse random U, V with entries in {-1, 0, 1}
            U_rand = np.zeros((n * m, r))
            V_rand = np.zeros((m * p, r))
            
            # Density based on rank
            density_U = min(0.5, 0.2 + 0.3 * (r / 60.0))
            density_V = min(0.5, 0.2 + 0.3 * (r / 60.0))
            
            mask_U = np.random.random((n * m, r)) < density_U
            U_rand[mask_U] = np.random.choice([-1, 1], size=mask_U.sum())
            
            mask_V = np.random.random((m * p, r)) < density_V
            V_rand[mask_V] = np.random.choice([-1, 1], size=mask_V.sum())
            
            # Build and solve
            A, b = build_constraint_matrix(U_rand, V_rand, r)
            
            try:
                W_vec, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=1e-10)
            except Exception:
                continue
            
            # Check residual
            if residuals is None or len(residuals) == 0:
                continue
            if residuals[0] > 1e-8:
                continue
            
            # Check if W is close to half-integers
            W_check = W_vec * 2
            W_rounded = np.round(W_check)
            diff = np.abs(W_check - W_rounded)
            
            if np.max(diff) < 1e-5:
                W_snap = W_rounded / 2.0
                W_mat = W_snap.reshape(p * n, r)
                
                # Verify
                if is_valid_decomposition(U_rand, V_rand, W_mat, r):
                    best_U = U_rand.copy()
                    best_V = V_rand.copy()
                    best_W = W_mat.copy()
                    best_rank = r
                    found = True
                    break
        
        if found:
            break
    
    # If we haven't found anything better, try a greedy set cover approach
    if best_rank >= 58 and time.time() - start_time < time_limit - 10:
        # Generate a pool of candidate rank-1 tensors
        # Each candidate is (u, v, w) where u in R^(n*m), v in R^(m*p), w in R^(p*n)
        # with small integer/half-integer entries
        
        # For each (i,j,k), the contribution of candidate c is u[i*m+j]*v[j*p+k]*w[k*n+i]
        # We need to select a subset S such that for all (i,j,k):
        # sum_{c in S} u_c[i*m+j]*v_c[j*p+k]*w_c[k*n+i] = 1
        
        # This is a linear system! Let x_c be 1 if candidate c is selected, 0 otherwise
        # For each (i,j,k): sum_c x_c * u_c[i*m+j]*v_c[j*p+k]*w_c[k*n+i] = 1
        
        # Generate candidates
        num_candidates = 500
        candidates = []
        
        for _ in range(num_candidates):
            # Generate random sparse u, v, w
            u = np.zeros(n * m)
            v = np.zeros(m * p)
            w = np.zeros(p * n)
            
            # Sparse with entries in {-1, 0, 1}
            mask_u = np.random.random(n * m) < 0.3
            u[mask_u] = np.random.choice([-1, 1], size=mask_u.sum())
            
            mask_v = np.random.random(m * p) < 0.3
            v[mask_v] = np.random.choice([-1, 1], size=mask_v.sum())
            
            mask_w = np.random.random(p * n) < 0.3
            w[mask_w] = np.random.choice([-1, 1], size=mask_w.sum())
            
            candidates.append((u, v, w))
        
        # Build the linear system
        # Variables: x_c for c in range(num_candidates)
        # Constraints: one for each (i,j,k)
        
        A_sc = np.zeros((n * m * p, num_candidates))
        b_sc = np.ones(n * m * p)
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c_idx, (u, v, w) in enumerate(candidates):
                        A_sc[idx, c_idx] = u[i*m+j] * v[j*p+k] * w[k*n+i]
                    idx += 1
        
        # Solve for x (should be 0 or 1, but let's allow any real and round)
        try:
            x_vec, residuals, rank_sc, sv = np.linalg.lstsq(A_sc, b_sc, rcond=1e-10)
        except Exception:
            x_vec = None
        
        if x_vec is not None:
            # Round to 0 or 1
            x_rounded = np.round(x_vec)
            
            # Check if this gives a valid decomposition
            # The selected candidates form a decomposition of rank = sum(x_rounded)
            # But we need to verify that the sum actually equals the matmul tensor
            
            selected = np.where(x_rounded == 1)[0]
            if len(selected) > 0 and len(selected) < best_rank:
                # Build U, V, W from selected candidates
                U_sc = np.zeros((n * m, len(selected)))
                V_sc = np.zeros((m * p, len(selected)))
                W_sc = np.zeros((p * n, len(selected)))
                
                for sel_idx, c_idx in enumerate(selected):
                    u, v, w = candidates[c_idx]
                    U_sc[:, sel_idx] = u
                    V_sc[:, sel_idx] = v
                    W_sc[:, sel_idx] = w
                
                # Verify
                if is_valid_decomposition(U_sc, V_sc, W_sc, len(selected)):
                    best_U = U_sc.copy()
                    best_V = V_sc.copy()
                    best_W = W_sc.copy()
                    best_rank = len(selected)
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
