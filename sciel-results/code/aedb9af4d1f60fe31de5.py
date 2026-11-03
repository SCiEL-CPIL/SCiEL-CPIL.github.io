import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    N_u = n * m  # 6
    N_v = m * p  # 9
    N_w = p * n  # 6
    
    # Build the target tensor
    T = np.zeros((N_u, N_v, N_w))
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
                indices.append((i * m + j, j * p + k, k * n + i))
    
    # The rank-17 solution (Strassen + schoolbook)
    U17 = np.zeros((N_u, 17))
    V17 = np.zeros((N_v, 17))
    W17 = np.zeros((N_w, 17))
    
    U17[0, 0] = 1; U17[4, 0] = 1
    V17[0, 0] = 1; V17[4, 0] = 1
    W17[0, 0] = 1; W17[3, 0] = 1
    
    U17[3, 1] = 1; U17[4, 1] = 1
    V17[0, 1] = 1
    W17[1, 1] = 1; W17[3, 1] = -1
    
    U17[0, 2] = 1
    V17[1, 2] = 1; V17[4, 2] = -1
    W17[2, 2] = 1; W17[3, 2] = 1
    
    U17[4, 3] = 1
    V17[3, 3] = 1; V17[0, 3] = -1
    W17[0, 3] = 1; W17[1, 3] = 1
    
    U17[0, 4] = 1; U17[1, 4] = 1
    V17[4, 4] = 1
    W17[0, 4] = -1; W17[2, 4] = 1
    
    U17[3, 5] = 1; U17[0, 5] = -1
    V17[0, 5] = 1; V17[1, 5] = 1
    W17[3, 5] = 1
    
    U17[1, 6] = 1; U17[4, 6] = -1
    V17[3, 6] = 1; V17[4, 6] = 1
    W17[0, 6] = 1
    
    U17[2, 7] = 1; V17[6, 7] = 1; W17[0, 7] = 1
    U17[2, 8] = 1; V17[7, 8] = 1; W17[2, 8] = 1
    U17[5, 9] = 1; V17[6, 9] = 1; W17[1, 9] = 1
    U17[5, 10] = 1; V17[7, 10] = 1; W17[3, 10] = 1
    
    U17[0, 11] = 1; V17[2, 11] = 1; W17[4, 11] = 1
    U17[1, 12] = 1; V17[5, 12] = 1; W17[4, 12] = 1
    U17[2, 13] = 1; V17[8, 13] = 1; W17[4, 13] = 1
    U17[3, 14] = 1; V17[2, 14] = 1; W17[5, 14] = 1
    U17[4, 15] = 1; V17[5, 15] = 1; W17[5, 15] = 1
    U17[5, 16] = 1; V17[8, 16] = 1; W17[5, 16] = 1
    
    # Verify the rank-17 decomposition is correct
    T_recon = np.einsum('ar,br,cr->abc', U17, V17, W17)
    if np.max(np.abs(T_recon - T)) > 1e-10:
        return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}
    
    def fast_residual(U, V, W):
        """Compute the residual for the tensor decomposition."""
        res = 0.0
        for u_idx, v_idx, w_idx in indices:
            val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
            res += (val - 1.0) ** 2
        return res
    
    # Strategy: Fix U to a sparse pattern, solve for V and W
    # The idea: if we fix U, the problem becomes:
    # sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
    # For each (a,b,c) with T[a,b,c]=1: sum_r U[a,r] * V[b,r] * W[c,r] = 1
    # This is still nonlinear in V,W, but if we fix V as well, it's linear in W
    # 
    # Better approach: Try specific sparse U patterns and solve the resulting system
    
    R = 16
    
    # Generate candidate sparse U patterns
    # Each column of U should be sparse (1-3 non-zeros)
    # Based on Strassen structure for 2x2x2
    
    def generate_sparse_U_patterns():
        """Generate candidate sparse U patterns for rank-16."""
        patterns = []
        
        # Pattern 1: Based on Strassen for 2x2x2 extended to 2x3x3
        # Strassen uses 7 terms for 2x2x2. For 2x3x3, we need more.
        # Let's try patterns where each column has 2 non-zeros
        
        # Define possible support sets for each column
        # U has shape (6, 16). Each column should be sparse.
        
        # Try a structured pattern:
        # Columns 0-6: Strassen-like for the 2x2 core (indices 0,1,3,4 in U)
        # Columns 7-15: Additional terms for the extra column
        
        # Strassen 2x2x2 pattern (7 terms):
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B01-B11)
        # M4 = A11*(B10-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B10)
        # M7 = (A01-A11)*(B01+B11)
        
        # For 2x3x3, map A00->U[0], A01->U[1], A10->U[3], A11->U[4]
        # The extra elements A02->U[2], A12->U[5] need additional terms
        
        # Let's try: first 7 columns follow Strassen, remaining 9 are for extra
        U_pat = np.zeros((6, 16))
        # Strassen-like columns
        U_pat[0, 0] = 1; U_pat[4, 0] = 1  # A00+A11
        U_pat[3, 1] = 1; U_pat[4, 1] = 1  # A10+A11
        U_pat[0, 2] = 1                     # A00
        U_pat[4, 3] = 1                     # A11
        U_pat[0, 4] = 1; U_pat[1, 4] = 1   # A00+A01
        U_pat[3, 5] = 1; U_pat[0, 5] = -1  # A10-A00
        U_pat[1, 6] = 1; U_pat[4, 6] = -1  # A01-A11
        
        # Additional columns for the extra column (index 2 and 5 in U)
        # These need to handle A02 and A12
        U_pat[2, 7] = 1
        U_pat[5, 8] = 1
        U_pat[2, 9] = 1
        U_pat[5, 10] = 1
        U_pat[0, 11] = 1
        U_pat[1, 12] = 1
        U_pat[2, 13] = 1
        U_pat[3, 14] = 1
        U_pat[4, 15] = 1
        
        patterns.append(U_pat)
        
        # Pattern 2: More balanced distribution
        U_pat2 = np.zeros((6, 16))
        # Each of the 6 rows should appear in about 16/6 ≈ 2.67 columns
        # Try: each row appears in 2 or 3 columns
        
        # Row 0 (A00): columns 0,1,4
        U_pat2[0, 0] = 1
        U_pat2[0, 1] = 1
        U_pat2[0, 4] = 1
        
        # Row 1 (A01): columns 2,3,5
        U_pat2[1, 2] = 1
        U_pat2[1, 3] = 1
        U_pat2[1, 5] = 1
        
        # Row 2 (A02): columns 6,7,8
        U_pat2[2, 6] = 1
        U_pat2[2, 7] = 1
        U_pat2[2, 8] = 1
        
        # Row 3 (A10): columns 9,10,11
        U_pat2[3, 9] = 1
        U_pat2[3, 10] = 1
        U_pat2[3, 11] = 1
        
        # Row 4 (A11): columns 12,13,14
        U_pat2[4, 12] = 1
        U_pat2[4, 13] = 1
        U_pat2[4, 14] = 1
        
        # Row 5 (A12): columns 15,0,1 (wrap around)
        U_pat2[5, 15] = 1
        U_pat2[5, 0] = 1
        U_pat2[5, 1] = 1
        
        patterns.append(U_pat2)
        
        return patterns
    
    def solve_for_VW(U_fixed):
        """Given a fixed U, try to find V and W such that the decomposition holds."""
        R_val = U_fixed.shape[1]
        
        # The constraint is: sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
        # This is a system of equations in V and W.
        # 
        # Approach: For a given V, the equations become linear in W.
        # For each (a,b,c) with T[a,b,c]=1:
        #   sum_r U[a,r] * V[b,r] * W[c,r] = 1
        #
        # This is: sum_r (U[a,r] * V[b,r]) * W[c,r] = 1
        #
        # Let M[a,b,c,r] = U[a,r] * V[b,r]. Then:
        #   sum_r M[a,b,c,r] * W[c,r] = 1
        #
        # But this is still coupled across different (a,b) for the same c.
        #
        # Better: Fix V as well, then it's linear in W.
        # Or: Use alternating minimization.
        
        # Try: Initialize V randomly, then solve for W, then solve for V, iterate.
        
        V = np.random.randn(N_v, R_val) * 0.5
        W = np.random.randn(N_w, R_val) * 0.5
        
        for iteration in range(50):
            if time.time() > deadline - 30:
                break
            
            # Solve for W given U, V
            # For each c: sum_r (sum_a U[a,r]*V[b,r]*delta) ... 
            # Actually: For each (a,b,c) with T=1:
            #   sum_r U[a,r] * V[b,r] * W[c,r] = 1
            #
            # Group by c: For fixed c, we have equations:
            #   sum_r W[c,r] * (U[a,r] * V[b,r]) = 1  for all (a,b) such that T[a,b,c]=1
            #
            # This is: W[c,:] @ A_c = 1  where A_c[a,b,r] = U[a,r]*V[b,r]
            # But W[c,:] has only R_val components, and we have multiple (a,b) constraints.
            
            # For each c, build a linear system for W[c,:]
            W_new = np.zeros_like(W)
            for c in range(N_w):
                # Find all (a,b) such that T[a,b,c] = 1
                eqs = []
                for a in range(N_u):
                    for b in range(N_v):
                        if T[a, b, c] != 0:
                            # sum_r U[a,r] * V[b,r] * W[c,r] = 1
                            row = U_fixed[a, :] * V[b, :]
                            eqs.append(row)
                
                if eqs:
                    A_sys = np.array(eqs)  # shape (num_eqs, R_val)
                    b_sys = np.ones(len(eqs))
                    # Solve A_sys @ W[c,:] = b_sys (least squares)
                    try:
                        W_sol, _, _, _ = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                        W_new[c, :] = W_sol
                    except:
                        W_new[c, :] = W[c, :]
            
            # Solve for V given U, W
            V_new = np.zeros_like(V)
            for b in range(N_v):
                eqs = []
                for a in range(N_u):
                    for c in range(N_w):
                        if T[a, b, c] != 0:
                            # sum_r U[a,r] * V[b,r] * W[c,r] = 1
                            row = U_fixed[a, :] * W[c, :]
                            eqs.append(row)
                
                if eqs:
                    A_sys = np.array(eqs)  # shape (num_eqs, R_val)
                    b_sys = np.ones(len(eqs))
                    try:
                        V_sol, _, _, _ = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                        V_new[b, :] = V_sol
                    except:
                        V_new[b, :] = V[b, :]
            
            # Check convergence
            W = W_new
            V = V_new
            
            # Compute residual
            T_recon = np.einsum('ar,br,cr->abc', U_fixed, V, W)
            res = np.sum((T_recon - T) ** 2)
            
            if res < 1e-10:
                return U_fixed, V, W, res
        
        # Final residual
        T_recon = np.einsum('ar,br,cr->abc', U_fixed, V, W)
        res = np.sum((T_recon - T) ** 2)
        return U_fixed, V, W, res
    
    # Try each U pattern
    best_solution = (U17.copy(), V17.copy(), W17.copy())
    best_res = fast_residual(U17, V17, W17)
    
    patterns = generate_sparse_U_patterns()
    
    for U_pat in patterns:
        if time.time() > deadline - 20:
            break
        
        # Try multiple random initializations for V, W
        for trial in range(10):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(42 + trial)
            U_test, V_test, W_test, res = solve_for_VW(U_pat)
            
            if res < best_res:
                best_res = res
                best_solution = (U_test.copy(), V_test.copy(), W_test.copy())
                if best_res < 1e-10:
                    # Try to round
                    U_round = np.round(U_test * 2) / 2
                    V_round = np.round(V_test * 2) / 2
                    W_round = np.round(W_test * 2) / 2
                    res_round = fast_residual(U_round, V_round, W_round)
                    if res_round < 1e-10:
                        return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
                    break
    
    # If we found a good rank-16 solution, try to round
    if best_res < 1e-8:
        U_best, V_best, W_best = best_solution
        
        # Try multiple rounding strategies
        rounding_methods = [
            ('half', lambda x: np.round(x * 2) / 2),
            ('int', lambda x: np.round(x)),
            ('quarter', lambda x: np.round(x * 4) / 4),
        ]
        
        for name, round_fn in rounding_methods:
            U_round = round_fn(U_best)
            V_round = round_fn(V_best)
            W_round = round_fn(W_best)
            
            res_round = fast_residual(U_round, V_round, W_round)
            if res_round < 1e-10:
                return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
        
        # Try mixed rounding
        for u_round in [lambda x: np.round(x * 2) / 2, lambda x: np.round(x)]:
            for v_round in [lambda x: np.round(x * 2) / 2, lambda x: np.round(x)]:
                for w_round in [lambda x: np.round(x * 2) / 2, lambda x: np.round(x)]:
                    U_r = u_round(U_best)
                    V_r = v_round(V_best)
                    W_r = w_round(W_best)
                    res_r = fast_residual(U_r, V_r, W_r)
                    if res_r < 1e-10:
                        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    # Fall back to rank-17
    return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}
