import numpy as np
from itertools import product


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Analytical reduction for 3-point support {0, 1, M}.
    
    P = Prob[X1 + X2 + X3 < 2*X4]
    
    For support {0, 1, M} with weights (a, b, c) where a+b+c=1:
    We enumerate all 81 combinations of (i,j,k,l) and determine which satisfy
    s_i + s_j + s_k < 2*s_l.
    
    For each M, the condition depends on whether M > 1 (which we assume).
    The 81 conditions break into specific cases. We can write P as a 
    cubic polynomial in (a, b, c) with coefficients determined by M.
    
    Then for each M, we do a 2D grid search over (a, c) with b = 1-a-c >= 0.
    """
    
    # Support points
    s = np.array([0.0, 1.0])  # base, M added as third
    
    # We'll enumerate all (i, j, k, l) where each index is in {0, 1, 2}
    # Index 2 corresponds to M
    
    # Precompute the condition for each M
    # For a given M, condition is: s_i + s_j + s_k < 2*s_l
    # where s = [0, 1, M]
    
    # The key insight: for M > 1, the condition s_i + s_j + s_k < 2*s_l
    # depends on how many of the indices are 2 (i.e., equal to M).
    
    # Let's enumerate all 81 cases and for each, determine the condition as a 
    # function of M, then group them.
    
    # For each (i,j,k,l), the LHS = s_i + s_j + s_k and RHS = 2*s_l
    # s_0 = 0, s_1 = 1, s_2 = M
    
    # LHS can be: 0, 1, 2, 3, M, M+1, M+2, M+3, 2M, 2M+1, 2M+2, 3M
    # RHS can be: 0, 2, 2M
    
    # Condition: LHS < RHS
    
    # Let's just build the coefficient matrix for each M by direct enumeration.
    # For efficiency, we'll do a grid search over M and (a,c).
    
    # For a given M, P(a,b,c) = sum_{i,j,k,l: cond} a^{[i==0]} b^{[i==1]} c^{[i==2]} 
    #   * a^{[j==0]} b^{[j==1]} c^{[j==2]} 
    #   * a^{[k==0]} b^{[k==1]} c^{[k==2]} 
    #   * a^{[l==0]} b^{[l==1]} c^{[l==2]}
    
    # This is a homogeneous cubic in (a,b,c) since each term has total degree 4? 
    # Wait, no - it's degree 4: one factor for each of i,j,k,l.
    
    # Actually P is degree 4 in the weights. But since a+b+c=1, we can reduce.
    
    # Let's just compute P directly for each (a,c) given M.
    
    # Grid over M
    M_vals = np.concatenate([
        np.linspace(1.01, 2.0, 50),
        np.linspace(2.0, 5.0, 50),
        np.linspace(5.0, 20.0, 40),
        np.linspace(20.0, 100.0, 30),
        np.array([200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 50000.0])
    ])
    
    n_grid = 300
    
    best_P = -1.0
    best_a, best_b, best_c, best_M = 0.0, 0.0, 0.0, 2.0
    
    for M in M_vals:
        if M <= 1.0 + 1e-9:
            continue
        
        s = np.array([0.0, 1.0, M])
        
        # Precompute which (i,j,k,l) satisfy the condition
        # Build a 3x3x3x3 boolean array
        cond = np.zeros((3, 3, 3, 3), dtype=bool)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        if s[i] + s[j] + s[k] < 2.0 * s[l]:
                            cond[i, j, k, l] = True
        
        # Now P = sum_{i,j,k,l: cond[i,j,k,l]} w_i w_j w_k w_l
        # where w = [a, b, c]
        
        # For a grid of (a, c), compute P
        a_vals = np.linspace(0.0, 1.0, n_grid)
        
        best_local_P = -1.0
        best_local_a = 0.0
        best_local_c = 0.0
        
        for a in a_vals:
            c_max = 1.0 - a
            if c_max < 0:
                continue
            c_vals = np.linspace(0.0, c_max, n_grid)
            b_vals = 1.0 - a - c_vals
            valid = b_vals >= -1e-12
            if not np.any(valid):
                continue
            c_vals = c_vals[valid]
            b_vals = np.maximum(b_vals[valid], 0.0)
            
            # w shape: (n_c, 3)
            w = np.stack([np.full_like(c_vals, a), b_vals, c_vals], axis=1)
            
            # Compute P for each row
            # P = sum_{i,j,k,l: cond} w_i w_j w_k w_l
            # Vectorize over the n_c rows
            
            P_vals = np.zeros(len(c_vals))
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        for l in range(3):
                            if cond[i, j, k, l]:
                                P_vals += w[:, i] * w[:, j] * w[:, k] * w[:, l]
            
            max_idx = np.argmax(P_vals)
            if P_vals[max_idx] > best_local_P:
                best_local_P = P_vals[max_idx]
                best_local_a = a
                best_local_c = c_vals[max_idx]
        
        if best_local_P > best_P:
            best_P = best_local_P
            best_a = best_local_a
            best_c = best_local_c
            best_M = M
    
    b = 1.0 - best_a - best_c
    if b < 0:
        b = 0.0
    
    support = np.array([0.0, 1.0, best_M], dtype=np.float64)
    weights = np.array([best_a, b, best_c], dtype=np.float64)
    weights = np.maximum(weights, 0.0)
    
    return support, weights
