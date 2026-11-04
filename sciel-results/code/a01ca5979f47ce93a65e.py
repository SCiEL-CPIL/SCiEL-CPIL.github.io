import numpy as np
import time
import random

def construct():
    n, m, p = 3, 3, 3
    deadline = time.time() + 110  # Safety margin
    
    # Start with the known rank-23 decomposition for 3x3 matrix multiplication
    # This uses the structure of the tensor to reduce from 27 to 23 terms
    
    # The key insight: we can combine some of the 27 terms using algebraic identities
    # Specifically, we use 4 "Strassen-like" combinations that each save 1 rank
    
    # For 3x3, we can use the following approach:
    # - 9 terms for the "main diagonal" products (i=j=k pattern doesn't work directly)
    # - Actually, let's use a systematic construction
    
    # Known rank-23 decomposition for 3x3 matrix multiplication:
    # We'll use 23 terms where some terms compute multiple products
    
    rank = 23
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Term 0-8: Direct terms for specific (i,j,k) combinations
    # We'll use a pattern that covers all 27 equations with 23 terms
    
    # Strategy: Use 9 terms for "easy" cases, and 14 terms for the rest
    # with some sharing
    
    # Let me use a known construction:
    # Terms 0-8: U[i*3+j, t] = 1, V[j*3+k, t] = 1, W[k*3+i, t] = 1 for specific t
    # But we need to be more clever
    
    # Actually, let me use the approach from the literature:
    # For 3x3, we can achieve rank 23 by using 4 additional "Strassen" terms
    # that each handle 4 products, saving 1 rank each compared to direct
    
    # Let me construct this explicitly:
    
    # Direct terms (19 of them):
    direct_terms = [
        (0,0,0), (0,0,1), (0,0,2),
        (0,1,0), (0,1,1), (0,1,2),
        (0,2,0), (0,2,1), (0,2,2),
        (1,0,0), (1,0,1), (1,0,2),
        (1,1,0), (1,1,1), (1,1,2),
        (1,2,0), (1,2,1), (1,2,2),
        (2,0,0), (2,0,1), (2,0,2),
        (2,1,0), (2,1,1), (2,1,2),
        (2,2,0), (2,2,1), (2,2,2),
    ]
    
    # That's 27, we need to reduce to 23
    # Remove 4 terms and replace with 4 "combined" terms
    
    # Remove these 4: (0,1,1), (1,0,1), (1,1,0), (1,1,2)
    # Replace with 4 Strassen-like terms
    
    # This is getting complex. Let me use a different approach:
    # Use random search but with better initialization and more trials
    
    best_rank = 27
    best_U, best_V, best_W = None, None, None
    
    # Seed with naive solution
    U_naive = np.zeros((n*m, 27), dtype=int)
    V_naive = np.zeros((m*p, 27), dtype=int)
    W_naive = np.zeros((p*n, 27), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_naive[i*m+j, col] = 1
                V_naive[j*p+k, col] = 1
                W_naive[k*n+i, col] = 1
                col += 1
    best_U, best_V, best_W = U_naive, V_naive, W_naive
    best_rank = 27
    
    # Try to find rank 23 using a structured approach
    # Use the fact that we can have U, V entries in {-1, 0, 1}
    # and W entries can be half-integers
    
    # Let me try a more systematic search for rank 23
    for rank in [23, 24, 25]:
        if time.time() > deadline:
            break
            
        found = False
        for trial in range(200):
            if time.time() > deadline:
                break
                
            # Generate sparse U and V with entries in {-1, 0, 1}
            U = np.zeros((n*m, rank), dtype=int)
            V = np.zeros((m*p, rank), dtype=int)
            
            # Use a structured pattern: each column has 2-3 non-zeros
            for c in range(rank):
                # Choose 2-3 rows for U
                num_u = random.choice([2, 3])
                u_rows = random.sample(range(n*m), num_u)
                for idx in u_rows:
                    U[idx, c] = random.choice([-1, 1])
                
                # Choose 2-3 rows for V
                num_v = random.choice([2, 3])
                v_rows = random.sample(range(m*p), num_v)
                for idx in v_rows:
                    V[idx, c] = random.choice([-1, 1])
                    
            # Build the system: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            # This is nonlinear in W, but for fixed U,V it's linear in W
            # M[e, (c, w_idx)] = U[i*m+j,c] * V[j*p+k,c] where e=(i,j,k), w_idx=k*n+i
            
            num_eqs = 27
            M = np.zeros((num_eqs, p*n * rank))
            
            eq_count = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        w_idx = k * n + i
                        for c in range(rank):
                            coeff = U[i*m+j, c] * V[j*p+k, c]
                            if coeff != 0:
                                M[eq_count, c * p*n + w_idx] = coeff
                        eq_count += 1
            
            # Solve M @ W_vec = 1
            try:
                W_vec, residuals, rank_m, s = np.linalg.lstsq(M, np.ones(num_eqs), rcond=None)
                W_check = W_vec.reshape(rank, p*n)
                
                # Check if W entries are close to half-integers
                W_doubled = W_check * 2
                W_rounded = np.round(W_doubled)
                error = np.max(np.abs(W_doubled - W_rounded))
                
                if error < 1e-4:
                    W_solution = W_rounded / 2.0
                    
                    # Verify exactly
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                w_idx = k * n + i
                                val = 0.0
                                for c in range(rank):
                                    val += U[i*m+j, c] * V[j*p+k, c] * W_solution[c, w_idx]
                                if abs(val - 1.0) > 1e-8:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                            
                    if valid:
                        best_rank = rank
                        best_U = U
                        best_V = V
                        best_W = W_solution
                        found = True
                        break
            except Exception:
                continue
        
        if found:
            break
    
    def to_list(mat):
        return mat.tolist()
        
    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
