"""Systematic search for rank-minimal 2x3*3x4 matrix mult tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 4
    start_time = time.time()
    deadline = start_time + 145.0

    # Build the multiplication tensor
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1

    def validate_fast(U, V, W):
        """Fast vectorized validation."""
        T_hat = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(T_hat, T, atol=1e-10)

    # Schoolbook solution: rank = n*m*p = 24
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb), dtype=float)
    V_sb = np.zeros((m*p, rank_sb), dtype=float)
    W_sb = np.zeros((p*n, rank_sb), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1

    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    # Try structured patterns for lower ranks
    # Pattern: For each (i, j, k) we need to enforce i_a=i_c, j_a=j_b, k_b=k_c
    # Let's try rank = n*p = 8 with clever structure
    
    # Actually, let's try a different approach: use the fact that we can have
    # components that partially match indices
    
    # Try rank = 16: for each (i, k), use 2 components (one for each j)
    # But that's still 24...
    
    # Let's try rank = 12: for each (j, k), one component that handles all i
    # U[i*m+j, r_{j,k}] = 1 for all i
    # V[j*p+k, r_{j,k}] = 1
    # W[k*n+i, r_{j,k}] = 1 for all i
    # This gives T_hat[a,b,c] = [j_a==j_b] * [k_b==k_c] * [number of i with i_a=i_c=i]
    # = [j_a==j_b] * [k_b==k_c] * [i_a==i_c] * 1 (if i_a=i_c)
    # Wait, that's actually correct! Let me verify.
    
    # For component r_{j,k}:
    # U[a, r] = 1 iff a = i*m+j for some i (i.e., the j-component of a matches j)
    # V[b, r] = 1 iff b = j*p+k (exact match)
    # W[c, r] = 1 iff c = k*n+i for some i (i.e., the k-component of c matches k)
    
    # T_hat[a,b,c] = sum_{j,k} [a's j-component == j] * [b == j*p+k] * [c's k-component == k]
    # = sum_{j,k} [j_a == j] * [j_b == j and k_b == k] * [k_c == k]
    # = [j_a == j_b] * [k_b == k_c] * (sum over j,k where j=j_a=j_b and k=k_b=k_c)
    # = [j_a == j_b] * [k_b == k_c] * 1 (if the conditions are consistent)
    
    # But we also need i_a == i_c! This pattern doesn't enforce that.
    # So T_hat would be 1 whenever j_a==j_b and k_b==k_c, regardless of i.
    # That's wrong.
    
    # The issue is that W[c,r] = 1 for all i when c's k-component matches k.
    # We need W to also depend on i.
    
    # Let's try rank = n*m*p = 24 is schoolbook. Can we do better?
    
    # Try rank = 18: some sharing between components
    # Or try rank = 16, 14, 12, 10, 8...
    
    # For a systematic search, let's fix the support pattern and solve linear equations
    # For rank r, fix U and V, then solve for W
    
    # The key constraint: for each c, we need
    # sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] for all (a,b)
    
    # Let's try random U, V with structured supports and solve for W
    
    def try_structured_rank(r, n_trials=3000):
        """Try structured random U, V and solve for W."""
        for trial in range(n_trials):
            if time.time() > deadline:
                return None
            
            # Create U with sparse structure: each row has ~1-2 nonzeros
            U_rand = np.zeros((n*m, r), dtype=float)
            V_rand = np.zeros((m*p, r), dtype=float)
            
            # For U: each row (i*m+j) should have nonzeros in columns that "make sense"
            # For V: each row (j*p+k) similarly
            # For simplicity, let's use random sparse matrices
            
            # Random support: each entry is nonzero with some probability
            for a in range(n*m):
                num_nonzero = np.random.randint(1, min(3, r+1))
                cols = np.random.choice(r, size=num_nonzero, replace=False)
                for c in cols:
                    U_rand[a, c] = np.random.choice([-1, 1])
            
            for b in range(m*p):
                num_nonzero = np.random.randint(1, min(3, r+1))
                cols = np.random.choice(r, size=num_nonzero, replace=False)
                for c in cols:
                    V_rand[b, c] = np.random.choice([-1, 1])
            
            # Build P matrix: P[ab, r] = U[a,r] * V[b,r]
            P = np.zeros((n*m * m*p, r))
            for a in range(n*m):
                for b in range(m*p):
                    P[a*m*p + b, :] = U_rand[a, :] * V_rand[b, :]
            
            # For each c, solve P * W_c = T[:, :, c].flatten()
            W_sol = np.zeros((p*n, r))
            valid = True
            for c in range(p*n):
                rhs = T[:, :, c].flatten()
                try:
                    sol, residuals, rank_p, sv = np.linalg.lstsq(P, rhs, rcond=None)
                    residual = np.linalg.norm(P @ sol - rhs)
                    if residual > 1e-10:
                        valid = False
                        break
                    W_sol[c, :] = sol
                except:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check if W_sol has "nice" values
            # and validate the full decomposition
            if validate_fast(U_rand, V_rand, W_sol):
                return U_rand, V_rand, W_sol
        
        return None

    # Try ranks from 12 to 23
    for r in range(12, 24):
        if time.time() > deadline:
            break
        result = try_structured_rank(r, n_trials=2000)
        if result is not None:
            U_found, V_found, W_found = result
            best_rank = r
            best_U, best_V, best_W = U_found, V_found, W_found
            break

    # If no better solution found, return schoolbook
    result = {
        'U': best_U.tolist(),
        'V': best_V.tolist(),
        'W': best_W.tolist()
    }

    return result
