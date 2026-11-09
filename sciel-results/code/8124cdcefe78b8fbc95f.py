import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s margin for safety

    # Schoolbook decomposition (Rank 30) as fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                d = k * n + i
                U_sb[a][col] = 1
                V_sb[b][col] = 1
                W_sb[d][col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    # Values to use for random initialization of U and V
    # Using fractions for exact arithmetic where possible, but numpy floats for speed in solving
    # We'll use a set of small integers and half-integers
    init_values = [0, 0.5, 1, -0.5, -1]
    
    def is_valid(U, V, W):
        """
        Exact verification of the decomposition.
        U: (n*m, R), V: (m*p, R), W: (p*n, R)
        """
        R = len(U[0])
        # Check all non-zero entries of the tensor
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    d = k * n + i
                    s = 0.0
                    for r in range(R):
                        s += U[a][r] * V[b][r] * W[d][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        # Check some zero entries to ensure no spurious non-zeros
        # The tensor T is 1 at (i*m+j, j*p+k, k*n+i) and 0 elsewhere.
        # We need to ensure that for any (a,b,d) not of the form above, the sum is 0.
        # There are (n*m)*(m*p)*(p*n) - n*m*p entries that should be 0.
        # For 2x3x5: 6*15*10 - 30 = 870 entries.
        # This check is expensive if done naively, but we can be smarter or rely on the structure.
        # Actually, if we solved for W exactly using the non-zero constraints, 
        # the zero constraints are not automatically satisfied unless the rank is high enough 
        # or the specific random U,V happen to satisfy them.
        # We must check them.
        
        # Efficient check for zeros:
        # For each d, we have W[d].
        # For each a, b, we compute S[a,b] = sum_r U[a,r] V[b,r] W[d,r].
        # We need S[a,b,d] = 0 for all (a,b,d) where T[a,b,d]=0.
        
        # Let's construct the full tensor in memory if R is small enough?
        # R is likely < 30. 6*15*10 = 900 entries.
        # We can compute the full reconstruction and compare.
        
        T_recon = np.zeros((n*m, m*p, p*n))
        for r in range(R):
            # u_r is (n*m, 1), v_r is (m*p, 1), w_r is (p*n, 1)
            # T_recon += outer(outer(u_r, v_r), w_r)
            # Using einsum is faster
            pass
        
        # Using numpy for speed in verification
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        # T_recon[a,b,d] = sum_r U[a,r] V[b,r] W[d,r]
        T_recon = np.einsum('ar,br,dr->abd', U_np, V_np, W_np)
        
        # Create target tensor
        T_target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    d = k * n + i
                    T_target[a, b, d] = 1.0
                    
        if not np.allclose(T_recon, T_target, atol=1e-9):
            return False
            
        return True

    def try_rank(target_rank):
        """
        Try to find a decomposition of target_rank using random U, V and solving for W.
        """
        if time.time() > deadline:
            return None
        
        # Number of trials
        # For small ranks, we need many trials.
        # 18 is the target.
        max_trials = 2000
        
        for trial in range(max_trials):
            if time.time() > deadline:
                return None
            
            # Randomly initialize U and V
            U = [[random.choice(init_values) for _ in range(target_rank)] for _ in range(n*m)]
            V = [[random.choice(init_values) for _ in range(target_rank)] for _ in range(m*p)]
            
            # Convert to numpy for solving
            U_np = np.array(U)
            V_np = np.array(V)
            
            W = [[0.0] * target_rank for _ in range(p*n)]
            valid = True
            
            for i in range(n):
                for k in range(p):
                    d = k * n + i
                    # System of equations for W[d, :]
                    # For each j in 0..m-1:
                    # sum_r W[d,r] * U[i*m+j, r] * V[j*p+k, r] = 1
                    
                    # Build matrix A of shape (m, target_rank)
                    A = np.zeros((m, target_rank))
                    for j in range(m):
                        a_idx = i * m + j
                        b_idx = j * p + k
                        # A[j, r] = U[a_idx, r] * V[b_idx, r]
                        A[j, :] = U_np[a_idx, :] * V_np[b_idx, :]
                    
                    b_vec = np.ones(m)
                    
                    # Solve A @ w = b_vec
                    # We want a solution w that is close to integers/half-integers.
                    # First, check if a solution exists in reals.
                    try:
                        # Use lstsq to handle potential rank deficiency or inconsistency
                        w_sol, residuals, rank_A, sv = np.linalg.lstsq(A, b_vec, rcond=None)
                        
                        # Check residual
                        if np.max(np.abs(A @ w_sol - b_vec)) > 1e-9:
                            valid = False
                            break
                            
                        # Try to round w_sol to nearest half-integer
                        # w_rounded = round(2 * w_sol) / 2
                        w_rounded = np.round(2 * w_sol) / 2.0
                        
                        # Verify the rounded solution exactly
                        if np.max(np.abs(A @ w_rounded - b_vec)) > 1e-9:
                            # Maybe try other rounding strategies?
                            # Or just fail this trial.
                            valid = False
                            break
                            
                        W[d] = w_rounded.tolist()
                        
                    except Exception:
                        valid = False
                        break
            
            if not valid:
                continue
            
            # If we got here, we have a candidate U, V, W
            # Verify the full tensor
            if is_valid(U, V, W):
                return (U, V, W)
                
        return None

    # Search for lower ranks
    # Conjectured rank is 18.
    # We iterate from 18 to schoolbook rank - 1.
    for r in range(18, rank_schoolbook):
        if time.time() > deadline:
            break
            
        res = try_rank(r)
        if res:
            U, V, W = res
            best_rank = r
            best_U, best_V, best_W = U, V, W
            break # Found the lowest rank in this sequence
            
    return {"U": best_U, "V": best_V, "W": best_W}
