"""Matrix multiplication 2x3x10 - rank-minimal decomposition search."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    if not U or not V or not W:
        return False
    
    r = len(U[0]) if U else 0
    # Check target entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                total = 0
                for col in range(r):
                    total += U[a][col] * V[b][col] * W[c][col]
                if abs(total - 1) > 1e-9:
                    return False
    
    # Check non-target entries are 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                for c_test in range(p * n):
                    if c_test == k * n + i:
                        continue
                    total = 0
                    for col in range(r):
                        total += U[a][col] * V[b][col] * W[c_test][col]
                    if abs(total) > 1e-9:
                        return False
    return True

def _schoolbook_decomposition(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _try_random_rank(n, m, p, r, num_trials=50, deadline=None):
    """
    Attempt to find a rank-r decomposition by randomizing supports and solving linear systems.
    Returns (U, V, W) if successful, else None.
    """
    # Pre-compute the constraints:
    # For each (i, j, k), we need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # And for all other (i, j, c_test) where c_test != k*n+i, sum_c ... = 0
    
    # This is a system of linear equations in the entries of W, given U and V.
    # However, U and V are also unknown.
    # Strategy:
    # 1. Randomly initialize U and V with sparse entries (mostly 0, some 1s or -1s).
    # 2. For each column c, we have a triplet (u_c, v_c, w_c).
    #    The contribution to entry (a, b, c_idx) is u_c[a] * v_c[b] * w_c[c_idx].
    #    We need sum_c u_c[a] * v_c[b] * w_c[c_idx] = delta(a,b,c_idx).
    
    # Let's fix U and V randomly (sparse) and solve for W.
    # If the system is consistent and the solution W has "nice" properties (or just exists), we have a decomposition.
    # Note: The problem asks for integer/half-integer entries. The linear solve might give rationals.
    # We can check if the solution is close to half-integers.
    
    A_size = n * m
    B_size = m * p
    C_size = p * n
    
    # Number of constraints: A_size * B_size * C_size
    # But only A_size * B_size * (C_size/A_size) = n*m*p are non-zero targets?
    # No, the tensor is dense in the sense that we must specify all zeros.
    # Total constraints: A_size * B_size * C_size.
    # Unknowns: r * C_size (entries of W).
    # For r < 60, r * C_size = r * 20.
    # Constraints = 6 * 30 * 20 = 3600.
    # If r=59, unknowns = 1180. Overdetermined.
    
    # Actually, we can exploit symmetry/structure.
    # Let's try to fix U and V to be "basis-like" or random sparse, and solve for W.
    
    for trial in range(num_trials):
        if deadline and time.time() > deadline:
            break
            
        # Generate random sparse U and V
        # Let's try a specific structure:
        # U and V have entries in {-1, 0, 1}.
        # W will be determined.
        
        U_rand = np.zeros((A_size, r))
        V_rand = np.zeros((B_size, r))
        
        # Randomly set some entries to 1 or -1
        # Density ~ 0.5
        for c in range(r):
            for a in range(A_size):
                if np.random.rand() < 0.5:
                    U_rand[a, c] = 1 if np.random.rand() < 0.5 else -1
            for b in range(B_size):
                if np.random.rand() < 0.5:
                    V_rand[b, c] = 1 if np.random.rand() < 0.5 else -1
        
        # Now solve for W.
        # The equation is: sum_c U_rand[a, c] * V_rand[b, c] * W[c, c_idx] = T[a, b, c_idx]
        # Let M[a, b, c] = U_rand[a, c] * V_rand[b, c].
        # Then sum_c M[a, b, c] * W[c, c_idx] = T[a, b, c_idx].
        # This is a linear system for each c_idx independently?
        # No, W is a matrix (C_size x r).
        # Let's reshape.
        # For a fixed c_idx (let's call it k_out), we have a vector w_k of length r.
        # The equation for all (a, b) is:
        # sum_c M[a, b, c] * w_k[c] = T[a, b, k_out]
        # This is a linear system M_vec * w_k = T_vec, where M_vec is the vectorized M.
        
        # Wait, M depends on a and b.
        # Let's vectorize the LHS.
        # LHS(a, b) = sum_c M[a, b, c] * w[c].
        # This is not a standard linear system in w for all (a,b) simultaneously unless we treat it as a bilinear form.
        # Actually, for each (a, b), we have one linear equation in w.
        # Total equations: A_size * B_size.
        # Unknowns: r.
        # If A_size * B_size >= r, we can solve for w for each k_out.
        
        # Let's construct the matrix for one k_out.
        # Rows correspond to (a, b) pairs.
        # Columns correspond to c (0..r-1).
        # Entry (ab, c) = U_rand[a, c] * V_rand[b, c].
        
        # We need to solve this for each of the C_size output indices.
        # But T[a, b, c_idx] is non-zero only for specific (a, b, c_idx) triples.
        # Specifically, T[a, b, c_idx] = 1 if (a,b,c_idx) is a valid index triple, else 0.
        
        # Let's build the constraint matrix A_sys of shape (A_size * B_size, r).
        A_sys = np.zeros((A_size * B_size, r))
        for a in range(A_size):
            for b in range(B_size):
                idx = a * B_size + b
                for c in range(r):
                    A_sys[idx, c] = U_rand[a, c] * V_rand[b, c]
        
        # Now, for each output index k_out (0..C_size-1), we have a target vector T_k_out of length A_size * B_size.
        # T_k_out[idx] = 1 if idx corresponds to (a, b) such that (a, b, k_out) is a target, else 0.
        
        # Let's precompute the target vectors.
        # Target condition:
        # a = i * m + j
        # b = j * p + k
        # c_idx = k * n + i
        # Note that k is determined by b and j: k = b // p? No, b = j*p + k, so k = b % p.
        # And j is determined by a and i: j = a % m.
        # So for a given (a, b), the target c_idx is uniquely determined IF the indices are consistent.
        # Let's check consistency:
        # j = a % m
        # i = a // m
        # k = b % p
        # j' = b // p
        # If j == j', then the target c_idx is k * n + i.
        # If j != j', then T[a, b, c_idx] = 0 for all c_idx.
        
        # So, for each k_out, the target vector T_k_out has 1s at positions (a, b) where:
        # 1. j = a % m == b // p
        # 2. k_out == (b % p) * n + (a // m)
        
        # Let's iterate through k_out and solve.
        W_sol = np.zeros((C_size, r))
        success = True
        
        for k_out in range(C_size):
            T_vec = np.zeros(A_size * B_size)
            for a in range(A_size):
                i = a // m
                j = a % m
                for b in range(B_size):
                    j_b = b // p
                    k_b = b % p
                    if j == j_b:
                        target_c = k_b * n + i
                        if target_c == k_out:
                            idx = a * B_size + b
                            T_vec[idx] = 1.0
            
            # Solve A_sys * w = T_vec
            # A_sys is (A_size*B_size) x r.
            # We want a least squares solution, but we need an EXACT solution.
            # If the system is consistent, np.linalg.lstsq will find it.
            # However, if the system is inconsistent (due to random U, V), there is no solution.
            # We can check the residual.
            
            try:
                w_sol, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, T_vec, rcond=None)
                # Check residual
                res = np.linalg.norm(A_sys.dot(w_sol) - T_vec)
                if res > 1e-9:
                    success = False
                    break
                W_sol[k_out, :] = w_sol
            except np.linalg.LinAlgError:
                success = False
                break
        
        if success:
            # Check if W_sol has "nice" values (integers or half-integers)
            # The problem allows integer/half-integer entries.
            # Let's check if 2 * W_sol is close to integers.
            W_check = 2 * W_sol
            if np.allclose(W_check, np.round(W_check), atol=1e-6):
                # Convert to list of lists of floats (which are half-integers)
                U_list = [[float(x) for x in row] for row in U_rand]
                V_list = [[float(x) for x in row] for row in V_rand]
                W_list = [[float(x) for x in row] for row in W_sol]
                
                # Final verification
                if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                    return U_list, V_list, W_list
    
    return None

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # Generate schoolbook as fallback
    U_sb, V_sb, W_sb = _schoolbook_decomposition(n, m, p)
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Start from rank 55 and go up to 59
    # The inspiration suggests 55 is possible.
    
    # We will try ranks in descending order of likelihood/difficulty?
    # Actually, lower rank is harder. Let's try 59, 58, 57, 56, 55.
    
    # To maximize chances, we can allocate time proportionally or just loop.
    # Let's try 55 first with more trials if we have time, or just iterate.
    
    for r in range(55, 60):
        if time.time() > deadline - 5:
            break
            
        # Number of trials depends on remaining time
        # Let's just do a fixed number of trials for each rank
        result = _try_random_rank(n, m, p, r, num_trials=20, deadline=deadline)
        
        if result is not None:
            U_try, V_try, W_try = result
            best_U, best_V, best_W = U_try, V_try, W_try
            best_rank = r
            # If we found a lower rank, we could break or try even lower?
            # The loop goes 55, 56, 57, 58, 59.
            # If we find 55, we return it immediately as it's the best possible guess from inspiration.
            if r == 55:
                break
            # If we find 59, we continue to try 58? No, the loop is ascending in range(55, 60).
            # Wait, range(55, 60) is 55, 56, 57, 58, 59.
            # If we find 55, we break.
            # If we find 56, we should probably try to find 55 again?
            # But 55 is the first in the loop. So if we didn't find 55, we try 56.
            # This logic is slightly off. We should try the target rank (55) extensively first.
            
            # Let's restructure: Try 55 with many trials. If fail, try 56, etc.
            pass

    # Re-structured search:
    # 1. Try rank 55 with high trial count.
    # 2. If fail, try rank 56.
    # 3. Etc.
    
    # Since the previous loop already ran 55, let's assume if it failed, we move on.
    # But the previous loop only did 20 trials. Let's do more for 55 if time permits.
    
    # Actually, let's just stick to the simple loop. If we found 55, we broke.
    # If we didn't find 55, we tried 56, etc.
    # The best_rank variable holds the best found.
    
    # One more attempt for rank 55 if we didn't find it and have time
    if best_rank > 55 and time.time() < deadline - 10:
        result = _try_random_rank(n, m, p, 55, num_trials=50, deadline=deadline)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = 55

    if _verify_decomposition(best_U, best_V, best_W, n, m, p):
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        return {"U": U_sb, "V": V_sb, "W": W_sb}

# EVOLVE-BLOCK-END
