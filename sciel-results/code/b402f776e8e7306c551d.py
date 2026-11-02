"""Tensor decomposition search for 3x4x7 matrix multiplication."""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 4, 7]
START_TIME = time.time()
TIME_LIMIT = 120.0

def _elapsed():
    return time.time() - START_TIME

def _schoolbook(n, m, p):
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
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    """Check if the decomposition exactly decomposes matrix multiplication."""
    if not U or not V or not W:
        return False
    
    rank = len(U[0]) if U else 0
    if len(V[0]) != rank or len(W[0]) != rank:
        return False
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # Build the tensor T
    T = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        u = U[:, c].reshape(n, m)
        v = V[:, c].reshape(m, p)
        w = W[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                if u[i, j] != 0:
                    for k in range(p):
                        if v[j, k] != 0 and w[k, i] != 0:
                            T[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    
    return np.allclose(T, 1.0, atol=1e-10)

def _try_algebraic_attack(n, m, p, max_rank=None):
    """
    Try to find a low-rank decomposition using linear algebra.
    
    Strategy: For each possible rank r, try to find U, V, W such that
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    
    We can fix U and V to have simple patterns and solve for W.
    """
    if max_rank is None:
        max_rank = n * m * p
    
    best_U, best_V, best_W, best_rank = None, None, None, max_rank
    
    # Try ranks from 1 up to max_rank
    for r in range(1, max_rank + 1):
        if _elapsed() > TIME_LIMIT - 10:
            break
        
        # Try different strategies for constructing U and V
        # Strategy 1: U has 1s in specific positions, V has 1s in specific positions
        # W is then determined by solving a linear system
        
        # For a given rank r, we need to assign which (i,j) pairs get which U values,
        # which (j,k) pairs get which V values, and then solve for W.
        
        # Simplified approach: Try to find r pairs (i_c, j_c) for U, (j'_c, k_c) for V
        # such that we can solve for W.
        
        # Actually, let's try a more direct approach:
        # Assume U[i*m+j, c] = 1 if (i,j) is in some set S_c, else 0
        # Assume V[j*p+k, c] = 1 if (j,k) is in some set T_c, else 0
        # Then W[k*n+i, c] must satisfy:
        # sum_c [1 if (i,j) in S_c] * [1 if (j,k) in T_c] * W[k*n+i, c] = 1
        
        # This is a linear system in W. Let's try small r values.
        
        # For r = n*m*p (schoolbook), we know it works.
        # For smaller r, we need to find if a solution exists.
        
        # Let's try r = n*m = 12 first
        if r <= n * m:
            # Try: U[i*m+j, c] = 1 if c == i*m+j (diagonal-like)
            # V[j*p+k, c] = 1 if j == c % m and k == c // m (but c < m*p)
            # This doesn't quite work. Let me think differently.
            
            # Alternative: Use the fact that we can have fractional entries.
            # For rank r, we need r vectors u_c in R^(n*m), v_c in R^(m*p), w_c in R^(p*n)
            # such that the tensor sum equals the multiplication tensor.
            
            # Let's try a randomized approach with linear algebra verification.
            np.random.seed(42)
            
            # Generate random U and V with small integer/half-integer entries
            for attempt in range(10):
                if _elapsed() > TIME_LIMIT - 5:
                    break
                
                # Generate random U and V with entries in {-1, 0, 1} or {-0.5, 0, 0.5, 1}
                U_rand = np.random.randint(-1, 2, size=(n * m, r))
                V_rand = np.random.randint(-1, 2, size=(m * p, r))
                
                # For each (i,j,k), we need:
                # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                # This is a linear system in W.
                
                # Build the linear system: A * W_vec = b
                # where W_vec is the flattened W matrix (p*n*r entries)
                # But W appears in a structured way.
                
                # Actually, for fixed U and V, the constraint for each (i,j,k) is:
                # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                # This couples W entries across different c for the same (k,i).
                
                # Let's group by (k,i): For each (k,i), we have:
                # sum_j [sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]] = ...
                # No, that's not right. The constraint is for each (i,j,k) separately.
                
                # The unknowns are W[k*n+i, c] for all k,i,c.
                # The constraints are: for each (i,j,k),
                # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                
                # This is a linear system. Let's set it up.
                # Number of unknowns: p * n * r
                # Number of constraints: n * m * p
                
                num_unknowns = p * n * r
                num_constraints = n * m * p
                
                if num_constraints > num_unknowns:
                    continue  # Overdetermined, likely no solution
                
                # Build A matrix and b vector
                A = np.zeros((num_constraints, num_unknowns))
                b = np.ones(num_constraints)
                
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for c in range(r):
                                coeff = U_rand[i*m+j, c] * V_rand[j*p+k, c]
                                if coeff != 0:
                                    # W[k*n+i, c] is at position (k*n+i)*r + c
                                    A[idx, (k*n+i)*r + c] = coeff
                            idx += 1
                
                # Solve the linear system
                try:
                    # Use least squares if overdetermined, or solve directly
                    W_sol, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=None)
                    
                    # Check if the solution is valid
                    if np.allclose(A @ W_sol, b, atol=1e-10):
                        # Reshape W_sol to W matrix
                        W_mat = W_sol.reshape(p * n, r)
                        
                        # Check if entries are integers or half-integers
                        # Round to nearest half-integer and verify
                        W_rounded = np.round(W_mat * 2) / 2
                        
                        # Verify with rounded values
                        U_list = U_rand.tolist()
                        V_list = V_rand.tolist()
                        W_list = W_rounded.tolist()
                        
                        if _verify(U_list, V_list, W_list, n, m, p):
                            return U_list, V_list, W_list, r
                except Exception:
                    pass
    
    return None, None, None, max_rank

def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook decomposition
    result = _schoolbook(n, m, p)
    U, V, W = result["U"], result["V"], result["W"]
    schoolbook_rank = n * m * p
    
    # Try algebraic attack to find lower rank
    if _elapsed() < TIME_LIMIT - 20:
        U_alt, V_alt, W_alt, rank_alt = _try_algebraic_attack(n, m, p, max_rank=schoolbook_rank)
        if U_alt is not None and rank_alt < schoolbook_rank:
            U, V, W = U_alt, V_alt, W_alt
    
    return {"U": U, "V": V, "W": W}
