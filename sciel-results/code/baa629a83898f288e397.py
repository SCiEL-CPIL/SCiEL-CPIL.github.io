import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank_schoolbook = n * m * p
    
    # Build the target tensor T of shape (nm, mp, pn)
    # T[a, b, d] = 1 if a=(i,j), b=(j,k), d=(k,i) with matching indices, else 0.
    # a = i*m + j
    # b = j*p + k
    # d = k*n + i
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                d = k * n + i
                T[a, b, d] = 1.0

    def check_solution(U, V, W):
        # Compute sum_c U[:, c] * V[:, c] * W[:, c]
        # This is the tensor product sum.
        # U is (nm, r), V is (mp, r), W is (pn, r)
        # Result is (nm, mp, pn)
        # We can compute this efficiently.
        # Reshape U to (nm, 1, r), V to (1, mp, r), W to (pn, 1, 1)? No.
        # Sum over c: U[a,c] * V[b,c] * W[d,c]
        # This is the trace of U^T (V * W^T)? No.
        # Let's just do it with einsum or loops if small.
        # r is ~70, nm=12, mp=24, pn=18.
        # 12*24*18*70 operations is ~360k, very fast.
        r = U.shape[1]
        T_hat = np.zeros((nm, mp, pn), dtype=np.float64)
        for c in range(r):
            T_hat += np.einsum('a,b,d->abd', U[:, c], V[:, c], W[:, c])
        return np.linalg.norm(T - T_hat)

    def solve_linear_for_W(U, V, T):
        # Solve for W such that sum_c U[:,c] V[:,c] W[:,c] approx T
        # This is a linear system for W.
        # Vectorize T to (nm*mp, pn).
        # Vectorize U*V to (nm*mp, r).
        # We want M @ W = T_vec
        # M[a*m+b, c] = U[a,c] * V[b,c]
        # T_vec[a*m+b, d] = T[a,b,d]
        r = U.shape[1]
        M = np.zeros((nm * mp, r), dtype=np.float64)
        for a in range(nm):
            for b in range(mp):
                for c in range(r):
                    M[a * mp + b, c] = U[a, c] * V[b, c]
        T_vec = T.reshape(nm * mp, pn)
        # Least squares: W = (M^T M)^-1 M^T T_vec
        # Use np.linalg.lstsq
        W, residuals, rank, s = np.linalg.lstsq(M, T_vec, rcond=None)
        return W.reshape(pn, r)

    def solve_linear_for_U(V, W, T):
        # Symmetric
        r = V.shape[1]
        M = np.zeros((nm * pn, r), dtype=np.float64)
        for a in range(nm):
            for d in range(pn):
                for c in range(r):
                    M[a * pn + d, c] = V[0, c] * W[d, c] # Wait, V index?
                    # Equation: sum_c U[a,c] V[b,c] W[d,c] = T[a,b,d]
                    # Fix V, W. Sum over b? No, T depends on b.
                    # This is not a simple linear system for U if we sum over b.
                    # We need to match for ALL b.
                    # So for each a, d, and all b, the equation must hold.
                    # This means the vector (sum_c U[a,c] V[b,c] W[d,c])_b must equal T[a,:,d].
                    # Let's construct the system differently.
                    # For fixed a, d, we want to find u = U[:, c] such that 
                    # sum_c u_c (V[:,c] * W[d,c]) = T[a,:,d]
                    # This is a linear system for u = U[a, :] ? No, U[a,c] is a scalar.
                    # The unknowns are U[a,c] for c=0..r-1.
                    # The equations are for each b: sum_c U[a,c] V[b,c] W[d,c] = T[a,b,d].
                    # This is a system of mp equations with r unknowns.
                    # We can solve this for each (a,d) pair.
                    pass
        
        # Let's implement the specific solver for U.
        U_sol = np.zeros((nm, r), dtype=np.float64)
        for a in range(nm):
            for d in range(pn):
                # Matrix A of shape (mp, r)
                A = np.zeros((mp, r), dtype=np.float64)
                for b in range(mp):
                    for c in range(r):
                        A[b, c] = V[b, c] * W[d, c]
                # RHS vector of shape (mp,)
                rhs = T[a, :, d]
                # Solve A x = rhs for x = U[a, :]
                x, residuals, rank, s = np.linalg.lstsq(A, rhs, rcond=None)
                U_sol[a, :] = x
        return U_sol

    # Similar for V
    def solve_linear_for_V(U, W, T):
        r = U.shape[1]
        V_sol = np.zeros((mp, r), dtype=np.float64)
        for b in range(mp):
            for d in range(pn):
                A = np.zeros((nm, r), dtype=np.float64)
                for a in range(nm):
                    for c in range(r):
                        A[a, c] = U[a, c] * W[d, c]
                rhs = T[:, b, d]
                x, residuals, rank, s = np.linalg.lstsq(A, rhs, rcond=None)
                V_sol[b, :] = x
        return V_sol

    # Start with schoolbook
    def get_schoolbook():
        r = rank_schoolbook
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    U, V, W = get_schoolbook()
    best_U, best_V, best_W = U, V, W
    best_rank = rank_schoolbook
    best_err = 0.0

    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin

    # Try to reduce rank
    # We will try ranks from 71 down to 50 (or until time runs out)
    # For each rank, run ALS for a few iterations
    
    for r in range(rank_schoolbook - 1, 40, -1):
        if time.time() > deadline:
            break
            
        # Initialize randomly
        np.random.seed(42 + r) # Reproducible
        U_curr = np.random.randn(nm, r)
        V_curr = np.random.randn(mp, r)
        W_curr = np.random.randn(pn, r)
        
        # Run ALS
        for iter in range(10):
            if time.time() > deadline:
                break
            W_curr = solve_linear_for_W(U_curr, V_curr, T)
            U_curr = solve_linear_for_U(V_curr, W_curr, T)
            V_curr = solve_linear_for_V(U_curr, W_curr, T)
            
            err = check_solution(U_curr, V_curr, W_curr)
            if err < 1e-9:
                # Found a valid decomposition
                best_U, best_V, best_W = U_curr, V_curr, W_curr
                best_rank = r
                break
        
        if best_rank < r:
            break # Found a smaller rank, can stop or continue? 
            # The loop goes downwards, so if we found r, we can continue to r-1.
            # But if we found r=71, we should try 70.
            # The loop structure handles this.

    # If no better rank found, return schoolbook
    if best_rank == rank_schoolbook:
        U, V, W = get_schoolbook()
    else:
        U, V, W = best_U, best_V, best_W

    # Convert to lists of lists for the interface
    def mat_to_list(M):
        return M.tolist()

    return {"U": mat_to_list(U), "V": mat_to_list(V), "W": mat_to_list(W)}
