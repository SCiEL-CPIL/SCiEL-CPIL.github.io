"""Matrix multiplication 3x5x7 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Trivial rank: n*m*p = 105
    trivial_rank = n * m * p
    
    # Build the trivial decomposition
    U = np.zeros((n * m, trivial_rank), dtype=float)
    V = np.zeros((m * p, trivial_rank), dtype=float)
    W = np.zeros((p * n, trivial_rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Precompute valid triples (a, b, c) where a=i*m+j, b=j*p+k, c=k*n+i
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                triples.append((i * m + j, j * p + k, k * n + i))
    
    # Index triples by first, second, third component for fast lookup
    triples_by_a = {}
    triples_by_b = {}
    triples_by_c = {}
    for (a, b, c) in triples:
        if a not in triples_by_a:
            triples_by_a[a] = []
        triples_by_a[a].append((b, c))
        if b not in triples_by_b:
            triples_by_b[b] = []
        triples_by_b[b].append((a, c))
        if c not in triples_by_c:
            triples_by_c[c] = []
        triples_by_c[c].append((a, b))
    
    def verify(U_mat, V_mat, W_mat, tol=1e-8):
        """Check if the decomposition correctly represents the multiplication tensor."""
        rank = U_mat.shape[1]
        # Check non-zero entries
        for (a, b, c) in triples:
            val = np.sum(U_mat[a] * V_mat[b] * W_mat[c])
            if abs(val - 1.0) > tol:
                return False
        
        # Check zero entries
        # Total entries: (n*m) * (m*p) * (p*n)
        # We only have 105 non-zero entries.
        # We can check a sample or all. 11025 is small.
        nm = n * m
        mp = m * p
        pn = p * n
        # Create a set of valid triples for fast lookup
        valid_triples = set(triples)
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    if (a, b, c) not in valid_triples:
                        val = np.sum(U_mat[a] * V_mat[b] * W_mat[c])
                        if abs(val) > tol:
                            return False
        return True
    
    def time_left():
        return deadline - time.time()
    
    def try_rank(r, max_iters=300, n_seeds=5):
        """Try to find a rank-r decomposition using alternating least squares with momentum."""
        if r >= trivial_rank:
            return None
        
        best_residual = float('inf')
        best_result = None
        
        for seed in range(n_seeds):
            if time_left() < 15:
                break
            
            np.random.seed(seed * 1000 + 42)
            
            # Random initialization with better scaling
            U_mat = np.random.randn(n * m, r)
            V_mat = np.random.randn(m * p, r)
            W_mat = np.random.randn(p * n, r)
            
            # Scale to make products O(1)
            scale = 1.0 / (r ** (1/6.0))
            U_mat *= scale
            V_mat *= scale
            W_mat *= scale
            
            # Momentum terms
            U_mom = np.zeros_like(U_mat)
            V_mom = np.zeros_like(V_mat)
            W_mom = np.zeros_like(W_mat)
            beta = 0.7  # momentum coefficient
            
            for it in range(max_iters):
                if time_left() < 5:
                    break
                
                # Update U: for each row a, solve linear system
                U_new = np.zeros_like(U_mat)
                for a in range(n * m):
                    constraints = []
                    rhs = []
                    for (b, c) in triples_by_a.get(a, []):
                        row = V_mat[b] * W_mat[c]
                        constraints.append(row)
                        rhs.append(1.0)
                    
                    if len(constraints) > 0:
                        A_sys = np.array(constraints)
                        b_sys = np.array(rhs)
                        # np.linalg.lstsq handles both overdetermined and underdetermined systems
                        # returning the minimum norm solution for underdetermined cases.
                        try:
                            x, _, _, _ = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                        except np.linalg.LinAlgError:
                            # Fallback to pseudo-inverse if lstsq fails
                            x = np.linalg.pinv(A_sys) @ b_sys
                        # Momentum update
                        U_mom[a] = beta * U_mom[a] + (x - U_mat[a])
                        U_new[a] = U_mat[a] + U_mom[a]
                
                U_mat = U_new
                
                # Update V: for each row b, solve linear system
                V_new = np.zeros_like(V_mat)
                for b in range(m * p):
                    constraints = []
                    rhs = []
                    for (a, c) in triples_by_b.get(b, []):
                        row = U_mat[a] * W_mat[c]
                        constraints.append(row)
                        rhs.append(1.0)
                    
                    if len(constraints) > 0:
                        A_sys = np.array(constraints)
                        b_sys = np.array(rhs)
                        try:
                            x, _, _, _ = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                        except np.linalg.LinAlgError:
                            x = np.linalg.pinv(A_sys) @ b_sys
                        V_mom[b] = beta * V_mom[b] + (x - V_mat[b])
                        V_new[b] = V_mat[b] + V_mom[b]
                
                V_mat = V_new
                
                # Update W: for each row c, solve linear system
                W_new = np.zeros_like(W_mat)
                for c in range(p * n):
                    constraints = []
                    rhs = []
                    for (a, b) in triples_by_c.get(c, []):
                        row = U_mat[a] * V_mat[b]
                        constraints.append(row)
                        rhs.append(1.0)
                    
                    if len(constraints) > 0:
                        A_sys = np.array(constraints)
                        b_sys = np.array(rhs)
                        try:
                            x, _, _, _ = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                        except np.linalg.LinAlgError:
                            x = np.linalg.pinv(A_sys) @ b_sys
                        W_mom[c] = beta * W_mom[c] + (x - W_mat[c])
                        W_new[c] = W_mat[c] + W_mom[c]
                
                W_mat = W_new
                
                # Check convergence periodically
                if it % 10 == 9:
                    residual = 0.0
                    for (a, b, c) in triples:
                        val = np.sum(U_mat[a] * V_mat[b] * W_mat[c])
                        residual += (val - 1.0) ** 2
                    
                    if residual < best_residual:
                        best_residual = residual
                        best_result = (U_mat.copy(), V_mat.copy(), W_mat.copy())
                    
                    if residual < 1e-10:
                        # Found a valid decomposition
                        # Try to clean up to get integer/half-integer entries
                        U_clean = np.round(U_mat * 2) / 2
                        V_clean = np.round(V_mat * 2) / 2
                        W_clean = np.round(W_mat * 2) / 2
                        if verify(U_clean, V_clean, W_clean, tol=1e-6):
                            return (U_clean, V_clean, W_clean)
                        else:
                            # Try rounding to integers
                            U_int = np.round(U_mat)
                            V_int = np.round(V_mat)
                            W_int = np.round(W_mat)
                            if verify(U_int, V_int, W_int, tol=1e-6):
                                return (U_int, V_int, W_int)
                            else:
                                # If we can't snap to half-integers, this rank is not "exact" per problem statement
                                # Do not return invalid floats. Return None to signal failure for this rank.
                                return None
        
        # Return best result found
        if best_result is not None:
            U_r, V_r, W_r = best_result
            # Try to clean up
            U_clean = np.round(U_r * 2) / 2
            V_clean = np.round(V_r * 2) / 2
            W_clean = np.round(W_r * 2) / 2
            if verify(U_clean, V_clean, W_clean, tol=1e-6):
                return (U_clean, V_clean, W_clean)
            U_int = np.round(U_r)
            V_int = np.round(V_r)
            W_int = np.round(W_r)
            if verify(U_int, V_int, W_int, tol=1e-6):
                return (U_int, V_int, W_int)
            # If we cannot snap to half-integers, we cannot claim an exact decomposition
            # with the required entry types. Return None.
            return None
        
        return None
    
    # Try ranks from lower bound up
    # Lower bound: max(n*m, m*p, p*n) = max(15, 35, 21) = 35
    lower_bound = max(n * m, m * p, p * n)
    
    best_result = None
    best_rank = trivial_rank
    
    # First, try the trivial decomposition as a fallback
    U_best = U.copy()
    V_best = V.copy()
    W_best = W.copy()
    
    # Try to find lower rank decompositions
    # Start from lower bound and go up
    for r in range(lower_bound, trivial_rank):
        if time_left() < 20:
            break
        
        # Use more iterations for lower ranks (harder problems)
        max_iters = 200 if r <= 50 else 100
        n_seeds = 3 if r <= 50 else 2
        
        result = try_rank(r, max_iters=max_iters, n_seeds=n_seeds)
        if result is not None:
            U_best, V_best, W_best = result
            best_rank = r
            break  # Found the lowest rank we can achieve
    
    # Convert to lists for the required interface
    U_list = U_best.tolist()
    V_list = V_best.tolist()
    W_list = W_best.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
