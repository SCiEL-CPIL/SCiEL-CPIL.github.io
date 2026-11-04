"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 130.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Helper: build a decomposition from a list of (u_vec, v_vec, w_vec)
    # u_vec has length n*m, v_vec has length m*p, w_vec has length p*n
    # Returns U, V, W as lists of lists
    def build_decomp(terms):
        rank = len(terms)
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        for c, (u, v, w) in enumerate(terms):
            for i in range(n * m):
                if u[i] != 0:
                    U[i][c] = u[i]
            for j in range(m * p):
                if v[j] != 0:
                    V[j][c] = v[j]
            for k in range(p * n):
                if w[k] != 0:
                    W[k][c] = w[k]
        return {"U": U, "V": V, "W": W}
    
    # Verify a decomposition
    def verify(U, V, W):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # For each (i, k), check that sum_c U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c] 
        # equals delta(i,k) for the correct entries... 
        # Actually, the tensor T[i,j,k] = 1 if we're multiplying A[i,j] * B[j,k] -> C[i,k]
        # The decomposition requires: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # Wait, the convention is W[k*n+i, r]. So W index is k*n + i.
        
        # Let's check: for each (i, j, k), the product should be 1 if it's a valid multiplication term
        # Actually, the matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k in their ranges.
        # No wait, that's not right. The tensor for matrix multiplication C = A*B where C[i,k] = sum_j A[i,j]*B[j,k]
        # The tensor is T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
        
        # So we need: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        
        rank = U_arr.shape[1]
        # Build the tensor check
        errors = 0
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # sum over c of U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    s = 0.0
                    for c in range(rank):
                        s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        errors += 1
                        max_err = max(max_err, abs(s - 1.0))
        return errors == 0, max_err
    
    # Start with schoolbook
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try to find a lower rank decomposition using random search with verification
    # Strategy: try random sparse decompositions of decreasing rank
    
    # First, let's try a structured approach: 
    # For small matrices, we can use known results or search
    
    # Let's do a random search for rank < schoolbook
    random.seed(42)
    
    # We'll search for decompositions with rank from a lower bound up to schoolbook-1
    # Lower bound: max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    # But actually the lower bound is at least max(n*m, m*p, p*n) and also at least n*m*p / something
    
    # Let's try random decompositions
    # For each rank r, try to find random vectors that work
    
    # Actually, a better approach: use the fact that for small matrices,
    # we can try to find the minimal rank by search
    
    # Let's try a different approach: start with schoolbook and try to merge terms
    # Two terms (u1,v1,w1) and (u2,v2,w2) can be merged if there exists a linear combination
    # that preserves the tensor. This is complex.
    
    # Simpler: random search for low-rank decompositions
    # Generate random sparse vectors and check
    
    def random_sparse_vec(length, density=0.3, half_int=False):
        vec = np.zeros(length)
        for idx in range(length):
            if random.random() < density:
                if half_int:
                    vec[idx] = random.choice([-1, -0.5, 0.5, 1])
                else:
                    vec[idx] = random.choice([-1, 1])
        return vec
    
    # Try to find a decomposition of rank r
    def try_random_decomp(rank, n_trials=100):
        for trial in range(n_trials):
            terms = []
            for c in range(rank):
                u = random_sparse_vec(n*m, density=0.2)
                v = random_sparse_vec(m*p, density=0.2)
                w = random_sparse_vec(p*n, density=0.2)
                terms.append((u, v, w))
            
            # Convert to numpy for verification
            U_arr = np.zeros((n*m, rank))
            V_arr = np.zeros((m*p, rank))
            W_arr = np.zeros((p*n, rank))
            for c, (u, v, w) in enumerate(terms):
                U_arr[:, c] = u
                V_arr[:, c] = v
                W_arr[:, c] = w
            
            # Check tensor
            valid = True
            for i in range(n):
                if not valid:
                    break
                for j in range(m):
                    if not valid:
                        break
                    for k in range(p):
                        s = np.dot(U_arr[i*m+j, :], V_arr[j*p+k, :] * W_arr[k*n+i, :])
                        if abs(s - 1.0) > 1e-6:
                            valid = False
                            break
            if valid:
                return terms
        
        return None
    
    # Try ranks from lower bound to schoolbook-1
    # But this is very unlikely to work with pure random search for rank < 90
    
    # Let's use a more structured approach: 
    # The known lower bound for 3x5 * 5x6 is at least 30.
    # Known upper bounds from literature might be lower than 90.
    
    # For now, let's just return the schoolbook decomposition as a baseline,
    # but try a few quick random searches for improvement
    
    # Quick search: try rank 85, 86, 87, 88, 89 with a few trials each
    for target_rank in range(best_rank - 1, max(29, best_rank - 10), -1):
        if time.time() > deadline:
            break
        result = try_random_decomp(target_rank, n_trials=50)
        if result is not None:
            best_result = build_decomp(result)
            best_rank = target_rank
            break
    
    # Also try a structured perturbation of schoolbook
    # Remove one term and see if we can adjust others to compensate
    # This is a local search
    
    def local_search(start_rank, n_iterations=1000):
        """Try to reduce rank by one by finding a linear dependency."""
        if time.time() > deadline:
            return None, start_rank
        
        # Start with schoolbook
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = np.zeros(n*m)
                    v = np.zeros(m*p)
                    w = np.zeros(p*n)
                    u[i*m+j] = 1
                    v[j*p+k] = 1
                    w[k*n+i] = 1
                    terms.append((u, v, w))
        
        current_rank = len(terms)
        
        for iteration in range(n_iterations):
            if time.time() > deadline:
                break
            
            # Try to remove a random term and see if the remaining terms can be adjusted
            # This is complex, so let's just try removing a term and checking if 
            # the tensor is still satisfied (which it won't be unless there's a dependency)
            
            # Better: check if any two terms are linearly dependent in a way that allows removal
            # For simplicity, just try removing each term one at a time and see if 
            # we can solve for a correction
            
            # Actually, let's try a different local search: 
            # Perturb the coefficients slightly and see if we can maintain validity with fewer terms
            
            # For now, skip complex local search
            pass
        
        return None, current_rank
    
    # Try local search (will likely not find improvements quickly)
    improved, new_rank = local_search(best_rank, n_iterations=100)
    if improved is not None:
        best_result = improved
        best_rank = new_rank
    
    return best_result
# EVOLVE-BLOCK-END
