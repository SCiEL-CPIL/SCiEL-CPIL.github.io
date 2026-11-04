import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify the tensor decomposition reconstructs matrix multiplication exactly."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > tol:
                    return False
    # Check zero constraints (sample)
    import random
    random.seed(42)
    for _ in range(1000):
        a = random.randint(0, n*m-1)
        b = random.randint(0, m*p-1)
        g = random.randint(0, p*n-1)
        i = a // m; j = a % m
        jp = b // p; k = b % p
        kn = g // n; ip = g % n
        if i != ip or j != jp or k != kn:
            val = 0.0
            for r in range(rank):
                val += U[a, r] * V[b, r] * W[g, r]
            if abs(val) > tol:
                return False
    return True

def _round_to_half_integer(M):
    """Round matrix entries to nearest 0.5."""
    return np.round(M * 2) / 2

def _sparse_skeleton_solve(n, m, p, target_rank, seed=42):
    """
    Try to find a rank-`target_rank` decomposition by:
    1. Fixing a sparse pattern for U and V (each row has few nonzeros)
    2. Solving for W via linear equations
    3. Verifying the result
    
    Strategy: Each "tensor slice" (i,j,k) needs contribution from at least one
    rank index where U[i*m+j,r] * V[j*p+k,r] != 0 and W[k*n+i,r] can be set.
    
    We use a structured sparse pattern based on partitioning.
    """
    from fractions import Fraction as Fr
    import random
    
    rng = random.Random(seed)
    
    # Build the list of required (a,b,g) triples where the tensor = 1
    required = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                g = k * n + i
                required.append((a, b, g))
    
    # Strategy: assign each (i,j,k) to a subset of rank indices
    # For a given rank r, we need to partition the work
    # 
    # Simple approach: try to find a pattern where each rank index r
    # covers a specific "block" of (i,j,k) triples.
    # 
    # For 4x5x6 = 120 entries, try ranks that divide nicely.
    # 
    # Let's try a more systematic approach:
    # Each rank index r has a "type" (i_r, j_r, k_r) meaning it's
    # active for row i_r, col j_r, and output k_r.
    # But that's just schoolbook.
    #
    # Better: allow each rank to cover multiple (i,j,k) triples.
    # The constraint is: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    #
    # If we fix U and V to have entries in {0, 1, -1}, then for each (g = k*n+i),
    # we need: sum_r U[i*m+j,r]*V[j*p+k,r]*W[g,r] = 1 for all (j,k) with this (i,k)
    #
    # This is a system of linear equations in W.
    #
    # Let's try: fix U and V with a structured sparse pattern, then solve for W.
    
    # Pattern: for each rank index r, assign it to cover a specific (i, j, k) triple
    # but allow some sharing.
    
    # Try different sparsity patterns
    patterns = [
        # (nnz_per_U_row, nnz_per_V_row, strategy)
        (1, 1, 'diagonal'),
        (2, 1, 'row_pair'),
        (1, 2, 'col_pair'),
        (2, 2, 'cross'),
    ]
    
    for nnz_u, nnz_v, strat in patterns:
        if target_rank < n * m * p // (nnz_u * nnz_v):
            continue
        
        # Initialize U and V with sparse random pattern
        U = np.zeros((n * m, target_rank))
        V = np.zeros((m * p, target_rank))
        
        # For each row of U, pick nnz_u random columns
        for a in range(n * m):
            cols = rng.sample(range(target_rank), min(nnz_u, target_rank))
            for c in cols:
                U[a, c] = rng.choice([1, -1])
        
        # For each row of V, pick nnz_v random columns
        for b in range(m * p):
            cols = rng.sample(range(target_rank), min(nnz_v, target_rank))
            for c in cols:
                V[b, c] = rng.choice([1, -1])
        
        # Now solve for W
        # For each g = k*n + i, we have equations:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[g, r] = 1 for all j
        # 
        # This is: (U_i * V_k) dot W_g = 1 where U_i is the i-th row-block of U
        # and V_k is the k-th row-block of V... wait, let me be more careful.
        #
        # For fixed (i, k), the equation for each j is:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        #
        # Let u_jr = U[i*m+j, r], v_jr = V[j*p+k, r], w_r = W[k*n+i, r]
        # sum_j sum_r u_jr * v_jr * w_r = 1  ... no, it's per (i,j,k)
        #
        # Actually each (i,j,k) gives one equation:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        #
        # For fixed (i,k), we have m equations (one per j) in the variables W[k*n+i, r].
        # These are: sum_r (U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r] = 1
        #
        # So for each (i,k), we have a system: A_{ik} * w_{ik} = 1_m
        # where A_{ik}[j, r] = U[i*m+j, r] * V[j*p+k, r]
        #
        # This is a linear system in w_{ik} (length target_rank) with m equations.
        # If m < target_rank, it's underdetermined - we need more constraints.
        # The zero constraints give more equations!
        #
        # For the zero constraints: for (i', j', k') not matching (i,j,k),
        # sum_r U[i'*m+j', r] * V[j'*p+k', r] * W[k'*n+i', r] = 0
        #
        # But W[k'*n+i', r] is a different variable set.
        #
        # Actually, the full system couples all W variables. Let me think differently.
        #
        # The full system is: for each (a, b, g) triple,
        # sum_r U[a, r] * V[b, r] * W[g, r] = T[a,b,g]
        # where T is the matrix multiplication tensor.
        #
        # If we fix U and V, this becomes a linear system in W.
        # But it's not a standard linear system because of the product U*V*W.
        #
        # For fixed g, the equations involving W[g, :] are:
        # For each (a, b) such that the tensor T[a,b,g] is defined:
        # sum_r U[a, r] * V[b, r] * W[g, r] = T[a,b,g]
        #
        # This is: (U_a * V_b) dot W_g = T[a,b,g] where * is element-wise
        #
        # So for each g, we have a system: A_g * W_g = t_g
        # where A_g is a matrix with rows indexed by valid (a,b) pairs for this g
        # and columns indexed by r.
        # A_g[(a,b), r] = U[a, r] * V[b, r]
        # t_g[(a,b)] = T[a,b,g]
        #
        # The valid (a,b) for a given g=k*n+i are: a=i*m+j, b=j*p+k for j=0..m-1
        # (these give T=1), and all other (a,b) pairs give T=0.
        #
        # So for each g, we have n*m*p - m equations (most are zero constraints)
        # in target_rank unknowns.
        
        # Build and solve the system for W
        W = np.zeros((p * n, target_rank))
        
        solved = True
        for g in range(p * n):
            k = g // n
            i = g % n
            
            # Build the system A * w = t
            # Rows: all (a, b) pairs except those where T=1 for this g
            # Actually, let's just use all (a,b) pairs
            # For (a=i*m+j, b=j*p+k): T=1
            # For all other (a,b): T=0
            
            # Number of (a,b) pairs: (n*m) * (m*p)
            num_ab = n * m * m * p
            A = np.zeros((num_ab, target_rank))
            t = np.zeros(num_ab)
            
            idx = 0
            for a in range(n * m):
                for b in range(m * p):
                    # Check if this is a "one" entry
                    ai = a // m
                    aj = a % m
                    bj = b // p
                    bk = b % p
                    if ai == i and bj == j and bk == k:  # This would be T=1
                        pass  # handled below
                    # A[idx, r] = U[a, r] * V[b, r]
                    A[idx, :] = U[a, :] * V[b, :]
                    # t[idx] = 1 if (a,b,g) is a one-entry, else 0
                    if ai == i and aj == bj and bk == k:
                        t[idx] = 1.0
                    else:
                        t[idx] = 0.0
                    idx += 1
            
            # Solve A * w = t using least squares
            try:
                w, residuals, rank_val, sv = np.linalg.lstsq(A, t, rcond=None)
                W[g, :] = w
            except:
                solved = False
                break
        
        if not solved:
            continue
        
        # Verify the solution
        if _verify_decomposition(U, V, W, n, m, p):
            return U, V, W
    
    return None

def _optimize_decomposition_fast(n, m, p, initial_rank=None, max_time=100):
    """
    Fast Alternating Least Squares using pre-computed Gram matrices.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    if initial_rank is None:
        initial_rank = n * m * p
    
    # Try decreasing ranks
    for target_rank in range(initial_rank - 1, max(10, initial_rank - 50), -1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for seed in range(5):
            if time.time() > deadline:
                break
            
            np.random.seed(seed)
            
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # Pre-compute indices for fast t-vector calculation
            a_to_bg = {}
            for i in range(n):
                for j in range(m):
                    a = i * m + j
                    pairs = []
                    for k in range(p):
                        b = j * p + k
                        g = k * n + i
                        pairs.append((b, g))
                    a_to_bg[a] = pairs

            b_to_ag = {}
            for j in range(m):
                for k in range(p):
                    b = j * p + k
                    pairs = []
                    for i in range(n):
                        a = i * m + j
                        g = k * n + i
                        pairs.append((a, g))
                    b_to_ag[b] = pairs

            g_to_ab = {}
            for k in range(p):
                for i in range(n):
                    g = k * n + i
                    pairs = []
                    for j in range(m):
                        a = i * m + j
                        b = j * p + k
                        pairs.append((a, b))
                    g_to_ab[g] = pairs

            # Alternating least squares
            for iteration in range(50):
                if time.time() > deadline:
                    break
                
                # Update U
                G_V = V.T @ V
                G_W = W.T @ W
                A_mat = G_V * G_W
                A_mat += 1e-6 * np.eye(target_rank)
                
                for a in range(n * m):
                    b_vec = np.zeros(target_rank)
                    for b, g in a_to_bg[a]:
                        b_vec += V[b, :] * W[g, :]
                    
                    try:
                        U[a, :] = np.linalg.solve(A_mat, b_vec)
                    except:
                        pass

                # Update V
                G_U = U.T @ U
                G_W = W.T @ W
                A_mat = G_U * G_W
                A_mat += 1e-6 * np.eye(target_rank)
                
                for b in range(m * p):
                    b_vec = np.zeros(target_rank)
                    for a, g in b_to_ag[b]:
                        b_vec += U[a, :] * W[g, :]
                    
                    try:
                        V[b, :] = np.linalg.solve(A_mat, b_vec)
                    except:
                        pass

                # Update W
                G_U = U.T @ U
                G_V = V.T @ V
                A_mat = G_U * G_V
                A_mat += 1e-6 * np.eye(target_rank)
                
                for g in range(p * n):
                    b_vec = np.zeros(target_rank)
                    for a, b in g_to_ab[g]:
                        b_vec += U[a, :] * V[b, :]
                    
                    try:
                        W[g, :] = np.linalg.solve(A_mat, b_vec)
                    except:
                        pass
                
                # Check convergence and round
                if iteration % 10 == 0:
                    error = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                a = i*m+j; b = j*p+k; g = k*n+i
                                val = np.sum(U[a, :] * V[b, :] * W[g, :])
                                error += (val - 1.0) ** 2
                    
                    np.random.seed(42)
                    for _ in range(200):
                        a = np.random.randint(0, n*m)
                        b = np.random.randint(0, m*p)
                        g = np.random.randint(0, p*n)
                        i = a // m; j = a % m
                        jp = b // p; k = b % p
                        kn = g // n; ip = g % n
                        if i != ip or j != jp or k != kn:
                            val = np.sum(U[a, :] * V[b, :] * W[g, :])
                            error += val ** 2
                    
                    if error < 1e-6:
                        U_rounded = _round_to_half_integer(U)
                        V_rounded = _round_to_half_integer(V)
                        W_rounded = _round_to_half_integer(W)
                        
                        if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p):
                            return U_rounded, V_rounded, W_rounded
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as baseline
    best_U, best_V, best_W = _build_schoolbook(n, m, p)
    best_rank = best_U.shape[1]
    
    # Try sparse skeleton approach for various ranks
    for target_rank in range(60, best_rank, 2):
        if time.time() > deadline - 30:
            break
        result = _sparse_skeleton_solve(n, m, p, target_rank, seed=target_rank)
        if result is not None:
            U, V, W = result
            if U.shape[1] < best_rank:
                best_U, best_V, best_W = U, V, W
                best_rank = U.shape[1]
    
    # Try ALS optimization
    if time.time() < deadline - 10:
        result = _optimize_decomposition_fast(n, m, p, initial_rank=best_rank, max_time=int(deadline - time.time() - 5))
        if result is not None:
            U, V, W = result
            if U.shape[1] < best_rank:
                best_U, best_V, best_W = U, V, W
                best_rank = U.shape[1]
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
