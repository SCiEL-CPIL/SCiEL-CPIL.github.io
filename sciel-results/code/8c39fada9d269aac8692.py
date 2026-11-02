"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds

def _generate_candidates(weights):
    """Generate all integer vectors with the specified weights."""
    candidates = []
    for w in weights:
        if w > D:
            continue
        for indices in itertools.combinations(range(D), w):
            for signs in itertools.product([1, -1], repeat=w):
                v = [0] * D
                for idx, sign in zip(indices, signs):
                    v[idx] = sign
                candidates.append(v)
    return np.array(candidates, dtype=np.int64)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin
    
    # Generate candidates with weight 3 and 4
    # Weight 3: N=3. Condition with other weight 3: dot <= 1.
    # Weight 4: N=4. Condition with other weight 4: dot <= 2.
    # Mixed 3,4: Condition is dist^2 >= max(N1, N2).
    
    candidates = _generate_candidates([3, 4])
    n_cand = len(candidates)
    
    # Precompute norms squared
    norms_sq = np.sum(candidates * candidates, axis=1)
    
    # Precompute pairwise dot products and check compatibility
    # dist^2 = n_i + n_j - 2*dot_ij
    # Condition: dist^2 >= max(n_i, n_j)
    # n_i + n_j - 2*dot_ij >= max(n_i, n_j)
    # If n_i == n_j: 2*n_i - 2*dot_ij >= n_i => n_i >= 2*dot_ij => dot_ij <= n_i/2
    # If n_i != n_j: e.g., n_i > n_j. n_i + n_j - 2*dot_ij >= n_i => n_j >= 2*dot_ij => dot_ij <= n_j/2
    
    # Vectorized check for all pairs
    # A[i, j] = dot(v_i, v_j)
    A = candidates @ candidates.T
    N = norms_sq[:, None]
    N_T = norms_sq[None, :]
    
    # Conflict matrix: C[i, j] = 1 if pair (i, j) is INVALID (i.e., they cannot both be selected)
    # We want to select a set where NO pair is invalid.
    # So constraint for ILP: if C[i, j] == 1, then x_i + x_j <= 1.
    
    # Calculate max_norm for each pair
    max_norms = np.maximum(N, N_T)
    dist_sq = N + N_T - 2 * A
    
    # Invalid if dist_sq < max_norms
    invalid_mask = dist_sq < max_norms
    
    # We don't need to consider self-pairs (i==j) as x_i + x_i <= 1 is just x_i <= 0.5 which implies x_i=0 for binary? 
    # No, standard MIS formulation uses x_i + x_j <= 1 for i != j.
    # For i == j, the constraint is trivial or handled by x_i <= 1.
    # Let's zero out the diagonal to avoid self-conflicts in the constraint generation logic if we iterate.
    np.fill_diagonal(invalid_mask, False)
    
    # Build ILP constraints
    # Maximize sum(x_i)
    # c = -1 (minimize -sum)
    c = np.ones(n_cand) * -1
    
    # Constraints: A_ineq @ x <= b
    # For each invalid pair (i, j), x_i + x_j <= 1
    # We can construct the sparse matrix for these constraints.
    
    # Get indices of invalid pairs (upper triangle to avoid duplicates)
    i_indices, j_indices = np.where(invalid_mask)
    
    # Number of constraints
    n_constraints = len(i_indices)
    
    if n_constraints > 0:
        # Create sparse matrix
        from scipy.sparse import coo_matrix, csr_matrix
        
        # Each row in the matrix corresponds to one constraint: x_i + x_j <= 1
        # Row k corresponds to constraint (i_indices[k], j_indices[k])
        # Non-zero entries are at (k, i_indices[k]) and (k, j_indices[k])
        
        rows = np.concatenate([np.arange(n_constraints), np.arange(n_constraints)])
        cols = np.concatenate([i_indices, j_indices])
        data = np.ones(n_constraints * 2)
        
        A_ineq = coo_matrix((data, (rows, cols)), shape=(n_constraints, n_cand)).tocsr()
        b_ineq = np.ones(n_constraints)
        
        constraints = LinearConstraint(A_ineq, 0, b_ineq)
    else:
        # No constraints, select all
        constraints = LinearConstraint(np.zeros((1, n_cand)), 0, 1)
        
    bounds = Bounds(0, 1)
    
    # Solve ILP
    # integrality: 1 for all variables (binary)
    integrality = np.ones(n_cand)
    
    try:
        # Time limit for solver
        # milp doesn't have a direct time_limit parameter in all versions, 
        # but we can check time after. 
        # However, for large n_cand, this might be slow.
        # n_cand for weight 3: C(11,3)*2^3 = 165*8 = 1320
        # n_cand for weight 4: C(11,4)*2^4 = 330*16 = 5280
        # Total ~6600 variables.
        # This is a large ILP. The solver might take a while.
        # We will run it and hope it finishes or returns a good incumbent.
        
        res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        # Check if a solution was found
        if res.success or (hasattr(res, 'x') and res.x is not None and np.any(res.x > 0.5)):
            # res.x is the solution vector
            selected = res.x > 0.5
            # Count selected
            k = np.sum(selected)
            if k > 0:
                # Return the selected vectors
                return candidates[selected].astype(np.int64)
            
    except Exception as e:
        print(f"ILP failed: {e}")
    
    # Fallback: Greedy approach if ILP fails or times out (though milp doesn't timeout easily without a signal)
    # To be safe, if we are close to deadline, we should return a greedy result.
    # But milp is synchronous. If it takes 100s, we are in trouble.
    # Let's add a simpler greedy fallback just in case the ILP is too slow or fails.
    
    if time.time() > deadline - 5:
        # Emergency greedy
        selected_indices = []
        selected_vectors = []
        # Sort by norm? Or just random?
        # Let's do a quick greedy with a fixed order
        for i in range(n_cand):
            v = candidates[i]
            n_v = norms_sq[i]
            if not selected_vectors:
                selected_indices.append(i)
                selected_vectors.append(v)
                continue
            sel_arr = np.array(selected_vectors)
            sel_norms = np.array([norms_sq[j] for j in selected_indices])
            dots = sel_arr @ v
            dist_sq_g = n_v + sel_norms - 2 * dots
            max_norms_g = np.maximum(n_v, sel_norms)
            if np.all(dist_sq_g >= max_norms_g):
                selected_indices.append(i)
                selected_vectors.append(v)
        return np.array(selected_vectors, dtype=np.int64)
        
    # If ILP returned something, we used it. 
    # If it failed, we fall through here.
    
    # Standard Greedy Fallback
    best_vectors = []
    best_count = 0
    rng = np.random.default_rng(42)
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
        perm = rng.permutation(n_cand)
        selected_indices = []
        selected_vectors = []
        
        for idx in perm:
            v = candidates[idx]
            n_v = norms_sq[idx]
            if not selected_vectors:
                selected_indices.append(idx)
                selected_vectors.append(v)
                continue
            sel_arr = np.array(selected_vectors)
            sel_norms = np.array([norms_sq[j] for j in selected_indices])
            dots = sel_arr @ v
            dist_sq_g = n_v + sel_norms - 2 * dots
            max_norms_g = np.maximum(n_v, sel_norms)
            if np.all(dist_sq_g >= max_norms_g):
                selected_indices.append(idx)
                selected_vectors.append(v)
                
        if len(selected_vectors) > best_count:
            best_count = len(selected_vectors)
            best_vectors = selected_vectors.copy()
            
    if best_vectors:
        return np.array(best_vectors, dtype=np.int64)
        
    # Final fallback: empty or single vector? Task says nonzero vectors.
    # If we found nothing, return a single valid vector.
    return candidates[:1].astype(np.int64)

# EVOLVE-BLOCK-END
