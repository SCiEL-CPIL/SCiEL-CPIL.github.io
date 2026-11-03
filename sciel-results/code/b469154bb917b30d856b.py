"""
Iterated Local Search (ILS) with Spectral Initialization and 3-Swap Neighborhood.

Strategy:
1. Spectral Initialization: Use leading eigenvectors of flow (A) and distance (B)
   matrices to create an initial assignment based on coordinate alignment in a
   low-dimensional latent space.
2. Local Search: Use a 3-swap neighborhood (permuting 3 facilities among 3 locations).
   This allows escaping local minima that are robust to 2-swaps.
3. Iteration: Perturb the best solution found by performing a few random 3-swaps,
   then run local search again. Keep the best solution found.
4. Time Management: Strict deadline guard to return best result before timeout.
"""

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _spectral_init(a, b, n):
    """
    Initialize permutation using spectral decomposition.
    Projects facilities and locations into the space of the top few eigenvectors
    and matches them based on Euclidean distance.
    """
    # For symmetric matrices, eigenvectors are orthogonal.
    # We take the top k eigenvectors.
    k = 16  # Number of dimensions to use
    
    try:
        # np.linalg.eigh is for symmetric/Hermitian matrices
        # We need the eigenvectors corresponding to the largest eigenvalues.
        # eigh returns eigenvalues in ascending order.
        evals_a, evecs_a = np.linalg.eigh(a)
        evals_b, evecs_b = np.linalg.eigh(b)
        
        # Take the last k eigenvectors (largest eigenvalues)
        # Note: For QAP, we typically look for structure in the largest eigenvalues
        # which correspond to the dominant modes of the matrix.
        U = evecs_a[:, -k:]
        V = evecs_b[:, -k:]
        
        # Compute coordinates for facilities (rows of A) and locations (rows of B)
        # Actually, the eigenvectors provide a basis.
        # The "position" of facility i in this space is given by the i-th row of U.
        # The "position" of location j in this space is given by the j-th row of V.
        
        # We want to assign facility i to location j such that U[i, :] is close to V[j, :]
        # This is an assignment problem. We can use the Hungarian algorithm (scipy.optimize.linear_sum_assignment)
        # or a simple greedy matching if we want to avoid scipy dependency issues, 
        # but scipy is allowed.
        
        from scipy.optimize import linear_sum_assignment
        
        # Cost matrix for assignment: squared Euclidean distance
        # dist[i, j] = || U[i] - V[j] ||^2
        # = ||U[i]||^2 + ||V[j]||^2 - 2 U[i] . V[j]
        
        # Since we just need a good initial guess, we can compute the full distance matrix.
        # n=256, k=16. 256x256 matrix.
        
        # Compute dot products
        dot = U @ V.T
        norm_u_sq = np.sum(U**2, axis=1)
        norm_v_sq = np.sum(V**2, axis=1)
        
        # dist[i, j] = norm_u_sq[i] + norm_v_sq[j] - 2*dot[i, j]
        dist = norm_u_sq[:, None] + norm_v_sq[None, :] - 2 * dot
        
        # Solve assignment
        row_ind, col_ind = linear_sum_assignment(dist)
        
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        
        return p
        
    except Exception:
        # Fallback to random if spectral fails
        return np.random.permutation(n)

def _swap3_delta(a, b, p, i, j, k):
    """
    Compute delta for a 3-swap: p[i], p[j], p[k] are permuted.
    Let x=p[i], y=p[j], z=p[k].
    We consider the 6 possible permutations of (x, y, z) assigned to (i, j, k).
    The current state is one of them (identity relative to current p).
    We calculate the cost change for the other 5 permutations.
    
    To do this efficiently, we isolate the terms involving indices i, j, k.
    Total Cost = Sum_{u,v} a[u,v] b[p[u], p[v]]
    
    Let S = {i, j, k}.
    Terms can be classified as:
    1. u, v not in S: Unchanged.
    2. u in S, v not in S: Depends on p[u].
    3. u not in S, v in S: Depends on p[v].
    4. u, v in S: Depends on p[u] and p[v].
    
    We can compute the "interaction" of the set S with the rest of the graph,
    and the internal cost of S.
    
    Let R = {0..n-1} \ S.
    
    Cost_S_to_R = Sum_{u in S, v in R} a[u,v] b[p[u], p[v]] + Sum_{u in R, v in S} a[u,v] b[p[u], p[v]]
    Since a,b symmetric, this is 2 * Sum_{u in S, v in R} a[u,v] b[p[u], p[v]] ?
    Wait, a[u,v] b[p[u], p[v]] is the term.
    The sum over u in S, v in R is half of the cross terms if we sum over all pairs.
    Actually, let's just compute the change directly.
    
    Old Cross Cost (S to R):
    C_old_cross = 0
    for u in S:
        for v in R:
            C_old_cross += a[u, v] * b[p[u], p[v]] + a[v, u] * b[p[v], p[u]]
    Since a,b symmetric: C_old_cross = 2 * sum_{u in S, v in R} a[u,v] b[p[u], p[v]]
    
    New Cross Cost (S to R) for a specific permutation perm of S:
    Let new_p be the permutation of p where p[i], p[j], p[k] are replaced by the permuted values.
    C_new_cross = 2 * sum_{u in S, v in R} a[u,v] b[new_p[u], p[v]]
    
    Old Internal Cost (S to S):
    C_old_int = sum_{u in S, v in S} a[u,v] b[p[u], p[v]]
    
    New Internal Cost (S to S):
    C_new_int = sum_{u in S, v in S} a[u,v] b[new_p[u], new_p[v]]
    
    Delta = (C_new_cross + C_new_int) - (C_old_cross + C_old_int)
    
    To make this fast (O(n) per 3-swap evaluation), we precompute:
    For each u in S, we need sum_{v in R} a[u,v] b[loc_u, p[v]].
    This is (sum_{v in all} a[u,v] b[loc_u, p[v]]) - (sum_{v in S} a[u,v] b[loc_u, p[v]])
    
    Let RowSum[u, loc] = sum_{v in all} a[u,v] b[loc, p[v]].
    This depends on p, so it changes.
    
    However, for a fixed p, we can compute the vector of costs for each facility u and each possible location l?
    No, that's O(n^2) memory and update.
    
    Alternative:
    Just compute the delta for the 3-swap by iterating over R (size n-3).
    For a specific permutation of S, say (i->y, j->z, k->x):
    Delta_cross = 2 * ( sum_{v in R} a[i,v] b[y, p[v]] + a[j,v] b[z, p[v]] + a[k,v] b[x, p[v]] )
                 - 2 * ( sum_{v in R} a[i,v] b[x, p[v]] + a[j,v] b[y, p[v]] + a[k,v] b[z, p[v]] )
    
    This can be vectorized over v in R.
    Let R_indices = np.array([idx for idx in range(n) if idx not in (i,j,k)])
    
    term_old = a[i, R_indices] * b[x, p[R_indices]] + a[j, R_indices] * b[y, p[R_indices]] + a[k, R_indices] * b[z, p[R_indices]]
    term_new = a[i, R_indices] * b[y, p[R_indices]] + a[j, R_indices] * b[z, p[R_indices]] + a[k, R_indices] * b[x, p[R_indices]]
    
    Delta_cross = 2 * (np.sum(term_new) - np.sum(term_old))
    
    Delta_int = (a[i,j] b[y,z] + a[j,i] b[z,y] + a[i,k] b[y,x] + a[k,i] b[x,y] + a[j,k] b[z,x] + a[k,j] b[x,z])
             - (a[i,j] b[x,y] + a[j,i] b[y,x] + a[i,k] b[x,z] + a[k,i] b[z,x] + a[j,k] b[y,z] + a[k,j] b[z,y])
             
    Since symmetric:
    Delta_int = 2 * ( a[i,j] b[y,z] + a[i,k] b[y,x] + a[j,k] b[z,x] ) - 2 * ( a[i,j] b[x,y] + a[i,k] b[x,z] + a[j,k] b[y,z] )
    
    Total Delta = Delta_cross + Delta_int
    """
    x, y, z = p[i], p[j], p[k]
    
    # If any location is the same, it's a 2-swap or no-op.
    if x == y or y == z or x == z:
        return 0 # Should not happen in valid perm
    
    # Create mask for R
    # For speed, we can use a boolean mask or just indices.
    # n=256, so creating a list of indices is fast enough if done in numpy.
    # But doing this 5 times per 3-swap candidate is expensive.
    
    # Optimization: Precompute the sum over all v, then subtract S terms.
    # Sum_all[u, loc] = sum_{v in 0..n-1} a[u,v] b[loc, p[v]]
    
    # We don't maintain Sum_all. We compute it on the fly for the 3 nodes i,j,k.
    # sum_{v in all} a[i,v] b[loc, p[v]]
    
    # Let's compute the "global" interaction of i, j, k with the current permutation p.
    # G_i[loc] = sum_{v} a[i,v] b[loc, p[v]]
    # G_j[loc] = sum_{v} a[j,v] b[loc, p[v]]
    # G_k[loc] = sum_{v} a[k,v] b[loc, p[v]]
    
    # This is O(n^2) to compute for 3 nodes? No, O(n) per node if we vectorize over v.
    # G_i = a[i, :] @ B_p ? No.
    # b[loc, p[v]] is an element of B.
    # G_i[loc] = sum_v a[i,v] B[loc, p[v]]
    # This is (a[i, :] * 1) ? No.
    # Let M be the matrix where M[loc, v] = B[loc, p[v]].
    # Then G_i = a[i, :] @ M.
    # Constructing M is O(n^2). We don't want to do that for every swap.
    
    # Let's stick to the R loop but vectorized.
    # R_indices = np.setdiff1d(np.arange(n), [i,j,k])
    # This allocation is costly.
    
    # Alternative: Use the fact that Delta_cross = 2 * ( New_Sum_R - Old_Sum_R )
    # Old_Sum_R = Sum_{v in R} (a[i,v] b[x, p[v]] + a[j,v] b[y, p[v]] + a[k,v] b[z, p[v]])
    # New_Sum_R = Sum_{v in R} (a[i,v] b[y, p[v]] + a[j,v] b[z, p[v]] + a[k,v] b[x, p[v]])
    
    # We can compute Sum_All and subtract Sum_S.
    # Sum_All_i_x = sum_{v in all} a[i,v] b[x, p[v]]
    # Sum_All_i_y = sum_{v in all} a[i,v] b[y, p[v]]
    # etc.
    
    # How to compute Sum_All_i_x fast?
    # It is the dot product of a[i, :] and the vector b[x, p[0:n]].
    # b[x, p[0:n]] is just a permutation of row x of B.
    
    # If we precompute nothing, each Sum_All is O(n).
    # We need 6 such sums (i with x,y; j with y,z; k with z,x for old; and new ones).
    # Actually, we need sums for:
    # i with x, y
    # j with y, z
    # k with z, x
    # That's 6 dot products of length n. O(6n).
    # Then subtract the terms for v in {i,j,k}. O(1).
    # Then compute Delta_int. O(1).
    
    # Total O(n) per 3-swap candidate.
    
    # Old terms:
    sum_i_x = np.dot(a[i, :], b[x, p])
    sum_j_y = np.dot(a[j, :], b[y, p])
    sum_k_z = np.dot(a[k, :], b[z, p])
    
    # New terms (for permutation i->y, j->z, k->x):
    sum_i_y = np.dot(a[i, :], b[y, p])
    sum_j_z = np.dot(a[j, :], b[z, p])
    sum_k_x = np.dot(a[k, :], b[x, p])
    
    # Subtract S terms from Old:
    # v=i: a[i,i] b[x, p[i]] + a[j,i] b[y, p[i]] + a[k,i] b[z, p[i]]
    # v=j: a[i,j] b[x, p[j]] + a[j,j] b[y, p[j]] + a[k,j] b[z, p[j]]
    # v=k: a[i,k] b[x, p[k]] + a[j,k] b[y, p[k]] + a[k,k] b[z, p[k]]
    
    old_sub = (a[i,i]*b[x,x] + a[j,i]*b[y,x] + a[k,i]*b[z,x] +
               a[i,j]*b[x,y] + a[j,j]*b[y,y] + a[k,j]*b[z,y] +
               a[i,k]*b[x,z] + a[j,k]*b[y,z] + a[k,k]*b[z,z])
    
    # Subtract S terms from New:
    # v=i: a[i,i] b[y, p[i]] + a[j,i] b[z, p[i]] + a[k,i] b[x, p[i]]
    # v=j: a[i,j] b[y, p[j]] + a[j,j] b[z, p[j]] + a[k,j] b[x, p[j]]
    # v=k: a[i,k] b[y, p[k]] + a[j,k] b[z, p[k]] + a[k,k] b[x, p[k]]
    
    new_sub = (a[i,i]*b[y,x] + a[j,i]*b[z,x] + a[k,i]*b[x,x] +
               a[i,j]*b[y,y] + a[j,j]*b[z,y] + a[k,j]*b[x,y] +
               a[i,k]*b[y,z] + a[j,k]*b[z,z] + a[k,k]*b[x,z])
    
    delta_cross = 2.0 * ( (sum_i_y + sum_j_z + sum_k_x - new_sub) - (sum_i_x + sum_j_y + sum_k_z - old_sub) )
    
    # Internal Delta
    # Old Int: a[i,j]b[x,y] + a[j,i]b[y,x] + a[i,k]b[x,z] + a[k,i]b[z,x] + a[j,k]b[y,z] + a[k,j]b[z,y]
    # New Int: a[i,j]b[y,z] + a[j,i]b[z,y] + a[i,k]b[y,x] + a[k,i]b[x,y] + a[j,k]b[z,x] + a[k,j]b[x,z]
    
    # Symmetric simplification:
    # Old Int = 2 * ( a[i,j]b[x,y] + a[i,k]b[x,z] + a[j,k]b[y,z] )
    # New Int = 2 * ( a[i,j]b[y,z] + a[i,k]b[y,x] + a[j,k]b[z,x] )
    
    delta_int = 2.0 * ( a[i,j]*(b[y,z] - b[x,y]) + a[i,k]*(b[y,x] - b[x,z]) + a[j,k]*(b[z,x] - b[y,z]) )
    
    return delta_cross + delta_int

def _local_search_3swap(a, b, p, time_limit=10.0, max_iterations=1000):
    """
    Local search using 3-swaps.
    Tries random 3-swaps and accepts improvements.
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    start_time = time.time()
    rng = np.random.default_rng(123)
    
    for _ in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
            
        # Pick 3 distinct indices
        i, j, k = rng.choice(n, size=3, replace=False)
        
        # Try all 5 non-identity permutations of (i, j, k)
        # Identity is (x,y,z) -> (x,y,z)
        # Permutations of locations (x,y,z) assigned to (i,j,k):
        # 1. (y, x, z) -> Swap i,j
        # 2. (z, y, x) -> Cycle i->j->k->i ? No, i gets y, j gets z, k gets x is (y,z,x)
        
        # Let's define the 6 permutations of the tuple (x, y, z) assigned to (i, j, k).
        # Current: i:x, j:y, k:z
        # Candidates:
        # 1. i:y, j:x, k:z  (Swap i,j)
        # 2. i:z, j:y, k:x  (Swap i,k)
        # 3. i:x, j:z, k:y  (Swap j,k)
        # 4. i:y, j:z, k:x  (Cycle)
        # 5. i:z, j:x, k:y  (Reverse Cycle)
        
        x, y, z = p[i], p[j], p[k]
        
        # Candidate 1: Swap i, j
        # Delta for swapping i, j is just a 2-swap. We can use the 3-swap function with k=i? No.
        # We can compute delta for specific assignments.
        
        # Let's just compute the delta for the 5 candidates using the formula derived.
        # The formula `_swap3_delta` computes delta for the specific permutation (i->y, j->z, k->x).
        # We need a generic delta calculator for any assignment of {x,y,z} to {i,j,k}.
        
        # To save time, let's just evaluate the full cost for the 5 neighbors?
        # Full cost is O(n^2) = 65k ops. 5 neighbors = 325k ops.
        # With n=256, this is very fast in NumPy.
        # 1000 iterations * 325k ops = 325M ops. ~0.5s.
        # This is much simpler and less error-prone than deriving 5 different delta formulas.
        
        best_neighbor_cost = current_cost
        best_neighbor_p = None
        
        # Generate the 5 neighbor permutations
        neighbors = [
            (y, x, z), # Swap i,j
            (z, y, x), # Swap i,k
            (x, z, y), # Swap j,k
            (y, z, x), # Cycle
            (z, x, y), # Reverse Cycle
        ]
        
        for nx, ny, nz in neighbors:
            p[i], p[j], p[k] = nx, ny, nz
            c = _qap_cost(a, b, p)
            if c < best_neighbor_cost:
                best_neighbor_cost = c
                best_neighbor_p = p.copy()
        
        # Restore p
        p[i], p[j], p[k] = x, y, z
        
        if best_neighbor_p is not None:
            p = best_neighbor_p
            current_cost = best_neighbor_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        else:
            # No improvement found in this neighborhood
            # In ILS, we might break or continue with random restarts.
            # For local search, we usually stop if no improvement in a full pass,
            # but here we are doing random sampling of the neighborhood.
            # We'll just continue.
            pass
            
    return best_p, best_cost

def _perturb(p, num_moves, rng):
    """
    Perturb the permutation by performing num_moves random 3-swaps.
    """
    n = len(p)
    p = p.copy()
    for _ in range(num_moves):
        i, j, k = rng.choice(n, size=3, replace=False)
        # Apply a random permutation of p[i], p[j], p[k]
        vals = [p[i], p[j], p[k]]
        rng.shuffle(vals)
        p[i], p[j], p[k] = vals
    return p

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 145.0  # Safety margin
    
    # 1. Spectral Initialization
    p = _spectral_init(a, b, n)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    rng = np.random.default_rng(42)
    
    # 2. Iterated Local Search
    while time.time() - start_time < time_limit:
        # Run local search on current p
        # Allocate time based on remaining budget
        remaining = time_limit - (time.time() - start_time)
        ls_time = min(5.0, remaining * 0.8)
        
        p, current_cost = _local_search_3swap(a, b, p, time_limit=ls_time)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
        
        # Check time again
        if time.time() - start_time > time_limit - 10.0:
            break
            
        # Perturb
        # Number of moves increases as we stagnate? Or keep constant.
        # Let's try a small perturbation first.
        num_perturb_moves = 3
        p = _perturb(p, num_perturb_moves, rng)
        
        # Optional: If we didn't improve, try a larger perturbation or restart
        # For simplicity, we just continue the loop.
        
    return best_p.tolist()

# EVOLVE-BLOCK-END
