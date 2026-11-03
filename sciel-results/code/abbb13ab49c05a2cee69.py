"""Search-based solver for QAPLIB wil100: Spectral Init + Vectorized SA."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _spectral_initial(a, b):
    """
    Initialize permutation using spectral embedding.
    Aligns eigenvectors of A (flow) and B (distance) to minimize cost.
    """
    n = a.shape[0]
    
    # Use symmetric parts to ensure real eigenvectors and stable ordering
    a_sym = (a + a.T) / 2.0
    b_sym = (b + b.T) / 2.0
    
    try:
        # Compute eigenvalues and eigenvectors
        # We want to match high 'centrality' or structural features.
        # For QAP, minimizing sum a_ij b_pi(pj) suggests matching 
        # eigenvectors of A and B in a way that aligns high flow with low distance (if B is distance)
        # or high flow with high distance (if B is benefit). 
        # Standard QAP minimizes flow * distance.
        # Let's align the principal components.
        
        w_a, v_a = np.linalg.eigh(a_sym)
        w_b, v_b = np.linalg.eigh(b_sym)
        
        # Sort by eigenvalue magnitude (descending) to align principal structures
        idx_a = np.argsort(np.abs(w_a))[::-1]
        idx_b = np.argsort(np.abs(w_b))[::-1]
        
        # Take the top few eigenvectors (or all)
        # Using all n eigenvectors for full alignment
        vecs_a = v_a[:, idx_a]
        vecs_b = v_b[:, idx_b]
        
        # Flip signs to ensure alignment (dot product positive)
        for k in range(n):
            if np.dot(vecs_a[:, k], vecs_b[:, k]) < 0:
                vecs_b[:, k] = -vecs_b[:, k]
                
        # Compute similarity matrix between facilities (rows of vecs_a) and locations (rows of vecs_b)
        # S[i, j] = sum_k vecs_a[i, k] * vecs_b[j, k]
        # This is effectively vecs_a @ vecs_b.T
        similarity = vecs_a @ vecs_b.T
        
        # We want to assign facility i to location j that maximizes similarity?
        # Cost is sum a_ij b_pi(j, pi(k)).
        # If A and B have similar structure, aligning eigenvectors reduces cost.
        # Let's use the Hungarian algorithm on -similarity to find the best matching.
        
        from scipy.optimize import linear_sum_assignment
        
        # Minimize cost, so we maximize similarity.
        # linear_sum_assignment minimizes the sum of the matrix entries.
        # So we pass -similarity.
        row_ind, col_ind = linear_sum_assignment(-similarity)
        
        p = np.empty(n, dtype=np.int64)
        # row_ind is facilities, col_ind is locations
        # p[facility] = location
        p[row_ind] = col_ind
        
        return p
        
    except Exception:
        # Fallback to random if spectral fails
        return np.random.permutation(n)

def _vectorized_swap_deltas(a, b, p):
    """
    Compute the delta in cost for every possible swap (i, j) where i < j.
    Returns a 2D array of deltas.
    """
    n = len(p)
    
    # We need to compute the change in cost for swapping p[i] and p[j].
    # Cost = sum_{u,v} a[u,v] b[p[u], p[v]]
    
    # Let's derive the vectorized formula for all pairs (i, j).
    # This is tricky to do in a single broadcast for O(n^3) terms without huge memory.
    # However, for n=100, an O(n^2) matrix of deltas is 10k entries.
    # The computation for each delta is O(n). Total O(n^3) = 1,000,000 ops.
    # NumPy can handle this.
    
    # Let pi = p[i], pj = p[j].
    # Delta(i, j) = 
    #   a[i,i]*(b[pj,pj]-b[pi,pi]) + a[j,j]*(b[pi,pi]-b[pj,pj])
    #   + a[i,j]*(b[pj,pi]-b[pi,pj]) + a[j,i]*(b[pi,pj]-b[pj,pi])
    #   + sum_{k != i,j} (a[i,k]-a[j,k])*(b[pj,p[k]]-b[pi,p[k]])
    #   + sum_{k != i,j} (a[k,i]-a[k,j])*(b[p[k],pj]-b[p[k],pi])
    
    # Let's precompute terms that depend on p.
    # B_pi = b[np.ix_(p, p)]
    B_pi = b[np.ix_(p, p)]
    
    # We want to compute D[i, j] for all i < j.
    # Let's use broadcasting.
    
    # Term 1: Diagonals
    # b_diag = np.diag(B_pi)
    # T1[i, j] = (a[i,i] - a[j,j]) * (b[pj, pj] - b[pi, pi])
    
    # Term 2: Off-diagonals between i and j
    # T2[i, j] = (a[i,j] - a[j,i]) * (b[pj, pi] - b[pi, pj])
    
    # Term 3 & 4: Interaction with others
    # T3[i, j] = sum_{k != i, j} (a[i,k] - a[j,k]) * (B_pi[pj_idx, p[k]] - B_pi[pi_idx, p[k]])
    # Note: B_pi[x, y] = b[p[x], p[y]]
    # So B_pi[pj, p[k]] is b[p[pj], p[p[k]]] which is b[pj_val, pk_val]
    # Wait, p is the permutation. p[i] is the location of facility i.
    # B_pi is the permuted distance matrix.
    # B_pi[u, v] = b[p[u], p[v]]
    
    # Let's re-evaluate the sum term:
    # sum_{k != i, j} (a[i,k] - a[j,k]) * (b[pj, p[k]] - b[pi, p[k]])
    # Here pj = p[j], pi = p[i].
    # Let A_diff[i, k] = a[i, k] - a[j, k] -- this depends on j, so it's 3D if we vectorize over j.
    
    # This is getting complex for a single vectorized call for ALL pairs.
    # Given n=100, a loop over i (100 iterations) and vectorized j (100) is O(n^2) Python loops.
    # 100 * 100 = 10,000 vector operations of size ~100. This is very fast in NumPy.
    
    deltas = np.zeros((n, n), dtype=np.float64)
    
    # Precompute row/col vectors of B_pi for fast access?
    # B_pi is 100x100.
    
    for i in range(n):
        pi = p[i]
        
        # Vectorize over all j != i
        # We want delta(i, j) for all j.
        
        # Let's compute the components for fixed i and all j.
        
        # 1. Diagonal terms involving i and j
        # b_pi_pi = b[pi, pi]
        # b_pj_pj = b[p[j], p[j]] -> np.diag(B_pi)[j]
        b_diag = np.diag(B_pi)
        b_pi_pi = b_diag[i]
        b_pj_pj = b_diag # array of b[p[j], p[j]] for all j
        
        # T1[i, j] = (a[i, i] - a[j, j]) * (b_pj_pj[j] - b_pi_pi)
        # a_diag = np.diag(a)
        a_diag = np.diag(a)
        a_ii = a_diag[i]
        a_jj = a_diag # array for all j
        
        t1 = (a_ii - a_jj) * (b_pj_pj - b_pi_pi)
        
        # 2. Off-diagonal terms between i and j
        # T2[i, j] = (a[i, j] - a[j, i]) * (b[p[j], p[i]] - b[p[i], p[j]])
        # b[p[j], p[i]] = B_pi[j, i]
        # b[p[i], p[j]] = B_pi[i, j]
        b_pj_pi = B_pi[:, i] # B_pi[j, i] for all j
        b_pi_pj = B_pi[i, :] # B_pi[i, j] for all j
        
        a_ij = a[i, :] # a[i, j] for all j
        a_ji = a[:, i] # a[j, i] for all j
        
        t2 = (a_ij - a_ji) * (b_pj_pi - b_pi_pj)
        
        # 3. Sum over k != i, j
        # Sum_{k != i, j} (a[i, k] - a[j, k]) * (B_pi[pj, p[k]] - B_pi[pi, p[k]])
        # Note: B_pi is indexed by facility indices.
        # B_pi[u, v] = b[p[u], p[v]]
        # So B_pi[j, k] = b[p[j], p[k]]
        # B_pi[i, k] = b[p[i], p[k]]
        
        # We need to sum over k.
        # Let M[j, k] = B_pi[j, k] - B_pi[i, k]
        # Then term is sum_{k != i, j} (a[i, k] - a[j, k]) * M[j, k]
        
        # Let's create a matrix M where M[j, k] = B_pi[j, k] - B_pi[i, k]
        # B_pi is (n, n). B_pi[i, :] is (n,).
        # M = B_pi - B_pi[i, :][np.newaxis, :]
        M = B_pi - B_pi[i, :][np.newaxis, :]
        
        # Now we need to compute S[j] = sum_{k != i, j} (a[i, k] - a[j, k]) * M[j, k]
        # Let A_diff[j, k] = a[i, k] - a[j, k]
        # A_diff = a[i, :][np.newaxis, :] - a
        A_diff = a[i, :][np.newaxis, :] - a
        
        # Product P[j, k] = A_diff[j, k] * M[j, k]
        P = A_diff * M
        
        # Sum over k, excluding k=i and k=j
        # Sum_k P[j, k]
        row_sums = P.sum(axis=1)
        
        # Subtract k=i term: P[j, i]
        p_ji = P[:, i]
        
        # Subtract k=j term: P[j, j]
        p_jj = np.diag(P)
        
        # S[j] = row_sums[j] - p_ji[j] - p_jj[j]
        S = row_sums - p_ji - p_jj
        
        # Combine terms
        deltas[i, :] = t1 + t2 + S
        
        # Symmetry? The delta for swap (i, j) is the same as (j, i).
        # But our formula for fixed i computes delta(i, j).
        # We only need i < j.
        
        # Zero out i=i
        deltas[i, i] = 0
        
    return deltas

def solve(a, b):
    n = len(a)
    deadline = 115.0  # Safety margin
    start_time = time.time()
    
    np.random.seed(42)
    
    # 1. Spectral Initialization
    p = _spectral_initial(a, b)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    # 2. Vectorized Simulated Annealing
    # We will perform many iterations.
    # In each iteration, we compute all swap deltas, pick a random swap (or best), and update.
    # Actually, computing ALL deltas is O(n^3) ~ 1e6 ops.
    # If we do this every step, it might be slow.
    # Let's estimate: 1e6 float ops is ~1ms in NumPy? Maybe 10ms.
    # 110 seconds / 10ms = 11,000 iterations.
    # This is decent.
    
    # Alternative: Compute deltas for a random subset of swaps?
    # Or just do random swaps with O(n) delta calculation in Python?
    # O(n) delta in Python is ~100 ops + overhead. ~10us?
    # 110s / 10us = 11,000,000 iterations.
    # Vectorized all-swaps is 11,000 iterations but gives global info.
    
    # Let's stick to the vectorized all-swaps approach but optimize the inner loop.
    # We can re-use the matrix M and A_diff construction?
    # No, they depend on i.
    
    # Let's try a hybrid:
    # 1. Compute all deltas once.
    # 2. Pick the best swap.
    # 3. Update p and cost.
    # 4. Repeat.
    
    # This is "Steepest Descent" with full evaluation.
    # It might get stuck in local optima quickly.
    # Let's add SA acceptance.
    
    t0 = 1000.0
    t_min = 1.0
    # Cooling schedule
    # Let's run for a fixed number of iterations based on time.
    
    # Precompute static parts?
    # a is static. b is static.
    
    # To speed up, we can avoid recomputing B_pi from scratch if we update it incrementally?
    # Updating B_pi for a swap (i, j) requires updating row i, col i, row j, col j.
    # B_pi[i, :] = b[p[i], p]
    # This is O(n) to update.
    # Then computing deltas is O(n^3).
    # The bottleneck is the O(n^3) delta calculation.
    
    # Let's implement the vectorized delta calculation efficiently.
    
    # Optimization:
    # We can compute the deltas for all i in a single loop over i, as done in _vectorized_swap_deltas.
    # This is 100 iterations of vectorized ops.
    
    while time.time() - start_time < deadline:
        # Compute all swap deltas
        deltas = _vectorized_swap_deltas(a, b, p)
        
        # Find the best swap (lowest delta)
        # Mask out i=j
        np.fill_diagonal(deltas, np.inf)
        
        min_val = np.min(deltas)
        
        # Temperature
        # Simple cooling: start high, drop to 0
        # Or fixed temperature?
        # Let's use a simple cooling schedule based on iteration count or time.
        # For now, let's just do steepest descent if min_val < 0, else random with SA.
        
        if min_val < 0:
            # Take the best step
            idx = np.unravel_index(np.argmin(deltas), deltas.shape)
            i, j = idx
            if i > j:
                i, j = j, i
            
            # Apply swap
            p[i], p[j] = p[j], p[i]
            current_cost += min_val
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                
        else:
            # No improving move. Use SA to escape.
            # Pick a random swap
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
                
            delta = deltas[i, j]
            
            # Temperature
            # Let's use a temperature that decays over time
            elapsed = time.time() - start_time
            frac = elapsed / deadline
            temp = t0 * (t_min / t0) ** frac
            
            if delta < 0 or np.random.random() < np.exp(-delta / max(temp, 1e-10)):
                p[i], p[j] = p[j], p[i]
                current_cost += delta
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
            else:
                # Rejection: we might still want to try another random swap in the same "step"?
                # No, just loop.
                pass
        
        # Check time
        if time.time() - start_time > deadline - 1:
            break

    return best_p.tolist()

# EVOLVE-BLOCK-END
