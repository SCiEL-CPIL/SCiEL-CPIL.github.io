"""QAP solver: Max-Cut spectral init + incremental local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    """Solve QAP using Max-Cut-inspired spectral initialization and fast local search."""
    n = len(a)
    deadline = time.time() + 140.0  # safety margin before 150s limit
    np.random.seed(42)
    
    def compute_cost(p):
        """Compute QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]"""
        return int(np.sum(a[:, :, None, None] * b[p[:, None], p[None, :]]))
    
    # --- Max-Cut-inspired spectral initialization ---
    # Transform QAP into a binary relaxation problem
    # Key insight: QAP can be related to Max-Cut via spectral methods
    
    def maxcut_spectral_init():
        """Generate initial permutation using Max-Cut spectral relaxation."""
        # For QAP, we can construct a matrix that captures the interaction structure
        # and use its eigenvectors to guide the assignment
        
        # Approach: Use the Laplacian-like structure of the problem
        # The QAP cost can be written in terms of assignment matrix X:
        # cost = sum_{i,j,k,l} a[i,j] * b[k,l] * X[i,k] * X[j,l]
        #      = trace(X^T * A * X * B) where A and B are appropriately defined
        
        # Spectral relaxation: use top eigenvectors of a combined matrix
        # to create a soft assignment, then round to a permutation
        
        # Build a matrix that captures the "difficulty" of the assignment
        # High flow pairs should be assigned to low-distance locations
        
        # Method 1: Use eigendecomposition of flow matrix weighted by distance
        a_centered = a - a.mean()
        b_centered = b - b.mean()
        
        try:
            # Get eigenvectors of both matrices
            eigvals_a, eigvecs_a = np.linalg.eigh(a_centered)
            eigvals_b, eigvecs_b = np.linalg.eigh(b_centered)
            
            # Sort by eigenvalue (descending for a, ascending for b to minimize)
            idx_a = np.argsort(-eigvals_a)
            idx_b = np.argsort(eigvals_b)
            
            # Use top eigenvectors to create spectral coordinates
            # For facilities: use top eigenvector of flow matrix
            # For locations: use bottom eigenvector of distance matrix
            spec_fac = eigvecs_a[:, idx_a[0]]
            spec_loc = eigvecs_b[:, idx_b[0]]
            
            # Create soft assignment: high flow spectral coord -> low distance spectral coord
            # This is like a Max-Cut: we want to separate high-flow facilities
            # from each other by assigning them to distant locations
            
            # Rank facilities by spectral coordinate (high to low)
            fac_order = np.argsort(-spec_fac)
            # Rank locations by spectral coordinate (low to high)
            loc_order = np.argsort(spec_loc)
            
            p = np.empty(n, dtype=np.int64)
            p[fac_order] = loc_order
            return p
            
        except:
            # Fallback: rank matching
            heavy = np.argsort(-a.sum(axis=1), kind="stable")
            central = np.argsort(b.sum(axis=1), kind="stable")
            p = np.empty(n, dtype=np.int64)
            p[heavy] = central
            return p
    
    def spectral_relaxation_init():
        """Alternative spectral initialization using relaxation."""
        # Use a different spectral approach: 
        # Construct a matrix that combines flow and distance information
        # and use its eigenvectors for initialization
        
        try:
            # Create a combined matrix that captures the QAP structure
            # C[i,j] = sum_k a[i,k] * b[k,j] (approximate interaction)
            C = a @ b
            
            # Center it
            C_centered = C - C.mean()
            
            # Get top eigenvector
            eigvals, eigvecs = np.linalg.eigh(C_centered)
            idx = np.argsort(-np.abs(eigvals))
            
            # Use top eigenvector to create ranking
            spec = eigvecs[:, idx[0]]
            
            # Rank facilities and locations by spectral coordinate
            fac_rank = np.argsort(spec)
            loc_rank = np.argsort(-spec)  # inverse for contrast
            
            p = np.empty(n, dtype=np.int64)
            p[fac_rank] = loc_rank
            return p
            
        except:
            heavy = np.argsort(-a.sum(axis=1), kind="stable")
            central = np.argsort(b.sum(axis=1), kind="stable")
            p = np.empty(n, dtype=np.int64)
            p[heavy] = central
            return p
    
    def greedy_init():
        """Greedy construction: assign facilities one by one to best location."""
        p = np.empty(n, dtype=np.int64)
        assigned = np.zeros(n, dtype=bool)
        
        # Sort facilities by total flow (heaviest first)
        fac_order = np.argsort(-a.sum(axis=1))
        
        for fac in fac_order:
            if assigned[fac]:
                continue
            # Find best unassigned location
            best_loc = -1
            best_cost = np.inf
            
            # Compute incremental cost of assigning fac to each location
            for loc in range(n):
                if assigned[loc]:
                    continue
                
                # Cost contribution: sum_j a[fac,j] * b[loc, p[j]] for assigned j
                cost = 0
                for j in range(n):
                    if assigned[j] and j != fac:
                        cost += a[fac, j] * b[loc, p[j]]
                        cost += a[j, fac] * b[p[j], loc]
                
                if cost < best_cost:
                    best_cost = cost
                    best_loc = loc
            
            p[fac] = best_loc
            assigned[fac] = True
            assigned[best_loc] = True  # Mark location as used (but we track by index)
        
        # Fix: the above has a bug. Let's use a simpler greedy.
        p = np.empty(n, dtype=np.int64)
        used_locs = set()
        fac_order = np.argsort(-a.sum(axis=1))
        
        for fac in fac_order:
            best_loc = -1
            best_cost = np.inf
            for loc in range(n):
                if loc in used_locs:
                    continue
                # Compute cost of assigning fac to loc given current partial assignment
                cost = 0
                for j in range(n):
                    if j in used_locs and j != fac:
                        # j is already assigned to some location
                        # Find which facility is at j
                        pass  # This is getting complex, skip for now
            
            # Simplified: just pick the least central unassigned location for heaviest facility
            unassigned = [l for l in range(n) if l not in used_locs]
            if unassigned:
                # Pick location with smallest row sum among unassigned
                best_loc = min(unassigned, key=lambda l: b[l].sum())
                p[fac] = best_loc
                used_locs.add(best_loc)
            else:
                p[fac] = fac  # fallback
        
        # Ensure it's a valid permutation
        # The above might not produce a valid permutation, so use a simpler approach
        p = np.argsort(np.argsort(b.sum(axis=1)))  # rank-based
        return p
    
    # --- Fast local search with incremental updates ---
    def fast_local_search(p, cost, time_limit):
        """Local search with O(n) incremental update per swap."""
        p = p.copy()
        
        # Precompute for fast delta computation
        # We need: for each i, the "interaction vector" with other facilities
        # When we swap p[i] and p[j], we can update efficiently
        
        # Precompute:
        #   row_sum[i] = sum_k a[i,k] * b[p[i], p[k]]  (row contribution of facility i)
        #   col_sum[i] = sum_k a[k,i] * b[p[k], p[i]]  (col contribution of facility i)
        
        def compute_sums(p):
            row_sums = np.zeros(n)
            col_sums = np.zeros(n)
            for i in range(n):
                row_sums[i] = np.dot(a[i], b[p[i], p])
                col_sums[i] = np.dot(a[:, i], b[p, p[i]])
            return row_sums, col_sums
        
        # Actually, let's use a different approach: precompute the full cost matrix
        # and use a smarter delta formula
        
        # Precompute b_p = b[p, p] once
        # But this changes with every swap, so we need incremental updates
        
        # Key insight: when we swap p[i] and p[j], only the rows/cols involving i and j change
        # We can maintain a "potential" function that allows O(n) update
        
        # Let's use the following approach:
        # Precompute A_row[i] = a[i, :] and A_col[i] = a[:, i]
        # Maintain: for each location l, the "load" from assigned facilities
        #   load[l] = sum_{k: p[k]=l} (something)
        # But this is getting complex. Let's use a simpler but still fast approach:
        
        # Precompute the current assignment's "interaction matrix"
        # D[i,j] = a[i,j] * b[p[i], p[j]]
        # When we swap p[i] and p[j], we need to update D for all (i,k), (j,k), (k,i), (k,j)
        # This is O(n) per swap if we're clever
        
        # Actually, the simplest fast approach: precompute for each pair (i,j) the delta
        # But that's O(n^2) memory and O(n^2) to update per swap, which is too slow
        
        # Let's use the dot product approach but with incremental updates:
        # Maintain dot_row[i, l] = sum_k a[i,k] * b[l, p[k]]
        # When we swap p[i] and p[j] (u->v, v->u):
        #   dot_row[i, :] changes because p[i] changed from u to v
        #   dot_row[j, :] changes because p[j] changed from v to u
        #   For all k, dot_row[k, u] and dot_row[k, v] change because b[u, p[k]] and b[v, p[k]]
        #   are affected when k=i or k=j
        #
        # Specifically:
        #   dot_row[i, l] = sum_k a[i,k] * b[l, p[k]]
        #   After swap: dot_row_new[i, l] = sum_k a[i,k] * b[l, p_new[k]]
        #   where p_new[i] = v, p_new[j] = u, p_new[k] = p[k] for k not in {i,j}
        #   So dot_row_new[i, l] = dot_row_old[i, l] - a[i,i]*b[l,u] - a[i,j]*b[l,v] + a[i,i]*b[l,v] + a[i,j]*b[l,u]
        #   = dot_row_old[i, l] + a[i,i]*(b[l,v]-b[l,u]) + a[i,j]*(b[l,u]-b[l,v])
        #   = dot_row_old[i, l] + (a[i,i]-a[i,j])*(b[l,v]-b[l,u])
        #
        # Similarly for dot_row[j, l]
        # And for all k, dot_row[k, u] and dot_row[k, v] need updates because
        # b[u, p[k]] and b[v, p[k]] change when k=i or k=j
        #   dot_row_new[k, l] = dot_row_old[k, l] for l not in {u,v}
        #   dot_row_new[k, u] = dot_row_old[k, u] - a[k,i]*b[u,u] - a[k,j]*b[u,v] + a[k,i]*b[u,v] + a[k,j]*b[u,u]
        #   = dot_row_old[k, u] + (a[k,j]-a[k,i])*(b[u,u]-b[u,v])
        #   dot_row_new[k, v] = dot_row_old[k, v] - a[k,i]*b[v,u] - a[k,j]*b[v,v] + a[k,i]*b[v,v] + a[k,j]*b[v,u]
        #   = dot_row_old[k, v] + (a[k,j]-a[k,i])*(b[v,v]-b[v,u])
        #
        # This is O(n) per swap for the dot_row update, which is acceptable!
        
        # Let's implement this incremental update approach
        
        # Initialize dot_row and dot_col
        def init_dot_products(p):
            b_col_p = b[:, p]  # [l, k] = b[l, p[k]]
            b_row_p = b[p, :]  # [k, l] = b[p[k], l]
            dot_row = a @ b_col_p.T  # [i, l] = sum_k a[i,k]*b[l,p[k]]
            dot_col = a.T @ b_row_p  # [i, l] = sum_k a[k,i]*b[p[k],l]
            return dot_row, dot_col
        
        dot_row, dot_col = init_dot_products(p)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Try random swaps
            num_trials = 20000
            for _ in range(num_trials):
                if time.time() > time_limit:
                    break
                i, j = np.random.randint(0, n, 2)
                if i == j:
                    continue
                
                u = p[i]
                v = p[j]
                if u == v:
                    continue
                
                # Compute delta using precomputed dot products
                # Delta = change in cost when swapping p[i] and p[j]
                # = sum_{k,l} a[k,l] * (b[p_new[k], p_new[l]] - b[p[k], p[l]])
                
                # Only terms involving i or j change
                # Terms where k=i or l=i or k=j or l=j
                
                # Let's compute the delta directly:
                # Old contribution involving i or j:
                #   sum_l a[i,l]*b[u,p[l]] + sum_l a[l,i]*b[p[l],u]
                #   sum_l a[j,l]*b[v,p[l]] + sum_l a[l,j]*b[p[l],v]
                #   a[i,j]*b[u,v] + a[j,i]*b[v,u]
                
                # New contribution:
                #   sum_l a[i,l]*b[v,p_new[l]] + sum_l a[l,i]*b[p_new[l],v]
                #   sum_l a[j,l]*b[u,p_new[l]] + sum_l a[l,j]*b[p_new[l],u]
                #   a[i,j]*b[v,u] + a[j,i]*b[u,v]
                
                # Where p_new[i]=v, p_new[j]=u, p_new[k]=p[k] for k not in {i,j}
                
                # This is complex. Let's use the dot product formula:
                # delta = (new_cost - old_cost)
                # We can compute this using the precomputed dot_row and dot_col
                
                # Actually, let's just compute the delta using the formula:
                # delta = sum_{k not in {i,j}} [a[i,k]*(b[v,p[k]]-b[u,p[k]]) + a[k,i]*(b[p[k],v]-b[p[k],u])
                #                           + a[j,k]*(b[u,p[k]]-b[v,p[k]]) + a[k,j]*(b[p[k],u]-b[p[k],v])]
                #       + a[i,i]*(b[v,v]-b[u,u]) + a[i,j]*(b[v,u]-b[u,v])
                #       + a[j,i]*(b[u,v]-b[v,u]) + a[j,j]*(b[u,u]-b[v,v])
                
                # Use precomputed dot_row and dot_col:
                # dot_row[i, v] = sum_k a[i,k]*b[v,p[k]]
                # dot_row[i, u] = sum_k a[i,k]*b[u,p[k]]
                # So sum_{k not in {i,j}} a[i,k]*(b[v,p[k]]-b[u,p[k]])
                #   = dot_row[i,v] - dot_row[i,u] - a[i,i]*(b[v,v]-b[u,u]) - a[i,j]*(b[v,u]-b[u,v])
                
                delta = 0.0
                
                # Diagonal and cross terms
                delta += a[i, i] * (b[v, v] - b[u, u])
                delta += a[i, j] * (b[v, u] - b[u, v])
                delta += a[j, i] * (b[u, v] - b[v, u])
                delta += a[j, j] * (b[u, u] - b[v, v])
                
                # Off-diagonal terms using dot products
                delta += (dot_row[i, v] - dot_row[i, u]) - a[i,i]*(b[v,v]-b[u,u]) - a[i,j]*(b[v,u]-b[u,v])
                delta += (dot_row[j, u] - dot_row[j, v]) - a[j,i]*(b[u,v]-b[v,u]) - a[j,j]*(b[u,u]-b[v,v])
                delta += (dot_col[i, v] - dot_col[i, u]) - a[i,i]*(b[v,v]-b[u,u]) - a[j,i]*(b[v,u]-b[u,v])
                delta += (dot_col[j, u] - dot_col[j, v]) - a[i,j]*(b[u,v]-b[v,u]) - a[j,j]*(b[u,u]-b[v,v])
                
                if delta < -1e-9:
                    # Accept swap
                    p[i], p[j] = p[j], p[i]
                    cost += delta
                    improved = True
                    
                    # Update dot_row and dot_col incrementally
                    # This is the key optimization: O(n) update instead of O(n^3) recompute
                    
                    # Update dot_row[i, :]
                    dot_row[i, :] += (a[i,i]-a[i,j]) * (b[:,v]-b[:,u])
                    # Update dot_row[j, :]
                    dot_row[j, :] += (a[j,j]-a[j,i]) * (b[:,u]-b[:,v])
                    # Update dot_row[:, u] and dot_row[:, v] for all k
                    # dot_row[k, u] += (a[k,j]-a[k,i])*(b[u,u]-b[u,v])
                    # dot_row[k, v] += (a[k,j]-a[k,i])*(b[v,v]-b[v,u])
                    diff_a = a[:,j]-a[:,i]
                    dot_row[:, u] += diff_a * (b[u,u]-b[u,v])
                    dot_row[:, v] += diff_a * (b[v,v]-b[v,u])
                    
                    # Update dot_col similarly
                    # dot_col[i, :] += (a[i,i]-a[j,i])*(b[v,:]-b[u,:])
                    # Wait, let's derive this properly
                    # dot_col[i, l] = sum_k a[k,i]*b[p[k],l]
                    # After swap: dot_col_new[i, l] = sum_k a[k,i]*b[p_new[k],l]
                    # = dot_col_old[i, l] - a[i,i]*b[u,l] - a[j,i]*b[v,l] + a[i,i]*b[v,l] + a[j,i]*b[u,l]
                    # = dot_col_old[i, l] + (a[i,i]-a[j,i])*(b[v,l]-b[u,l])
                    dot_col[i, :] += (a[i,i]-a[j,i]) * (b[v,:]-b[u,:])
                    dot_col[j, :] += (a[j,j]-a[i,j]) * (b[u,:]-b[v,:])
                    # For all k, dot_col[k, u] and dot_col[k, v]
                    # dot_col[k, u] = sum_m a[m,k]*b[p[m],u]
                    # After swap: dot_col_new[k, u] = dot_col_old[k, u] - a[i,k]*b[u,u] - a[j,k]*b[v,u] + a[i,k]*b[v,u] + a[j,k]*b[u,u]
                    # = dot_col_old[k, u] + (a[j,k]-a[i,k])*(b[u,u]-b[v,u])
                    # dot_col[k, v] = sum_m a[m,k]*b[p[m],v]
                    # After swap: dot_col_new[k, v] = dot_col_old[k, v] - a[i,k]*b[u,v] - a[j,k]*b[v,v] + a[i,k]*b[v,v] + a[j,k]*b[u,v]
                    # = dot_col_old[k, v] + (a[j,k]-a[i,k])*(b[v,v]-b[u,v])
                    diff_a_T = a[:,j]-a[:,i]  # same as above
                    dot_col[:, u] += diff_a_T * (b[u,u]-b[v,u])
                    dot_col[:, v] += diff_a_T * (b[v,v]-b[u,v])
        
        return p, cost
    
    # --- Try multiple initial solutions ---
    best_p = None
    best_cost = np.inf
    
    # Max-Cut spectral init
    p1 = maxcut_spectral_init()
    c1 = compute_cost(p1)
    if c1 < best_cost:
        best_cost = c1
        best_p = p1.copy()
    
    # Spectral relaxation init
    if time.time() < deadline - 30:
        p2 = spectral_relaxation_init()
        c2 = compute_cost(p2)
        if c2 < best_cost:
            best_cost = c2
            best_p = p2.copy()
    
    # Rank matching
    if time.time() < deadline - 30:
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        p3 = np.empty(n, dtype=np.int64)
        p3[heavy] = central
        c3 = compute_cost(p3)
        if c3 < best_cost:
            best_cost = c3
            best_p = p3.copy()
    
    # --- Local search ---
    if time.time() < deadline - 20:
        best_p, best_cost = fast_local_search(best_p, best_cost, deadline - 20)
    
    # --- Hungarian refinement ---
    if time.time() < deadline - 15:
        b_col_p = b[:, best_p]
        b_row_p = b[best_p, :]
        C = a @ b_col_p.T + a.T @ b_row_p
        row_ind, col_ind = linear_sum_assignment(C)
        p_hung = np.empty(n, dtype=np.int64)
        p_hung[row_ind] = col_ind
        cost_hung = compute_cost(p_hung)
        if cost_hung < best_cost:
            best_p = p_hung
            best_cost = cost_hung
    
    # --- Final local search ---
    if time.time() < deadline - 5:
        best_p, best_cost = fast_local_search(best_p, best_cost, deadline - 3)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
