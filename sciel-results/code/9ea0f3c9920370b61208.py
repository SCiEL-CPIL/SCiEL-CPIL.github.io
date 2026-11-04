"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist


def _compute_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    return np.sum(a * b[np.ix_(p, p)])


def _swap_cost(a, b, p, i, j):
    """
    Compute the cost change of swapping positions i and j in permutation p.
    p[i] = facility at location i, p[j] = facility at location j.
    After swap: p'[i] = p[j], p'[j] = p[i].
    """
    fi = p[i]
    fj = p[j]
    
    # Cost contribution involving fi at position i and fj at position j
    # Original: sum over all k != i,j: a[fi,k]*b[i,p[k]] + a[k,fj]*b[p[k],j] 
    #           + a[fi,fj]*b[i,j] + a[fj,fi]*b[j,i]
    #           + a[fi,fi]*b[i,i] + a[fj,fj]*b[j,j]
    # After swap: fi at j, fj at i
    # New: sum over all k != i,j: a[fi,k]*b[j,p[k]] + a[k,fj]*b[p[k],i]
    #      + a[fi,fj]*b[j,i] + a[fj,fi]*b[i,j]
    #      + a[fi,fi]*b[j,j] + a[fj,fj]*b[i,i]
    
    # Let's compute delta = new_cost - old_cost
    # Terms that change are those involving fi or fj interacting with positions i or j,
    # and fi-fj interaction.
    
    # For all k not in {i,j}:
    # Old: a[fi,k]*b[i,p[k]] + a[k,fj]*b[p[k],j]
    # New: a[fi,k]*b[j,p[k]] + a[k,fj]*b[p[k],i]
    # Delta_k = a[fi,k]*(b[j,p[k]] - b[i,p[k]]) + a[k,fj]*(b[p[k],i] - b[p[k],j])
    
    # We can vectorize over all k, then subtract k=i and k=j contributions.
    
    # Let's compute for all k first
    # delta_all[k] = a[fi,k]*(b[j,p[k]] - b[i,p[k]]) + a[k,fj]*(b[p[k],i] - b[p[k],j])
    
    # But we need to be careful: when k=i or k=j, the "other" facility changes position too.
    # Actually, let's just compute the full delta by considering:
    # 1. fi moving from i to j
    # 2. fj moving from j to i
    # 3. The interaction between fi and fj changes
    
    # Contribution of fi at position i (old) vs fi at position j (new), interacting with all others:
    # Old fi at i: sum_k a[fi,k] * b[i, p[k]]
    # New fi at j: sum_k a[fi,k] * b[j, p[k]]
    # But p changes: p[i] becomes fj, p[j] becomes fi
    # So we need to be careful.
    
    # Let me just compute it directly by evaluating the affected terms.
    
    # The cost is sum_{x,y} a[x,y] * b[p[x], p[y]]
    # Terms that change are those where x or y is in {fi, fj} (since their positions change)
    
    # Actually, simpler: let's compute the cost contribution of all pairs (x,y) where 
    # x in {fi,fj} or y in {fi,fj}.
    
    # Old cost contribution:
    # sum_{y} a[fi,y]*b[i,p[y]] + sum_{y} a[fj,y]*b[j,p[y]]
    # - but we double count a[fi,fj]*b[i,j] and a[fj,fi]*b[j,i]
    # Actually no, the sum is over all x,y so:
    # sum_{y} a[fi,y]*b[i,p[y]] includes y=fj: a[fi,fj]*b[i,p[j]] = a[fi,fj]*b[i,fj]
    # sum_{y} a[fj,y]*b[j,p[y]] includes y=fi: a[fj,fi]*b[j,p[i]] = a[fj,fi]*b[j,fi]
    
    # So old contribution from fi,fj rows:
    # C_old = sum_y a[fi,y]*b[i,p[y]] + sum_y a[fj,y]*b[j,p[y]]
    
    # New: fi at j, fj at i
    # C_new = sum_y a[fi,y]*b[j,p_new[y]] + sum_y a[fj,y]*b[i,p_new[y]]
    # where p_new[i]=fj, p_new[j]=fi, p_new[k]=p[k] for k!=i,j
    
    # This is getting complex. Let me just do it with arrays.
    
    n = len(p)
    p_arr = np.asarray(p)
    
    # Build b_p = b[p_arr, p_arr]
    # Actually, let's just compute the delta directly.
    
    # delta = C_new - C_old
    # C_old = sum_{x,y} a[x,y] * b[p[x],p[y]]
    # C_new = sum_{x,y} a[x,y] * b[p_new[x],p_new[y]]
    
    # Only terms where p_new[x] != p[x] or p_new[y] != p[y] differ.
    # p_new differs from p only at positions i and j.
    # So p_new[x] != p[x] only when x in {i, j}.
    # Similarly for y.
    
    # So delta = sum over x in {i,j} or y in {i,j} of a[x,y]*(b[p_new[x],p_new[y]] - b[p[x],p[y]])
    
    # Let's enumerate:
    # (i,i): a[i,i]*(b[p_new[i],p_new[i]] - b[p[i],p[i]]) = a[i,i]*(b[fj,fj] - b[fi,fi])
    # (i,j): a[i,j]*(b[p_new[i],p_new[j]] - b[p[i],p[j]]) = a[i,j]*(b[fj,fi] - b[fi,fj])
    # (j,i): a[j,i]*(b[p_new[j],p_new[i]] - b[p[j],p[i]]) = a[j,i]*(b[fi,fj] - b[fj,fi])
    # (j,j): a[j,j]*(b[p_new[j],p_new[j]] - b[p[j],p[j]]) = a[j,j]*(b[fi,fi] - b[fj,fj])
    # (i,k) for k!=i,j: a[i,k]*(b[p_new[i],p_new[k]] - b[p[i],p[k]]) = a[i,k]*(b[fj,p[k]] - b[fi,p[k]])
    # (j,k) for k!=i,j: a[j,k]*(b[p_new[j],p_new[k]] - b[p[j],p[k]]) = a[j,k]*(b[fi,p[k]] - b[fj,p[k]])
    # (k,i) for k!=i,j: a[k,i]*(b[p_new[k],p_new[i]] - b[p[k],p[i]]) = a[k,i]*(b[p[k],fj] - b[p[k],fi])
    # (k,j) for k!=i,j: a[k,j]*(b[p_new[k],p_new[j]] - b[p[k],p[j]]) = a[k,j]*(b[p[k],fi] - b[p[k],fj])
    
    # Note: a here is the flow matrix, indexed by facilities. But p maps facility->location.
    # Wait, I'm confusing myself. Let me re-read the problem.
    
    # "Facility i is placed at location p[i]."
    # Cost: sum_{i,j} a[i,j] * b[p[i], p[j]]
    # So a is facility-flow, b is location-distance.
    # p[i] = location of facility i.
    
    # So in the cost formula, i and j are facility indices, p[i] and p[j] are locations.
    
    # A swap swaps the locations of two facilities. Let's say we swap facilities u and v.
    # So p[u] and p[v] are exchanged.
    
    # Let me redo: swap facilities u and v (not positions i,j).
    # p_new[u] = p[v], p_new[v] = p[u], p_new[x] = p[x] for x not in {u,v}.
    
    # delta = sum_{x,y} a[x,y] * (b[p_new[x],p_new[y]] - b[p[x],p[y]])
    # Only nonzero when x in {u,v} or y in {u,v}.
    
    # Let lu = p[u], lv = p[v].
    # (u,u): a[u,u]*(b[lv,lv] - b[lu,lu])
    # (u,v): a[u,v]*(b[lv,lu] - b[lu,lv])
    # (v,u): a[v,u]*(b[lu,lv] - b[lv,lu])
    # (v,v): a[v,v]*(b[lu,lu] - b[lv,lv])
    # (u,w) for w not in {u,v}: a[u,w]*(b[lv,p[w]] - b[lu,p[w]])
    # (v,w) for w not in {u,v}: a[v,w]*(b[lu,p[w]] - b[lv,p[w]])
    # (w,u) for w not in {u,v}: a[w,u]*(b[p[w],lv] - b[p[w],lu])
    # (w,v) for w not in {u,v}: a[w,v]*(b[p[w],lu] - b[p[w],lv])
    
    # This is a lot. For n=150, we can compute this in O(n) per swap evaluation.
    
    # Let's precompute for efficiency. But for now, let's just implement it.
    
    u = i  # facility index
    v = j  # facility index
    lu = p[u]
    lv = p[v]
    
    delta = 0.0
    
    # Diagonal and cross terms
    delta += a[u,u] * (b[lv,lv] - b[lu,lu])
    delta += a[u,v] * (b[lv,lu] - b[lu,lv])
    delta += a[v,u] * (b[lu,lv] - b[lv,lu])
    delta += a[v,v] * (b[lu,lu] - b[lv,lv])
    
    # Terms with u as first index, w as second
    for w in range(n):
        if w == u or w == v:
            continue
        delta += a[u,w] * (b[lv, p[w]] - b[lu, p[w]])
        delta += a[v,w] * (b[lu, p[w]] - b[lv, p[w]])
        delta += a[w,u] * (b[p[w], lv] - b[p[w], lu])
        delta += a[w,v] * (b[p[w], lu] - b[p[w], lv])
    
    return delta


def _local_search_2opt(a, b, p, time_limit):
    """
    2-opt local search: repeatedly swap pairs of facilities if it reduces cost.
    Use a steepest-descent or first-improvement strategy.
    """
    n = len(p)
    p = list(p)
    current_cost = _compute_cost(a, b, p)
    best_cost = current_cost
    best_p = list(p)
    
    start_time = time.time()
    improved = True
    
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        best_delta = 0
        best_pair = None
        
        # Try all pairs
        for i in range(n):
            if (time.time() - start_time) > time_limit:
                break
            for j in range(i+1, n):
                delta = _swap_cost(a, b, p, i, j)
                if delta < best_delta:
                    best_delta = delta
                    best_pair = (i, j)
        
        if best_pair is not None:
            i, j = best_pair
            p[i], p[j] = p[j], p[i]
            current_cost += best_delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = list(p)
            improved = True
    
    return best_p, best_cost


def solve(a, b):
    n = len(a)
    deadline = time.time() + 145.0  # 5s safety margin
    
    # Step 1: MDS + Procrustes for initial solution
    # Compute MDS coordinates for both matrices
    # For matrix A (flow), we want to embed facilities in space such that 
    # distances reflect the "importance" of flows.
    # For matrix B (distance), we want to embed locations in space.
    
    # Use classical MDS: eigen-decomposition of -1/2 * L * D * L where L is centering matrix
    # But for large matrices, this is expensive. Let's use a simpler approach.
    
    # Alternative: use the row sums or some other heuristic.
    # Actually, let's use a proper MDS implementation.
    
    def _mds_embedding(M, dim=2):
        """Compute MDS embedding of matrix M (interpreted as distances)."""
        m = len(M)
        # Center the matrix
        J = np.eye(m) - np.ones((m, m)) / m
        B = -0.5 * J @ (M ** 2) @ J
        # Eigen-decomposition
        eigvals, eigvecs = np.linalg.eigh(B)
        # Take largest dim eigenvectors
        idx = np.argsort(-eigvals)[:dim]
        coords = eigvecs[:, idx] * np.sqrt(np.maximum(eigvals[idx], 0))
        return coords
    
    # For the flow matrix A, we can interpret it as a "similarity" matrix.
    # High flow means facilities should be close. So distance = max_flow - flow.
    max_a = np.max(a)
    A_dist = max_a - a
    
    # For the distance matrix B, it's already a distance matrix.
    B_dist = b.copy()
    
    # Compute MDS embeddings
    A_coords = _mds_embedding(A_dist, dim=3)
    B_coords = _mds_embedding(B_dist, dim=3)
    
    # Procrustes: find optimal rotation/translation to align A_coords to B_coords
    # Minimize ||A_coords @ R + t - B_coords||_F
    
    # Center both
    A_centered = A_coords - A_coords.mean(axis=0)
    B_centered = B_coords - B_coords.mean(axis=0)
    
    # SVD for Procrustes
    U, S, Vt = np.linalg.svd(B_centered.T @ A_centered)
    R = Vt.T @ U.T
    
    # Optimal alignment
    A_aligned = A_centered @ R + B_centered.mean(axis=0)
    
    # Match facilities to locations by nearest neighbor
    from scipy.spatial.distance import cdist as _cdist
    dists = _cdist(A_aligned, B_centered + B_coords.mean(axis=0))
    # Greedy matching
    p = [0] * n
    used = set()
    for i in range(n):
        # Find closest unused location
        best_j = -1
        best_d = np.inf
        for j in range(n):
            if j not in used and dists[i, j] < best_d:
                best_d = dists[i, j]
                best_j = j
        p[i] = best_j
        used.add(best_j)
    
    # Step 2: Local search with 2-opt swaps
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        p, cost = _local_search_2opt(a, b, p, remaining_time - 5)
    
    # Step 3: Final polish with more careful local search if time permits
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        # Try another round of local search
        p, cost = _local_search_2opt(a, b, p, remaining_time - 5)
    
    return p


# EVOLVE-BLOCK-END
