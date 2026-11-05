import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115 seconds safety margin
    
    best_n = 11
    # Fallback solution n=11 (from seed)
    n_fallback = 11
    colors_fallback = [min(j - i, n_fallback - j + i) - 1 for i in range(n_fallback) for j in range(i + 1, n_fallback)]
    best_colors = colors_fallback

    def build_adjacency(colors, n):
        """Build adjacency matrices and common-neighbor matrices for each color."""
        mats = [np.zeros((n, n), dtype=np.int8) for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                mats[c][i, j] = 1
                mats[c][j, i] = 1
                idx += 1
        
        # Compute common neighbor matrices M_c = A_c @ A_c
        common_mats = []
        for m in mats:
            M = m @ m
            common_mats.append(M)
        return mats, common_mats

    def compute_cost_from_common(common_mats):
        """Compute total C4 count from common-neighbor matrices."""
        total_cost = 0.0
        for M in common_mats:
            diag_M = np.diag(M)
            sum_M_sq = np.sum(M**2)
            sum_M = np.sum(M)
            sum_diag_sq = np.sum(diag_M**2)
            sum_diag = np.sum(diag_M)
            sum_off_diag_sq = (sum_M_sq - sum_diag_sq) / 2.0
            sum_off_diag = (sum_M - sum_diag) / 2.0
            cost_c = (sum_off_diag_sq - sum_off_diag) / 2.0
            total_cost += cost_c
        return total_cost

    def get_cost(colors, n):
        """Calculate the total number of C4s across all 5 color classes."""
        num_edges = n * (n - 1) // 2
        if len(colors) != num_edges:
            return float('inf')
        mats, common_mats = build_adjacency(colors, n)
        return compute_cost_from_common(common_mats)

    # Precompute edge index to (u, v) mapping
    def edge_to_uv(e_idx, n):
        """Convert edge index to (u, v) pair in lexicographic order."""
        # For vertex i, edges are (i, i+1), (i, i+2), ..., (i, n-1)
        # Cumulative: after vertex i, we've used sum_{k=0}^{i} (n-1-k) edges
        # We need to find which vertex i this edge belongs to
        # Edges starting at vertex i: n - 1 - i
        # Cumulative edges before vertex i: sum_{k=0}^{i-1} (n-1-k) = i*(n-1) - i*(i-1)//2
        # Let's precompute a lookup table for efficiency
        # Actually, let's just compute it directly
        # For a given e_idx, find i such that:
        # sum_{k=0}^{i-1} (n-1-k) <= e_idx < sum_{k=0}^{i} (n-1-k)
        # sum_{k=0}^{i-1} (n-1-k) = i*(n-1) - i*(i-1)//2
        # Let's binary search or just iterate (n is small)
        for i in range(n):
            count = n - 1 - i
            if e_idx < count:
                j = i + 1 + e_idx
                return (i, j)
            e_idx -= count
        # Should not reach here
        return (0, 1)

    # Build lookup table for edge index -> (u, v)
    def build_edge_lookup(n):
        lookup = []
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                lookup.append((i, j))
                idx += 1
        return lookup

    # Try to find solutions for larger n, starting from high to low
    for n in range(12, 27):
        if time.time() > deadline - 10:
            break
        
        num_edges = n * (n - 1) // 2
        edge_lookup = build_edge_lookup(n)
        
        # Random initialization
        colors = [random.randint(0, 4) for _ in range(num_edges)]
        
        # Build initial adjacency and common-neighbor matrices
        mats, common_mats = build_adjacency(colors, n)
        current_cost = compute_cost_from_common(common_mats)
        
        # Simple Hill Climbing with Random Restarts
        for restart in range(10):
            if time.time() > deadline - 5:
                break
                
            if current_cost == 0:
                best_n = n
                best_colors = colors[:]
                break # Found valid for this n, try next n
                
            # Hill climb with incremental updates
            stuck_count = 0
            max_stuck = 2000
            while stuck_count < max_stuck:
                if time.time() > deadline - 5:
                    break
                    
                if current_cost == 0:
                    break
                    
                # Pick random edge
                e_idx = random.randint(0, num_edges - 1)
                old_c = colors[e_idx]
                new_c = random.randint(0, 4)
                if old_c == new_c:
                    continue
                    
                u, v = edge_lookup[e_idx]
                
                # Update common-neighbor matrices incrementally
                # When we change edge (u,v) from color old_c to new_c:
                # For old_c: remove edge (u,v)
                # For new_c: add edge (u,v)
                
                # The common-neighbor matrix M_c[i][j] = number of common neighbors of i and j in color c
                # Adding edge (u,v) to color c:
                #   M_c[u][v] += 1, M_c[v][u] += 1  (no wait, this is wrong)
                # Actually, M_c = A_c @ A_c, so M_c[i][j] = sum_k A_c[i][k] * A_c[k][j]
                # When we add edge (u,v) to A_c:
                #   A_c[u][v] = 1, A_c[v][u] = 1
                #   M_c[i][j] changes for all i, j
                #   Specifically, M_c[i][j] += A_c[i][u]*A_c[u][j] + A_c[i][v]*A_c[v][j]
                #   But A_c[u][j] is 1 only if j==v, and A_c[v][j] is 1 only if j==u
                #   So M_c[i][j] += A_c[i][u]*(j==v ? 1 : 0) + A_c[i][v]*(j==u ? 1 : 0)
                #   = (j==v) * A_c[i][u] + (j==u) * A_c[i][v]
                #   So for fixed i:
                #     M_c[i][v] += A_c[i][u]
                #     M_c[i][u] += A_c[i][v]
                #   Similarly for j fixed... 
                #   Actually, let me think again.
                #   M_c[i][j] = sum_k A_c[i][k] * A_c[k][j]
                #   When we add edge (u,v), A_c[u][v] and A_c[v][u] become 1 (were 0).
                #   So M_c[i][j] increases by:
                #     A_c[i][u] * A_c[u][j] (new term from k=u)
                #     A_c[i][v] * A_c[v][j] (new term from k=v)
                #   A_c[u][j] = 1 if j == v, else 0 (since we just added (u,v))
                #   A_c[v][j] = 1 if j == u, else 0 (since we just added (v,u))
                #   So M_c[i][j] += A_c[i][u] * (j==v) + A_c[i][v] * (j==u)
                #   For j = v: M_c[i][v] += A_c[i][u]
                #   For j = u: M_c[i][u] += A_c[i][v]
                #   For other j: no change
                #   So only column v and column u of M_c change.
                #   Specifically:
                #     For all i: M_c[i][v] += A_c[i][u]
                #     For all i: M_c[i][u] += A_c[i][v]
                #   But note A_c[i][u] is 1 if there's an edge (i,u) in color c, else 0.
                #   And A_c[i][v] is 1 if there's an edge (i,v) in color c, else 0.
                #   Also, M_c is symmetric, so we should update both M_c[i][v] and M_c[v][i], etc.
                #   Wait, M_c = A_c @ A_c, and A_c is symmetric, so M_c is symmetric.
                #   M_c[i][v] = sum_k A_c[i][k] * A_c[k][v]
                #   When we add (u,v): A_c[u][v]=1, A_c[v][u]=1
                #   M_c[i][v] = sum_k A_c[i][k] * A_c[k][v]
                #   The new terms are when k=u: A_c[i][u] * A_c[u][v] = A_c[i][u] * 1 = A_c[i][u]
                #   And when k=v: A_c[i][v] * A_c[v][v] = A_c[i][v] * 0 = 0
                #   So M_c[i][v] += A_c[i][u]
                #   Similarly, M_c[v][i] = sum_k A_c[v][k] * A_c[k][i]
                #   New term when k=u: A_c[v][u] * A_c[u][i] = 1 * A_c[u][i] = A_c[i][u]
                #   So M_c[v][i] += A_c[i][u]
                #   By symmetry, M_c[i][v] = M_c[v][i], consistent.
                #   Similarly, M_c[i][u] += A_c[i][v] and M_c[u][i] += A_c[i][v]
                
                # So the update is:
                # For color old_c (removing edge (u,v)):
                #   For all i: M_old[i][v] -= A_old[i][u], M_old[v][i] -= A_old[i][u]
                #   For all i: M_old[i][u] -= A_old[i][v], M_old[u][i] -= A_old[i][v]
                # For color new_c (adding edge (u,v)):
                #   For all i: M_new[i][v] += A_new[i][u], M_new[v][i] += A_new[i][u]
                #   For all i: M_new[i][u] += A_new[i][v], M_new[u][i] += A_new[i][v]
                
                # But we need to be careful: when we remove the edge from old_c, 
                # A_old[i][u] and A_old[i][v] are the values BEFORE the removal.
                # Since we're removing (u,v), A_old[u][v] and A_old[v][u] go from 1 to 0.
                # For i != u,v: A_old[i][u] and A_old[i][v] don't change.
                # For i = u: A_old[u][u] = 0, A_old[u][v] = 1 (before removal)
                # For i = v: A_old[v][u] = 1 (before removal), A_old[v][v] = 0
                
                # Let's just update the matrices directly.
                # First, save the old adjacency values we need
                A_old = mats[old_c]
                A_new = mats[new_c]
                
                # Get the columns we need
                # For old_c: we need A_old[:, u] and A_old[:, v] (before modification)
                col_u_old = A_old[:, u].copy()
                col_v_old = A_old[:, v].copy()
                
                # For new_c: we need A_new[:, u] and A_new[:, v] (before modification)
                col_u_new = A_new[:, u].copy()
                col_v_new = A_new[:, v].copy()
                
                # Now update M matrices
                M_old = common_mats[old_c]
                M_new = common_mats[new_c]
                
                # Remove edge from old_c
                # M_old[i][v] -= col_u_old[i] for all i
                # M_old[v][i] -= col_u_old[i] for all i  (but M is symmetric, so M_old[i][v] = M_old[v][i])
                # Actually, since M is symmetric, updating M_old[:, v] and M_old[v, :] is redundant if we do both.
                # Let's just update column v and row v, and column u and row u.
                # M_old[:, v] -= col_u_old  (this updates M_old[i][v] for all i)
                # M_old[v, :] -= col_u_old  (this updates M_old[v][i] for all i)
                # But since M is symmetric, M_old[i][v] = M_old[v][i], so these are the same update.
                # We should only do one of them to avoid double-counting? No, they update different entries.
                # M_old[:, v] updates entries (i, v) for all i
                # M_old[v, :] updates entries (v, i) for all i
                # These are different entries unless i = v.
                # For i = v: M_old[v][v] is updated by both, so we'd be double-counting.
                # Let's handle it carefully.
                
                # Actually, let's just do:
                # M_old[:, v] -= col_u_old
                # M_old[v, :] -= col_u_old
                # This will update M_old[v][v] twice: once from col_u_old[v] and once from col_u_old[v].
                # col_u_old[v] = A_old[v][u] = 1 (since (u,v) is in old_c before removal)
                # So M_old[v][v] -= 2, but it should only -= 1? 
                # Let's check: M_old[v][v] = sum_k A_old[v][k] * A_old[k][v]
                # When we remove (u,v), A_old[v][u] and A_old[u][v] go from 1 to 0.
                # The term k=u: A_old[v][u] * A_old[u][v] = 1*1 = 1, this term is removed.
                # So M_old[v][v] decreases by 1.
                # But our update does M_old[v][v] -= col_u_old[v] = 1, and M_old[v][v] -= col_u_old[v] = 1 again.
                # So we're subtracting 2 instead of 1. Bug!
                
                # The issue is that when i = v in M_old[:, v] -= col_u_old, we get M_old[v][v] -= col_u_old[v]
                # And when i = v in M_old[v, :] -= col_u_old, we get M_old[v][v] -= col_u_old[v]
                # So we're double-counting the diagonal.
                
                # Fix: only update off-diagonal entries, or handle diagonal separately.
                # Let's update M_old[:, v] and M_old[v, :] but skip the diagonal.
                # Or, more simply, just update M_old[:, v] and rely on symmetry? No, M is stored explicitly.
                
                # Let's do it properly:
                # For all i != v: M_old[i][v] -= col_u_old[i]
                # For all i != u: M_old[i][u] -= col_v_old[i]
                # And by symmetry, these updates automatically handle the other side.
                # But we're storing the full matrix, so we need to update both M_old[i][v] and M_old[v][i].
                
                # Actually, the cleanest way is:
                # M_old[:, v] -= col_u_old
                # M_old[v, :] -= col_u_old
                # M_old[:, u] -= col_v_old
                # M_old[u, :] -= col_v_old
                # But this double-counts M_old[v][v], M_old[u][u], M_old[u][v], M_old[v][u].
                
                # Let me think differently. The change in M_c when adding edge (u,v) is:
                # Delta_M_c[i][j] = A_c[i][u] * (j==v) + A_c[i][v] * (j==u)
                # So Delta_M_c is a matrix where:
                #   Delta_M_c[i][v] = A_c[i][u]
                #   Delta_M_c[i][u] = A_c[i][v]
                #   Delta_M_c[i][j] = 0 for j != u, v
                # This is NOT symmetric in general! Wait, but M_c should be symmetric.
                # Let's check: Delta_M_c[i][j] = A_c[i][u] * delta_{j,v} + A_c[i][v] * delta_{j,u}
                # Delta_M_c[j][i] = A_c[j][u] * delta_{i,v} + A_c[j][v] * delta_{i,u}
                # For i != j, i != u, j != v: both are 0.
                # For i = a, j = v: Delta_M_c[a][v] = A_c[a][u], Delta_M_c[v][a] = A_c[v][u] * delta_{a,v} + A_c[v][v] * delta_{a,u} = 0 + 0 = 0 (if a != v, u)
                # That's not symmetric! So my formula is wrong.
                
                # Let me re-derive. M_c = A_c @ A_c, so M_c[i][j] = sum_k A_c[i][k] * A_c[k][j]
                # When we add edge (u,v), A_c[u][v] and A_c[v][u] increase by 1.
                # Delta_M_c[i][j] = sum_k Delta_A_c[i][k] * A_c[k][j] + A_c[i][k] * Delta_A_c[k][j]
                # Delta_A_c[i][k] is nonzero only when (i,k) = (u,v) or (v,u)
                # So Delta_A_c[u][v] = 1, Delta_A_c[v][u] = 1, else 0.
                # Delta_M_c[i][j] = Delta_A_c[i][u] * A_c[u][j] + Delta_A_c[i][v] * A_c[v][j] + A_c[i][u] * Delta_A_c[u][j] + A_c[i][v] * Delta_A_c[v][j]
                # = (i==u) * A_c[u][j] + (i==v) * A_c[v][j] + A_c[i][u] * (j==v) + A_c[i][v] * (j==u)
                # For i != u, v: Delta_M_c[i][j] = A_c[i][u] * (j==v) + A_c[i][v] * (j==u)
                # For i = u: Delta_M_c[u][j] = A_c[u][j] + A_c[u][u] * (j==v) + A_c[u][v] * (j==u) = A_c[u][j] + 0 + 1 * (j==u) = A_c[u][j] + (j==u)
                #   Since A_c[u][u] = 0, and A_c[u][v] = 1 (before addition, but we're adding it, so this is tricky)
                #   Actually, when we add (u,v), A_c[u][v] goes from 0 to 1. So in the formula, A_c[u][v] should be the OLD value, which is 0.
                #   So Delta_M_c[u][j] = A_c[u][j] + A_c[u][v] * (j==u) = A_c[u][j] + 0 * (j==u) = A_c[u][j]
                #   But wait, A_c[u][j] is the OLD value. For j = v, A_c[u][v] = 0 (old), so Delta_M_c[u][v] = 0.
                #   For j = u, A_c[u][u] = 0, so Delta_M_c[u][u] = 0.
                #   For other j, A_c[u][j] is whatever it was.
                # This is getting complicated. Let me just recompute M for the two affected colors.
                
                # Given the complexity, let's just recompute the common-neighbor matrices for old_c and new_c.
                # This is still much faster than recomputing all 5.
                
                # Modify adjacency matrices
                A_old[u, v] = 0
                A_old[v, u] = 0
                A_new[u, v] = 1
                A_new[v, u] = 1
                
                # Recompute common-neighbor matrices for old_c and new_c
                common_mats[old_c] = A_old @ A_old
                common_mats[new_c] = A_new @ A_new
                
                # Recompute cost
                new_cost = compute_cost_from_common(common_mats)
                
                if new_cost < current_cost:
                    current_cost = new_cost
                    colors[e_idx] = new_c
                    stuck_count = 0
                else:
                    # Revert
                    A_old[u, v] = 1
                    A_old[v, u] = 1
                    A_new[u, v] = 0
                    A_new[v, u] = 0
                    common_mats[old_c] = A_old @ A_old
                    common_mats[new_c] = A_new @ A_new
                    stuck_count += 1
                    
                if stuck_count > max_stuck:
                    break # Stuck, restart
                    
            if current_cost == 0:
                best_n = n
                best_colors = colors[:]
                break # Found valid for this n, try next n

    return {'n': best_n, 'colors': best_colors}
