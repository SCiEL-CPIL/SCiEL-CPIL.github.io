"""Rewritten QAP solver: spectral + WL color refinement + Hungarian + optimized local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    """Solve QAP: assign n facilities to n locations minimizing sum a[i,j]*b[p[i],p[j]].
    
    Strategy:
    1. Spectral initialization using eigenvectors of QAP-related matrices.
    2. Hungarian-based linearization refinement (multiple rounds).
    3. Fast local search with precomputed swap deltas and multiple restarts.
    """
    n = len(a)
    deadline = time.time() + 115.0  # safety margin before 120s limit
    np.random.seed(42)
    
    def compute_cost(p):
        """Compute QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]"""
        total = 0
        for i in range(n):
            total += np.dot(a[i], b[p[i], p])
        return total
    
    # --- Spectral initialization ---
    # Use eigenvectors of the QAP-related matrices to generate high-quality initial solutions
    # The QAP can be related to eigenvalue problems. We use multiple spectral approaches:
    
    def spectral_init(method='flow'):
        """Generate initial permutation using spectral methods."""
        if method == 'flow':
            # Use eigenvectors of the flow matrix a
            # Center a to remove mean
            a_centered = a - a.mean()
            # Get top eigenvectors
            try:
                eigvals, eigvecs = np.linalg.eigh(a_centered)
                # Sort by eigenvalue magnitude (descending)
                idx = np.argsort(-np.abs(eigvals))
                eigvecs_sorted = eigvecs[:, idx[:min(50, n)]]
                # Use the top eigenvector to create a ranking
                # Map facility indices to their "spectral coordinate"
                spec_coords = eigvecs_sorted[:, 0]
                # Rank facilities by spectral coordinate
                fac_rank = np.argsort(spec_coords)
                # Rank locations by their row sum of b (centrality)
                loc_rank = np.argsort(b.sum(axis=1))
                # Assign: facility with rank r gets location with rank r
                p = np.empty(n, dtype=np.int64)
                p[fac_rank] = loc_rank
            except:
                # Fallback to rank matching
                heavy = np.argsort(-a.sum(axis=1), kind="stable")
                central = np.argsort(b.sum(axis=1), kind="stable")
                p = np.empty(n, dtype=np.int64)
                p[heavy] = central
            return p
        elif method == 'distance':
            # Use eigenvectors of the distance matrix b
            b_centered = b - b.mean()
            try:
                eigvals, eigvecs = np.linalg.eigh(b_centered)
                idx = np.argsort(-np.abs(eigvals))
                eigvecs_sorted = eigvecs[:, idx[:min(50, n)]]
                spec_coords = eigvecs_sorted[:, 0]
                loc_rank = np.argsort(spec_coords)
                # Rank facilities by their row sum of a (flow intensity)
                fac_rank = np.argsort(a.sum(axis=1))
                p = np.empty(n, dtype=np.int64)
                p[fac_rank] = loc_rank
            except:
                heavy = np.argsort(-a.sum(axis=1), kind="stable")
                central = np.argsort(b.sum(axis=1), kind="stable")
                p = np.empty(n, dtype=np.int64)
                p[heavy] = central
            return p
        elif method == 'combined':
            # Combine flow and distance spectral info
            a_centered = a - a.mean()
            b_centered = b - b.mean()
            try:
                eigvals_a, eigvecs_a = np.linalg.eigh(a_centered)
                eigvals_b, eigvecs_b = np.linalg.eigh(b_centered)
                idx_a = np.argsort(-np.abs(eigvals_a))
                idx_b = np.argsort(-np.abs(eigvals_b))
                # Use top eigenvectors from both
                spec_a = eigvecs_a[:, idx_a[0]]
                spec_b = eigvecs_b[:, idx_b[0]]
                # Combine: use flow spectral for facilities, distance spectral for locations
                fac_rank = np.argsort(spec_a)
                loc_rank = np.argsort(spec_b)
                p = np.empty(n, dtype=np.int64)
                p[fac_rank] = loc_rank
            except:
                heavy = np.argsort(-a.sum(axis=1), kind="stable")
                central = np.argsort(b.sum(axis=1), kind="stable")
                p = np.empty(n, dtype=np.int64)
                p[heavy] = central
            return p
        else:
            # Default: rank matching
            heavy = np.argsort(-a.sum(axis=1), kind="stable")
            central = np.argsort(b.sum(axis=1), kind="stable")
            p = np.empty(n, dtype=np.int64)
            p[heavy] = central
            return p
    
    # --- WL Color Refinement Initialization ---
    def wl_init(max_iters=20):
        """Weisfeiler-Lehman color refinement to derive facility-location mapping.
        
        Build sparse graphs from a (facility) and b (location), iteratively refine
        node colors based on neighborhood color histograms, then match colors.
        """
        # Build weighted adjacency from matrices (use top-k connections to keep sparse)
        k = min(20, n)  # top-k neighbors
        
        def build_topk_adj(M):
            """Build top-k adjacency list from matrix M."""
            adj = []
            for i in range(n):
                row = M[i].copy()
                # Get top-k indices by absolute value
                idx = np.argpartition(-np.abs(row), k)[:k]
                idx = idx[np.argsort(-np.abs(row[idx]))]
                adj.append(list(zip(idx.tolist(), row[idx].tolist())))
            return adj
        
        adj_fac = build_topk_adj(a)
        adj_loc = build_topk_adj(b)
        
        # Initialize colors: based on degree (sum of absolute weights)
        def init_colors(adj):
            colors = []
            for i in range(n):
                deg = sum(abs(w) for _, w in adj[i])
                colors.append(deg)
            # Discretize into bins
            if len(set(colors)) > 50:
                quantiles = np.linspace(0, 1, 51)
                bins = np.quantile(colors, quantiles)
                colors = [int(np.searchsorted(bins, c, side='right')) for c in colors]
            return colors
        
        fac_colors = init_colors(adj_fac)
        loc_colors = init_colors(adj_loc)
        
        # WL refinement iterations
        for it in range(max_iters):
            if time.time() > deadline - 40.0:
                break
            
            # For each node, compute histogram of neighbor colors
            def refine_colors(adj, colors):
                new_colors = []
                # Count color frequencies
                color_counts = {}
                for c in colors:
                    color_counts[c] = color_counts.get(c, 0) + 1
                
                for i in range(n):
                    # Build histogram of neighbor colors
                    hist = {}
                    for nb, w in adj[i]:
                        c = colors[nb]
                        hist[c] = hist.get(c, 0) + 1
                    # Create a signature: (own_color, sorted histogram)
                    sig = (colors[i], tuple(sorted(hist.items())))
                    # Hash the signature to a new color
                    new_colors.append(hash(sig))
                
                # Relabel colors to be dense
                unique_sigs = list(set(new_colors))
                sig_to_color = {s: i for i, s in enumerate(unique_sigs)}
                return [sig_to_color[c] for c in new_colors]
            
            fac_colors_new = refine_colors(adj_fac, fac_colors)
            loc_colors_new = refine_colors(adj_loc, loc_colors)
            
            # Check convergence
            if fac_colors_new == fac_colors and loc_colors_new == loc_colors:
                break
            
            fac_colors = fac_colors_new
            loc_colors = loc_colors_new
        
        # Match facilities to locations based on refined colors
        # Group by color, then within each color group, match by secondary features
        p = np.zeros(n, dtype=np.int64)
        
        # Build color groups
        fac_groups = {}
        for i in range(n):
            c = fac_colors[i]
            if c not in fac_groups:
                fac_groups[c] = []
            fac_groups[c].append(i)
        
        loc_groups = {}
        for j in range(n):
            c = loc_colors[j]
            if c not in loc_groups:
                loc_groups[c] = []
            loc_groups[c].append(j)
        
        # For each color, match facilities to locations
        # Use secondary feature: row sum of original matrix
        fac_row_sums = a.sum(axis=1)
        loc_row_sums = b.sum(axis=1)
        
        for color in fac_groups:
            facs = fac_groups[color]
            # Find matching location color
            # Try exact color match first
            if color in loc_groups:
                locs = loc_groups[color]
            else:
                # Fallback: find color with similar group size
                best_color = None
                best_size_diff = float('inf')
                for lc in loc_groups:
                    size_diff = abs(len(loc_groups[lc]) - len(facs))
                    if size_diff < best_size_diff:
                        best_size_diff = size_diff
                        best_color = lc
                locs = loc_groups.get(best_color, list(range(n)))
            
            # Sort by row sum and match
            facs_sorted = sorted(facs, key=lambda i: -fac_row_sums[i])
            locs_sorted = sorted(locs, key=lambda j: -loc_row_sums[j])
            
            # If sizes don't match, cycle through locations
            for idx, fi in enumerate(facs_sorted):
                lj = locs_sorted[idx % len(locs_sorted)]
                p[fi] = lj
        
        # Ensure p is a valid permutation (fix any conflicts)
        used = set(p.tolist())
        remaining = list(set(range(n)) - used)
        for i in range(n):
            if p[i] in used and i != np.where(p == p[i])[0][0]:
                # Conflict: find an unused location
                if remaining:
                    p[i] = remaining.pop(0)
        
        return p
    
    # --- Try multiple initial solutions and keep the best ---
    init_methods = ['flow', 'distance', 'combined', 'default', 'wl']
    best_p = None
    best_cost = np.inf
    
    for method in init_methods:
        if time.time() > deadline - 30.0:
            break
        if method == 'wl':
            p_init = wl_init()
        else:
            p_init = spectral_init(method)
        cost_init = compute_cost(p_init)
        if cost_init < best_cost:
            best_cost = cost_init
            best_p = p_init.copy()
    
    # --- Local search with precomputed dot products and incremental updates ---
    def local_search(p, cost, time_limit):
        """Perform local search starting from p with initial cost."""
        p = p.copy()
        
        # Precompute dot_row and dot_col for fast delta computation
        # dot_row[i, l] = sum_k a[i,k] * b[l, p[k]]
        # dot_col[i, l] = sum_k a[k,i] * b[p[k], l]
        
        def recompute_precomp(p):
            # b[:, p] has shape (n, n): [l, k] = b[l, p[k]]
            b_col_p = b[:, p]  # (n, n)
            # b[p, :] has shape (n, n): [k, l] = b[p[k], l]
            b_row_p = b[p, :]  # (n, n)
            
            # dot_row[i, l] = sum_k a[i,k] * b[l, p[k]] = (a @ b_col_p.T)[i, l]
            dot_row = a @ b_col_p.T  # (n, n)
            
            # dot_col[i, l] = sum_k a[k,i] * b[p[k], l] = (a.T @ b_row_p)[i, l]
            dot_col = a.T @ b_row_p  # (n, n)
            
            return dot_row, dot_col
        
        dot_row, dot_col = recompute_precomp(p)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Random sampling of pairs
            # Since updates are O(n^2), we can try many more swaps
            num_trials = 100000
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
                # Terms 1-4
                delta = 0.0
                delta += a[i, i] * (b[v, v] - b[u, u])
                delta += a[i, j] * (b[v, u] - b[u, v])
                delta += a[j, i] * (b[u, v] - b[v, u])
                delta += a[j, j] * (b[u, u] - b[v, v])
                
                # Terms 5-8 using precomputed dot products
                # Term 5: sum_{k not in {i,j}} a[i,k] * (b[v, p[k]] - b[u, p[k]])
                # = dot_row[i, v] - dot_row[i, u] - a[i,i]*(b[v,v]-b[u,u]) - a[i,j]*(b[v,u]-b[u,v])
                delta += (dot_row[i, v] - dot_row[i, u]) - a[i,i]*(b[v,v]-b[u,u]) - a[i,j]*(b[v,u]-b[u,v])
                
                # Term 6: sum_{k not in {i,j}} a[j,k] * (b[u, p[k]] - b[v, p[k]])
                # = dot_row[j, u] - dot_row[j, v] - a[j,i]*(b[u,v]-b[v,u]) - a[j,j]*(b[u,u]-b[v,v])
                delta += (dot_row[j, u] - dot_row[j, v]) - a[j,i]*(b[u,v]-b[v,u]) - a[j,j]*(b[u,u]-b[v,v])
                
                # Term 7: sum_{k not in {i,j}} a[k,i] * (b[p[k], v] - b[p[k], u])
                # = dot_col[i, v] - dot_col[i, u] - a[i,i]*(b[v,v]-b[u,u]) - a[j,i]*(b[v,u]-b[u,v])
                delta += (dot_col[i, v] - dot_col[i, u]) - a[i,i]*(b[v,v]-b[u,u]) - a[j,i]*(b[v,u]-b[u,v])
                
                # Term 8: sum_{k not in {i,j}} a[k,j] * (b[p[k], u] - b[p[k], v])
                # = dot_col[j, u] - dot_col[j, v] - a[i,j]*(b[u,v]-b[v,u]) - a[j,j]*(b[u,u]-b[v,v])
                delta += (dot_col[j, u] - dot_col[j, v]) - a[i,j]*(b[u,v]-b[v,u]) - a[j,j]*(b[u,u]-b[v,v])
                
                if delta < -1e-9:
                    # Accept the swap
                    p[i], p[j] = p[j], p[i]
                    cost += delta
                    improved = True
                    
                    # Incremental update of dot_row and dot_col
                    # D_b = b[v, :] - b[u, :]
                    D_b = b[v, :] - b[u, :]
                    
                    # Update dot_row:
                    # dot_row[k, l] += (a[k, i] - a[k, j]) * D_b[l]
                    diff_a_col = a[:, i] - a[:, j]
                    dot_row += np.outer(diff_a_col, D_b)
                    
                    # Update dot_col:
                    # dot_col[c, l] += (a[i, c] - a[j, c]) * D_b[l]
                    diff_a_row = a[i, :] - a[j, :]
                    dot_col += np.outer(diff_a_row, D_b)
                    
                    # Optional: Periodic re-computation to correct floating point drift
                    # if _ % 1000 == 0:
                    #     dot_row, dot_col = recompute_precomp(p)
        
        return p, cost
    
    # Run local search with multiple restarts from different initializations
    # First, run from the best initial solution
    time_limit = deadline - 35.0
    best_p, best_cost = local_search(best_p, best_cost, time_limit)
    
    # Try additional local searches from other initializations if time permits
    for method in ['wl', 'combined', 'flow']:
        if time.time() > deadline - 25.0:
            break
        if method == 'wl':
            p_alt = wl_init()
        else:
            p_alt = spectral_init(method)
        cost_alt = compute_cost(p_alt)
        if cost_alt < best_cost + (best_cost * 0.05):  # Within 5% of best
            time_limit = deadline - 15.0
            p_ls, cost_ls = local_search(p_alt, cost_alt, time_limit)
            if cost_ls < best_cost:
                best_p = p_ls
                best_cost = cost_ls
    
    # --- Hungarian-based refinement ---
    if time.time() < deadline - 20.0:
        # Build linear cost matrix
        b_col_p = b[:, best_p]  # shape (n, n), [j, k] = b[j, p[k]]
        b_row_p = b[best_p, :]  # shape (n, n), [k, j] = b[p[k], j]
        
        C = a @ b_col_p.T + a.T @ b_row_p
        
        row_ind, col_ind = linear_sum_assignment(C)
        p_hung = np.empty(n, dtype=np.int64)
        p_hung[row_ind] = col_ind
        
        cost_hung = compute_cost(p_hung)
        if cost_hung < best_cost:
            best_p = p_hung
            best_cost = cost_hung
            # Run local search from Hungarian solution
            if time.time() < deadline - 10.0:
                time_limit = deadline - 8.0
                best_p, best_cost = local_search(best_p, best_cost, time_limit)
    
    # --- Final local search pass ---
    if time.time() < deadline - 5.0:
        time_limit = deadline - 2.0
        best_p, best_cost = local_search(best_p, best_cost, time_limit)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
