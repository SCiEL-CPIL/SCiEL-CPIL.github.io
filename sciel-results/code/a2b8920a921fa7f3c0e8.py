"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.linalg as la

def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    
    start_time = time.time()
    timeout = 118.0  # Safety margin before 120s kill

    def calc_cost(p):
        b_perm = b[np.ix_(p, p)]
        return int(np.sum(a * b_perm))

    def get_delta(p, i, j):
        u = p[i]
        v = p[j]
        
        # b[v, p[k]] - b[u, p[k]]
        diff_b = b[v, p] - b[u, p]
        diff_a = a[i] - a[j]
        
        sum_all = np.dot(diff_a, diff_b)
        
        term_i = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
        term_j = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
        
        delta = 2 * (sum_all - term_i - term_j)
        return delta

    def local_search(p, current_cost, max_time=None):
        # Vectorized First Improvement Local Search
        # We scan i from 0 to n-1. For each i, we vectorize the delta calculation for all j > i.
        # If any delta < 0, we pick the best one (or first one? Best is usually better for quality, First for speed).
        # Given the 120s limit, let's try Best Improvement per i to ensure quality, but it's O(n^3) total per pass.
        # Wait, vectorizing over j for a fixed i is O(n^2) operations per i, so O(n^3) per pass.
        # n=100, n^3 = 1e6 operations. Very fast.
        
        while True:
            if time.time() - start_time > timeout - 2.0:
                break
            
            improved = False
            for i in range(n):
                u = p[i]
                a_i = a[i]
                b_u_p = b[u, p]
                
                # We want to find j > i that minimizes delta.
                # delta_j = 2 * (dot(a_i - a[j], b[p[j], p] - b[u, p]) - term_i_j - term_j_j)
                
                # Let's vectorize over j.
                # Indices j > i
                js = np.arange(i + 1, n)
                if len(js) == 0:
                    continue
                
                vs = p[js] # locations for j > i
                
                # diff_b_matrix: shape (n-i-1, n)
                # diff_b_matrix[k, :] = b[vs[k], p] - b_u_p
                # b[vs, p] is (n-i-1, n)
                diff_b_mat = b[vs, p] - b_u_p
                
                # diff_a_matrix: shape (n-i-1, n)
                # diff_a_matrix[k, :] = a_i - a[js[k]]
                diff_a_mat = a_i - a[js]
                
                # sum_all: shape (n-i-1,)
                sum_all = np.sum(diff_a_mat * diff_b_mat, axis=1)
                
                # term_i: shape (n-i-1,)
                # term_i_k = (a[i, i] - a[js[k], i]) * (b[vs[k], u] - b[u, u])
                # a[js, i] is (n-i-1,)
                # b[vs, u] is (n-i-1,)
                term_i = (a[i, i] - a[js, i]) * (b[vs, u] - b[u, u])
                
                # term_j: shape (n-i-1,)
                # term_j_k = (a[i, js[k]] - a[js[k], js[k]]) * (b[vs[k], vs[k]] - b[u, vs[k]])
                # a[i, js] is (n-i-1,)
                # a[js, js] is (n-i-1,)
                # b[vs, vs] is (n-i-1,)
                # b[u, vs] is (n-i-1,)
                term_j = (a[i, js] - np.diag(a[js, :][:, js])) * (np.diag(b[js, :][:, js]) - b[u, vs])
                
                # Wait, np.diag(a[js, :][:, js]) is not quite right.
                # a[js, js] needs to be extracted.
                # a[js, js] can be done as a[np.ix_(js, js)].diagonal()
                
                a_js_js = a[np.ix_(js, js)].diagonal()
                b_vs_vs = b[np.ix_(vs, vs)].diagonal()
                
                term_j = (a[i, js] - a_js_js) * (b_vs_vs - b[u, vs])
                
                deltas = 2 * (sum_all - term_i - term_j)
                
                min_idx = np.argmin(deltas)
                min_delta = deltas[min_idx]
                
                if min_delta < 0:
                    j = js[min_idx]
                    p[i], p[j] = p[j], p[i]
                    current_cost += min_delta
                    improved = True
                    # Restart scan from i=0 to be safe, or continue?
                    # Restarting is safer for finding the global best in the neighborhood.
                    break
            
            if not improved:
                break
        
        return p, current_cost

    def spectral_init():
        # Eigenvector alignment
        # Center matrices to remove mean, which often helps with alignment
        a_centered = a - a.mean(axis=1, keepdims=True)
        b_centered = b - b.mean(axis=1, keepdims=True)
        
        # Compute eigenvectors
        # For QAP, we often look at the dominant eigenvectors of the Laplacian or the matrix itself.
        # Let's use the eigenvectors of the matrices.
        try:
            # Symmetrize for stability? QAP matrices are often symmetric but not always.
            # scipy.linalg.eigh requires symmetric.
            # Let's use eigh on the symmetrized versions.
            a_sym = (a_centered + a_centered.T) / 2.0
            b_sym = (b_centered + b_centered.T) / 2.0
            
            # Get eigenvectors corresponding to largest eigenvalues?
            # Or smallest?
            # Usually, we align the "structure".
            # Let's try aligning the first few eigenvectors.
            
            # Actually, a simpler spectral method:
            # Solve the eigenvalue problem for the Laplacian of the graph defined by b?
            # Or just use the eigenvectors of b to define "location types" and a to define "facility types".
            
            # Let's use the eigenvectors of the centered matrices.
            # We want to match high-flow facilities to low-distance locations?
            # Or match the geometric structure.
            
            # Let's compute eigenvectors of b (distances).
            # Low eigenvalues of the Laplacian correspond to smooth modes.
            # Let's just use the first few eigenvectors of the centered b.
            
            # Since eigh returns in ascending order of eigenvalues.
            # The first eigenvector is constant (for Laplacian).
            # For general matrices, it's just the direction of least variance?
            
            # Let's try a different approach:
            # Project facilities and locations onto the first K principal components (eigenvectors).
            # Then assign based on distance in this projected space?
            
            # Simpler:
            # Use the eigenvectors of the centered b matrix.
            # Sort facilities by their projection onto the dominant eigenvector of a.
            # Sort locations by their projection onto the dominant eigenvector of b.
            
            # Let's get the eigenvector with the largest absolute eigenvalue?
            # Or the one with the largest variance?
            
            # Let's use the eigenvector corresponding to the largest eigenvalue of the centered matrix.
            # This captures the most "structure".
            
            w_a, v_a = la.eigh(a_sym)
            w_b, v_b = la.eigh(b_sym)
            
            # Take the eigenvector with the largest absolute eigenvalue
            idx_a = np.argmax(np.abs(w_a))
            idx_b = np.argmax(np.abs(w_b))
            
            proj_a = v_a[:, idx_a]
            proj_b = v_b[:, idx_b]
            
            # Sort facilities by projection
            # Sort locations by projection
            # We want to match high proj_a with low proj_b? Or high with high?
            # Since we want to minimize sum a_ij b_pi_pj, and a, b are centered,
            # if a_ij is positive, we want b_pi_pj to be negative (or small).
            # This is tricky.
            
            # Let's try sorting both in the same order and see.
            # Or opposite order.
            
            # Let's generate a few candidates and pick the best.
            
            candidates = []
            
            # Candidate 1: Same order
            order_a = np.argsort(proj_a)
            order_b = np.argsort(proj_b)
            p1 = np.empty(n, dtype=np.int64)
            p1[order_a] = order_b
            
            # Candidate 2: Opposite order
            p2 = np.empty(n, dtype=np.int64)
            p2[order_a] = order_b[::-1]
            
            # Candidate 3: Rank matching (from current)
            flow_sums = a.sum(axis=1)
            dist_sums = b.sum(axis=1)
            heavy_facilities = np.argsort(-flow_sums, kind="stable")
            central_locations = np.argsort(dist_sums, kind="stable")
            p3 = np.empty(n, dtype=np.int64)
            p3[heavy_facilities] = central_locations
            
            # Candidate 4: Opposite rank matching
            p4 = np.empty(n, dtype=np.int64)
            p4[heavy_facilities] = central_locations[::-1]
            
            candidates = [p1, p2, p3, p4]
            
            # Evaluate and return the best
            best_p = None
            best_cost = np.inf
            for p in candidates:
                cost = calc_cost(p)
                if cost < best_cost:
                    best_cost = cost
                    best_p = p.copy()
            
            return best_p, best_cost
            
        except Exception:
            # Fallback to rank matching
            flow_sums = a.sum(axis=1)
            dist_sums = b.sum(axis=1)
            heavy_facilities = np.argsort(-flow_sums, kind="stable")
            central_locations = np.argsort(dist_sums, kind="stable")
            p = np.empty(n, dtype=np.int64)
            p[heavy_facilities] = central_locations
            cost = calc_cost(p)
            return p, cost

    # 1. Initialization
    p, current_cost = spectral_init()
    best_p = p.copy()
    best_cost = current_cost
    
    # 2. Local Search on initial solution
    p, current_cost = local_search(p, current_cost)
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()
    
    # 3. ILS Loop
    p = best_p.copy()
    current_cost = best_cost
    no_improve_count = 0
    
    # Use a random number generator for reproducibility if needed, but np.random is fine
    while time.time() - start_time < timeout - 5.0:
        # Perturbation
        # Adaptive number of swaps
        if no_improve_count < 5:
            num_swaps = 3
        elif no_improve_count < 15:
            num_swaps = 7
        else:
            num_swaps = 12
        
        for _ in range(num_swaps):
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            delta = get_delta(p, i, j)
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            
        # Local Search
        p, current_cost = local_search(p, current_cost)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            # Reset to best
            p = best_p.copy()
            current_cost = best_cost
        
        # Check timeout
        if time.time() - start_time > timeout - 5.0:
            break
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
