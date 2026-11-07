import time
import numpy as np
from scipy.optimize import minimize

def sign_edges(graph, budget):
    deadline = time.time() + 85.0  # Safety margin for 90s limit
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    
    # Build index mapping for fast signed adjacency construction
    edge_pos = {}
    for idx, (u, v) in enumerate(edges):
        edge_pos[(u, v)] = idx
        edge_pos[(v, u)] = idx
        
    # Precompute edge endpoints arrays for vectorized operations
    U = np.array([e[0] for e in edges], dtype=np.int32)
    V = np.array([e[1] for e in edges], dtype=np.int32)

    def build_signed_continuous(x):
        """Build signed adjacency matrix from continuous vector x."""
        S = np.zeros((n, n), dtype=np.float64)
        # Vectorized assignment is tricky with repeated indices if not careful, 
        # but here edges are unique pairs (u,v) in the list.
        # We can use np.add.at or just a loop since num_edges is small.
        # Loop is faster for small N than overhead of advanced indexing in some cases?
        # Let's try a loop for clarity and speed on small N.
        for i in range(num_edges):
            s = x[i]
            S[U[i], V[i]] = s
            S[V[i], U[i]] = s
        return S

    def objective_and_grad(x):
        """Compute spectral norm and gradient."""
        S = build_signed_continuous(x)
        # Use eigvalsh for symmetric matrix
        # We need eigenvalues and eigenvectors
        vals, vecs = np.linalg.eigh(S)
        # Spectral norm is max(abs(vals))
        abs_vals = np.abs(vals)
        max_abs = np.max(abs_vals)
        
        # Gradient of spectral norm
        # If S is symmetric, S = V diag(w) V^T
        # ||S||_2 = max |w_i|
        # Let k be index of max |w_k|.
        # grad = (1/||S||) * (v_k v_k^T) * sign(w_k) ? 
        # Actually, derivative of max |w_i| w.r.t S is v_k v_k^T * sign(w_k) / ||S||
        # Wait, if w_k > 0, grad S = v_k v_k^T. If w_k < 0, grad S = -v_k v_k^T.
        # So grad S = sign(w_k) * v_k v_k^T / ||S||
        
        # If there are multiple max values, subgradient is convex combination. 
        # For optimization, picking one is often sufficient, or we can sum them.
        # Let's pick the one with largest absolute value.
        k = np.argmax(abs_vals)
        w_k = vals[k]
        v_k = vecs[:, k]
        
        # grad_S = sign(w_k) * outer(v_k, v_k) / max_abs
        # If max_abs is 0, gradient is undefined, but we can return 0.
        if max_abs < 1e-12:
            return 0.0, np.zeros_like(x)
            
        sign_w = 1.0 if w_k > 0 else -1.0
        grad_S = (sign_w / max_abs) * np.outer(v_k, v_k)
        
        # Project gradient to x space
        # x_i affects S[u,v] and S[v,u]
        # grad_x[i] = grad_S[u,v] + grad_S[v,u]
        # Since S is symmetric, grad_S is symmetric, so grad_S[u,v] == grad_S[v,u]
        # So grad_x[i] = 2 * grad_S[U[i], V[i]]
        
        grad_x = 2.0 * grad_S[U, V]
        
        return max_abs, grad_x

    def objective_only(x):
        S = build_signed_continuous(x)
        vals = np.linalg.eigvalsh(S)
        return float(np.max(np.abs(vals)))

    def sign_edges_continuous_to_discrete(x):
        """Round continuous vector to signs."""
        return np.sign(x).astype(np.int8)

    best_signs = None
    best_obj = float('inf')
    
    def update_best(signs, obj):
        nonlocal best_signs, best_obj
        if obj < best_obj:
            best_obj = obj
            best_signs = signs.copy()

    # Helper to run local search from a starting point
    def local_search(start_signs, max_iters=500):
        cur_signs = start_signs.copy()
        S_cur = build_signed_continuous(cur_signs.astype(np.float64))
        cur_obj = objective_only(cur_signs)
        update_best(cur_signs, cur_obj)
        
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 2.0:
            improved = False
            best_flip = -1
            best_flip_obj = cur_obj
            
            # Greedy single flip
            for e in range(num_edges):
                u, v = edges[e]
                old_s = cur_signs[e]
                new_s = -old_s
                cur_signs[e] = new_s
                S_cur[u, v] = new_s
                S_cur[v, u] = new_s
                o = objective_only(cur_signs)
                # Undo
                cur_signs[e] = old_s
                S_cur[u, v] = old_s
                S_cur[v, u] = old_s
                
                if o < best_flip_obj - 1e-12:
                    best_flip_obj = o
                    best_flip = e
            
            if best_flip >= 0:
                # Apply best flip
                e = best_flip
                u, v = edges[e]
                old_s = cur_signs[e]
                cur_signs[e] = -old_s
                S_cur[u, v] = -old_s
                S_cur[v, u] = -old_s
                cur_obj = best_flip_obj
                update_best(cur_signs, cur_obj)
                improved = True
                iters += 1
            else:
                # Try pair flips if stuck
                found = False
                for e1 in range(num_edges):
                    if time.time() > deadline - 2.0: break
                    for e2 in range(e1 + 1, num_edges):
                        if time.time() > deadline - 2.0: break
                        u1, v1 = edges[e1]
                        u2, v2 = edges[e2]
                        s1_old = cur_signs[e1]
                        s2_old = cur_signs[e2]
                        s1_new = -s1_old
                        s2_new = -s2_old
                        
                        cur_signs[e1] = s1_new
                        cur_signs[e2] = s2_new
                        S_cur[u1, v1] = s1_new
                        S_cur[v1, u1] = s1_new
                        S_cur[u2, v2] = s2_new
                        S_cur[v2, u2] = s2_new
                        
                        o = objective_only(cur_signs)
                        
                        if o < cur_obj - 1e-12:
                            cur_obj = o
                            update_best(cur_signs, cur_obj)
                            found = True
                            improved = True
                            break
                        else:
                            # Undo
                            cur_signs[e1] = s1_old
                            cur_signs[e2] = s2_old
                            S_cur[u1, v1] = s1_old
                            S_cur[v1, u1] = s1_old
                            S_cur[u2, v2] = s2_old
                            S_cur[v2, u2] = s2_old
                            
                    if improved: break
                if not found:
                    break
        return cur_signs

    # 1. Try structured patterns
    m = n // 2
    top_edges = []
    bottom_edges = []
    match_edges = []
    for idx, (u, v) in enumerate(edges):
        if u < m and v < m:
            top_edges.append(idx)
        elif u >= m and v >= m:
            bottom_edges.append(idx)
        else:
            match_edges.append(idx)

    def get_pattern(pattern_id):
        signs = np.ones(num_edges, dtype=np.int8)
        if pattern_id == 0:
            pass
        elif pattern_id == 1:
            for i, idx in enumerate(top_edges):
                signs[idx] = 1 if i % 2 == 0 else -1
            for i, idx in enumerate(bottom_edges):
                signs[idx] = 1 if i % 2 == 0 else -1
        elif pattern_id == 2:
            for i, idx in enumerate(match_edges):
                signs[idx] = 1 if i % 2 == 0 else -1
        elif pattern_id == 3:
            for i, idx in enumerate(top_edges):
                signs[idx] = 1 if i % 2 == 0 else -1
            for i, idx in enumerate(bottom_edges):
                signs[idx] = -1 if i % 2 == 0 else 1
            for i, idx in enumerate(match_edges):
                signs[idx] = 1 if i % 2 == 0 else -1
        return signs

    for p in range(4):
        if time.time() > deadline - 10.0: break
        pattern = get_pattern(p)
        local_search(pattern)

    # 2. Continuous Relaxation + Local Search
    if time.time() < deadline - 10.0:
        rng = np.random.default_rng(3)
        # Try a few random continuous starting points
        for _ in range(10):
            if time.time() > deadline - 5.0: break
            x0 = rng.choice([-1.0, 1.0], size=num_edges)
            
            # Optimize continuous relaxation
            # We use L-BFGS-B
            try:
                res = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True, 
                               options={'maxiter': 100, 'ftol': 1e-8})
                x_opt = res.x
                # Clip to [-1, 1]
                x_opt = np.clip(x_opt, -1.0, 1.0)
                # Round to signs
                signs_rounded = np.sign(x_opt).astype(np.int8)
                # Ensure no zeros (if exactly 0, pick 1)
                signs_rounded[signs_rounded == 0] = 1
                
                # Refine with local search
                local_search(signs_rounded)
            except Exception:
                pass

    # 3. Random Discrete Restarts
    if time.time() < deadline - 5.0:
        rng = np.random.default_rng(42)
        for _ in range(20):
            if time.time() > deadline - 2.0: break
            signs = rng.choice([-1, 1], size=num_edges).astype(np.int8)
            local_search(signs)

    if best_signs is None:
        best_signs = np.array([1] * num_edges, dtype=np.int8)
    
    return best_signs.astype(np.int8)
