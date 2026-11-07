import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh

def sign_edges(graph, budget):
    start = time.time()
    deadline = start + 82.0  # safety margin for 90s limit

    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    m = n // 2

    # Build edge index lookup
    edge_idx = {}
    for idx, (u, v) in enumerate(edges):
        edge_idx[(u, v)] = idx
        edge_idx[(v, u)] = idx

    # Identify edge categories
    # Top cycle edges: (i, i+1) for i in 0..m-1 (mod m), all vertices < m
    # Bottom cycle edges: (m+i, m+i+1), all vertices >= m
    # Matching edges: (i, m+i)
    top_edge_idx = []
    bottom_edge_idx = []
    match_edge_idx = []
    for idx, (u, v) in enumerate(edges):
        if u < m and v < m:
            top_edge_idx.append(idx)
        elif u >= m and v >= m:
            bottom_edge_idx.append(idx)
        else:
            match_edge_idx.append(idx)

    # For each top cycle edge, determine which "position" it is
    # Top cycle: vertices 0..m-1, edges connect i to (i+1)%m
    # We need to identify: edge (0,1) -> pos 0, edge (m-1, 0) -> pos m-1
    top_pos = {}  # edge_idx -> position in cycle
    bottom_pos = {}
    match_pos = {}
    for idx in top_edge_idx:
        u, v = edges[idx]
        # position: the smaller vertex index (for edge (i, i+1), pos = i; for (m-1,0), pos = m-1)
        if u == 0:
            pos = m - 1  # edge (m-1, 0)
        elif v == 0:
            pos = m - 1
        else:
            pos = min(u, v)
        top_pos[idx] = pos
    for idx in bottom_edge_idx:
        u, v = edges[idx]
        u2, v2 = u - m, v - m
        if u2 == 0:
            pos = m - 1
        elif v2 == 0:
            pos = m - 1
        else:
            pos = min(u2, v2)
        bottom_pos[idx] = pos
    for idx in match_edge_idx:
        u, v = edges[idx]
        pos = min(u, v)
        match_pos[idx] = pos

    def build_signed_adj(params):
        """
        params: array of length m+4
          params[0] = sign of top edge (0,1)  [continuous, will be rounded]
          params[1] = sign of top edge (m-1,0)
          params[2] = sign of bottom edge (0,1)
          params[3] = sign of bottom edge (m-1,0)
          params[4+i] = sign of matching edge (i, m+i) for i=0..m-1
        """
        S = np.zeros((n, n), dtype=np.float64)
        # Top cycle
        for idx in top_edge_idx:
            pos = top_pos[idx]
            if pos == 0:
                s = params[0]
            elif pos == m - 1:
                s = params[1]
            else:
                # For other positions, we need a sign. 
                # Actually, in a circulant, all edges of the same "type" have the same sign.
                # But in a cycle, edge (i, i+1) and edge (i+1, i+2) are the same type.
                # The only two types in a cycle are: forward edges and the "wrap" edge.
                # Wait, in a cycle C_m, all m edges are equivalent under rotation.
                # So actually the top cycle has only 1 degree of freedom for signs!
                # Let me reconsider.
                # A signed cycle: the sign pattern is determined up to gauge.
                # The product of signs around the cycle determines if it's "balanced".
                # For spectral radius, what matters is the actual signs.
                # In a circulant matrix, A[i, i+1] = a for all i (mod m).
                # So all top cycle edges have the SAME sign.
                s = params[0]
            S[u, v] = s
            S[v, u] = s
        # Bottom cycle
        for idx in bottom_edge_idx:
            pos = bottom_pos[idx]
            s = params[2]
            u, v = edges[idx]
            S[u, v] = s
            S[v, u] = s
        # Matching
        for idx in match_edge_idx:
            pos = match_pos[idx]
            s = params[4 + pos]
            u, v = edges[idx]
            S[u, v] = s
            S[v, u] = s
        return S

    def spectral_radius(S):
        ev = np.linalg.eigvalsh(S)
        return float(np.max(np.abs(ev)))

    def is_connected(S):
        # Convert to graph and check connectivity
        G = csr_matrix(np.abs(S))
        ncomp, _ = connected_components(G, directed=False)
        return ncomp == 1

    def objective_and_check(params):
        S = build_signed_adj(params)
        if not is_connected(S):
            return float('inf'), None
        r = spectral_radius(S)
        return r, S

    def round_and_validate(params):
        """Round continuous params to ±1, check validity, return signs array."""
        rounded = np.sign(params)
        rounded[rounded == 0] = 1.0
        S = build_signed_adj(rounded)
        if not is_connected(S):
            return None, float('inf')
        r = spectral_radius(S)
        # Build the sign array for output
        signs = np.ones(num_edges, dtype=np.int8)
        for idx in top_edge_idx:
            signs[idx] = int(rounded[0])
        for idx in bottom_edge_idx:
            signs[idx] = int(rounded[2])
        for idx in match_edge_idx:
            pos = match_pos[idx]
            signs[idx] = int(rounded[4 + pos])
        return signs, r

    def gradient_descent(params0, lr=0.5, momentum=0.9, max_iter=2000):
        """Momentum gradient descent on the reduced parameter space."""
        params = params0.copy()
        v = np.zeros_like(params)
        best_params = params.copy()
        best_obj = objective_and_check(params)[0]
        
        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            # Compute gradient via finite differences
            grad = np.zeros_like(params)
            eps = 0.01
            for i in range(len(params)):
                params_plus = params.copy()
                params_plus[i] += eps
                obj_plus = objective_and_check(params_plus)[0]
                params_minus = params.copy()
                params_minus[i] -= eps
                obj_minus = objective_and_check(params_minus)[0]
                grad[i] = (obj_plus - obj_minus) / (2 * eps)
            
            v = momentum * v - lr * grad
            params = params + v
            # Clip to reasonable range
            params = np.clip(params, -2.0, 2.0)
            
            obj = objective_and_check(params)[0]
            if obj < best_obj:
                best_obj = obj
                best_params = params.copy()
        
        return best_params, best_obj

    # Initial best
    best_signs = None
    best_radius = float('inf')

    def update_best(params):
        nonlocal best_signs, best_radius
        signs, r = round_and_validate(params)
        if signs is not None and r < best_radius:
            best_radius = r
            best_signs = signs

    # Start from several initial points
    rng = np.random.default_rng(42)
    
    # Pattern 1: all +1
    update_best(np.ones(m + 4))
    
    # Pattern 2: alternating matching
    p2 = np.ones(m + 4)
    for i in range(m):
        p2[4 + i] = 1.0 if i % 2 == 0 else -1.0
    update_best(p2)

    # Pattern 3: negative top cycle
    p3 = np.ones(m + 4)
    p3[0] = -1.0
    update_best(p3)

    # Pattern 4: negative bottom cycle
    p4 = np.ones(m + 4)
    p4[2] = -1.0
    update_best(p4)

    # Now do gradient descent from several starting points
    start_points = [
        np.ones(m + 4),
        -np.ones(m + 4),
        np.ones(m + 4) * 0.5,
    ]
    
    # Add some random starts
    for _ in range(8):
        start_points.append(rng.uniform(-1, 1, size=m + 4))
    
    # Also add structured starts
    for _ in range(6):
        p = np.ones(m + 4)
        # Random subset of matching edges negative
        neg_mask = rng.random(m) < 0.5
        p[4:] = np.where(neg_mask, -1.0, 1.0)
        start_points.append(p)
    
    for sp in start_points:
        if time.time() > deadline - 10.0:
            break
        # Try multiple learning rates
        for lr in [0.3, 0.5, 0.8, 1.2]:
            if time.time() > deadline - 5.0:
                break
            best_p, best_o = gradient_descent(sp, lr=lr, momentum=0.9, max_iter=800)
            update_best(best_p)
            # Also try from the rounded version of best_p
            rounded_p = np.sign(best_p)
            rounded_p[rounded_p == 0] = 1.0
            update_best(rounded_p)
            # Continue descent from rounded point
            if time.time() < deadline - 5.0:
                best_p2, best_o2 = gradient_descent(rounded_p, lr=0.3, momentum=0.9, max_iter=500)
                update_best(best_p2)

    # Final: also try a few random sign vectors directly
    if time.time() < deadline - 15.0:
        rng2 = np.random.default_rng(3)
        for _ in range(20):
            if time.time() > deadline - 5.0:
                break
            signs = rng2.choice([-1, 1], size=num_edges)
            # Build S and check
            S = np.zeros((n, n), dtype=np.float64)
            for idx, (u, v) in enumerate(edges):
                s = signs[idx]
                S[u, v] = s
                S[v, u] = s
            if is_connected(S):
                r = spectral_radius(S)
                if r < best_radius:
                    best_radius = r
                    best_signs = signs.astype(np.int8)

    # Fallback
    if best_signs is None:
        best_signs = np.ones(num_edges, dtype=np.int8)

    return best_signs.astype(np.int8)
