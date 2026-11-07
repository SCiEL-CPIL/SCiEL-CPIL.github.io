import time
import numpy as np
from collections import deque

def sign_edges(graph, budget):
    deadline = time.time() + 85.0  # 5s safety margin
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    m_val = n // 2
    
    # Precompute edge indices
    edge_idx = {}
    for i, (u, v) in enumerate(edges):
        edge_idx[(u, v)] = i
        edge_idx[(v, u)] = i
    
    # Classify edges: top cycle, bottom cycle, matching
    top_edges = []    # edges within 0..m-1
    bottom_edges = [] # edges within m..2m-1
    match_edges = []  # matching edges (i, i+m)
    
    for i, (u, v) in enumerate(edges):
        if u < m_val and v < m_val:
            top_edges.append(i)
        elif u >= m_val and v >= m_val:
            bottom_edges.append(i)
        else:
            match_edges.append(i)
    
    # Build adjacency for fast connectivity via cycle parity
    # The lift is connected iff the product of signs around any fundamental cycle is -1
    # For prism: check top cycle product, bottom cycle product, and matching parity
    
    def get_cycle_products(signs):
        """Return (top_product, bottom_product) for the two m-cycles"""
        top_prod = 1
        for i in range(m_val):
            u = i
            v = (i + 1) % m_val
            e = edge_idx.get((u, v))
            if e is not None:
                top_prod *= signs[e]
        
        bottom_prod = 1
        for i in range(m_val):
            u = m_val + i
            v = m_val + (i + 1) % m_val
            e = edge_idx.get((u, v))
            if e is not None:
                bottom_prod *= signs[e]
        
        return top_prod, bottom_prod
    
    def is_connected_fast(signs):
        """Fast connectivity check using cycle parity.
        The 2-lift of a prism graph is connected iff:
        - The product of signs around the top cycle is -1 OR
        - The product of signs around the bottom cycle is -1
        (Actually, we need the lift to be connected, which for a prism means
        the two cycles must have opposite parity or the matching connects them properly)
        
        For a prism graph with matching edges, the lift is connected iff
        the top cycle product * bottom cycle product = -1
        (i.e., one is +1 and the other is -1, or both are -1... let me think)
        
        Actually, the simplest check: build the lift graph and do BFS, but fast.
        """
        # For prism graphs, the 2-lift is connected iff the product of signs
        # around the "outer" cycle (going around one side, across matching, 
        # around the other side back) is -1.
        # 
        # The prism has a fundamental cycle: 0-1-2-...-(m-1)-(m-1+m)-(m-2+m)-...-0
        # This is a 2m-cycle. The lift is connected iff the product of signs
        # around this 2m-cycle is -1.
        
        # Let's compute: go 0 -> 1 -> ... -> m-1 -> (m-1+m) -> (m-2+m) -> ... -> m -> 0
        prod = 1
        # Top cycle forward
        for i in range(m_val - 1):
            e = edge_idx.get((i, i + 1))
            if e is not None:
                prod *= signs[e]
        # Matching edge (m-1, m-1+m)
        e = edge_idx.get((m_val - 1, 2 * m_val - 1))
        if e is not None:
            prod *= signs[e]
        # Bottom cycle backward
        for i in range(m_val - 2, -1, -1):
            e = edge_idx.get((m_val + i, m_val + i + 1))
            if e is not None:
                prod *= signs[e]
        # Matching edge (0, m)
        e = edge_idx.get((0, m_val))
        if e is not None:
            prod *= signs[e]
        
        return prod == -1
    
    def get_objective(signs):
        S = np.zeros((n, n), dtype=np.float64)
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            S[u, v] = s
            S[v, u] = s
        ev = np.linalg.eigvalsh(S)
        return float(np.max(np.abs(ev)))
    
    def get_objective_fast(signs):
        """Compute spectral radius using circulant structure for speed."""
        # For prism graphs, the signed adjacency matrix has a block structure
        # that can be diagonalized using DFT on each cycle.
        # But for simplicity and correctness, use eigvalsh directly.
        S = np.zeros((n, n), dtype=np.float64)
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            S[u, v] = s
            S[v, u] = s
        ev = np.linalg.eigvalsh(S)
        return float(np.max(np.abs(ev)))
    
    # Precompute 4-cycles for cycle flips
    cycles_4 = []
    for i in range(m_val):
        v1 = i
        v2 = (i + 1) % m_val
        v3 = m_val + i
        v4 = m_val + (i + 1) % m_val
        
        e1 = edge_idx.get((v1, v2))
        e2 = edge_idx.get((v2, v4))
        e3 = edge_idx.get((v4, v3))
        e4 = edge_idx.get((v3, v1))
        
        if e1 is not None and e2 is not None and e3 is not None and e4 is not None:
            cycles_4.append([e1, e2, e3, e4])
    
    rng = np.random.default_rng(3)
    
    best_signs = None
    best_obj = float('inf')
    
    def update_best(signs, obj):
        nonlocal best_signs, best_obj
        if obj < best_obj:
            best_obj = obj
            best_signs = signs.copy()
    
    def local_search(initial_signs, time_limit=None):
        nonlocal best_signs, best_obj
        if time_limit is None:
            time_limit = deadline
        current_signs = initial_signs.copy()
        
        if not is_connected_fast(current_signs):
            return
        
        current_obj = get_objective(current_signs)
        update_best(current_signs, current_obj)
        
        while time.time() < time_limit:
            improved = False
            
            # 1. Try single flips - use fast connectivity
            best_flip_idx = -1
            best_flip_obj = current_obj
            for i in range(num_edges):
                old = current_signs[i]
                current_signs[i] = -old
                if is_connected_fast(current_signs):
                    obj = get_objective(current_signs)
                    if obj < best_flip_obj - 1e-12:
                        best_flip_obj = obj
                        best_flip_idx = i
                current_signs[i] = old
                
            if best_flip_idx != -1:
                current_signs[best_flip_idx] = -current_signs[best_flip_idx]
                current_obj = best_flip_obj
                update_best(current_signs, current_obj)
                improved = True
            else:
                # 2. Try 4-cycle flips
                best_cycle_idx = -1
                best_cycle_obj = current_obj
                for c_idx, cyc in enumerate(cycles_4):
                    for e in cyc:
                        current_signs[e] = -current_signs[e]
                    if is_connected_fast(current_signs):
                        obj = get_objective(current_signs)
                        if obj < best_cycle_obj - 1e-12:
                            best_cycle_obj = obj
                            best_cycle_idx = c_idx
                    for e in cyc:
                        current_signs[e] = -current_signs[e]
                        
                if best_cycle_idx != -1:
                    for e in cycles_4[best_cycle_idx]:
                        current_signs[e] = -current_signs[e]
                    current_obj = best_cycle_obj
                    update_best(current_signs, current_obj)
                    improved = True
                else:
                    # 3. Try pair flips
                    found_pair = False
                    for _ in range(30):
                        if time.time() > time_limit: break
                        e1 = rng.integers(num_edges)
                        e2 = rng.integers(num_edges)
                        if e1 == e2: continue
                        current_signs[e1] = -current_signs[e1]
                        current_signs[e2] = -current_signs[e2]
                        if is_connected_fast(current_signs):
                            obj = get_objective(current_signs)
                            if obj < current_obj - 1e-12:
                                current_obj = obj
                                update_best(current_signs, current_obj)
                                found_pair = True
                                improved = True
                                break
                        else:
                            current_signs[e1] = -current_signs[e1]
                            current_signs[e2] = -current_signs[e2]
                            
                    if not found_pair:
                        break
    
    def polish(signs):
        """High-precision polish"""
        nonlocal best_signs, best_obj
        current_signs = signs.copy()
        if not is_connected_fast(current_signs):
            return
        current_obj = get_objective(current_signs)
        update_best(current_signs, current_obj)
        
        for _ in range(30):
            if time.time() > deadline - 2.0:
                break
            improved = False
            
            # Single flip sweep
            for i in range(num_edges):
                if time.time() > deadline - 2.0:
                    break
                old = current_signs[i]
                current_signs[i] = -old
                if is_connected_fast(current_signs):
                    obj = get_objective(current_signs)
                    if obj < current_obj - 1e-14:
                        current_obj = obj
                        improved = True
                    else:
                        current_signs[i] = old
                else:
                    current_signs[i] = old
            
            if not improved:
                # Pair flips
                for _ in range(50):
                    if time.time() > deadline - 2.0:
                        break
                    e1 = rng.integers(num_edges)
                    e2 = rng.integers(num_edges)
                    if e1 == e2: continue
                    current_signs[e1] = -current_signs[e1]
                    current_signs[e2] = -current_signs[e2]
                    if is_connected_fast(current_signs):
                        obj = get_objective(current_signs)
                        if obj < current_obj - 1e-14:
                            current_obj = obj
                            improved = True
                            break
                    else:
                        current_signs[e1] = -current_signs[e1]
                        current_signs[e2] = -current_signs[e2]
            
            if not improved:
                break
        
        update_best(current_signs, current_obj)
    
    def continuous_optimization(time_limit):
        """Use continuous phase optimization to find good sign assignments.
        
        For each edge, assign a phase theta_i in [0, 2*pi). The sign is
        +1 if cos(theta_i) > 0, -1 otherwise. We optimize the spectral radius
        of the matrix with entries exp(i*theta_i) on edges, then round.
        
        Actually, a better approach: use the fact that for the prism graph,
        the eigenvalues can be computed analytically from the DFT of the
        edge signs on each cycle.
        """
        nonlocal best_signs, best_obj
        
        # For prism graph, we can parametrize:
        # Top cycle edges: signs t_0, t_1, ..., t_{m-1} (product must be -1 for connectivity)
        # Bottom cycle edges: signs b_0, b_1, ..., b_{m-1}
        # Matching edges: signs m_0, m_1, ..., m_{m-1}
        
        # The signed adjacency matrix in Fourier basis:
        # For each frequency k, we get a 2x2 matrix
        # [[sum_top_k, sum_match_k], [sum_match_k, sum_bottom_k]]
        # where sum_top_k = sum_j t_j * exp(2*pi*i*k*j/m)
        
        # Let's use this to compute eigenvalues faster
        def compute_eigvals_fast(signs):
            # Extract top, bottom, matching signs
            top_s = np.zeros(m_val, dtype=np.float64)
            bottom_s = np.zeros(m_val, dtype=np.float64)
            match_s = np.zeros(m_val, dtype=np.float64)
            
            for i, (u, v) in enumerate(edges):
                s = signs[i]
                if u < m_val and v < m_val:
                    # Top cycle edge from u to v
                    idx = u % m_val
                    top_s[idx] = s
                elif u >= m_val and v >= m_val:
                    idx = (u - m_val) % m_val
                    bottom_s[idx] = s
                else:
                    # Matching edge
                    idx = min(u, v) % m_val
                    match_s[idx] = s
            
            # DFT
            top_fft = np.fft.fft(top_s)
            bottom_fft = np.fft.fft(bottom_s)
            match_fft = np.fft.fft(match_s)
            
            max_eig = 0.0
            for k in range(m_val):
                a = real_part(top_fft[k])
                b = real_part(match_fft[k])
                c = real_part(bottom_fft[k])
                # 2x2 matrix [[a, b], [b, c]]
                # Eigenvalues: (a+c)/2 +- sqrt(((a-c)/2)^2 + b^2)
                trace = a + c
                det = a * c - b * b
                disc = trace * trace - 4 * det
                if disc < 0:
                    disc = 0
                sqrt_disc = np.sqrt(disc)
                e1 = (trace + sqrt_disc) / 2
                e2 = (trace - sqrt_disc) / 2
                max_eig = max(max_eig, abs(e1), abs(e2))
            
            return max_eig
        
        def real_part(z):
            return z.real
        
        # Random restarts in continuous space
        for restart in range(50):
            if time.time() > time_limit:
                break
            
            # Initialize with random signs
            signs = rng.choice([-1, 1], size=num_edges).astype(np.float64)
            if not is_connected_fast(signs):
                continue
            
            obj = compute_eigvals_fast(signs)
            if obj < best_obj:
                best_obj = obj
                best_signs = signs.copy().astype(np.int8)
            
            # Gradient-like descent: try flipping edges that reduce objective
            for iteration in range(100):
                if time.time() > time_limit:
                    break
                
                best_flip = -1
                best_obj_local = obj
                for i in range(num_edges):
                    signs[i] = -signs[i]
                    if is_connected_fast(signs):
                        o = compute_eigvals_fast(signs)
                        if o < best_obj_local - 1e-12:
                            best_obj_local = o
                            best_flip = i
                    signs[i] = -signs[i]
                
                if best_flip == -1:
                    break
                
                signs[best_flip] = -signs[best_flip]
                obj = best_obj_local
                if obj < best_obj:
                    best_obj = obj
                    best_signs = signs.copy().astype(np.int8)
    
    # Strategy:
    # 1. Quick structured initializations + local search
    # 2. Continuous optimization with more restarts
    # 3. Final polish
    
    # 1. All 1s
    signs = np.ones(num_edges, dtype=np.int8)
    if is_connected_fast(signs):
        local_search(signs, deadline - 40.0)
    
    # 2. Alternating top
    signs = np.ones(num_edges, dtype=np.int8)
    for i, (u, v) in enumerate(edges):
        if u < m_val and v < m_val:
            signs[i] = 1 if (u % 2 == 0) else -1
    if is_connected_fast(signs):
        local_search(signs, deadline - 30.0)
    
    # 3. Alternating bottom
    signs = np.ones(num_edges, dtype=np.int8)
    for i, (u, v) in enumerate(edges):
        if u >= m_val and v >= m_val:
            signs[i] = 1 if ((u - m_val) % 2 == 0) else -1
    if is_connected_fast(signs):
        local_search(signs, deadline - 20.0)
    
    # 4. Continuous optimization
    if time.time() < deadline - 10.0:
        continuous_optimization(deadline - 5.0)
    
    # 5. More random restarts with local search
    for _ in range(200):
        if time.time() > deadline - 5.0:
            break
        signs = rng.choice([-1, 1], size=num_edges).astype(np.int8)
        local_search(signs, deadline - 5.0)
    
    # Final polish
    if best_signs is not None:
        polish(best_signs)
    
    if best_signs is None:
        best_signs = np.ones(num_edges, dtype=np.int8)
        
    return best_signs
