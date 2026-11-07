import time
import numpy as np
from collections import deque

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 88.0
    
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    m = n // 2
    
    edge_list = [(u, v) for u, v in edges]
    
    # Precompute edge indices for fast connectivity check
    top_cycle_edge_indices = []
    bottom_cycle_edge_indices = []
    match_edge_indices = []
    
    edge_map = {}
    for i, (u, v) in enumerate(edge_list):
        edge_map[(u, v)] = i
        edge_map[(v, u)] = i
        
    for i in range(m):
        u = i
        v = (i + 1) % m
        top_cycle_edge_indices.append(edge_map[(u, v)])
        
    for i in range(m):
        u = m + i
        v = m + (i + 1) % m
        bottom_cycle_edge_indices.append(edge_map[(u, v)])
        
    for i in range(m):
        u = i
        v = m + i
        match_edge_indices.append(edge_map[(u, v)])

    def is_connected_lift_fast(signs):
        prod_top = 1
        for idx in top_cycle_edge_indices:
            prod_top *= signs[idx]
        if prod_top == -1:
            return True
            
        prod_bot = 1
        for idx in bottom_cycle_edge_indices:
            prod_bot *= signs[idx]
        if prod_bot == -1:
            return True
            
        # Check consistency
        t_top = np.ones(m, dtype=int)
        for i in range(1, m):
            t_top[i] = t_top[i-1] * signs[top_cycle_edge_indices[i-1]]
            
        t_bot = np.ones(m, dtype=int)
        for i in range(1, m):
            t_bot[i] = t_bot[i-1] * signs[bottom_cycle_edge_indices[i-1]]
            
        for i in range(m):
            s_match = signs[match_edge_indices[i]]
            expected = t_top[i] * t_bot[i]
            if s_match != expected:
                return True
                
        return False

    def spectral_radius(signs):
        A = np.zeros((n, n))
        for i, (u, v) in enumerate(edge_list):
            s = signs[i]
            A[u, v] = s
            A[v, u] = s
        evals = np.linalg.eigvalsh(A)
        return np.max(np.abs(evals))

    best_radius = float('inf')
    best_signs = None
    
    rng = np.random.default_rng(3)
    
    # Phase 1: Random restarts with fast connectivity check
    max_restarts = 500 
    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
        
        signs = None
        for _ in range(50):
            candidate = rng.choice([-1, 1], size=num_edges)
            if is_connected_lift_fast(candidate):
                signs = candidate
                break
        
        if signs is None:
            continue
        
        current_radius = spectral_radius(signs)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_signs = signs.copy()
            
        improved = True
        max_passes = 20
        for _pass in range(max_passes):
            if time.time() > deadline - 2.0:
                break
            improved = False
            
            A = np.zeros((n, n))
            for i, (u, v) in enumerate(edge_list):
                s = signs[i]
                A[u, v] = s
                A[v, u] = s
            evals, evecs = np.linalg.eigh(A)
            v_top = evecs[:, -1]
            
            priorities = np.zeros(num_edges)
            for i, (u, v) in enumerate(edge_list):
                priorities[i] = signs[i] * v_top[u] * v_top[v]
                
            edge_order = np.argsort(-priorities)
            
            for i in edge_order:
                if time.time() > deadline - 1.0:
                    break
                
                if priorities[i] <= 0:
                    continue
                    
                old_sign = signs[i]
                signs[i] = -old_sign
                if is_connected_lift_fast(signs):
                    new_radius = spectral_radius(signs)
                    if new_radius < current_radius:
                        current_radius = new_radius
                        improved = True
                        if current_radius < best_radius:
                            best_radius = current_radius
                            best_signs = signs.copy()
                    else:
                        signs[i] = old_sign
                else:
                    signs[i] = old_sign
                    
            if not improved:
                break
                
        if current_radius < best_radius:
            best_radius = current_radius
            best_signs = signs.copy()
            
    # Final polish
    if best_signs is not None and time.time() < deadline - 1.0:
        for _ in range(3):
            improved = False
            for i in range(num_edges):
                if time.time() > deadline - 0.5:
                    break
                old_sign = best_signs[i]
                best_signs[i] = -old_sign
                if is_connected_lift_fast(best_signs):
                    new_radius = spectral_radius(best_signs)
                    if new_radius < best_radius:
                        best_radius = new_radius
                        improved = True
                    else:
                        best_signs[i] = old_sign
                else:
                    best_signs[i] = old_sign
            if not improved:
                break

    return best_signs
