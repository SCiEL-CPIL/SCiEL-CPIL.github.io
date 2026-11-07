"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import deque

def sign_edges(graph, budget):
    start_time = time.time()
    DEADLINE = 55  # Safety margin before 60s limit
    
    m = graph['n'] // 2
    edges = graph['edges']
    n_edges = len(edges)
    n_vertices = 2 * m
    
    # Identify edge types: cycle edges vs matching edges
    # For prism: top cycle (0..m-1), bottom cycle (m..2m-1), matching (i, i+m)
    # Determine which edges are matching
    matching_set = set()
    cycle_edges_top = []
    cycle_edges_bottom = []
    
    for idx, (u, v) in enumerate(edges):
        if abs(u - v) == m:
            matching_set.add(idx)
        elif u < m and v < m:
            cycle_edges_top.append(idx)
        elif u >= m and v >= m:
            cycle_edges_bottom.append(idx)
    
    # For fast eigenvalue computation, we exploit the prism structure.
    # The signed adjacency matrix for a prism graph has a block structure.
    # Eigenvalues can be computed from the top cycle signs, bottom cycle signs, and matching signs.
    # 
    # Actually, let's use a direct approach: build the matrix and use eigvalsh,
    # but cache and optimize. For small m (4,5,7,9), eigvalsh on 8x8 to 18x18 is fast.
    
    def compute_radius_fast(signs):
        """Compute spectral radius using the prism structure."""
        # Build 2m x 2m signed adjacency
        A = np.zeros((n_vertices, n_vertices), dtype=np.float64)
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            A[u, v] = s
            A[v, u] = s
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    
    def check_connectivity(signs):
        """Check if the 2-lift is connected using BFS."""
        # Build adjacency of the lift
        # Lift has 2*n_vertices nodes: (v,0) and (v,1) for v in 0..n_vertices-1
        # Edge (u,v) with sign s: if s=+1, (u,0)-(v,0) and (u,1)-(v,1)
        # if s=-1, (u,0)-(v,1) and (u,1)-(v,0)
        
        n_lift = 2 * n_vertices
        adj = [[] for _ in range(n_lift)]
        
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            if s > 0:
                # Same side
                adj[u * 2].append(v * 2)
                adj[v * 2].append(u * 2)
                adj[u * 2 + 1].append(v * 2 + 1)
                adj[v * 2 + 1].append(u * 2 + 1)
            else:
                # Opposite side
                adj[u * 2].append(v * 2 + 1)
                adj[v * 2 + 1].append(u * 2)
                adj[u * 2 + 1].append(v * 2)
                adj[v * 2].append(u * 2 + 1)
        
        # BFS from node 0
        visited = [False] * n_lift
        queue = deque([0])
        visited[0] = True
        count = 1
        
        while queue:
            node = queue.popleft()
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    count += 1
                    queue.append(neighbor)
        
        return count == n_lift
    
    def evaluate(signs):
        """Evaluate signs: return (is_valid, radius)."""
        if not check_connectivity(signs):
            return False, float('inf')
        radius = compute_radius_fast(signs)
        return True, radius
    
    # Initial random signs
    rng = np.random.default_rng(42)
    
    best_signs = None
    best_radius = float('inf')
    best_valid = False
    
    # Generate initial candidates with various strategies
    def generate_candidates():
        """Generate diverse initial sign configurations."""
        candidates = []
        
        # All +1
        candidates.append(np.ones(n_edges, dtype=np.float64))
        # All -1
        candidates.append(-np.ones(n_edges, dtype=np.float64))
        
        # Alternating patterns on cycles
        for phase in range(2):
            signs = np.ones(n_edges, dtype=np.float64)
            for idx in cycle_edges_top:
                if (idx // 1) % 2 != phase:  # approximate
                    signs[idx] = -1
            for idx in cycle_edges_bottom:
                if (idx // 1) % 2 != phase:
                    signs[idx] = -1
            candidates.append(signs.copy())
        
        # Random
        for _ in range(30):
            candidates.append(rng.choice([-1, 1], size=n_edges).astype(np.float64))
        
        # Structured: alternate matching signs
        for match_pattern in range(4):
            signs = np.ones(n_edges, dtype=np.float64)
            matching_list = sorted(matching_set)
            for j, idx in enumerate(matching_list):
                if j % (match_pattern + 1) == 0:
                    signs[idx] = -1
            candidates.append(signs.copy())
        
        return candidates
    
    # Evaluate initial candidates
    for signs in generate_candidates():
        if time.time() - start_time > DEADLINE:
            break
        valid, radius = evaluate(signs)
        if valid and radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
            best_valid = True
    
    # Local search with single flips
    if best_valid:
        signs = best_signs.copy()
        no_improve_streak = 0
        max_no_improve = 50
        
        while time.time() - start_time < DEADLINE and no_improve_streak < max_no_improve:
            improved = False
            # Try all single flips
            for i in range(n_edges):
                if time.time() - start_time > DEADLINE:
                    break
                
                signs[i] = -signs[i]
                valid, new_radius = evaluate(signs)
                
                if valid and new_radius < best_radius - 1e-12:
                    best_radius = new_radius
                    best_signs = signs.copy()
                    best_valid = True
                    improved = True
                    no_improve_streak = 0
                else:
                    signs[i] = -signs[i]
            
            # Try double flips (pairs of edges)
            if time.time() - start_time < DEADLINE:
                for i in range(n_edges):
                    if time.time() - start_time > DEADLINE:
                        break
                    for j in range(i + 1, n_edges):
                        if time.time() - start_time > DEADLINE:
                            break
                        
                        signs[i] = -signs[i]
                        signs[j] = -signs[j]
                        valid, new_radius = evaluate(signs)
                        
                        if valid and new_radius < best_radius - 1e-12:
                            best_radius = new_radius
                            best_signs = signs.copy()
                            best_valid = True
                            improved = True
                            no_improve_streak = 0
                        else:
                            signs[i] = -signs[i]
                            signs[j] = -signs[j]
            
            if not improved:
                no_improve_streak += 1
    
    # If still no valid, try more random restarts
    if not best_valid:
        for _ in range(200):
            if time.time() - start_time > DEADLINE:
                break
            signs = rng.choice([-1, 1], size=n_edges).astype(np.float64)
            valid, radius = evaluate(signs)
            if valid and radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
                best_valid = True
    
    # Final polish with triple flips if time remains
    if best_valid and time.time() - start_time < DEADLINE - 5:
        signs = best_signs.copy()
        for _ in range(20):
            if time.time() - start_time > DEADLINE:
                break
            improved = False
            for i in range(n_edges):
                if time.time() - start_time > DEADLINE:
                    break
                for j in range(i + 1, n_edges):
                    if time.time() - start_time > DEADLINE:
                        break
                    for k in range(j + 1, n_edges):
                        if time.time() - start_time > DEADLINE:
                            break
                        
                        signs[i] = -signs[i]
                        signs[j] = -signs[j]
                        signs[k] = -signs[k]
                        valid, new_radius = evaluate(signs)
                        
                        if valid and new_radius < best_radius - 1e-12:
                            best_radius = new_radius
                            best_signs = signs.copy()
                            best_valid = True
                            improved = True
                        else:
                            signs[i] = -signs[i]
                            signs[j] = -signs[j]
                            signs[k] = -signs[k]
            
            if not improved:
                break
    
    # Return as int array
    if best_signs is None:
        # Fallback: return all +1
        return np.ones(n_edges, dtype=int)
    
    return best_signs.astype(int)
# EVOLVE-BLOCK-END
