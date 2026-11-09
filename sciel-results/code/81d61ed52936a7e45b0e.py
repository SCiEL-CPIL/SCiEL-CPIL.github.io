import numpy as np
import time

def sign_edges(graph, budget):
    from autoresearch_suite.science_tcs_expansion import check_lift
    
    deadline = time.time() + 55.0  # safety margin before 60s limit
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    
    # Build adjacency structure for fast evaluation
    # Signed adjacency matrix: A[u,v] = sign for edge (u,v)
    
    def build_matrix(signs):
        A = np.zeros((n, n), dtype=np.float64)
        for i, (u, v) in enumerate(edges):
            A[u, v] = signs[i]
            A[v, u] = signs[i]
        return A
    
    def eval_radius(signs):
        A = build_matrix(signs)
        try:
            check_lift(graph, signs)  # validate
        except Exception:
            return None
        eigvals = np.linalg.eigvalsh(A)
        return max(abs(eigvals[0]), abs(eigvals[-1]))
    
    def is_valid(signs):
        try:
            check_lift(graph, signs)
            return True
        except Exception:
            return False
    
    best_signs = None
    best_radius = float('inf')
    
    # Multiple random restarts with local search
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        # Random start
        signs = rng.choice([-1, 1], size=num_edges)
        
        # Check validity
        if not is_valid(signs):
            continue
        
        radius = eval_radius(signs)
        if radius is None:
            continue
        
        if radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
        
        # Local search: flip one sign at a time
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_radius = radius
            for i in range(num_edges):
                if time.time() > deadline:
                    break
                # Try flipping sign i
                old_sign = signs[i]
                signs[i] = -old_sign
                
                if not is_valid(signs):
                    signs[i] = old_sign
                    continue
                
                r = eval_radius(signs)
                if r is not None and r < current_radius - 1e-12:
                    current_radius = r
                    improved = True
                    if r < best_radius:
                        best_radius = r
                        best_signs = signs.copy()
                else:
                    signs[i] = old_sign
        
        # If we found a good solution, try more sophisticated moves
        if best_signs is not None and time.time() < deadline:
            # Try double flips
            signs = best_signs.copy()
            current_radius = best_radius
            improved_double = True
            while improved_double and time.time() < deadline:
                improved_double = False
                for i in range(num_edges):
                    if time.time() > deadline:
                        break
                    for j in range(i+1, num_edges):
                        if time.time() > deadline:
                            break
                        signs[i] = -signs[i]
                        signs[j] = -signs[j]
                        
                        if not is_valid(signs):
                            signs[i] = -signs[i]
                            signs[j] = -signs[j]
                            continue
                        
                        r = eval_radius(signs)
                        if r is not None and r < current_radius - 1e-12:
                            current_radius = r
                            improved_double = True
                            if r < best_radius:
                                best_radius = r
                                best_signs = signs.copy()
                        else:
                            signs[i] = -signs[i]
                            signs[j] = -signs[j]
    
    if best_signs is None:
        # Fallback: random valid solution
        for _ in range(1000):
            signs = rng.choice([-1, 1], size=num_edges)
            if is_valid(signs):
                best_signs = signs
                break
    
    return best_signs
