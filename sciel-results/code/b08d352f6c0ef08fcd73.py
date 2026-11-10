import numpy as np
import time

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Finds a large square-difference-free set modulo 205 using Integer Linear Programming (ILP).
    This is an exact optimization approach, distinct from stochastic heuristics.
    """
    m = 205
    
    # 1. Compute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # 2. Build conflict graph edges
    # Edge (u, v) exists if (u-v) in F or (v-u) in F
    edges = []
    for u in range(m):
        for v in range(u + 1, m):
            d1 = (u - v) % m
            d2 = (v - u) % m
            if d1 in forbidden or d2 in forbidden:
                edges.append((u, v))
                
    if not edges:
        return np.array(list(range(m)), dtype=np.int64)
        
    # 3. Try to solve using ILP (scipy.optimize.milp)
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        n_vars = m
        c = np.ones(n_vars)
        
        A_ub = np.zeros((len(edges), n_vars))
        b_ub = np.ones(len(edges))
        
        for i, (u, v) in enumerate(edges):
            A_ub[i, u] = 1
            A_ub[i, v] = 1
            
        constraints = [LinearConstraint(A_ub, ub=b_ub)]
        integrality = np.ones(n_vars)
        bounds = Bounds(lb=0, ub=1)
        
        # Set a time limit to ensure we finish within 600s
        # We use 500s to be safe.
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds, options={'time_limit': 500})
        
        # Status 0: Optimal, 2: Best solution found within time limit
        if res.status in [0, 2] and res.x is not None:
            result = []
            for i in range(m):
                if res.x[i] > 0.5:
                    result.append(i)
            if result:
                return np.array(result, dtype=np.int64)
                
    except Exception:
        pass
        
    # 4. Fallback: Deterministic Greedy + Local Search
    # If ILP is unavailable or fails, use a deterministic heuristic.
    
    # Calculate degrees
    degrees = [0] * m
    for u, v in edges:
        degrees[u] += 1
        degrees[v] += 1
        
    current_set = set()
    
    # Greedy: Add nodes with lowest degree first
    order = sorted(range(m), key=lambda x: degrees[x])
    for v in order:
        # Check compatibility
        compatible = True
        for u in current_set:
            d1 = (u - v) % m
            d2 = (v - u) % m
            if d1 in forbidden or d2 in forbidden:
                compatible = False
                break
        if compatible:
            current_set.add(v)
            
    # Local Search: Try to add any remaining node
    improved = True
    while improved:
        improved = False
        for v in range(m):
            if v in current_set:
                continue
            compatible = True
            for u in current_set:
                d1 = (u - v) % m
                d2 = (v - u) % m
                if d1 in forbidden or d2 in forbidden:
                    compatible = False
                    break
            if compatible:
                current_set.add(v)
                improved = True
                
    return np.array(list(current_set), dtype=np.int64)
