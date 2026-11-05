import math
import time

def construct():
    start_time = time.time()
    # Hard limit to ensure we return before 90s
    # We check time periodically or just rely on MILP being fast.
    # If MILP is slow, we might need to abort. 
    # But N=255 is tiny for MILP.
    
    nodes = list(range(2, 257))
    n = len(nodes)
    
    # Calculate weights
    weights = [1.0 / v for v in nodes]
    
    # Build conflict edges
    # Edge (i, j) if lcm(nodes[i], nodes[j]) <= 256
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            a = nodes[i]
            b = nodes[j]
            # Efficient lcm check
            # lcm(a, b) = a * b // gcd(a, b)
            # To avoid large intermediate numbers (though a,b <= 256, product <= 65536, safe)
            g = math.gcd(a, b)
            l = (a // g) * b
            if l <= 256:
                edges.append((i, j))
                
    best_solution = []
    best_score = 0.0
    
    # Try MILP
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np
        
        c = -np.array(weights)
        
        if not edges:
            # No conflicts, take all
            best_solution = nodes
            best_score = sum(weights)
        else:
            m = len(edges)
            A_ub = np.zeros((m, n))
            for k, (i, j) in enumerate(edges):
                A_ub[k, i] = 1.0
                A_ub[k, j] = 1.0
            
            b_ub = np.ones(m)
            
            integrality = np.ones(n)
            bounds = Bounds(lb=0, ub=1)
            
            # Solve
            res = milp(
                c=c,
                constraints=[LinearConstraint(A_ub, ub=b_ub)],
                integrality=integrality,
                bounds=bounds
            )
            
            if res.status == 0:
                x = res.x
                # Threshold to handle floating point errors
                selected_indices = [i for i in range(n) if x[i] > 0.5]
                best_solution = [nodes[i] for i in selected_indices]
                best_score = sum(weights[i] for i in selected_indices)
            else:
                # If MILP failed, fallback
                pass
                
    except Exception:
        # Fallback if SciPy/MILP not available or failed
        pass
        
    if not best_solution:
        # Heuristic Fallback
        # Greedy: Iterate through nodes, add if compatible
        # Order: By weight descending (smaller v first)
        # This is what the seed does, but we can try to optimize order or use local search
        
        # Let's try a slightly smarter greedy:
        # Sort by weight descending (v ascending)
        # But maybe large v's are "cheaper" in conflicts?
        # Actually, small v's have high weight. We want them if possible.
        
        current_set = set()
        # Try adding in order of v
        for v in nodes:
            if all(math.lcm(v, u) > 256 for u in current_set):
                current_set.add(v)
                
        # Local Search: Try to add any missing compatible node
        improved = True
        while improved:
            improved = False
            for v in nodes:
                if v not in current_set:
                    if all(math.lcm(v, u) > 256 for u in current_set):
                        current_set.add(v)
                        improved = True
                        break # Restart scan after change? Or just continue. 
                        # Breaking and restarting ensures we don't miss interactions? 
                        # Actually, if we add v, it might block others, but we are only adding.
                        # So just continue is fine, but checking all again is safer for "maximal" set.
        
        # Try swapping to improve weight
        # If we have u in set, and v not in set, and v compatible with set \ {u},
        # and 1/v > 1/u, swap.
        improved = True
        while improved:
            improved = False
            for u in list(current_set):
                for v in nodes:
                    if v in current_set: continue
                    # Check if v is compatible with current_set - {u}
                    if all(math.lcm(v, w) > 256 for w in current_set if w != u):
                        if 1.0/v > 1.0/u:
                            current_set.remove(u)
                            current_set.add(v)
                            improved = True
                            break
                if improved: break
        
        best_solution = list(current_set)
        best_score = sum(1.0/v for v in best_solution)
        
    return best_solution
