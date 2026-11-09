import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    N = 256
    
    # 1. Filter nodes: 2*v must not be a perfect square
    valid_nodes = []
    for v in range(1, N + 1):
        s = 2 * v
        r = int(math.isqrt(s))
        if r * r != s:
            valid_nodes.append(v)
    
    if not valid_nodes:
        return [1]
        
    M = len(valid_nodes)
    
    # 2. Build adjacency list
    # adj[i] is a set of indices j such that valid_nodes[i] + valid_nodes[j] is a square
    adj = [set() for _ in range(M)]
    
    for i in range(M):
        vi = valid_nodes[i]
        for j in range(i + 1, M):
            vj = valid_nodes[j]
            s = vi + vj
            r = int(math.isqrt(s))
            if r * r == s:
                adj[i].add(j)
                adj[j].add(i)
                
    # 3. Use MILP to find maximum independent set
    # Formulation: maximize sum(x_i) subject to x_i + x_j <= 1 for each edge (i,j)
    # x_i in {0, 1}
    
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np
        
        # Build constraint matrix for edge constraints: x_i + x_j <= 1
        edges = []
        seen = set()
        for i in range(M):
            for j in adj[i]:
                if i < j and (i, j) not in seen:
                    edges.append((i, j))
                    seen.add((i, j))
        
        E = len(edges)
        
        if E == 0:
            # No conflicts, take all nodes
            return valid_nodes[:]
        
        # Objective: maximize sum(x_i), so c = -1 for each variable
        c = -np.ones(M)
        
        # Constraints: x_i + x_j <= 1 for each edge
        # A_ub * x <= b_ub
        A_ub = np.zeros((E, M))
        for k, (i, j) in enumerate(edges):
            A_ub[k, i] = 1.0
            A_ub[k, j] = 1.0
        
        b_ub = np.ones(E)
        
        # Bounds: 0 <= x_i <= 1
        lb = np.zeros(M)
        ub = np.ones(M)
        
        # Integrality: all variables are binary
        integrality = np.ones(M)
        
        constraint = LinearConstraint(A_ub, ub=b_ub)
        bounds = Bounds(lb, ub)
        
        res = milp(c=c, constraints=[constraint], integrality=integrality, bounds=bounds)
        
        if res.status == 0 and res.x is not None:
            # Extract solution
            solution = res.x
            independent_set_indices = [i for i in range(M) if solution[i] > 0.5]
            
            if independent_set_indices:
                result = [valid_nodes[i] for i in independent_set_indices]
                return result
        
        # Fallback if MILP fails
        raise Exception("MILP failed")
        
    except Exception:
        # Fallback: greedy minimum degree heuristic
        remaining = set(range(M))
        independent_set_indices = []
        
        while remaining:
            if time.time() > deadline:
                break
                
            min_deg_node = None
            min_deg = float('inf')
            
            for u in remaining:
                deg = len(adj[u] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    min_deg_node = u
                    
            if min_deg_node is None:
                break
                
            independent_set_indices.append(min_deg_node)
            remaining.discard(min_deg_node)
            for v in list(adj[min_deg_node]):
                remaining.discard(v)
                
        result = [valid_nodes[i] for i in independent_set_indices]
        
        if not result:
            return [valid_nodes[0]]
            
        return result
