import time
import random
import itertools
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit

    # Try to build graphs of decreasing target size
    for n in [21, 20, 19, 18, 17, 16, 15]:
        if time.time() > deadline - 5:
            break
        
        # Allocate more time for larger n
        time_budget = 120 if n >= 21 else 60
        search_deadline = min(deadline - 2, time.time() + time_budget)
        
        graph = _find_graph(n, search_deadline)
        if graph is not None:
            return graph
            
    # Fallback: return a small valid graph
    return {'n': 5, 'edges': []}

def _find_graph(n, deadline):
    rng = random.Random(42)
    
    max_restarts = 30
    best_graph = None
    best_cost = float('inf')
    
    for restart in range(max_restarts):
        if time.time() > deadline - 1:
            break
            
        # Random initial graph
        p = 0.5
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        current_cost = _cost(adj, n)
        
        if current_cost == 0:
            return _adj_to_edges(adj, n)
            
        T = 1.0
        cooling = 0.9995
        steps = 0
        max_steps = 20000
        
        while steps < max_steps and time.time() < deadline - 1:
            # Find a violation to guide the move
            viol = _find_one_violation(adj, n)
            if viol is None:
                return _adj_to_edges(adj, n)
                
            # Pick a random edge in the violation
            u, v = rng.sample(viol, 2)
            
            has_edge = (adj[u] >> v) & 1
            if has_edge:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
            new_cost = _cost(adj, n)
            delta = new_cost - current_cost
            
            if delta < 0 or rng.random() < math.exp(-delta / T):
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_graph = adj[:]
            else:
                # Revert
                if has_edge:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    
            T *= cooling
            steps += 1
            
            if current_cost == 0:
                return _adj_to_edges(adj, n)
                
    if best_graph is not None and best_cost == 0:
        return _adj_to_edges(best_graph, n)
        
    return None

def _cost(adj, n):
    """
    Count regular induced subgraphs of size >= 6.
    Returns early if count > 50.
    """
    count = 0
    for size in range(6, n + 1):
        for subset in itertools.combinations(range(n), size):
            if _is_regular(adj, subset):
                count += 1
                if count > 50:
                    return count
    return count

def _is_regular(adj, subset):
    """
    Check if induced subgraph on subset is regular.
    """
    mask = 0
    for v in subset:
        mask |= (1 << v)
        
    degs = []
    for v in subset:
        deg = bin(adj[v] & mask).count('1')
        degs.append(deg)
        
    return all(d == degs[0] for d in degs)

def _find_one_violation(adj, n):
    """
    Find one regular induced subgraph of size >= 6.
    """
    for size in range(6, n + 1):
        for subset in itertools.combinations(range(n), size):
            if _is_regular(adj, subset):
                return list(subset)
    return None

def _adj_to_edges(adj, n):
    """
    Convert adjacency list of bitmasks to list of edges.
    """
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                edges.append([i, j])
    return edges
