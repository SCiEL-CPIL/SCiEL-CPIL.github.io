import time
import random
import math

def _get_penalty(adj, n, min_deg):
    c4_count = 0
    for u in range(n):
        for v in range(u + 1, n):
            common = len(adj[u] & adj[v])
            if common >= 2:
                c4_count += common * (common - 1) // 2
    c4_count //= 2
    
    deg_violation = 0
    for i in range(n):
        if len(adj[i]) < min_deg:
            deg_violation += min_deg - len(adj[i])
            
    return c4_count * 100 + deg_violation

def _construct_local_search(n, min_deg, time_limit):
    start_time = time.time()
    
    adj = [set() for _ in range(n)]
    
    # Initial random graph
    # Target degree min_deg.
    # Random edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_edges)
    
    # Start with approx min_deg * n / 2 edges
    target_edges = (n * min_deg) // 2
    # Add some extra to allow removal
    initial_edges = min(len(all_edges), int(target_edges * 1.2))
    
    for u, v in all_edges[:initial_edges]:
        adj[u].add(v)
        adj[v].add(u)
        
    current_penalty = _get_penalty(adj, n, min_deg)
    
    T = 5.0
    cooling_rate = 0.999
    
    while time.time() - start_time < time_limit:
        if current_penalty == 0:
            edge_list = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edge_list.append([u, v])
            return {'n': n, 'edges': edge_list}
            
        # Pick random edge
        u, v = random.choice(all_edges)
        
        # Flip
        if v in adj[u]:
            adj[u].remove(v)
            adj[v].remove(u)
        else:
            adj[u].add(v)
            adj[v].add(u)
            
        new_penalty = _get_penalty(adj, n, min_deg)
        delta = new_penalty - current_penalty
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_penalty = new_penalty
        else:
            # Revert
            if v in adj[u]:
                adj[u].remove(v)
                adj[v].remove(u)
            else:
                adj[u].add(v)
                adj[v].add(u)
                
        T *= cooling_rate
        if T < 0.01:
            T = 0.01
            
    return None

def construct():
    start_time = time.time()
    deadline = 115.0
    
    for n in range(46, 38, -1):
        if time.time() - start_time > deadline:
            break
            
        remaining = deadline - (time.time() - start_time)
        if remaining < 2:
            break
            
        min_deg = n - 39
        if min_deg < 1:
            min_deg = 1
            
        t_limit = min(remaining - 1, 30.0) # Give 30s per n
        
        result = _construct_local_search(n, min_deg, t_limit)
        if result is not None:
            return result
            
    return {'n': 39, 'edges': []}
