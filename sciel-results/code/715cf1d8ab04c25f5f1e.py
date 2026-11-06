import random
import math

def construct():
    n = 16
    random.seed(42)
    
    # Helper: 3-Colorability check
    def is_3_colorable(adj):
        colors = [-1] * n
        order = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            used = set()
            for u in adj[v]:
                if colors[u] != -1:
                    used.add(colors[u])
            
            for c in range(3):
                if c not in used:
                    colors[v] = c
                    if backtrack(idx + 1):
                        return True
                    colors[v] = -1
            return False
        
        return backtrack(0)

    # Initialize with Grötzsch Graph (Vertices 0-10)
    # C5: 0-1-2-3-4-0
    # Copies: 5-9 (5-0, 6-1, 7-2, 8-3, 9-4)
    # Root: 10 connected to 5-9
    # Edges:
    # C5: (0,1), (1,2), (2,3), (3,4), (4,0)
    # Mycielski edges:
    # 5 connected to 1, 4 (neighbors of 0)
    # 6 connected to 2, 0
    # 7 connected to 3, 1
    # 8 connected to 4, 2
    # 9 connected to 0, 3
    # 10 connected to 5, 6, 7, 8, 9
    
    edge_set = set()
    adj = {i: set() for i in range(n)}
    
    def add_edge(u, v):
        if u == v: return
        if (u, v) in edge_set or (v, u) in edge_set: return
        # Check triangle
        if adj[u] & adj[v]:
            return False
        edge_set.add((u, v))
        edge_set.add((v, u))
        adj[u].add(v)
        adj[v].add(u)
        return True
        
    # Note: The add_edge above checks triangle. 
    # For Grötzsch, we know it's triangle free.
    # Let's just build it directly.
    
    c5_edges = [(0,1), (1,2), (2,3), (3,4), (4,0)]
    myc_edges = [(5,1), (5,4), (6,2), (6,0), (7,3), (7,1), (8,4), (8,2), (9,0), (9,3)]
    root_edges = [(10,5), (10,6), (10,7), (10,8), (10,9)]
    
    all_edges = c5_edges + myc_edges + root_edges
    
    for u, v in all_edges:
        add_edge(u, v)
        
    # Add vertices 11-15
    # Connect them to random independent sets initially
    for v in range(11, 16):
        # Find a large independent set in current graph
        # Simple greedy
        iset = set()
        nodes = list(range(v))
        random.shuffle(nodes)
        for u in nodes:
            if not (adj[u] & iset):
                iset.add(u)
        for u in iset:
            add_edge(v, u)
            
    # Simulated Annealing
    best_edges = list(edge_set)
    best_score = len(edge_set) if not is_3_colorable(adj) else -1000
    
    current_es = set(edge_set)
    current_ad = {i: set() for i in range(n)}
    for e in current_es:
        u, v = e
        current_ad[u].add(v)
        current_ad[v].add(u)
        
    def get_fitness(es, ad):
        if is_3_colorable(ad):
            return -1000 + len(es)
        return len(es)
        
    current_fitness = get_fitness(current_es, current_ad)
    
    T = 1.0
    cooling = 0.99
    iterations = 10000
    
    for i in range(iterations):
        if T < 0.01:
            break
            
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v:
            continue
            
        if (u, v) in current_es:
            # Remove
            new_es = current_es - {(u, v), (v, u)}
            new_ad = {k: set(val) for k, val in current_ad.items()}
            new_ad[u].remove(v)
            new_ad[v].remove(u)
        else:
            # Add
            if current_ad[u] & current_ad[v]:
                continue
            new_es = current_es | {(u, v), (v, u)}
            new_ad = {k: set(val) for k, val in current_ad.items()}
            new_ad[u].add(v)
            new_ad[v].add(u)
            
        new_fitness = get_fitness(new_es, new_ad)
        delta = new_fitness - current_fitness
        
        if delta > 0 or random.random() < math.exp(delta / T):
            current_es = new_es
            current_ad = new_ad
            current_fitness = new_fitness
            
            if current_fitness > best_score:
                best_score = current_fitness
                best_edges = list(current_es)
                
        T *= cooling
        
    # Format output
    result = []
    seen = set()
    for u, v in best_edges:
        if (u, v) not in seen and (v, u) not in seen:
            result.append([u, v])
            seen.add((u, v))
            seen.add((v, u))
            
    return result
