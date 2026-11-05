import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill
    
    n = 40
    target_edges = 401
    
    # Adjacency as list of bitmasks
    adj = [0] * n
    
    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
    def has_edge(u, v):
        return bool(adj[u] & (1 << v))
        
    def get_book(u, v):
        # Common neighbors of u and v
        return (adj[u] & adj[v]).bit_count()
        
    def compute_max_book(edges):
        # edges is a set of (u, v) with u < v
        max_b = 0
        for u, v in edges:
            b = get_book(u, v)
            if b > max_b:
                max_b = b
        return max_b

    def get_all_edges():
        edges = set()
        for u in range(n):
            mask = adj[u]
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                if u < v:
                    edges.add((u, v))
                mask -= lsb
        return edges

    # Initialization
    # Strategy: Start with a random graph, but bias towards bipartite structure?
    # Or just random.
    random.seed(42)
    all_possible_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    random.shuffle(all_possible_edges)
    initial_edges = all_possible_edges[:target_edges]
    
    for u, v in initial_edges:
        add_edge(u, v)
        
    current_edges = set(initial_edges)
    current_max = compute_max_book(current_edges)
    
    best_edges = set(current_edges)
    best_max = current_max
    
    # Hybrid Greedy + SA
    # Phase 1: Greedy
    # Try to improve by swapping the worst edge with a better non-edge
    greedy_time_limit = start_time + 30.0
    while time.time() < greedy_time_limit:
        # Find worst edge
        max_b = -1
        worst_edge = None
        for u, v in current_edges:
            b = get_book(u, v)
            if b > max_b:
                max_b = b
                worst_edge = (u, v)
                
        if worst_edge is None:
            break
            
        u, v = worst_edge
        
        best_candidate = None
        best_candidate_max = current_max + 100
        
        # Try 20 random non-edges
        for _ in range(20):
            a = random.randint(0, n - 1)
            b = random.randint(0, n - 1)
            if a == b:
                continue
            if a > b:
                a, b = b, a
            if not has_edge(a, b):
                new_edge = (a, b)
                
                # Simulate swap
                remove_edge(u, v)
                add_edge(a, b)
                
                new_edges = set(current_edges)
                new_edges.remove(worst_edge)
                new_edges.add(new_edge)
                
                new_max = compute_max_book(new_edges)
                
                # Revert simulation
                remove_edge(a, b)
                add_edge(u, v)
                
                if new_max < best_candidate_max:
                    best_candidate_max = new_max
                    best_candidate = new_edge
                    
        if best_candidate is not None and best_candidate_max < current_max:
            # Accept
            remove_edge(u, v)
            add_edge(*best_candidate)
            current_edges.remove(worst_edge)
            current_edges.add(best_candidate)
            current_max = best_candidate_max
            if current_max < best_max:
                best_max = current_max
                best_edges = set(current_edges)
        else:
            # No greedy improvement found, break to SA
            break
            
    # Phase 2: Simulated Annealing
    temp = 5.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Find worst edge
        max_b = -1
        worst_edge = None
        for u, v in current_edges:
            b = get_book(u, v)
            if b > max_b:
                max_b = b
                worst_edge = (u, v)
                
        if worst_edge is None:
            break
            
        u, v = worst_edge
        
        # Generate candidate non-edge
        found_new_edge = False
        for _ in range(20):
            a = random.randint(0, n - 1)
            b = random.randint(0, n - 1)
            if a == b:
                continue
            if a > b:
                a, b = b, a
            if not has_edge(a, b):
                new_edge = (a, b)
                found_new_edge = True
                break
                
        if not found_new_edge:
            break
            
        # Perform swap
        remove_edge(u, v)
        add_edge(*new_edge)
        
        new_edges = set(current_edges)
        new_edges.remove(worst_edge)
        new_edges.add(new_edge)
        
        new_max = compute_max_book(new_edges)
        
        delta = new_max - current_max
        
        if delta <= 0 or random.random() < pow(2.718, -delta / temp):
            current_edges = new_edges
            current_max = new_max
            if new_max < best_max:
                best_max = new_max
                best_edges = set(new_edges)
        else:
            # Revert
            remove_edge(*new_edge)
            add_edge(u, v)
            
        temp *= cooling
        
    # Convert best_edges to list of lists
    return [[u, v] for u, v in best_edges]
