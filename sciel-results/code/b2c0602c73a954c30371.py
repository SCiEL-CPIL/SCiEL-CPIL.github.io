import random
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Initial graph: 5 disjoint K4s
    edges = set()
    neighbors = [set() for _ in range(20)]
    
    def add_edge(u, v):
        edges.add((min(u,v), max(u,v)))
        neighbors[u].add(v)
        neighbors[v].add(u)
        
    # Add K4s
    for k in range(5):
        for a in range(4):
            for b in range(a+1, 4):
                add_edge(4*k + a, 4*k + b)
                
    def can_add(u, v):
        if (min(u,v), max(u,v)) in edges:
            return False
        # Check constraints
        # For w in N(v), N(u) and N(w) must share at most 1 neighbor (so that adding v makes it 2)
        # Actually, N(u) does not contain v yet.
        # N(w) might contain v.
        # If v in N(w), then v will be a common neighbor.
        # So current common neighbors + 1 <= 2 => current <= 1.
        for w in neighbors[v]:
            if len(neighbors[u] & neighbors[w]) >= 2:
                return False
        for w in neighbors[u]:
            if len(neighbors[v] & neighbors[w]) >= 2:
                return False
        return True

    def get_edge_count():
        return len(edges)
    
    best_edges = set(edges)
    
    # Greedy improvement
    # Try multiple random orderings
    all_possible = [(i, j) for i in range(20) for j in range(i+1, 20)]
    
    while time.time() < deadline:
        # Reset to best found so far? Or just continue from current?
        # Better to restart from the initial K4s or the best found.
        # Let's restart from the initial K4s to explore different landscapes.
        # Or, keep the best found and try to improve it further?
        # Greedy is monotonic. If we restart, we might find a better basin.
        
        # Let's try a few restarts from the base.
        current_edges = set(best_edges)
        current_neighbors = [set(n) for n in neighbors] # Deep copy?
        # Actually, let's just re-run the greedy from scratch each time with a new shuffle
        # But that discards the previous good edges.
        # Better: Start from base, run greedy. If better than best, update best.
        
        # Re-init
        edges.clear()
        for i in range(20): neighbors[i].clear()
        for k in range(5):
            for a in range(4):
                for b in range(a+1, 4):
                    add_edge(4*k + a, 4*k + b)
                    
        random.shuffle(all_possible)
        
        for u, v in all_possible:
            if time.time() > deadline:
                break
            if can_add(u, v):
                add_edge(u, v)
                
        if len(edges) > len(best_edges):
            best_edges = set(edges)
            # Update neighbors? No, we just need the edge list at the end.
            
    # Convert best_edges to list of lists
    return [[u, v] for u, v in best_edges]
