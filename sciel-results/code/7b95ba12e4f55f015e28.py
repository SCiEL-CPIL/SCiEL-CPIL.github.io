import random
import time

def construct():
    n = 40
    start_time = time.time()
    max_time = 145.0  # Safety margin
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    best_edges = []
    best_count = 0
    
    # We can try to seed with the K5 structure to get a head start, or start empty.
    # Starting with K5s gives 80 edges. Greedy from scratch might find a different local optimum.
    # Let's try starting from the K5 seed and adding edges.
    
    def get_seed_edges():
        edges = []
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    edges.append((5 * k + a, 5 * k + b))
        return edges

    seed_edges = get_seed_edges()
    
    def creates_c6(u, v, adj):
        # Check if adding (u, v) creates a 6-cycle
        # Cycle: u - a - b - c - d - v - u
        # a in N(u), d in N(v)
        # Path a - b - c - d in G \ {u, v}
        
        # Optimization: iterate over smaller neighbor set first?
        # Actually, we can just iterate.
        for a in adj[u]:
            if a == v:
                continue
            for b in adj[a]:
                if b == u or b == v or b == a:
                    continue
                for c in adj[b]:
                    if c == u or c == v or c == a or c == b:
                        continue
                    for d in adj[c]:
                        if d == u or d == v or d == a or d == b or d == c:
                            continue
                        if d in adj[v]:
                            return True
        return False

    def try_add_edges(initial_edges, random_seed):
        random.seed(random_seed)
        adj = [set() for _ in range(n)]
        edge_set = set()
        
        # Add initial edges
        for u, v in initial_edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v))
        
        # Generate candidate edges not in initial set
        candidates = [e for e in all_edges if e not in edge_set]
        random.shuffle(candidates)
        
        current_edges = list(initial_edges)
        
        for u, v in candidates:
            if not creates_c6(u, v, adj):
                adj[u].add(v)
                adj[v].add(u)
                edge_set.add((u, v))
                current_edges.append((u, v))
                
        return current_edges

    # Run multiple restarts
    # Start with the seed
    current_best = try_add_edges(seed_edges, 42)
    best_count = len(current_best)
    best_edges = current_best
    
    # Try starting from empty as well, maybe it finds a denser structure
    # But empty start is slower to reach high density.
    # Let's try a few random restarts from the seed, shuffling the candidate order.
    
    num_restarts = 10
    for i in range(num_restarts):
        if time.time() - start_time > max_time:
            break
        # Vary the seed for randomness
        restart_seed = 100 + i
        # We can also try different initial seeds if we had them, but we only have one.
        # Just re-shuffling the candidate edges is the main variation.
        # To diversify, we can also randomly remove a few edges from the seed before adding?
        # Or just rely on the shuffle.
        
        candidate_edges = try_add_edges(seed_edges, restart_seed)
        if len(candidate_edges) > best_count:
            best_count = len(candidate_edges)
            best_edges = candidate_edges
            
    # Try a few runs starting from empty, but limit time
    # Empty start might be slow, but let's try 2-3 quick ones
    for i in range(3):
        if time.time() - start_time > max_time - 10:
            break
        # For empty start, we need to be careful with time.
        # We can use a faster check or just try it.
        # The check is O(Delta^4). Delta starts small, so it's fast initially.
        candidate_edges = try_add_edges([], 1000 + i)
        if len(candidate_edges) > best_count:
            best_count = len(candidate_edges)
            best_edges = candidate_edges

    return [[u, v] for u, v in best_edges]

if __name__ == "__main__":
    # For local testing
    edges = construct()
    print(f"Edges: {len(edges)}")
