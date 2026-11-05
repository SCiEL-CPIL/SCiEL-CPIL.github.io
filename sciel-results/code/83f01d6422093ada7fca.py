import random
import time
import itertools

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_edges = []
    best_count = 0
    
    def has_c6(adj, u, v):
        # Check if adding edge (u,v) creates a C6
        # Path: u - a - b - c - d - v
        # a in N(u) \ {v}
        # b in N(a) \ {u, v, a}
        # c in N(b) \ {a, u, v, b}
        # d in N(c) \ {b, u, v, c}
        # Check if (d, v) in E
        
        # Optimization: Pre-compute neighbors
        # We assume adj is a list of sets or lists of neighbors
        
        # If u and v are already connected, we are checking if there is ANOTHER path?
        # No, we are checking if adding (u,v) to a graph NOT containing (u,v) creates a C6.
        # If (u,v) is already in E, we don't call this.
        
        # Fast fail: if no common neighbors? No, C6 length 5 path.
        
        # Iterate a
        for a in adj[u]:
            if a == v:
                continue
            # Iterate b
            for b in adj[a]:
                if b == u or b == v or b == a:
                    continue
                # Iterate c
                for c in adj[b]:
                    if c == a or c == u or c == v or c == b:
                        continue
                    # Iterate d
                    for d in adj[c]:
                        if d == b or d == u or d == v or d == c:
                            continue
                        if d in adj[v]:
                            return True
        return False

    def try_add_edge(adj, u, v):
        if u == v:
            return False
        if v in adj[u]:
            return True # Already there
        if has_c6(adj, u, v):
            return False
        adj[u].add(v)
        adj[v].add(u)
        return True

    def count_edges(adj):
        return sum(len(adj[i]) for i in range(N)) // 2

    def local_search(adj, max_iters=1000):
        # Try to improve by swapping edges
        # Remove one edge, add two? Or just try to add blocked edges by removing conflicts?
        # Simpler: Random edge removal and re-adding
        improved = True
        while improved and (time.time() < deadline):
            improved = False
            # Try all missing edges
            for u in range(N):
                for v in range(u + 1, N):
                    if v not in adj[u]:
                        if try_add_edge(adj, u, v):
                            improved = True
                            # Restart loop to check for new opportunities
                            # break # Don't break, keep trying to add more
            if improved:
                continue
            
            # If no edge can be added, try a swap: remove e1, add e2, e3
            # This is expensive. Let's try a simpler random perturbation.
            # Pick a random edge to remove, then try to add as many as possible.
            if random.random() < 0.5: # 50% chance to perturb
                edges = [(u, v) for u in range(N) for v in range(u + 1, N) if v in adj[u]]
                if not edges:
                    break
                e_to_remove = random.choice(edges)
                u, v = e_to_remove
                adj[u].remove(v)
                adj[v].remove(u)
                improved = True # Force another round of greedy add
                # Also try to add the removed edge back later?
                # The greedy loop above will try to add it back if it's valid.
                # But since it was removed because it was blocked or to make space, 
                # maybe it's still blocked.
                # We rely on the greedy phase to fill up.
        
    def run_greedy():
        adj = [set() for _ in range(N)]
        # Start with a random sparse graph to break symmetry
        # Or start empty
        # Let's start with a random matching or just empty
        # Random initialization can help escape local optima
        num_initial = random.randint(0, 20)
        for _ in range(num_initial):
            u, v = random.sample(range(N), 2)
            if v not in adj[u]:
                if not has_c6(adj, u, v):
                    adj[u].add(v)
                    adj[v].add(u)
        
        # Greedy fill
        # Order edges randomly
        all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if v not in adj[u]:
                if try_add_edge(adj, u, v):
                    pass
        
        # Local search phase
        local_search(adj)
        
        return count_edges(adj), adj

    # Multiple restarts
    while time.time() < deadline:
        count, adj = run_greedy()
        if count > best_count:
            best_count = count
            # Convert adj to list of edges
            best_edges = [(u, v) for u in range(N) for v in range(u + 1, N) if v in adj[u]]
            
        # If we found 105 or more, we can stop early?
        # The task says "Maximize". 105 is best known.
        # If we find 106, that's a discovery.
        # If we find 105, we can keep searching for 106.
        # But 105 is the target for score 1.125+.
        # Let's keep going.

    # Return the best edges found
    return best_edges
