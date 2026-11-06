import random
import time

def construct():
    # Try to find a graph on n=40 with alpha < 10
    # If fails, n=39, etc.
    
    def greedy_alpha(adj, n, iters=20):
        min_a = n
        for _ in range(iters):
            rem = set(range(n))
            ind = 0
            # Randomized greedy
            while rem:
                # Pick random vertex from rem
                v = random.choice(list(rem))
                ind += 1
                rem -= adj[v]
                rem.discard(v)
            if ind < min_a:
                min_a = ind
        return min_a

    def has_independent_set_10(adj, n):
        # Backtracking to find independent set of size 10
        # Prune if len(rem) + current < 10
        def bt(rem, curr):
            if curr == 10:
                return True
            if len(rem) + curr < 10:
                return False
            # Heuristic: pick vertex with min degree in rem
            # Or just first
            v = next(iter(rem))
            rem.discard(v)
            if bt(rem - adj[v], curr + 1):
                return True
            if bt(rem, curr):
                return True
            return False
        return bt(set(range(n)), 0)

    for n in [40, 39]:
        start = time.time()
        # Init: Maximal triangle-free graph
        adj = [set() for _ in range(n)]
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(edges)
        for u, v in edges:
            if not (adj[u] & adj[v]):
                adj[u].add(v)
                adj[v].add(u)
        
        best_alpha = greedy_alpha(adj, n, 50)
        
        # Local Search
        for i in range(2000):
            if time.time() - start > 50:
                break
            
            # Try to add an edge
            # Sample 100 pairs
            improved = False
            for _ in range(100):
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v or v in adj[u]:
                    continue
                if adj[u] & adj[v]:
                    continue
                
                # Add edge
                adj[u].add(v)
                adj[v].add(u)
                new_alpha = greedy_alpha(adj, n, 20)
                
                if new_alpha <= best_alpha:
                    best_alpha = new_alpha
                    improved = True
                else:
                    adj[u].remove(v)
                    adj[v].remove(u)
                
                if improved:
                    break
            
            if not improved:
                # Try to remove an edge to allow new additions?
                # Or just restart?
                # Let's just break and restart
                break
        
        if best_alpha < 10:
            if not has_independent_set_10(adj, n):
                edges_list = [[u, v] for u in range(n) for v in range(u+1, n) if v in adj[u]]
                return {'n': n, 'edges': edges_list}
        
        # If 40 fails, try 39 in next loop iteration
        # But we need to return something.
        
    # Fallback to A9
    n = 26
    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}

