import random
import time

def construct():
    N = 16
    
    def popcount(x):
        return bin(x).count('1')
        
    def get_clebsch_edges():
        edges = []
        for u in range(N):
            for v in range(u + 1, N):
                if popcount(u ^ v) == 2:
                    edges.append([u, v])
        return edges

    best_edges = get_clebsch_edges()
    best_count = len(best_edges)
    
    def build_bitset_adj(edges):
        adj = [0] * N
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def is_3_colorable(adj):
        color = [-1] * N
        degrees = [bin(x).count('1') for x in adj]
        order = sorted(range(N), key=lambda i: degrees[i], reverse=True)
        
        def backtrack(idx):
            if idx == N:
                return True
            i = order[idx]
            used = [False, False, False]
            for w in range(N):
                if (adj[i] >> w) & 1:
                    if color[w] != -1:
                        used[color[w]] = True
            for c in range(3):
                if not used[c]:
                    color[i] = c
                    if backtrack(idx + 1):
                        return True
                    color[i] = -1
            return False
        
        return backtrack(0)
        
    def creates_triangle(adj, u, v):
        return (adj[u] & adj[v]) != 0
        
    start_time = time.time()
    deadline = start_time + 115
    
    # Multiple restarts
    for restart in range(50):
        if time.time() > deadline:
            break
            
        # Start from Clebsch
        current_edges = get_clebsch_edges()
        current_adj = build_bitset_adj(current_edges)
        
        # Randomize initial state slightly?
        # No, Clebsch is a good start.
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to add edges
            # Shuffle potential non-edges
            non_edges = []
            for u in range(N):
                for v in range(u + 1, N):
                    if not ((current_adj[u] >> v) & 1):
                        non_edges.append((u, v))
                        
            random.shuffle(non_edges)
            
            for u, v in non_edges:
                if time.time() > deadline:
                    break
                    
                if creates_triangle(current_adj, u, v):
                    continue
                    
                # Add
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                
                if not is_3_colorable(current_adj):
                    current_edges.append([u, v])
                    if len(current_edges) > best_count:
                        best_count = len(current_edges)
                        best_edges = current_edges[:]
                    improved = True
                    break
                else:
                    # Rollback
                    current_adj[u] &= ~(1 << v)
                    current_adj[v] &= ~(1 << u)
                    
            if not improved:
                # Try removing an edge to allow new additions
                if current_edges:
                    # Remove random edge
                    idx = random.randint(0, len(current_edges) - 1)
                    u, v = current_edges[idx]
                    current_edges.pop(idx)
                    current_adj[u] &= ~(1 << v)
                    current_adj[v] &= ~(1 << u)
                    # Allow loop to continue to try adding again
                    improved = True # Force another iteration
                    # But limit removals?
                    # If we remove too many, we lose density.
                    # Let's say if we remove 5 times in a row without adding, break restart.
                    # Simple heuristic: if len(current_edges) < best_count - 2, break restart?
                    # No, just let it run.
                    
    return best_edges
