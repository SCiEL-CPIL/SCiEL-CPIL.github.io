import time
import sys

sys.setrecursionlimit(10000)

N = 20
M = N * (N - 1) // 2
ALL_EDGES = [(u, v) for u in range(N) for v in range(u + 1, N)]

def construct():
    best_count = 0
    best_edges = []
    start_time = time.time()
    time_limit = 115  # Leave 5s buffer

    # Initial greedy to get a good lower bound
    def greedy_init():
        adj = [0] * N
        common = [[0] * N for _ in range(N)]
        edges = []
        
        # Try a few random orderings
        best_local = 0
        best_local_edges = []
        
        for _ in range(50):
            adj = [0] * N
            common = [[0] * N for _ in range(N)]
            edges = []
            shuffled = ALL_EDGES[:]
            import random
            random.shuffle(shuffled)
            
            for u, v in shuffled:
                if can_add(adj, common, u, v):
                    add_edge(adj, common, u, v)
                    edges.append((u, v))
                    
            if len(edges) > best_local:
                best_local = len(edges)
                best_local_edges = edges[:]
                
        return best_local, best_local_edges

    def can_add(adj, common, u, v):
        if common[u][v] > 2:
            return False
        
        mask_v = adj[v]
        while mask_v:
            lsb = mask_v & -mask_v
            w = lsb.bit_length() - 1
            if common[u][w] > 1:
                return False
            mask_v ^= lsb
            
        mask_u = adj[u]
        while mask_u:
            lsb = mask_u & -mask_u
            w = lsb.bit_length() - 1
            if common[v][w] > 1:
                return False
            mask_u ^= lsb
            
        return True

    def add_edge(adj, common, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        mask_u = adj[u] & ~(1 << v)
        while mask_u:
            lsb = mask_u & -mask_u
            w = lsb.bit_length() - 1
            common[v][w] += 1
            common[w][v] += 1
            mask_u ^= lsb
            
        mask_v = adj[v] & ~(1 << u)
        while mask_v:
            lsb = mask_v & -mask_v
            w = lsb.bit_length() - 1
            common[u][w] += 1
            common[w][u] += 1
            mask_v ^= lsb

    def remove_edge(adj, common, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
        mask_u = adj[u]
        while mask_u:
            lsb = mask_u & -mask_u
            w = lsb.bit_length() - 1
            common[v][w] -= 1
            common[w][v] -= 1
            mask_u ^= lsb
            
        mask_v = adj[v]
        while mask_v:
            lsb = mask_v & -mask_v
            w = lsb.bit_length() - 1
            common[u][w] -= 1
            common[w][u] -= 1
            mask_v ^= lsb

    # Get initial solution
    init_count, init_edges = greedy_init()
    best_count = init_count
    best_edges = init_edges
    
    # B&B
    adj = [0] * N
    common = [[0] * N for _ in range(N)]
    current_edges = []
    
    # Sort edges to improve pruning? 
    # Edges that are "harder" to place should be considered later? Or earlier?
    # Usually, variables with high impact first.
    # Let's just use the standard order for now, or shuffle if we want.
    # Standard order is fine.
    
    def solve(idx, count):
        nonlocal best_count, best_edges, adj, common, current_edges
        
        if time.time() - start_time > time_limit:
            return
            
        if count + (M - idx) <= best_count:
            return
            
        if idx == M:
            if count > best_count:
                best_count = count
                best_edges = current_edges[:]
            return
            
        u, v = ALL_EDGES[idx]
        
        # Try adding edge (u, v)
        if can_add(adj, common, u, v):
            add_edge(adj, common, u, v)
            current_edges.append((u, v))
            solve(idx + 1, count + 1)
            current_edges.pop()
            remove_edge(adj, common, u, v)
            
        # Try skipping edge (u, v)
        solve(idx + 1, count)

    solve(0, 0)
    
    return [[u, v] for u, v in best_edges]
