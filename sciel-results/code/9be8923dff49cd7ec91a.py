import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s
    
    n = 15
    best_edges = None
    best_count = 0
    
    def get_degrees(adj):
        degs = [0] * n
        for u in range(n):
            degs[u] = adj[u].bit_count()
        return degs
    
    def is_valid(adj):
        degs = get_degrees(adj)
        # Group vertices by degree
        deg_groups = defaultdict(list)
        for v in range(n):
            deg_groups[degs[v]].append(v)
        
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] >> v) & 1:
                    continue
                # Check edge (u,v)
                # N(u)\{v} and N(v)\{u}
                nu_mask = adj[u] & ~(1 << v)
                nv_mask = adj[v] & ~(1 << u)
                
                # Get degrees of neighbors
                nu_degs = set()
                nm = nu_mask
                while nm:
                    s = nm & (-nm).bit_length() - 1
                    nm &= nm - 1
                    nu_degs.add(degs[s])
                
                nv_degs = set()
                nm = nv_mask
                while nm:
                    t = nm & (-nm).bit_length() - 1
                    nm &= nm - 1
                    nv_degs.add(degs[t])
                
                common = nu_degs & nv_degs
                for d in common:
                    # Check if there exist s in N(u)\{v} with deg d, t in N(v)\{u} with deg d, s != t
                    s_list = [s for s in range(n) if (nu_mask >> s) & 1 and degs[s] == d]
                    t_list = [t for t in range(n) if (nv_mask >> t) & 1 and degs[t] == d]
                    if len(s_list) == 1 and len(t_list) == 1 and s_list[0] == t_list[0]:
                        continue
                    return False
        return True
    
    def edge_count(adj):
        return sum(adj[u].bit_count() for u in range(n)) // 2
    
    def random_start():
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < 0.5:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj
    
    # Try several random restarts
    random.seed(42)
    num_restarts = 50
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        adj = random_start()
        
        # Local search: try adding edges
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding each missing edge
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj[u] >> v) & 1:
                        continue
                    # Try adding edge (u,v)
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    if is_valid(adj):
                        new_count = edge_count(adj)
                        if best_edges is None or new_count > best_count:
                            best_edges = [adj[i] for i in range(n)]
                            best_count = new_count
                        improved = True
                    else:
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)
        
        # Try removing edges from current best to allow more additions
        if best_edges is not None:
            adj = [best_edges[i] for i in range(n)]
            # Try removing one edge then re-adding
            for u in range(n):
                for v in range(u + 1, n):
                    if time.time() > deadline:
                        break
                    if not ((adj[u] >> v) & 1):
                        continue
                    # Remove edge (u,v)
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                    if is_valid(adj):
                        # Now try adding other edges
                        for u2 in range(n):
                            for v2 in range(u2 + 1, n):
                                if time.time() > deadline:
                                    break
                                if (adj[u2] >> v2) & 1:
                                    continue
                                adj[u2] |= (1 << v2)
                                adj[v2] |= (1 << u2)
                                if is_valid(adj):
                                    new_count = edge_count(adj)
                                    if new_count > best_count:
                                        best_edges = [adj[i] for i in range(n)]
                                        best_count = new_count
                                else:
                                    adj[u2] ^= (1 << v2)
                                    adj[v2] ^= (1 << u2)
                    # Restore edge
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
    
    # Convert best to edge list
    if best_edges is None:
        # Fallback: star
        return [[0, v] for v in range(1, n)]
    
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_edges[u] >> v) & 1:
                edges.append([int(u), int(v)])
    
    return edges
