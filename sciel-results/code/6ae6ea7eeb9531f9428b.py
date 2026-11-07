import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
N = 15

def _is_valid(adj, degrees):
    """
    Check if the graph represented by adjacency sets is valid.
    Valid means: for every simple path s-u-v-t (s!=t), deg(s) != deg(t).
    This is equivalent to: no two vertices at distance 2 have the same degree.
    """
    for u in range(N):
        for v in adj[u]:
            if v <= u:
                continue
            # s is a neighbor of u, t is a neighbor of v
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u:
                        continue
                    if s == t:
                        continue
                    if degrees[s] == degrees[t]:
                        return False
    return True

def _count_edges(adj):
    return sum(len(a) for a in adj) // 2

def _compute_degrees(adj):
    return [len(a) for a in adj]

def _try_add_edge(adj, degrees, u, v):
    """Try to add edge (u,v). Return True if valid after addition."""
    if v in adj[u]:
        return True
    # Tentatively add
    adj[u].add(v)
    adj[v].add(u)
    degrees[u] += 1
    degrees[v] += 1
    
    if _is_valid(adj, degrees):
        return True
    else:
        # Revert
        adj[u].discard(v)
        adj[v].discard(u)
        degrees[u] -= 1
        degrees[v] -= 1
        return False

def _greedy_fill(adj, degrees, order):
    """Add edges in the given order if possible."""
    for (u, v) in order:
        if v not in adj[u]:
            _try_add_edge(adj, degrees, u, v)

def _construct_bipartite_baseline():
    """K_{7,8} bipartite graph: 7*8 = 56 edges."""
    adj = [set() for _ in range(N)]
    degrees = [0] * N
    # Partition: 0-6 in A, 7-14 in B
    for a in range(7):
        for b in range(7, 15):
            adj[a].add(b)
            adj[b].add(a)
    for a in range(7):
        degrees[a] = 8
    for b in range(7, 15):
        degrees[b] = 7
    return adj, degrees

def _local_search(adj, degrees, time_limit):
    """
    Local search: try adding edges, and if stuck, remove a random edge and re-greedy.
    """
    start = time.time()
    best_adj = [set(a) for a in adj]
    best_degrees = list(degrees)
    best_count = _count_edges(adj)
    
    current_adj = [set(a) for a in adj]
    current_degrees = list(degrees)
    current_count = _count_edges(adj)
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(N) for v in range(u+1, N)]
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() - start < time_limit:
        improved = False
        
        # Try to add any missing edge
        for (u, v) in all_edges:
            if v not in current_adj[u]:
                if _try_add_edge(current_adj, current_degrees, u, v):
                    current_count += 1
                    improved = True
                    if current_count > best_count:
                        best_count = current_count
                        best_adj = [set(a) for a in current_adj]
                        best_degrees = list(current_degrees)
        
        if improved:
            no_improve_count = 0
            continue
        
        # No edge could be added. Try removing a random edge and re-filling.
        # Remove a random existing edge
        existing_edges = [(u, v) for u in range(N) for v in current_adj[u] if v > u]
        if not existing_edges:
            break
        
        # Try removing several random edges
        for _ in range(5):
            u, v = random.choice(existing_edges)
            if v in current_adj[u]:
                current_adj[u].discard(v)
                current_adj[v].discard(u)
                current_degrees[u] -= 1
                current_degrees[v] -= 1
                current_count -= 1
                
                # Re-greedy fill
                random.shuffle(all_edges)
                for (au, av) in all_edges:
                    if av not in current_adj[au]:
                        if _try_add_edge(current_adj, current_degrees, au, av):
                            current_count += 1
                
                if current_count > best_count:
                    best_count = current_count
                    best_adj = [set(a) for a in current_adj]
                    best_degrees = list(current_degrees)
                
                # Restore the removed edge if it helps
                if v not in current_adj[u]:
                    _try_add_edge(current_adj, current_degrees, u, v)
                    if v in current_adj[u]:
                        current_count += 1
                
                if current_count > best_count:
                    best_count = current_count
                    best_adj = [set(a) for a in current_adj]
                    best_degrees = list(current_degrees)
    
    return best_adj, best_degrees, best_count

def _random_restart_search(time_limit):
    """
    Multiple random restarts with greedy + local search.
    """
    start = time.time()
    best_adj = None
    best_degrees = None
    best_count = 0
    
    # Start with bipartite baseline
    adj, degrees = _construct_bipartite_baseline()
    count = _count_edges(adj)
    if count > best_count:
        best_adj = [set(a) for a in adj]
        best_degrees = list(degrees)
        best_count = count
    
    # Local search from baseline
    adj_ls, deg_ls, count_ls = _local_search(adj, degrees, time_limit - (time.time() - start))
    if count_ls > best_count:
        best_adj = adj_ls
        best_degrees = deg_ls
        best_count = count_ls
    
    # Random restarts
    elapsed = time.time() - start
    remaining = time_limit - elapsed
    if remaining > 1:
        for _ in range(20):
            if time.time() - start > time_limit - 1:
                break
            # Start from a random subset of edges
            adj_r = [set() for _ in range(N)]
            deg_r = [0] * N
            all_edges = [(u, v) for u in range(N) for v in range(u+1, N)]
            random.shuffle(all_edges)
            for (u, v) in all_edges:
                if random.random() < 0.3:
                    _try_add_edge(adj_r, deg_r, u, v)
            
            # Greedy fill
            random.shuffle(all_edges)
            for (u, v) in all_edges:
                if v not in adj_r[u]:
                    _try_add_edge(adj_r, deg_r, u, v)
            
            count_r = _count_edges(adj_r)
            
            # Local search
            adj_ls, deg_ls, count_ls = _local_search(adj_r, deg_r, min(remaining - (time.time() - start), 2))
            if count_ls > best_count:
                best_adj = adj_ls
                best_degrees = deg_ls
                best_count = count_ls
    
    return best_adj, best_degrees, best_count

def construct():
    start = time.time()
    time_limit = 85  # Safety margin
    
    adj, degrees, count = _random_restart_search(time_limit)
    
    # Final verification
    if adj is None:
        adj, degrees = _construct_bipartite_baseline()
    
    # Convert to list of [u, v]
    edges = []
    for u in range(N):
        for v in adj[u]:
            if u < v:
                edges.append([int(u), int(v)])
    
    return edges
# EVOLVE-BLOCK-END
