import time
import random
import math

N = 15
DEADLINE = 85.0

def _is_valid(adj, n):
    deg = [len(adj[i]) for i in range(n)]
    for u in range(n):
        for v in adj[u]:
            if u < v:
                for s in adj[u]:
                    if s == v:
                        continue
                    for t in adj[v]:
                        if t == u:
                            continue
                        if s == t:
                            continue
                        if deg[s] == deg[t]:
                            return False
    return True

def _edge_count(adj, n):
    return sum(len(adj[i]) for i in range(n)) // 2

def _is_threshold_graph(adj, n):
    """Check if graph is a threshold graph by verifying the degree sequence
    satisfies the threshold graph characterization."""
    deg = sorted([len(adj[i]) for i in range(n)], reverse=True)
    # For threshold graphs, the degree sequence must satisfy:
    # For each vertex, its neighbors form a prefix in the degree ordering
    # We check: if we order by degree, vertex i should be adjacent to
    # all vertices with degree > deg[i] (except itself)
    order = sorted(range(n), key=lambda i: -len(adj[i]))
    for pos, v in enumerate(order):
        for pos2, w in enumerate(order):
            if pos2 == pos:
                continue
            if len(adj[w]) > len(adj[v]):
                # w has higher degree, should be adjacent to v
                if w not in adj[v]:
                    return False
            elif len(adj[w]) < len(adj[v]):
                # w has lower degree, may or may not be adjacent
                pass
            else:
                # Same degree - in threshold graph, vertices with same degree
                # are either both adjacent to all higher-degree or neither
                # This is automatically satisfied if the above holds
                pass
    return True

def construct():
    start = time.time()
    
    # Start with K_{7,8} as baseline
    adj = [set() for _ in range(N)]
    for i in range(7):
        for j in range(7, N):
            adj[i].add(j)
            adj[j].add(i)
            
    best_adj = [set(s) for s in adj]
    best_count = _edge_count(best_adj, N)
    
    rng = random.Random(42)
    
    # Try to build a threshold graph with more edges
    # A threshold graph on n vertices has at most n*(n-1)/2 edges
    # but must satisfy the threshold property
    
    # Generate random threshold graphs and check validity
    # Threshold graph: build by adding vertices one at a time,
    # each new vertex is either isolated or connected to all previous
    
    best_threshold_adj = None
    best_threshold_count = 0
    
    for trial in range(2000):
        if time.time() > DEADLINE - 10:
            break
            
        # Generate a random threshold graph
        # Build sequence: '0' = isolated, '1' = dominating
        seq = []
        for i in range(N):
            # Bias towards more '1's for denser graphs
            if rng.random() < 0.6:
                seq.append(1)
            else:
                seq.append(0)
        
        # Build the graph from this sequence
        t_adj = [set() for _ in range(N)]
        for i in range(N):
            if seq[i] == 1:
                # Connect to all previous vertices
                for j in range(i):
                    t_adj[i].add(j)
                    t_adj[j].add(i)
        
        if _is_valid(t_adj, N):
            cnt = _edge_count(t_adj, N)
            if cnt > best_threshold_count:
                best_threshold_count = cnt
                best_threshold_adj = [set(s) for s in t_adj]
    
    # Use the better of bipartite and threshold
    if best_threshold_adj is not None and best_threshold_count > best_count:
        best_adj = best_threshold_adj
        best_count = best_threshold_count
    
    # Local search from best_adj to add more edges while maintaining validity
    current_adj = [set(s) for s in best_adj]
    current_count = best_count
    
    for step in range(50000):
        if time.time() > DEADLINE - 5:
            break
            
        i = rng.randint(0, N - 1)
        j = rng.randint(0, N - 1)
        if i == j:
            continue
            
        if j in current_adj[i]:
            # Remove
            current_adj[i].discard(j)
            current_adj[j].discard(i)
        else:
            # Add
            current_adj[i].add(j)
            current_adj[j].add(i)
            
        if _is_valid(current_adj, N):
            new_count = _edge_count(current_adj, N)
            if new_count > best_count:
                best_count = new_count
                best_adj = [set(s) for s in current_adj]
        else:
            # Revert
            if j in current_adj[i]:
                current_adj[i].discard(j)
                current_adj[j].discard(i)
            else:
                current_adj[i].add(j)
                current_adj[j].add(i)
                
    # Final check
    if not _is_valid(best_adj, N):
        # Fallback to K_{7,8}
        best_adj = [set() for _ in range(N)]
        for i in range(7):
            for j in range(7, N):
                best_adj[i].add(j)
                best_adj[j].add(i)
        best_count = 56

    edges = []
    for i in range(N):
        for j in best_adj[i]:
            if i < j:
                edges.append([int(i), int(j)])
                
    return edges
