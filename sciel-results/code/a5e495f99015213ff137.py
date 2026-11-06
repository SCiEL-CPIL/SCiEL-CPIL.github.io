import time
import random

N = 20
DEADLINE_FRACTION = 0.95

def construct():
    start_time = time.time()
    deadline = start_time + 120 * DEADLINE_FRACTION

    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    def can_add(adj, u, v):
        if adj[u] & (1 << v):
            return False
        
        # Case 1: v becomes a common neighbor of {u, a, b}
        # a, b must be neighbors of v.
        # Check common neighbors of u, a, b.
        for a in range(N):
            if a == u or a == v:
                continue
            if not (adj[a] & (1 << v)):
                continue
            for b in range(a + 1, N):
                if b == u or b == v:
                    continue
                if not (adj[b] & (1 << v)):
                    continue
                common = adj[u] & adj[a] & adj[b]
                if common.bit_count() > 1:
                    return False
        
        # Case 2: u becomes a common neighbor of {v, a, b}
        # a, b must be neighbors of u.
        for a in range(N):
            if a == u or a == v:
                continue
            if not (adj[a] & (1 << u)):
                continue
            for b in range(a + 1, N):
                if b == u or b == v:
                    continue
                if not (adj[b] & (1 << u)):
                    continue
                common = adj[v] & adj[a] & adj[b]
                if common.bit_count() > 1:
                    return False
        
        return True

    def get_edge_list(adj):
        edges = []
        for i in range(N):
            for j in range(i + 1, N):
                if adj[i] & (1 << j):
                    edges.append((i, j))
        return edges

    def greedy_fill(adj, time_limit=None):
        missing = [(i, j) for i in range(N) for j in range(i + 1, N) if not (adj[i] & (1 << j))]
        random.shuffle(missing)
        
        for u, v in missing:
            if time_limit and time.time() > time_limit:
                break
            if can_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def find_k33(adj):
        """
        Finds a K3,3 in the graph.
        Returns (u, v, w, x, y, z) such that u,v,w have common neighbors x,y,z.
        Returns None if K3,3-free.
        """
        for u in range(N):
            for v in range(u + 1, N):
                for w in range(v + 1, N):
                    common = adj[u] & adj[v] & adj[w]
                    if common.bit_count() >= 3:
                        # Extract 3 neighbors
                        neighbors = []
                        for i in range(N):
                            if (common >> i) & 1:
                                neighbors.append(i)
                                if len(neighbors) == 3:
                                    break
                        if len(neighbors) == 3:
                            return (u, v, w, neighbors[0], neighbors[1], neighbors[2])
        return None

    def prune_k33(adj, time_limit=None):
        """
        Removes edges to destroy all K3,3s.
        """
        while time_limit is None or time.time() < time_limit:
            k33 = find_k33(adj)
            if not k33:
                break
            u, v, w, x, y, z = k33
            # Edges in this K3,3
            candidates = [
                (u, x), (u, y), (u, z),
                (v, x), (v, y), (v, z),
                (w, x), (w, y), (w, z)
            ]
            # Heuristic: Remove edge with max sum of degrees
            best_edge = None
            best_score = -1
            for a, b in candidates:
                if not (adj[a] & (1 << b)):
                    continue
                score = adj[a].bit_count() + adj[b].bit_count()
                if score > best_score:
                    best_score = score
                    best_edge = (a, b)
            
            if best_edge:
                a, b = best_edge
                adj[a] &= ~(1 << b)
                adj[b] &= ~(1 << a)
            else:
                # Should not happen
                break
                
        return adj

    def local_search(adj, max_time=1.0):
        ls_start = time.time()
        current_edges = sum(x.bit_count() for x in adj) // 2
        best_adj = adj[:]
        best_edges = current_edges
        
        adj = greedy_fill(adj, deadline)
        current_edges = sum(x.bit_count() for x in adj) // 2
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = adj[:]
            
        missing = [(i, j) for i in range(N) for j in range(i + 1, N) if not (adj[i] & (1 << j))]
        
        no_improve_count = 0
        max_no_improve = 100
        
        while time.time() - ls_start < max_time and no_improve_count < max_no_improve:
            if time.time() > deadline - 2:
                break
                
            if not missing:
                break
                
            idx = random.randint(0, len(missing) - 1)
            u, v = missing[idx]
            
            if can_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges += 1
                missing.pop(idx)
                no_improve_count = 0
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = adj[:]
            else:
                kick_size = random.choice([1, 1, 1, 2, 2, 3])
                existing = get_edge_list(adj)
                if not existing:
                    break
                    
                removed = []
                for _ in range(min(kick_size, len(existing))):
                    if not existing:
                        break
                    idx2 = random.randint(0, len(existing) - 1)
                    a, b = existing[idx2]
                    adj[a] &= ~(1 << b)
                    adj[b] &= ~(1 << a)
                    current_edges -= 1
                    removed.append((a, b))
                    existing.pop(idx2)
                    
                adj = greedy_fill(adj, deadline)
                current_edges = sum(x.bit_count() for x in adj) // 2
                
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = adj[:]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    
                missing = [(i, j) for i in range(N) for j in range(i + 1, N) if not (adj[i] & (1 << j))]
                
        return best_adj, best_edges

    best_adj = None
    best_count = 0

    # Strategy: Multiple restarts of "Random Dense -> Prune -> Greedy Fill -> Local Search"
    # Each restart gets a chunk of time
    
    while time.time() < deadline - 5:
        # 1. Initialize with random dense graph
        adj = [0] * N
        # Density 0.6 to 0.8
        p = random.uniform(0.6, 0.8)
        for u in range(N):
            for v in range(u + 1, N):
                if random.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        # 2. Prune K3,3s
        adj = prune_k33(adj, deadline - 5)
        
        # 3. Greedy Fill
        adj = greedy_fill(adj, deadline - 5)
        
        count = sum(x.bit_count() for x in adj) // 2
        
        # 4. Local Search
        if count > best_count or best_adj is None:
            # Only run local search if we have a decent candidate
            # To save time, maybe run LS only if count is close to best or better
            adj_ls, count_ls = local_search(adj, max_time=1.0)
            if count_ls > best_count:
                best_count = count_ls
                best_adj = adj_ls[:]
        else:
            # Even if not better, try a quick LS?
            # Maybe skip to save time for more restarts
            pass

    # Final polish
    if best_adj:
        best_adj = greedy_fill(best_adj, deadline)
        final_count = sum(x.bit_count() for x in best_adj) // 2
        if final_count > best_count:
            best_count = final_count
            # best_adj is updated in place

    if best_adj is None:
        return []

    edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
    return edges
