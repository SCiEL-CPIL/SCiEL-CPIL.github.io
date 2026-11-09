def _local_search(adj, edges, c4, seed, time_limit):
    random.seed(seed)
    n = 31
    current_adj = adj[:]
    current_edges = edges[:]
    current_c4 = c4
    
    best_adj = current_adj[:]
    best_edges = current_edges[:]
    best_c4 = current_c4

    t_start = time.time()
    T = 1.0
    cooling_rate = 0.999

    def count_c4_local(adj_local):
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                k = (adj_local[i] & adj_local[j]).bit_count()
                if k > 1:
                    total += k * (k - 1) // 2
        return total // 2

    while time.time() - t_start < time_limit:
        if random.random() < 0.2 and len(current_edges) >= 2:
            # 2-swap
            idx1 = random.randint(0, len(current_edges) - 1)
            idx2 = random.randint(0, len(current_edges) - 1)
            if idx1 == idx2:
                continue
            u1, v1 = current_edges[idx1]
            u2, v2 = current_edges[idx2]
            
            # Try to add (u1, u2) and (v1, v2)
            # Check if these are non-edges
            if ((current_adj[u1] >> u2) & 1) or ((current_adj[v1] >> v2) & 1):
                # Try other pairing: (u1, v2) and (u2, v1)
                if ((current_adj[u1] >> v2) & 1) or ((current_adj[u2] >> v1) & 1):
                    continue
                else:
                    new_u1, new_v1 = u1, v2
                    new_u2, new_v2 = u2, v1
            else:
                new_u1, new_v1 = u1, u2
                new_u2, new_v2 = v1, v2

            # Remove old
            current_adj[u1] &= ~(1 << v1)
            current_adj[v1] &= ~(1 << u1)
            current_adj[u2] &= ~(1 << v2)
            current_adj[v2] &= ~(1 << u2)
            
            # Add new
            current_adj[new_u1] |= (1 << new_v1)
            current_adj[new_v1] |= (1 << new_u1)
            current_adj[new_u2] |= (1 << new_v2)
            current_adj[new_v2] |= (1 << new_u2)
            
            new_c4 = count_c4_local(current_adj)
            delta = new_c4 - current_c4

            accept = False
            if delta < 0:
                accept = True
            else:
                if T > 0.01:
                    if random.random() < math.exp(-delta / T):
                        accept = True

            if accept:
                current_c4 = new_c4
                current_edges[idx1] = (new_u1, new_v1)
                current_edges[idx2] = (new_u2, new_v2)
                if new_c4 < best_c4:
                    best_c4 = new_c4
                    best_adj = current_adj[:]
                    best_edges = current_edges[:]
                T *= cooling_rate
            else:
                # Revert
                current_adj[new_u1] &= ~(1 << new_v1)
                current_adj[new_v1] &= ~(1 << new_u1)
                current_adj[new_u2] &= ~(1 << new_v2)
                current_adj[new_v2] &= ~(1 << new_u2)
                
                current_adj[u1] |= (1 << v1)
                current_adj[v1] |= (1 << u1)
                current_adj[u2] |= (1 << v2)
                current_adj[v2] |= (1 << u2)
        else:
            # 1-swap
            idx = random.randint(0, len(current_edges) - 1)
            u1, v1 = current_edges[idx]
            
            while True:
                u2 = random.randint(0, n - 1)
                v2 = random.randint(0, n - 1)
                if u2 == v2:
                    continue
                if u2 > v2:
                    u2, v2 = v2, u2
                if not ((current_adj[u2] >> v2) & 1):
                    break
            
            if (u1, v1) == (u2, v2) or (v1, u1) == (u2, v2):
                continue

            current_adj[u1] &= ~(1 << v1)
            current_adj[v1] &= ~(1 << u1)
            current_adj[u2] |= (1 << v2)
            current_adj[v2] |= (1 << u2)
            
            new_c4 = count_c4_local(current_adj)
            delta = new_c4 - current_c4

            accept = False
            if delta < 0:
                accept = True
            else:
                if T > 0.01:
                    if random.random() < math.exp(-delta / T):
                        accept = True

            if accept:
                current_c4 = new_c4
                current_edges[idx] = (u2, v2)
                if new_c4 < best_c4:
                    best_c4 = new_c4
                    best_adj = current_adj[:]
                    best_edges = current_edges[:]
                T *= cooling_rate
            else:
                current_adj[u2] &= ~(1 << v2)
                current_adj[v2] &= ~(1 << u2)
                current_adj[u1] |= (1 << v1)
                current_adj[v1] |= (1 << u1)

    return best_adj, best_edges, best_c4
