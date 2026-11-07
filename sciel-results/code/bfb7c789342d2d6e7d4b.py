def _is_diam_le_4(adj, n):
    target_mask = (1 << n) - 1
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        visited = 1 << src
        frontier = 1 << src
        d = 0
        while frontier:
            d += 1
            if d > 4:
                return False
            next_frontier = 0
            f = frontier
            while f:
                lsb = f & -f
                u = lsb.bit_length() - 1
                f ^= lsb
                next_frontier |= adj[u]
            next_frontier &= ~visited
            if not next_frontier:
                break
            visited |= next_frontier
            frontier = next_frontier
        
        if visited != target_mask:
            return False
            
    return True
