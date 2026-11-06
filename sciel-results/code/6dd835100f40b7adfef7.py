"""Clique partitioning of dense random graphs via greedy max-clique extraction."""
import time
from random import Random


def _random_graph(n, p, seed):
    """Generate random graph using the same RNG as the verifier."""
    rng = Random(seed)
    adj = [0] * n  # bitmask adjacency
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    return adj


def _max_clique(adj, start, n):
    """Greedy max clique starting from `start` using bitmasks."""
    clique = [start]
    # Candidate set: neighbors of start
    cand = adj[start]
    
    while cand:
        # Find vertex in cand with max common neighbors within cand
        best = None
        best_common = -1
        temp = cand
        while temp:
            low = temp & (-temp)
            u = low.bit_length() - 1
            common = (adj[u] & cand).bit_count()
            if common > best_common:
                best_common = common
                best = u
            if common == 0:
                break
            temp ^= low
        if best is None:
            break
        clique.append(best)
        cand &= adj[best]
    
    return clique if len(clique) >= 2 else None


def _solve_instance(adj, n, deadline):
    """Find clique partition for a single graph. Returns list of cliques."""
    # Remaining edges bitmask per vertex
    rem = [adj[i] for i in range(n)]
    remaining_edges = sum(rem[i].bit_count() for i in range(n)) // 2
    cliques = []
    
    if remaining_edges == 0:
        return cliques
    
    while remaining_edges > 0:
        # Deadline guard
        if time.time() > deadline:
            break
        
        # Find vertex with max degree in remaining graph
        max_deg = -1
        v = -1
        for i in range(n):
            d = rem[i].bit_count()
            if d > max_deg:
                max_deg = d
                v = i
            if d == 0:
                continue
        
        if v == -1 or max_deg == 0:
            # Shouldn't happen if remaining_edges > 0, but safety
            break
        
        if max_deg == 1:
            # Just add the single edge
            low = rem[v] & (-rem[v])
            u = low.bit_length() - 1
            cliques.append([v, u])
            rem[v] &= ~(1 << u)
            rem[u] &= ~(1 << v)
            remaining_edges -= 1
            continue
        
        # Try to build a maximal clique containing v
        clique = _max_clique(rem, v, n)
        if clique is None:
            # Fallback: just add one edge
            low = rem[v] & (-rem[v])
            u = low.bit_length() - 1
            cliques.append([v, u])
            rem[v] &= ~(1 << u)
            rem[u] &= ~(1 << v)
            remaining_edges -= 1
            continue
        
        # Add the clique and remove all its edges
        m = len(clique)
        cliques.append(clique)
        for i in range(m):
            for j in range(i + 1, m):
                a, b = clique[i], clique[j]
                rem[a] &= ~(1 << b)
                rem[b] &= ~(1 << a)
        remaining_edges -= m * (m - 1) // 2
    
    # Handle any remaining edges (if deadline hit)
    if remaining_edges > 0:
        for i in range(n):
            if time.time() > deadline:
                break
            temp = rem[i]
            while temp:
                if time.time() > deadline:
                    break
                low = temp & (-temp)
                j = low.bit_length() - 1
                cliques.append([i, j])
                rem[i] &= ~(1 << j)
                rem[j] &= ~(1 << i)
                temp &= temp - 1
                # Clear bit in rem[i]
                rem[i] &= ~(1 << j)
    
    return cliques


def construct():
    deadline = time.time() + 140  # 10s safety margin
    
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline:
            # Can't even generate; return empty (invalid but avoids crash)
            results.append([])
            continue
        
        adj = _random_graph(n, p, seed)
        cliques = _solve_instance(adj, n, deadline)
        results.append(cliques)
    
    return results
