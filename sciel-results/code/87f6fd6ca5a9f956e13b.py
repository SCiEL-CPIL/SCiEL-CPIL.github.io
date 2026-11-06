"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _is_3_colorable_fast(adj_masks, n):
    """Faster 3-colorability check using color-class bitmasks with MRV heuristic."""
    # Precompute vertex ordering by degree (descending) for better pruning
    deg = [bin(adj_masks[i]).count('1') for i in range(n)]
    order = sorted(range(n), key=lambda x: -deg[x])
    
    # color_class[c] = bitmask of vertices colored c
    color = [-1] * n
    cc = [0, 0, 0]  # color class bitmasks
    colored_count = 0
    
    def bt(pos):
        nonlocal colored_count
        if pos == n:
            return True
        v = order[pos]
        nb = adj_masks[v]
        
        # Try colors, prefer those with fewer conflicts among uncolored neighbors
        # Use simple order: 0, 1, 2
        for c in range(3):
            # If any neighbor is in color class c, conflict
            if nb & cc[c]:
                continue
            color[v] = c
            cc[c] |= (1 << v)
            colored_count += 1
            if bt(pos + 1):
                return True
            cc[c] &= ~(1 << v)
            colored_count -= 1
            color[v] = -1
        return False
    
    return bt(0)


def _is_3_colorable_mrv(adj_masks, n):
    """3-colorability check with MRV + forward checking."""
    color = [-1] * n
    cc = [0, 0, 0]
    
    def bt(uncolored_mask):
        if uncolored_mask == 0:
            return True
        
        # Select uncolored vertex with fewest available colors (MRV)
        best_v = -1
        best_avail = 4
        m = uncolored_mask
        while m:
            v = (m & (-m)).bit_length() - 1
            m &= m - 1
            nb = adj_masks[v]
            avail = 3
            for c in range(3):
                if nb & cc[c]:
                    avail -= 1
            if avail == 0:
                return False
            if avail < best_avail:
                best_avail = avail
                best_v = v
                if avail == 1:
                    break
        
        v = best_v
        nb = adj_masks[v]
        # Try colors in order of least constraining (fewest uncolored neighbors in that color)
        color_order = []
        for c in range(3):
            if nb & cc[c]:
                continue
            # Count uncolored neighbors that would be constrained
            uncolored_nb = nb & uncolored_mask
            uncolored_nb &= ~(1 << v)
            color_order.append(c)
        
        for c in color_order:
            cc[c] |= (1 << v)
            if bt(uncolored_mask & ~(1 << v)):
                return True
            cc[c] &= ~(1 << v)
        return False
    
    return bt((1 << n) - 1)


def _triangle_free(adj_masks, n):
    """Check if graph is triangle-free using bitmasks."""
    for u in range(n):
        for v in range(u + 1, n):
            if (adj_masks[u] >> v) & 1:
                common = adj_masks[u] & adj_masks[v]
                common &= ~(1 << u)
                common &= ~(1 << v)
                if common:
                    return False
    return True


def _grötzsch_edges():
    """Return edges of the Grötzsch graph (Mycielski of C5) on vertices 0..10."""
    cycle = [(i, (i + 1) % 5) for i in range(5)]
    edges = [list(e) for e in cycle]
    for u, w in cycle:
        edges += [[u, 5 + w], [w, 5 + u]]
    edges += [[5 + u, 10] for u in range(5)]
    return edges


def _edges_to_adj(edges, n):
    """Convert edge list to adjacency bitmasks."""
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj


def _adj_to_edges(adj, n):
    """Convert adjacency bitmasks to edge list."""
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (adj[u] >> v) & 1:
                edges.append([u, v])
    return edges


def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n = 16
    
    # Start with Grötzsch graph on vertices 0..10, vertices 11..15 isolated
    base_edges = _grötzsch_edges()
    adj = _edges_to_adj(base_edges, n)
    
    assert _triangle_free(adj, n), "Initial graph not triangle-free"
    assert not _is_3_colorable_mrv(adj, n), "Initial graph is 3-colorable"
    
    best_edges = _adj_to_edges(adj, n)
    best_count = len(best_edges)
    
    def try_add_edge(adj, u, v, n):
        """Try to add edge (u,v). Return new adj if valid, else None."""
        common = adj[u] & adj[v]
        common &= ~(1 << u)
        common &= ~(1 << v)
        if common:
            return None
        
        new_adj = adj[:]
        new_adj[u] |= (1 << v)
        new_adj[v] |= (1 << u)
        
        if _is_3_colorable_mrv(new_adj, n):
            return None
        
        return new_adj
    
    # Systematic greedy: repeatedly scan all non-edges, add if possible
    # Order non-edges by potential impact (high degree pairs first)
    def get_non_edges_ordered(adj, n):
        non_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    continue
                deg_u = bin(adj[u]).count('1')
                deg_v = bin(adj[v]).count('1')
                non_edges.append((-(deg_u + deg_v), u, v))
        non_edges.sort()
        return [(u, v) for _, u, v in non_edges]
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        non_edges = get_non_edges_ordered(adj, n)
        for u, v in non_edges:
            if time.time() > deadline:
                break
            new_adj = try_add_edge(adj, u, v, n)
            if new_adj is not None:
                adj = new_adj
                changed = True
                best_edges = _adj_to_edges(adj, n)
                best_count = len(best_edges)
    
    # Local search with simulated annealing
    rng = random.Random(42)
    current_adj = adj[:]
    current_edges_count = best_count
    
    for iteration in range(2000):
        if time.time() > deadline:
            break
        
        # Pick a move type
        move_type = rng.random()
        
        if move_type < 0.5:
            # Try adding a random non-edge
            u = rng.randint(0, n - 1)
            v = rng.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            if (current_adj[u] >> v) & 1:
                continue
            new_adj = try_add_edge(current_adj, u, v, n)
            if new_adj is not None:
                current_adj = new_adj
                current_edges_count += 1
                if current_edges_count > best_count:
                    best_count = current_edges_count
                    best_edges = _adj_to_edges(current_adj, n)
                    adj = current_adj[:]
        elif move_type < 0.8:
            # Remove one edge, then greedily try to add 1-3 edges
            edges = _adj_to_edges(current_adj, n)
            if not edges:
                continue
            ru, rv = rng.choice(edges)
            
            new_adj = current_adj[:]
            new_adj[ru] &= ~(1 << rv)
            new_adj[rv] &= ~(1 << ru)
            
            # Try to add edges back, prioritizing high-degree pairs
            added = 0
            for _ in range(30):
                u = rng.randint(0, n - 1)
                v = rng.randint(0, n - 1)
                if u == v or u > v:
                    continue
                if (new_adj[u] >> v) & 1:
                    continue
                common = new_adj[u] & new_adj[v]
                common &= ~(1 << u)
                common &= ~(1 << v)
                if common:
                    continue
                test_adj = new_adj[:]
                test_adj[u] |= (1 << v)
                test_adj[v] |= (1 << u)
                if not _is_3_colorable_mrv(test_adj, n):
                    new_adj = test_adj
                    added += 1
                    if added >= 3:
                        break
            
            new_count = len(_adj_to_edges(new_adj, n))
            # Accept if better or with some probability (annealing)
            if new_count >= current_edges_count:
                current_adj = new_adj
                current_edges_count = new_count
                if current_edges_count > best_count:
                    best_count = current_edges_count
                    best_edges = _adj_to_edges(current_adj, n)
                    adj = current_adj[:]
            else:
                # Accept worse moves with decreasing probability
                prob = 0.1 * (1.0 - iteration / 2000.0)
                if rng.random() < prob:
                    current_adj = new_adj
                    current_edges_count = new_count
        else:
            # 2-opt style: remove 2 edges, add 2-4 different edges
            edges = _adj_to_edges(current_adj, n)
            if len(edges) < 2:
                continue
            e1, e2 = rng.sample(edges, 2)
            
            new_adj = current_adj[:]
            new_adj[e1[0]] &= ~(1 << e1[1])
            new_adj[e1[1]] &= ~(1 << e1[0])
            new_adj[e2[0]] &= ~(1 << e2[1])
            new_adj[e2[1]] &= ~(1 << e2[0])
            
            added = 0
            for _ in range(40):
                u = rng.randint(0, n - 1)
                v = rng.randint(0, n - 1)
                if u == v or u > v:
                    continue
                if (new_adj[u] >> v) & 1:
                    continue
                common = new_adj[u] & new_adj[v]
                common &= ~(1 << u)
                common &= ~(1 << v)
                if common:
                    continue
                test_adj = new_adj[:]
                test_adj[u] |= (1 << v)
                test_adj[v] |= (1 << u)
                if not _is_3_colorable_mrv(test_adj, n):
                    new_adj = test_adj
                    added += 1
                    if added >= 4:
                        break
            
            new_count = len(_adj_to_edges(new_adj, n))
            if new_count >= current_edges_count:
                current_adj = new_adj
                current_edges_count = new_count
                if current_edges_count > best_count:
                    best_count = current_edges_count
                    best_edges = _adj_to_edges(current_adj, n)
                    adj = current_adj[:]
            else:
                prob = 0.05 * (1.0 - iteration / 2000.0)
                if rng.random() < prob:
                    current_adj = new_adj
                    current_edges_count = new_count
    
    # Final polish: systematic greedy pass on the best found
    if time.time() < deadline - 5:
        changed = True
        while changed and time.time() < deadline - 1:
            changed = False
            non_edges = get_non_edges_ordered(adj, n)
            for u, v in non_edges:
                if time.time() > deadline - 1:
                    break
                new_adj = try_add_edge(adj, u, v, n)
                if new_adj is not None:
                    adj = new_adj
                    changed = True
                    best_edges = _adj_to_edges(adj, n)
                    best_count = len(best_edges)
    
    return best_edges

# EVOLVE-BLOCK-END
