import time
import random

_START = time.time()
_DEADLINE = _START + 115

def _has_path_5(u, v, adj_masks, n=40):
    """
    Returns True if there is a simple path of length 5 (i.e. a 6-cycle) through u-v.
    Equivalent: is there a simple path of length 4 from u to v?
    Path: u - a - b - c - d - v  (6 vertices, 5 edges)
    So we need: u-a, a-b, b-c, c-d, d-v all present, all distinct.
    
    Strategy: DFS from u up to depth 3 (u-a-b-c), then check if c has a neighbor d
    such that d is adjacent to v and d is not in {u,a,b,c,v}.
    """
    # BFS/DFS from u, track path of length up to 3
    # At depth 3 (node c), check: exists d in adj(c) \ visited \ {v} such that d in adj(v)
    stack = [(u, 0, 1 << u)]
    while stack:
        node, depth, mask = stack.pop()
        if depth == 3:
            # Need: exists d in adj(node) & adj(v) & ~mask & ~(1<<v)
            # d must not be v (d is the 5th vertex, v is the 6th)
            # Actually d must be distinct from u,a,b,c and also d != v
            # mask contains u,a,b,c. v is not in mask (unless v==u which can't happen in simple graph path)
            # Wait, v could be in mask if the path visited v. But we're looking for a path FROM u TO v.
            # If v is in mask, that means we already visited v, which would mean a shorter path exists.
            # But we want exactly length 4 from u to v. If v is in mask at depth 3, 
            # that means there's a path of length 3 from u to v, which is not what we want.
            # Actually for C6, we need a path of length 4 from u to v (plus edge u-v makes 6-cycle).
            # The path u-a-b-c-d-v has 5 edges. So from u to v via 4 intermediate nodes.
            # Let me re-think: C6 is a cycle of 6 vertices. If edge (u,v) is added, 
            # we need a path of length 5 from u to v in the existing graph.
            # Path: u - x1 - x2 - x3 - x4 - v (5 edges, 6 vertices total including u and v)
            # So x1,x2,x3,x4 are distinct and not u or v.
            
            # At depth 3 from u, we're at x3. We need x4 such that:
            # x4 in adj(x3), x4 in adj(v), x4 not in mask (mask has u,x1,x2,x3), x4 != v
            candidates = adj_masks[node] & adj_masks[v] & ~mask & ~(1 << v)
            if candidates:
                return True
            continue
        
        if depth == 2:
            # At x2. Need to go to x3, then x4, then v.
            # We can precompute: for each neighbor x3 of x2 (not in mask, x3 != v),
            # check if adj(x3) & adj(v) & ~(mask | (1<<x3)) & ~(1<<v) is non-empty.
            # But that's the same as depth 3 check. Let's just push to stack.
            neighbors = adj_masks[node] & ~mask & ~(1 << v)
            while neighbors:
                bit = neighbors & -neighbors
                nb = bit.bit_length() - 1
                neighbors ^= bit
                stack.append((nb, depth + 1, mask | bit))
            continue
        
        if depth == 1:
            # At x1. Go to x2.
            neighbors = adj_masks[node] & ~mask & ~(1 << v)
            while neighbors:
                bit = neighbors & -neighbors
                nb = bit.bit_length() - 1
                neighbors ^= bit
                stack.append((nb, depth + 1, mask | bit))
            continue
        
        # depth 0: at u
        neighbors = adj_masks[node] & ~mask & ~(1 << v)
        while neighbors:
            bit = neighbors & -neighbors
            nb = bit.bit_length() - 1
            neighbors ^= bit
            stack.append((nb, depth + 1, mask | bit))
    
    return False


def _greedy_fill(adj_masks, edges, n=40, deadline=None):
    """
    Greedily add all possible non-edges that don't create a C6.
    Returns the set of edges.
    """
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    improved = True
    while improved:
        if deadline and time.time() > deadline:
            break
        improved = False
        for u, v in all_edges:
            if (u, v) in edges:
                continue
            if not _has_path_5(u, v, adj_masks, n):
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
                edges.add((u, v))
                improved = True
    return edges


def _remove_random_edge(adj_masks, edges, rng, n=40):
    """Remove a random edge. Returns (u, v) of removed edge."""
    edge_list = list(edges)
    if not edge_list:
        return None
    u, v = rng.choice(edge_list)
    adj_masks[u] &= ~(1 << v)
    adj_masks[v] &= ~(1 << u)
    edges.discard((u, v))
    return (u, v)


def _make_c6_free(adj_masks, edges, rng, n=40, deadline=None):
    """
    Remove edges until the graph is C6-free.
    """
    while True:
        if deadline and time.time() > deadline:
            return edges
        # Find a C6
        found = False
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        for u, v in all_edges:
            if (u, v) not in edges:
                continue
            # Check if removing this edge would be needed
            # Actually, we need to find ANY C6 and remove an edge from it.
            # Let's find a C6 by checking each edge.
            # For edge (u,v), check if there's a path of length 5 from u to v NOT using (u,v).
            # Since (u,v) is in the graph, we need to check the graph WITHOUT (u,v).
            # Temporarily remove (u,v)
            adj_masks[u] &= ~(1 << v)
            adj_masks[v] &= ~(1 << u)
            
            if _has_path_5(u, v, adj_masks, n):
                # There's a C6 containing edge (u,v). Remove this edge.
                edges.discard((u, v))
                found = True
                break
            else:
                # Restore
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
        
        if not found:
            break
    
    return edges


def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 115
    
    n = 40
    best_edges = set()
    best_count = 0
    
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    rng = random.Random(42)
    
    # Strategy 1: Start from dense random graph, remove edges to make C6-free, then greedy fill
    for attempt in range(500):
        if time.time() > _DEADLINE - 3:
            break
        
        # Start with a random graph with varying density
        # Higher density means more edges to remove, but potentially better result after fill
        density = 0.3 + 0.4 * (attempt % 5) / 5.0  # 0.3 to 0.7
        
        adj_masks = [0] * n
        edges = set()
        
        for u, v in all_edges:
            if rng.random() < density:
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
                edges.add((u, v))
        
        # Remove edges to make C6-free
        edges = _make_c6_free(adj_masks, edges, rng, n, _DEADLINE - 5)
        
        # Greedy fill
        edges = _greedy_fill(adj_masks, edges, n, _DEADLINE - 5)
        
        count = len(edges)
        if count > best_count:
            best_count = count
            best_edges = set(edges)
        
        # ILS perturbation: if we have a good graph, perturb it
        if best_count > 0 and attempt % 3 == 0:
            if time.time() > _DEADLINE - 10:
                break
            
            # Work on a copy of the best graph
            adj_masks2 = adj_masks[:]
            edges2 = set(best_edges)
            
            # Perturb: remove k random edges and add some random non-edges
            k_remove = rng.randint(3, 8)
            removed = []
            for _ in range(k_remove):
                if not edges2:
                    break
                edge_list = list(edges2)
                u, v = rng.choice(edge_list)
                adj_masks2[u] &= ~(1 << v)
                adj_masks2[v] &= ~(1 << u)
                edges2.discard((u, v))
                removed.append((u, v))
            
            # Add some random non-edges (if they don't create C6 immediately)
            non_edges = [(u, v) for u, v in all_edges if (u, v) not in edges2]
            rng.shuffle(non_edges)
            for u, v in non_edges[:20]:
                if not _has_path_5(u, v, adj_masks2, n):
                    adj_masks2[u] |= (1 << v)
                    adj_masks2[v] |= (1 << u)
                    edges2.add((u, v))
            
            # Make C6-free again
            edges2 = _make_c6_free(adj_masks2, edges2, rng, n, _DEADLINE - 5)
            
            # Greedy fill
            edges2 = _greedy_fill(adj_masks2, edges2, n, _DEADLINE - 5)
            
            count2 = len(edges2)
            if count2 > best_count:
                best_count = count2
                best_edges = set(edges2)
    
    # Final greedy fill on the best graph to ensure local optimum
    if best_edges and time.time() < _DEADLINE - 2:
        adj_masks_final = [0] * n
        for u, v in best_edges:
            adj_masks_final[u] |= (1 << v)
            adj_masks_final[v] |= (1 << u)
        best_edges = _greedy_fill(adj_masks_final, best_edges, n, _DEADLINE - 1)
        best_count = len(best_edges)
    
    return [[u, v] for u, v in best_edges]
