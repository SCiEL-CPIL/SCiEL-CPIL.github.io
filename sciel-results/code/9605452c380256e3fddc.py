import time
import random
import sys

def construct():
    start = time.time()
    deadline = start + 110.0  # safety margin before 120s
    
    best_result = None
    best_n = 0
    
    # Try n from 35 down to 18
    for n in range(35, 17, -1):
        if time.time() > deadline:
            break
        
        # Try to find a valid graph on n vertices
        result = _solve_n(n, deadline)
        if result is not None:
            return result
        
        # If we couldn't find n, we still want to keep going
        # but we should be careful about time
    
    # Fallback: return best found or a known valid graph
    if best_result is not None:
        return best_result
    
    # Final fallback: Paley 17
    return _paley_17()


def _paley_17():
    squares = {x * x % 17 for x in range(1, 17)}
    edges = []
    for i in range(17):
        for j in range(i + 1, 17):
            if (j - i) % 17 in squares:
                edges.append([i, j])
    return {'n': 17, 'edges': edges}


def _solve_n(n, deadline):
    """Try to find a valid graph on n vertices using ILS."""
    rng = random.Random()
    
    for restart in range(200):
        if time.time() > deadline:
            return None
        
        # Initialize random graph
        adj = _random_graph(n, rng)
        
        # Local search
        for step in range(50000):
            if time.time() > deadline:
                return None
            
            # Check for K4
            k4 = _find_k4_fast(adj, n)
            if k4 is not None:
                # Remove an edge from this K4
                u, v, w, x = k4
                # Pick the edge that appears in the fewest other K4s (approximate: just pick random)
                candidates = [(u,v), (u,w), (u,x), (v,w), (v,x), (w,x)]
                e = rng.choice(candidates)
                adj[e[0]] &= ~(1 << e[1])
                adj[e[1]] &= ~(1 << e[0])
                continue
            
            # Check for independent 6-set
            i6 = _find_independent_6_fast(adj, n)
            if i6 is not None:
                # Add an edge to break this independent set
                u, v = rng.sample(i6, 2)
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                continue
            
            # Valid!
            return _to_edges(adj, n)
        
        # Perturb: flip 5-15 random edges
        perturb_count = rng.randint(5, 15)
        for _ in range(perturb_count):
            u = rng.randint(0, n - 1)
            v = rng.randint(0, n - 1)
            if u != v:
                if (adj[u] >> v) & 1:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
    
    return None


def _random_graph(n, rng):
    """Generate a random graph with edge probability ~0.5."""
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj


def _find_k4_fast(adj, n):
    """Find a K4 in the graph using bitmasks. Returns (u,v,w,x) or None."""
    for u in range(n):
        for v in range(u + 1, n):
            if not (adj[u] >> v) & 1:
                continue
            # Common neighbors of u and v
            common = adj[u] & adj[v]
            # We need to find two connected vertices w, x in common
            # Iterate over w in common
            w_mask = common
            while w_mask:
                # Get lowest set bit
                lb = w_mask & (-w_mask)
                w = lb.bit_length() - 1
                w_mask ^= lb
                
                # Now find x in common that is connected to w
                x_mask = w_mask & adj[w]
                if x_mask:
                    # Get any x
                    lb_x = x_mask & (-x_mask)
                    x = lb_x.bit_length() - 1
                    return (u, v, w, x)
    return None


def _find_independent_6_fast(adj, n):
    """Find an independent set of size 6. Returns list of 6 vertices or None."""
    # Use branch and bound with bitmasks
    # A vertex v is "available" for an independent set if it's not adjacent to any selected vertex
    
    def _dfs(selected, available_mask, depth):
        if depth == 6:
            return selected[:]
        
        # If we can't reach depth 6, prune
        if bin(available_mask).count('1') < (6 - depth):
            return None
        
        # Get list of available vertices
        avail_list = []
        mask = available_mask
        while mask:
            lb = mask & (-mask)
            v = lb.bit_length() - 1
            avail_list.append(v)
            mask ^= lb
        
        # Try each vertex
        for v in avail_list:
            # Check v is not adjacent to any in selected
            ok = True
            for u in selected:
                if (adj[u] >> v) & 1:
                    ok = False
                    break
            if not ok:
                continue
            
            new_selected = selected + [v]
            # New available: remove v and all neighbors of v
            new_avail = available_mask & ~(adj[v] | (1 << v))
            # Also remove vertices already in selected (they shouldn't be in avail anyway)
            result = _dfs(new_selected, new_avail, depth + 1)
            if result is not None:
                return result
        
        return None
    
    full_mask = (1 << n) - 1
    result = _dfs([], full_mask, 0)
    return result


def _to_edges(adj, n):
    """Convert adjacency bitmasks to edge list."""
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                edges.append([i, j])
    return edges
