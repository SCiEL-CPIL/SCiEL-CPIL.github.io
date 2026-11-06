import random
import time

def construct():
    deadline = time.time() + 110  # Safety margin before 150s kill

    # Try to find a graph for n=40 down to n=26
    for n in range(40, 25, -1):
        if time.time() > deadline:
            break

        # Try multiple random restarts for this n
        max_restarts = 200
        for r in range(max_restarts):
            if time.time() > deadline:
                break

            # 1. Initialize a random maximal triangle-free graph
            adj = [0] * n
            # Create list of all edges
            edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(edges)

            for u, v in edges:
                # Check if adding (u, v) creates a triangle
                # Common neighbors of u and v
                if (adj[u] & adj[v]) == 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)

            # 2. Check if alpha < 10 using efficient branch and bound
            # We check if there is an independent set of size 10
            alpha = _alpha_exact_bnb(adj, n, limit=10)
            if alpha < 10:
                # Convert adj to edge list
                edge_list = []
                for u in range(n):
                    mask = adj[u]
                    while mask:
                        lsb = mask & -mask
                        v = lsb.bit_length() - 1
                        if u < v:
                            edge_list.append([u, v])
                        mask -= lsb
                return {'n': n, 'edges': edge_list}

            # 3. If alpha >= 10, try to improve via local search
            # Attempt to reduce alpha by flipping edges
            adj, alpha = _optimize_edges(adj, n, deadline)
            if alpha < 10:
                edge_list = []
                for u in range(n):
                    mask = adj[u]
                    while mask:
                        lsb = mask & -mask
                        v = lsb.bit_length() - 1
                        if u < v:
                            edge_list.append([u, v])
                        mask -= lsb
                return {'n': n, 'edges': edge_list}

    # Fallback: Return a known valid graph (smaller n)
    # Use a simple construction or a hardcoded known graph
    # For n=26, we can use a specific structure or just return a smaller valid one
    # Let's try to find a small valid one quickly if we haven't
    for n in range(30, 25, -1):
        if time.time() > deadline:
            break
        for r in range(100):
            if time.time() > deadline:
                break
            adj = [0] * n
            edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(edges)
            for u, v in edges:
                if (adj[u] & adj[v]) == 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            alpha = _alpha_exact_bnb(adj, n, limit=10)
            if alpha < 10:
                edge_list = []
                for u in range(n):
                    mask = adj[u]
                    while mask:
                        lsb = mask & -mask
                        v = lsb.bit_length() - 1
                        if u < v:
                            edge_list.append([u, v])
                        mask -= lsb
                return {'n': n, 'edges': edge_list}

    # Last resort: return a trivially small graph that is valid
    # n=1 is always valid (no triangle, no independent set of 10)
    return {'n': 1, 'edges': []}

def _alpha_exact_bnb(adj, n, limit=10):
    """
    Exact max independent set size via branch and bound.
    Returns the size if it's < limit, or limit if it's >= limit (for early termination).
    Actually, to be precise for the check, we just need to know if it's >= limit.
    But returning the exact value up to limit is useful for optimization.
    """
    best = [0]
    
    def bnb(chosen_count, candidates):
        if chosen_count + candidates.bit_count() <= best[0]:
            return
        if chosen_count >= limit:
            best[0] = chosen_count
            return
        if candidates == 0:
            if chosen_count > best[0]:
                best[0] = chosen_count
            return
        
        # Heuristic: pick vertex with max degree in candidates to prune
        # Or just pick the first one for simplicity, but max degree is better
        # Let's stick to a simple bit manipulation heuristic
        lb = candidates & (-candidates)
        v = lb.bit_length() - 1
        rest = candidates ^ lb
        
        # Branch: include v
        new_cands = rest & ~adj[v]
        bnb(chosen_count + 1, new_cands)
        
        # Branch: exclude v
        bnb(chosen_count, rest)
        
    bnb(0, (1 << n) - 1)
    return best[0]

def _optimize_edges(adj, n, deadline):
    """
    Local search to reduce alpha while maintaining triangle-free property.
    Returns (adj, alpha).
    """
    adj = list(adj)
    best_alpha = _alpha_exact_bnb(adj, n, limit=10)
    
    # If already valid, return
    if best_alpha < 10:
        return adj, best_alpha

    # Try to add edges to reduce alpha
    # Adding an edge (u,v) is valid if no common neighbor exists (triangle-free)
    # It reduces alpha if it breaks all independent sets of size >= best_alpha?
    # Actually, adding edges makes it harder to form independent sets, so alpha can only decrease or stay same.
    # We want to reach alpha < 10.
    
    while time.time() < deadline:
        improved = False
        # Iterate through all non-edges
        for i in range(n):
            for j in range(i + 1, n):
                if time.time() > deadline:
                    return adj, best_alpha
                if adj[i] & (1 << j):
                    continue
                # Check triangle constraint
                if adj[i] & adj[j]:
                    continue
                # Try adding edge
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                new_alpha = _alpha_exact_bnb(adj, n, limit=10)
                if new_alpha < best_alpha:
                    best_alpha = new_alpha
                    improved = True
                    if best_alpha < 10:
                        return adj, best_alpha
                    break
                else:
                    # Revert
                    adj[i] ^= (1 << j)
                    adj[j] ^= (1 << i)
            if improved:
                break
        
        if not improved:
            # If no single edge addition helps, we are stuck in a local minimum
            # or no valid edge addition exists. Break.
            break
            
    return adj, best_alpha
