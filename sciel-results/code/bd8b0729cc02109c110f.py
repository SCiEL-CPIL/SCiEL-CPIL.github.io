"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Construct a dense graph on vertices 0..15 with diameter exactly two,
    where every edge is essential (deleting any single edge increases diameter
    above two or disconnects the graph). Maximize edge count.

    Strategy: Start with K_16 (120 edges) and iteratively remove edges that
    are NOT essential. An edge (u,v) is essential if removing it causes the
    distance between u and v to become > 2 (or disconnects them). We want to
    keep as many edges as possible while ensuring every remaining edge is essential.

    Approach: 
    - Begin with complete graph K_16.
    - For each edge, check if it's essential. If not, remove it.
    - The order of checking/removing matters; use a fixed order for determinism.
    - After removing non-essential edges, verify the result.
    """
    n = 16
    # Build adjacency set for K_16
    adj = [set() for _ in range(n)]
    edges = set()
    for u in range(n):
        for v in range(u+1, n):
            adj[u].add(v)
            adj[v].add(u)
            edges.add((u, v))

    # BFS to compute all-pairs distances (only need up to distance 2)
    def is_essential(u, v, adj, n):
        """Check if edge (u,v) is essential: removing it makes dist(u,v) > 2 or disconnected."""
        # Temporarily remove edge
        adj[u].discard(v)
        adj[v].discard(u)
        
        # BFS from u to see if v is reachable within distance 2
        # Quick check: if v is in adj[u], distance is 1 (but we just removed it, so no)
        # Distance 1: no (edge removed)
        # Distance 2: exists w such that w in adj[u] and w in adj[v]
        common = adj[u] & adj[v]
        if common:
            # Distance is 2, so edge was NOT essential
            adj[u].add(v)
            adj[v].add(u)
            return False
        
        # Check if v is reachable at distance 2 via any path
        # Actually, if no common neighbor, distance > 2 (or disconnected)
        # So edge IS essential
        adj[u].add(v)
        adj[v].add(u)
        return True

    # Iterate to remove non-essential edges
    # Use a fixed order: sort edges
    edge_list = sorted(edges)
    changed = True
    while changed:
        changed = False
        for (u, v) in edge_list:
            if (u, v) not in edges:
                continue
            if not is_essential(u, v, adj, n):
                # Remove this edge
                adj[u].discard(v)
                adj[v].discard(u)
                edges.discard((u, v))
                changed = True

    # Verify: check diameter is exactly 2
    def verify(adj, n):
        # Check diameter <= 2
        for u in range(n):
            for v in range(n):
                if u == v:
                    continue
                if v in adj[u]:
                    continue  # distance 1
                # Check distance 2
                common = adj[u] & adj[v]
                if not common:
                    return False
        return True

    if not verify(adj, n):
        # Fallback: return a known valid graph (e.g., K_8,8 or similar)
        # For now, return empty as a safe fallback (will score 0)
        pass

    # Convert to list of [u, v] lists
    result = []
    for (u, v) in sorted(edges):
        result.append([int(u), int(v)])
    
    return result
# EVOLVE-BLOCK-END
