"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    n = 15
    start = time.time()
    deadline = start + 50.0  # safety margin under 60s

    # Start with complete graph
    edges = set()
    for i in range(n):
        for j in range(i+1, n):
            edges.add((i, j))

    def get_adj(edges):
        adj = [set() for _ in range(n)]
        for (u, v) in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def get_deg(adj):
        return [len(adj[i]) for i in range(n)]

    def find_violations(edges, adj, deg):
        """Return list of (s, u, v, t) violating paths, and the offending edge (u,v)."""
        violations = []
        for (u, v) in edges:
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        violations.append((s, u, v, t, (u, v)))
        return violations

    # Iteratively remove edges to fix violations
    # Priority: remove edges that are in many violations
    removed = 0
    while time.time() < deadline:
        adj = get_adj(edges)
        deg = get_deg(adj)
        violations = find_violations(edges, adj, deg)
        if not violations:
            break
        # Count how many violations each edge is involved in
        from collections import Counter
        edge_count = Counter()
        for (s, u, v, t, e) in violations:
            edge_count[e] += 1
        # Remove the edge with max violations
        worst_edge = max(edge_count, key=lambda e: edge_count[e])
        edges.discard(worst_edge)
        removed += 1

    # Build final adjacency and verify
    adj = get_adj(edges)
    deg = get_deg(adj)
    # Final verification
    ok = True
    for (u, v) in edges:
        for s in adj[u]:
            if s == v:
                continue
            for t in adj[v]:
                if t == u or t == s:
                    continue
            if deg[s] == deg[t]:
                ok = False
                break
        if not ok:
            break

    result = []
    for (u, v) in edges:
        result.append([int(u), int(v)])
    return result
# EVOLVE-BLOCK-END
