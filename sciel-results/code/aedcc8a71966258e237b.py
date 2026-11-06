"""Combined program: seeded C6-free construction with local search."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    n = 40
    start_time = time.time()
    deadline = 150
    safety_margin = 5

    # Start with 8 disjoint K5's: 8 * 10 = 80 edges, definitely C6-free.
    best_adj = [set() for _ in range(n)]
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                best_adj[base + a].add(base + b)
                best_adj[base + b].add(base + a)

    def edge_count(adj):
        return sum(len(s) for s in adj) // 2

    best_count = edge_count(best_adj)
    best_adj_copy = [set(s) for s in best_adj]

    def is_safe_to_add(adj, u, v):
        """
        Check if adding (u, v) creates a C6.
        A new C6 must use the new edge (u, v).
        So we need a path of length 5 from u to v in the current graph.
        Path: u - w - z - y - x - v
        w in N(u), x in N(v).
        Need a path of length 3 from w to x: w - z - y - x.
        This exists if there is a z in N(w) such that N(z) intersects N(x) 
        (and the path is simple).
        For simplicity and speed, we check if there exists z in N(w) 
        such that (N(z) & N(x)) is non-empty.
        We must be careful about simple paths, but for dense graphs, 
        if such a z exists, it's highly likely a simple path exists.
        Actually, if z == x, path is w-x, length 1.
        If z == w, impossible as z in N(w).
        If y == w, then z-w, so w in N(z).
        If y == x, then x in N(z).
        The condition (N(z) & N(x)) non-empty means there is a common neighbor y.
        If y == w, then w in N(x). So x in N(w). Then dist(w,x)=1.
        If y == x, then x in N(z). So z-x. Path w-z-x. dist(w,x)=2.
        If y is distinct from w and x, then w-z-y-x is a simple path of length 3.
        So we just need to check if there is a z in N(w) such that (N(z) & N(x)) is non-empty.
        Wait, if the only common neighbor is w or x, it doesn't form a length 3 path.
        But if N(z) & N(x) is non-empty, and the intersection contains only w or x, 
        then dist(w,x) < 3.
        However, if dist(w,x) < 3, adding (u,v) might create a shorter cycle, 
        which is fine as long as it's not C6.
        We ONLY care about C6.
        So we need to check if there is a SIMPLE path of length 5.
        The check above might give false positives if the only paths are shorter.
        But if there is a path of length 5, there is also a path of length 3 between w and x.
        So the condition is necessary. Is it sufficient?
        If there is a z such that N(z) & N(x) is non-empty, does it imply a C6?
        Not necessarily. But it's a good heuristic.
        Let's refine:
        For w in N(u):
            For x in N(v):
                if w == x: continue
                # Check if dist(w, x) == 3
                # This is equivalent to: exists z in N(w) such that (N(z) & N(x)) is non-empty
                # AND the common neighbor y is not w and not x.
                # Actually, if (N(z) & N(x)) is non-empty, let y be in the intersection.
                # If y == w, then w in N(x), so dist(w,x)=1.
                # If y == x, then x in N(z), so dist(w,x)=2.
                # If y != w and y != x, then dist(w,x)=3.
                # So we need to check if there is a z in N(w) such that 
                # (N(z) & N(x)) - {w, x} is non-empty.
                common = adj[z] & adj[x]
                if common - {w, x}:
                    return False
        return True
        """
        for w in adj[u]:
            if w == v:
                continue
            for x in adj[v]:
                if x == u:
                    continue
                if w == x:
                    continue
                # Check if there is a path of length 3 from w to x
                for z in adj[w]:
                    if z == x or z == u or z == v:
                        continue
                    # Check if N(z) intersects N(x) with a vertex not in {w, x}
                    # Actually, we just need to see if there is a common neighbor y
                    # such that y != w and y != x.
                    # If such a y exists, then w-z-y-x is a path of length 3.
                    # Then u-w-z-y-x-v-u is a C6.
                    # We can check if (adj[z] & adj[x]) is non-empty.
                    # If it is, we need to verify that the common neighbor is not w or x.
                    # But if the common neighbor is w, then w in adj[x], so x in adj[w].
                    # If the common neighbor is x, then x in adj[z], so z in adj[x].
                    # In either case, we can just check the set difference.
                    if (adj[z] & adj[x]) - {w, x}:
                        return False
        return True

    def try_add_edge(adj, u, v):
        if v not in adj[u]:
            if is_safe_to_add(adj, u, v):
                adj[u].add(v)
                adj[v].add(u)
                return True
        return False

    # Local search loop
    while time.time() - start_time < deadline - safety_margin:
        improved = False
        # Shuffle vertices to avoid bias
        verts = list(range(n))
        random.shuffle(verts)
        
        for u in verts:
            for v in verts:
                if u >= v:
                    continue
                if try_add_edge(best_adj, u, v):
                    improved = True
                    if edge_count(best_adj) > best_count:
                        best_count = edge_count(best_adj)
                        best_adj_copy = [set(s) for s in best_adj]
        
        if not improved:
            # Try removing an edge to allow others? (2-opt style)
            # For now, just break if no improvement.
            # We can also try a random restart with a different seed.
            # But for now, let's just try to remove edges that are "blocking".
            # This is complex. Let's stick to greedy addition.
            # If we can't add any edge, we are at a local maximum.
            # We can try to remove an edge and see if we can add two.
            removed = False
            for u in range(n):
                for v in list(best_adj[u]):
                    if u < v:
                        # Remove edge
                        best_adj[u].remove(v)
                        best_adj[v].remove(u)
                        removed = True
                        # Try to add edges
                        added = 0
                        for x in range(n):
                            for y in range(x+1, n):
                                if try_add_edge(best_adj, x, y):
                                    added += 1
                        if added <= 1:
                            # Re-add the edge if we didn't gain much
                            best_adj[u].add(v)
                            best_adj[v].add(u)
                        else:
                            if edge_count(best_adj) > best_count:
                                best_count = edge_count(best_adj)
                                best_adj_copy = [set(s) for s in best_adj]
                        break
                if removed:
                    break
            if not removed:
                break

    # Return the best edges found
    edges = []
    for u in range(n):
        for v in best_adj_copy[u]:
            if u < v:
                edges.append([u, v])
    return edges
# EVOLVE-BLOCK-END
