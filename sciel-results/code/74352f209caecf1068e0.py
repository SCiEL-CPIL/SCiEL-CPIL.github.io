"""Construct a 3-regular graph with girth >= g on as few vertices as possible."""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Construct a k-regular graph with girth >= g using a voltage lift of K_{3,3}.
    
    Strategy:
    1. Base graph is K_{3,3} (6 vertices, 3-regular).
    2. Lift over Z_m with random voltages.
    3. Search for smallest m (and thus n=6m) such that girth >= g.
    """
    deadline = time.time() + 145
    
    def check_girth_bfs(n, adj, k, g):
        """Check if graph has girth >= g using BFS from each vertex."""
        radius = g // 2 - 1
        if radius <= 0:
            return True
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            count = [0] * (radius + 1)
            count[0] = 1
            while queue:
                u = queue.popleft()
                if dist[u] >= radius:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] <= radius:
                            count[dist[v]] += 1
                            queue.append(v)
            # Expected counts for a tree of depth 'radius' in a k-regular graph
            expected = [1] + [k * (k-1)**i for i in range(radius)]
            if count != expected:
                return False
        return True

    # Base graph: K_{3,3}
    # Vertices 0,1,2 in set U, 3,4,5 in set V
    # Edges: (u, v) for u in U, v in V
    n_base = 6
    base_adj = [[] for _ in range(n_base)]
    for u in range(3):
        for v in range(3, 6):
            base_adj[u].append(v)
            base_adj[v].append(u)
            
    def build_lift(m, voltages):
        """
        Build the voltage lift.
        voltages: dict mapping (u, v) -> voltage in Z_m for directed edge u->v
        """
        n = n_base * m
        adj = [[] for _ in range(n)]
        
        for u in range(n_base):
            for v in base_adj[u]:
                w = voltages.get((u, v), 0)
                for a in range(m):
                    src = u * m + a
                    dst = v * m + ((a + w) % m)
                    adj[src].append(dst)
        
        # The lift of a simple graph is simple if the voltages are chosen such that
        # no two edges between the same pair of base vertices have the same voltage difference?
        # Actually, K_{3,3} has only one edge between any u in U and v in V.
        # So there are no parallel edges in the base.
        # The lift will be simple if for any u, v, the mapping a -> a+w is a permutation (it is).
        # And we don't add the same edge twice.
        # Since base is simple, lift is simple.
        # We just need to ensure it's undirected properly.
        # The loop above adds both directions if we iterate over all directed edges.
        # Let's verify:
        # If we add (u, v) with w, we add src=u*m+a, dst=v*m+(a+w).
        # If we add (v, u) with -w, we add src=v*m+b, dst=u*m+(b-w).
        # Let b = a+w. Then dst = u*m + (a+w-w) = u*m+a = src.
        # So the edge is added twice (once from u to v, once from v to u).
        # We need to deduplicate or just build the adjacency list correctly.
        
        # Let's rebuild adj to be clean.
        final_adj = [[] for _ in range(n)]
        for u in range(n_base):
            for v in base_adj[u]:
                if u < v: # Process each undirected edge once
                    w_uv = voltages.get((u, v), 0)
                    for a in range(m):
                        src = u * m + a
                        dst = v * m + ((a + w_uv) % m)
                        final_adj[src].append(dst)
                        final_adj[dst].append(src)
        
        # Check 3-regular
        for v in range(n):
            if len(final_adj[v]) != 3:
                return None
        
        return final_adj
    
    best_result = None
    best_n = float('inf')
    
    # Lower bound for n is 512. n = 6m. So m >= 86.
    # Upper bound? Known best is 936, so m=156 works.
    # We search m from 86 to 160.
    
    rng = random.Random(42)
    
    for m in range(86, 161):
        if time.time() > deadline:
            break
            
        n = 6 * m
        if n >= best_n:
            continue
            
        # Try a few random voltage assignments
        # Number of trials can increase as m gets larger if time permits
        trials = 5
        for t in range(trials):
            if time.time() > deadline:
                break
                
            voltages = {}
            # Assign random voltages to edges (u, v) with u < v
            for u in range(3):
                for v in range(3, 6):
                    w = rng.randint(0, m - 1)
                    voltages[(u, v)] = w
                    voltages[(v, u)] = (-w) % m
            
            adj = build_lift(m, voltages)
            if adj is None:
                continue
            
            # Quick connectivity check
            visited = [False] * n
            queue = deque([0])
            visited[0] = True
            count = 1
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        count += 1
                        queue.append(v)
            if count != n:
                continue
            
            # Check girth
            if check_girth_bfs(n, adj, k, g):
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                best_result = (n, edges)
                best_n = n
                # If we found one, we can break inner loop to try smaller m?
                # No, we are iterating m upwards. We want the smallest n.
                # So if we find a valid graph for m, that's the best for this m.
                # We continue to next m? No, we want the GLOBAL smallest n.
                # Since m increases, n increases.
                # So the first valid m we find is the best.
                # We can return immediately?
                # Wait, maybe a larger m has a smaller n? No, n=6m.
                # So smallest m gives smallest n.
                # So we can return immediately upon finding the first valid graph.
                return best_result
                
    if best_result is not None:
        return best_result
        
    # Fallback: If no small m works, try a larger m with more trials or return a known large graph?
    # The task requires a valid graph. If we fail to find one in the search space, we should return a valid one.
    # Let's try m=156 (known to work for 936 vertices) with more trials.
    for m in range(160, 200):
        if time.time() > deadline:
            break
        n = 6 * m
        if n >= best_n:
            continue
        for t in range(20):
            if time.time() > deadline:
                break
            voltages = {}
            for u in range(3):
                for v in range(3, 6):
                    w = rng.randint(0, m - 1)
                    voltages[(u, v)] = w
                    voltages[(v, u)] = (-w) % m
            adj = build_lift(m, voltages)
            if adj is None:
                continue
            visited = [False] * n
            queue = deque([0])
            visited[0] = True
            count = 1
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        count += 1
                        queue.append(v)
            if count != n:
                continue
            if check_girth_bfs(n, adj, k, g):
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return (n, edges)
                
    # Emergency fallback: Return a valid graph if we have one, otherwise raise error?
    # The contract says "invalid result ... scores 0".
    # We must return a valid graph.
    # If we haven't found one, we might need a deterministic construction that is guaranteed to work but might be larger.
    # Or we can just return the best we found. If best is inf, we have a problem.
    # Let's assume the search space covers a working m.
    if best_result is not None:
        return best_result
        
    # Last resort: construct a large random regular graph and check?
    # This is risky. Let's just return the best found or a trivial valid graph if possible.
    # For k=3, g=16, we know 936 works.
    # If we didn't find it, we might be stuck.
    # But the search range 86-160 should likely find a smaller one if it exists, or 156 if not.
    
    raise ValueError("Failed to construct graph")
# EVOLVE-BLOCK-END
