import time
import random
import sys
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Strategy:
    1. Attempt to find a small graph via Random Lift construction.
       - Start with a small base 3-regular graph.
       - Apply random voltage lifts to increase size and potentially girth.
       - Verify girth exactly.
    2. If no valid lift is found within time budget, fall back to a standard 
       LUW construction with minimal depth d that guarantees girth >= g.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin

    def verify_girth(n, edges, k, g):
        """
        Returns True if girth >= g.
        Uses non-backtracking walks from each vertex up to depth g//2.
        """
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        for i in range(n):
            if len(adj[i]) != k:
                return False
        
        max_depth = g // 2
        
        for start in range(n):
            # State: (current_node, prev_node)
            # We track visited states to detect cycles
            visited = {}  # (u, v) -> depth
            queue = deque()
            
            for v in adj[start]:
                queue.append((v, start, 1))
                visited[(v, start)] = 1
            
            while queue:
                u, prev, depth = queue.popleft()
                
                if depth == max_depth:
                    continue
                
                for v in adj[u]:
                    if v == prev:
                        continue
                    
                    # If we hit the start node, we have a cycle
                    if v == start:
                        if depth + 1 < g:
                            return False
                        continue
                    
                    state = (v, u)
                    if state in visited:
                        # Cycle found
                        # The cycle length is depth - visited[state] + 1?
                        # Actually, if we see (v, u) again, the closed walk length is depth - visited[state].
                        # The shortest cycle in this walk is <= depth - visited[state].
                        if depth - visited[state] < g:
                            return False
                    else:
                        visited[state] = depth
                        queue.append((v, u, depth + 1))
                        
        return True

    def random_3_regular_base(n):
        """Generate a random 3-regular graph of size n (n even)."""
        if n % 2 != 0:
            return None
        if n < 4:
            return None
        
        # Configuration model with retries for simplicity
        for _ in range(100):
            half_edges = [i for i in range(n) for _ in range(3)]
            random.shuffle(half_edges)
            adj = [[] for _ in range(n)]
            valid = True
            for i in range(0, len(half_edges), 2):
                u, v = half_edges[i], half_edges[i+1]
                if u == v:
                    valid = False
                    break
                if v in adj[u]:
                    valid = False
                    break
                adj[u].append(v)
                adj[v].append(u)
            
            if not valid:
                continue
            
            # Check degrees
            if all(len(adj[i]) == 3 for i in range(n)):
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges, adj
        return None

    def lift_graph(base_n, base_edges, base_adj, m):
        """
        Apply a random voltage lift of size m to the base graph.
        Returns the lifted graph's edges and adjacency.
        """
        # Each vertex in base becomes a fiber of m vertices.
        # Total vertices: base_n * m
        # Edges: For each edge (u, v) in base, and for each x in 0..m-1,
        # connect (u, x) to (v, (x + perm_uv(x)) % m)?
        # Simpler: Use random permutations for each directed edge.
        
        n_lifted = base_n * m
        # Map base vertex u and layer x to lifted vertex u * m + x
        
        # We need to assign a permutation (or voltage) to each directed edge (u, v).
        # For 3-regular, each vertex has 3 outgoing edges.
        # We can just assign a random shift for each edge instance.
        
        lifted_adj = [[] for _ in range(n_lifted)]
        
        # Assign random shifts to each edge in the base graph
        # Base edges are undirected, but lift requires direction.
        # For each undirected edge (u, v), we need a permutation P_uv of Z_m.
        # The inverse permutation P_vu = P_uv^{-1}.
        
        edge_perms = {}
        for u, v in base_edges:
            # Generate a random permutation of 0..m-1
            perm = list(range(m))
            random.shuffle(perm)
            # Inverse permutation
            inv_perm = [0] * m
            for i, p in enumerate(perm):
                inv_perm[p] = i
            edge_perms[(u, v)] = perm
            edge_perms[(v, u)] = inv_perm
            
        for u, v in base_edges:
            perm = edge_perms[(u, v)]
            for x in range(m):
                u_lifted = u * m + x
                v_lifted = v * m + perm[x]
                lifted_adj[u_lifted].append(v_lifted)
                lifted_adj[v_lifted].append(u_lifted)
                
        # Check if simple (no multi-edges)
        # In a lift, multi-edges can occur if the base had multi-edges (it doesn't) 
        # or if different edges map to the same pair.
        # Since base is simple, and permutations are bijective, 
        # edges from different base edges might collide?
        # If (u, v) and (u, w) are edges, and perm_uv(x) = y, perm_uw(x) = y, 
        # then (u, x) connects to (v, y) and (w, y). These are distinct unless v=w.
        # So no multi-edges from same base vertex.
        # What about (u, x) to (v, y) and (v, y') to (u, x')?
        # It's a simple graph if the base is simple and we use proper lifts.
        
        edges = []
        for i in range(n_lifted):
            for j in lifted_adj[i]:
                if i < j:
                    edges.append((i, j))
                    
        return n_lifted, edges, lifted_adj

    best_n = float('inf')
    best_edges = []
    
    # --- Phase 1: Random Lift Search ---
    # Try to find a graph with n < 2000
    # Base sizes: 4, 6, 8, 10, 12, 14, 16, 18, 20
    # Lift sizes m: such that base_n * m is in [512, 2000]
    
    base_sizes = [4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 30]
    
    for base_n in base_sizes:
        if time.time() > deadline:
            break
            
        # Generate a base graph
        base_res = random_3_regular_base(base_n)
        if not base_res:
            continue
        b_n, b_edges, b_adj = base_res
        
        # Determine possible lift sizes m
        min_m = max(1, 512 // base_n)
        max_m = min(2000 // base_n, 50) # Cap m to keep verification fast
        
        # Try a few random lifts for each m
        # Prioritize m that result in n close to 512-1000
        candidate_ms = []
        for m in range(min_m, max_m + 1):
            n_total = base_n * m
            if 512 <= n_total <= 2000:
                candidate_ms.append(m)
                
        random.shuffle(candidate_ms)
        
        # Limit attempts per base size to keep time down
        attempts_per_m = 5
        for m in candidate_ms[:5]: # Try at most 5 different m values
            if time.time() > deadline:
                break
                
            for _ in range(attempts_per_m):
                if time.time() > deadline:
                    break
                    
                n_lifted, edges_lifted, adj_lifted = lift_graph(base_n, b_edges, b_adj, m)
                
                # Quick check: Girth
                if verify_girth(n_lifted, edges_lifted, k, g):
                    if n_lifted < best_n:
                        best_n = n_lifted
                        best_edges = edges_lifted
                        
    # --- Phase 2: Fallback to LUW ---
    # If we didn't find a small graph, use LUW.
    # For k=3, g=16, LUW with d=7 gives n=4374, girth=16.
    # LUW with d=6 gives n=1458, girth=14 (insufficient).
    
    if best_n == float('inf') or best_n > 4374:
        # Implement LUW construction for k=3
        def luw_construct_k3(d):
            q = 3
            # Rule table for k=3
            # Standard LUW rules
            rule = {}
            for c in range(2, d + 1):
                # Simple heuristic rule for k=3
                # We need to ensure girth.
                # Standard LUW uses specific linear recurrences.
                # For simplicity and reliability, we use a known good rule set.
                # If this fails, we return None.
                rule[c] = ((c - 2) % q, (c - 1) % q) # Placeholder, might not guarantee girth
            
            # This simple LUW might not guarantee girth 16 for d=7.
            # To be safe, we should use the verified LUW from the prompt if it works, 
            # or just return a large random graph? No, must be valid.
            
            # Let's use the prompt's LUW logic but with d=7.
            # The prompt's LUW was:
            # rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
            # Let's replicate that specific logic for d=7.
            
            if d < 5:
                return None, None
                
            # Re-implementing the specific LUW from the prompt to ensure it works
            # as it was the "current" strategy.
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]] # For q=4? No, q=3.
            # For q=3, we use mod 3.
            add = lambda x, y: (x + y) % 3
            sub = lambda x, y: (x - y) % 3
            mul = lambda x, y: x * y % 3
            
            rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
            for c in range(5, d + 1):
                r, c0 = (c - 5) % 4, c - (c - 5) % 4
                rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

            def neighbours(vertex):
                side, x = vertex
                for first in range(3):
                    y = [None, first] + [0] * (d - 1)
                    for c in range(2, d + 1):
                        a, b = rule[c]
                        if side == 0:
                            y[c] = add(x[c], mul(y[a], x[b]))
                        else:
                            y[c] = sub(x[c], mul(x[a], y[b]))
                    yield 1 - side, tuple(y)

            start = (0, (None,) + (0,) * d)
            index, order, edges = {start: 0}, [start], []
            for vertex in order:
                for other in neighbours(vertex):
                    if other not in index:
                        index[other] = len(order)
                        order.append(other)
                    if vertex[0] == 0:
                        edges.append((index[vertex], index[other]))
            return len(order), edges

        n_luw, edges_luw = luw_construct_k3(7)
        if n_luw is not None:
            # Verify it
            if verify_girth(n_luw, edges_luw, k, g):
                if n_luw < best_n:
                    best_n = n_luw
                    best_edges = edges_luw

    if best_n == float('inf'):
        # Last resort: Return a valid graph even if large
        # Try LUW d=8
        n_luw, edges_luw = luw_construct_k3(8)
        if n_luw and verify_girth(n_luw, edges_luw, k, g):
            return n_luw, edges_luw
            
        # If all else fails, return a trivial graph (will score 0, but better than crash)
        # Actually, we must return something.
        return 0, []

    return best_n, best_edges

if __name__ == "__main__":
    n, edges = construct(3, 16)
    print(f"Constructed graph with {n} vertices")
