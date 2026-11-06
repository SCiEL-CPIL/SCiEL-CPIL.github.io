import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    n = 20

    def is_k33_free(adj):
        """Full check for verification."""
        for a in range(n):
            for b in range(a + 1, n):
                common_ab = adj[a] & adj[b]
                if len(common_ab) < 3:
                    continue
                for c in range(b + 1, n):
                    if len(common_ab & adj[c]) >= 3:
                        return False
        return True

    def find_violation(adj):
        """
        Find a triple (u, v, w) such that |N(u) ∩ N(v) ∩ N(w)| >= 3.
        Returns (u, v, w, common_neighbours_set) or None.
        """
        for u in range(n):
            for v in range(u + 1, n):
                common_uv = adj[u] & adj[v]
                if len(common_uv) < 3:
                    continue
                for w in range(v + 1, n):
                    common_uvw = common_uv & adj[w]
                    if len(common_uvw) >= 3:
                        return (u, v, w, common_uvw)
        return None

    def clean_graph(adj, rng):
        """
        Remove edges to make the graph K3,3-free.
        """
        changed = True
        while changed:
            changed = False
            violation = find_violation(adj)
            if violation:
                u, v, w, common = violation
                # Pick a random common neighbor x
                x = rng.choice(list(common))
                # Remove one of the edges (u,x), (v,x), (w,x)
                # This reduces the intersection size by 1
                # We choose randomly which edge to remove
                edge_to_remove = rng.choice([(u, x), (v, x), (w, x)])
                a, b = edge_to_remove
                adj[a].discard(b)
                adj[b].discard(a)
                changed = True
        return adj

    def get_edge_count(adj):
        return sum(len(s) for s in adj) // 2

    def try_add_edge(adj, u, v):
        """
        Attempt to add edge (u,v).
        Returns True if successful, False otherwise.
        """
        if u == v:
            return False
        if v in adj[u]:
            return True
        
        adj[u].add(v)
        adj[v].add(u)
        
        # Check if this addition created a K3,3
        # Only triples involving u or v can be affected
        # Triples (u, x, y): N(u) ∩ N(x) ∩ N(y)
        for x in range(n):
            if x == u:
                continue
            for y in range(x + 1, n):
                if y == u:
                    continue
                if len(adj[u] & adj[x] & adj[y]) > 2:
                    adj[u].discard(v)
                    adj[v].discard(u)
                    return False
        
        # Triples (v, x, y): N(v) ∩ N(x) ∩ N(y)
        for x in range(n):
            if x == v:
                continue
            for y in range(x + 1, n):
                if y == v:
                    continue
                if len(adj[v] & adj[x] & adj[y]) > 2:
                    adj[u].discard(v)
                    adj[v].discard(u)
                    return False
        
        return True

    def local_search(adj, rng, time_limit):
        """
        Local search to maximize edges.
        """
        start = time.time()
        improved = True
        while improved and time.time() - start < time_limit:
            improved = False
            
            # 1. Try to add any missing edge
            missing = [(u, v) for u in range(n) for v in range(u + 1, n) if v not in adj[u]]
            rng.shuffle(missing)
            for u, v in missing:
                if try_add_edge(adj, u, v):
                    improved = True
            
            # 2. Try swaps
            if time.time() - start > time_limit * 0.8:
                break
                
            existing = [(u, v) for u in range(n) for v in range(u + 1, n) if v in adj[u]]
            missing = [(u, v) for u in range(n) for v in range(u + 1, n) if v not in adj[u]]
            
            if not existing or not missing:
                break

            rng.shuffle(existing)
            rng.shuffle(missing)
            
            swaps_to_try = 200
            for _ in range(swaps_to_try):
                if not existing or not missing:
                    break
                a, b = rng.choice(existing)
                c, d = rng.choice(missing)
                
                # Remove (a,b)
                adj[a].discard(b)
                adj[b].discard(a)
                
                # Try add (c,d)
                if try_add_edge(adj, c, d):
                    improved = True
                    # We successfully swapped.
                    # Update existing and missing lists? 
                    # For efficiency, we break and recompute in the next loop.
                    break
                else:
                    # Restore (a,b)
                    adj[a].add(b)
                    adj[b].add(a)

        return adj

    # Main Loop
    rng = random.Random(42)
    
    best_adj = None
    best_count = 0
    
    # Try multiple dense initializations
    for _ in range(10):
        if time.time() > deadline - 10:
            break
        
        # Initialize with dense random graph
        p = 0.8 + 0.05 * rng.random() # p in [0.8, 0.85]
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < p:
                    adj[i].add(j)
                    adj[j].add(i)
        
        # Clean it up
        adj = clean_graph(adj, rng)
        count = get_edge_count(adj)
        
        if count > best_count:
            best_adj = adj
            best_count = count

    if best_adj is None:
        # Fallback
        best_adj = [set() for _ in range(n)]
        for i in range(2):
            for j in range(2, 20):
                best_adj[i].add(j)
                best_adj[j].add(i)
        best_count = get_edge_count(best_adj)

    # Local Search
    if time.time() < deadline - 5:
        time_left = deadline - time.time() - 2
        if time_left > 1:
            best_adj = local_search(best_adj, rng, min(30.0, time_left))
            best_count = get_edge_count(best_adj)

    # Final Verification
    if not is_k33_free(best_adj):
        # Fallback to a known valid graph
        edges = [[a, b] for a in range(2) for b in range(2, 20)]
        return edges

    # Convert to edge list
    edges = []
    for u in range(n):
        for v in sorted(best_adj[u]):
            if u < v:
                edges.append([u, v])
                
    return edges
