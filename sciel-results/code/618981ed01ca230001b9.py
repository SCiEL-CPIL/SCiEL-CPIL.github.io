import random
import time
import math

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Moore bound for k=3, g=16 is 194. 
    # Best known is 936. 
    # Let's start from a reasonable range.
    # If we find a graph with n < seed_n, we win.
    # Seed n is ~ 10^5. 
    # Let's try to find n around 1000-2000.
    
    def generate_random_3_regular(n):
        # n must be even
        if n % 2 != 0:
            n += 1
        # Configuration model
        # Each vertex has 3 stubs
        stubs = []
        for i in range(n):
            for _ in range(3):
                stubs.append(i)
        
        random.shuffle(stubs)
        adj = {i: set() for i in range(n)}
        
        # Pair up stubs
        i = 0
        while i < len(stubs):
            u = stubs[i]
            v = stubs[i+1]
            i += 2
            if u == v:
                # Self loop, retry or fix
                # For simplicity, restart generation if loop occurs
                # But restarts are expensive. 
                # Better: swap with another stub to avoid loop?
                # Or just check at the end.
                pass
            if u in adj[v]:
                # Multi-edge, retry
                pass
            adj[u].add(v)
            adj[v].add(u)
            
        # Check validity
        valid = True
        for u in range(n):
            if len(adj[u]) != 3:
                valid = False
                break
            for v in adj[u]:
                if u == v or u not in adj[v]:
                    valid = False
                    break
        if not valid:
            return None
        return adj

    def compute_girth(adj, n):
        # BFS from each vertex
        min_girth = float('inf')
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = [start]
            for v in queue:
                for u in adj[v]:
                    if dist[u] == -1:
                        dist[u] = dist[v] + 1
                        queue.append(u)
                    else:
                        # Found a cycle
                        # The cycle length is dist[v] + dist[u] + 1? No.
                        # If we find a back edge, it's not simple BFS.
                        # Standard way: For each edge (u, v), shortest path u->v not using (u, v) + 1.
                        # This is O(n * (n+m)) which is O(n^2).
                        # For n=1000, 10^6 is fine.
                        pass
        
        # More efficient girth check:
        # For each edge (u, v), remove it, find dist(u, v), add 1.
        # But doing this for all edges is O(n * (n+m)).
        # Alternative: BFS from each node, but stop when depth > current min_girth / 2?
        # Actually, for each node u, BFS to find shortest cycle through u.
        # This is done by BFS from u, and for each neighbor v, if we reach v again, cycle is dist+1.
        # But BFS tree doesn't capture non-tree edges well for "shortest cycle through u".
        # Correct method: For each edge (u, v), find shortest path from u to v in G - {(u, v)}.
        # Let's implement this. It's robust.
        
        min_cycle = float('inf')
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
        
        for u, v in edges:
            # BFS from u to v avoiding edge (u, v)
            dist = [-1] * n
            dist[u] = 0
            queue = [u]
            found = False
            for x in queue:
                if dist[x] >= min_cycle: # Pruning
                    continue
                for y in adj[x]:
                    if (x == u and y == v) or (x == v and y == u):
                        continue
                    if dist[y] == -1:
                        dist[y] = dist[x] + 1
                        queue.append(y)
                        if y == v:
                            found = True
                            min_cycle = min(min_cycle, dist[y] + 1)
                            # We can break early if we found a very short cycle?
                            # But we need the shortest one.
                            # Actually, BFS guarantees shortest path.
                            # So once we reach v, dist[v] is the shortest.
                            break
                if found:
                    break
            if dist[v] != -1:
                min_cycle = min(min_cycle, dist[v] + 1)
        
        return min_cycle

    def local_repair(adj, n, target_girth):
        # Try to increase girth
        for _ in range(1000): # Max repair steps
            if time.time() > deadline:
                return adj, False
            curr_girth = compute_girth(adj, n)
            if curr_girth >= target_girth:
                return adj, True
            
            # Find a shortest cycle
            # We need to find the actual cycle, not just length.
            # Re-do BFS to find the path.
            # For simplicity, pick a random edge and try to swap it.
            # This is less effective but simpler.
            
            # Let's find a specific short cycle.
            # Use the edge loop above to find the edge (u, v) that forms the shortest cycle.
            # Then find the path.
            
            # For now, let's just try random swaps.
            # Pick a random edge (u, v).
            # Pick a random edge (x, y) not incident to u or v.
            # Swap.
            
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            
            if not edges:
                return adj, False
                
            random.shuffle(edges)
            
            swapped = False
            for u, v in edges:
                # Find neighbors of u and v
                nu = list(adj[u])
                nv = list(adj[v])
                
                # Try to swap (u, v) with some (x, y)
                # We need x in N(u) \ {v} and y in N(v) \ {u}
                # And (x, y) must be an edge? No, we add (u, x) and (v, y) and remove (u, v) and (x, y)?
                # No, standard 2-switch:
                # Remove (u, v) and (x, y). Add (u, x) and (v, y).
                # x must be neighbor of u? No, x can be any vertex?
                # No, to keep degree 3, x must be a neighbor of u or v?
                # If we remove (u, v), u and v have degree 2.
                # We must add an edge to u and an edge to v.
                # So we pick an edge (x, y) from the rest of the graph.
                # Remove (x, y). Add (u, x) and (v, y).
                # This keeps degrees: u (+1), v (+1), x (-1+1=0), y (-1+1=0).
                # Correct.
                
                # Find a random edge (x, y) not involving u or v.
                candidates = [e for e in edges if u not in e and v not in e]
                if not candidates:
                    continue
                
                x, y = random.choice(candidates)
                
                # Check if (u, x) and (v, y) are valid (not loops, not multi-edges)
                if x == u or y == v or x == v or y == u:
                    continue
                if x in adj[v] or y in adj[u]: # Check if new edges are already present?
                    # If (u, x) is already an edge, we'd have multi-edge.
                    # But (u, x) is not an edge because x is not in N(u) (x is from a disjoint edge).
                    # Wait, x is a vertex. Is x in N(u)?
                    # If x in N(u), then (u, x) is an edge.
                    # We are adding (u, x). If it exists, it's a multi-edge.
                    # So we must ensure x not in N(u) and y not in N(v).
                    # And x not in N(v) and y not in N(u) to avoid creating (v, x) or (u, y)?
                    # No, we only add (u, x) and (v, y).
                    # So we need:
                    # 1. x != u, y != v (no loops)
                    # 2. x not in adj[u] (no multi-edge u-x)
                    # 3. y not in adj[v] (no multi-edge v-y)
                    # 4. x != y (we removed x-y, so x!=y is guaranteed if simple graph)
                    # 5. Also, we must not create a loop? x!=u, y!=v covers that.
                    
                    pass # Logic above is sufficient if we check adj sets.
                
                if x in adj[u] or y in adj[v]:
                    continue
                
                # Perform swap
                adj[u].remove(v)
                adj[v].remove(u)
                adj[x].remove(y)
                adj[y].remove(x)
                
                adj[u].add(x)
                adj[x].add(u)
                adj[v].add(y)
                adj[y].add(v)
                
                swapped = True
                break
            
            if not swapped:
                return adj, False
                
        return adj, False

    # Main loop
    # Start from a lower bound. 
    # Moore bound 194. 
    # Let's try n from 500 to 2000.
    # If we find a valid graph, return it.
    
    best_n = None
    best_edges = None
    
    # Try a few random restarts for each n
    for n in range(500, 2000, 10):
        if time.time() > deadline:
            break
            
        # Try to find a graph with this n
        found = False
        for _ in range(5): # 5 restarts per n
            if time.time() > deadline:
                break
                
            adj = generate_random_3_regular(n)
            if adj is None:
                continue
                
            adj, valid = local_repair(adj, n, g)
            if valid:
                # Convert to edge list
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges
        # If not found, try next n
        # This might be too slow if 500-1000 don't work.
        # But 936 is the target.
        
    # Fallback: return the seed-like result or a large random graph?
    # If we didn't find one, we must return *something*.
    # The seed returns a valid graph. 
    # If our search fails, we should probably return a known valid graph or just a large random one that *might* work?
    # No, it must be valid.
    # If we can't find a small one, we can try to generate a large one and hope it has high girth?
    # Unlikely.
    
    # Let's try one more thing: if we fail, return the LUW graph from the seed?
    # No, we are rewriting from scratch.
    # We should ensure we return a valid graph.
    # If local repair fails, the graph might have girth < 16.
    # We must check validity before returning.
    
    # Let's modify the loop to store the best candidate and verify it.
    
    # Actually, the loop above returns immediately if valid.
    # If it finishes without returning, we need a fallback.
    # Let's generate a large graph (e.g. n=10000) and return it, hoping it's valid?
    # No, that's risky.
    
    # Better: If we don't find a small one, we can try to run the repair longer on a single n?
    # Or just return the first valid graph found in a broader search.
    
    # Let's assume we find one. The probability of finding a 936-vertex graph with girth 16 via random search is low.
    # But finding *any* graph with n < 10000 is likely if we search enough.
    # However, 150s is short.
    
    # Let's add a fallback: return a known small graph if we have one?
    # We don't.
    
    # Let's just return the first valid graph we find, even if n is large.
    # If we find nothing, we return a random 3-regular graph and hope? No, it must be valid.
    
    # Let's try to search up to n=5000.
    # And increase repair steps.
    
    # Re-structure:
    # For n in [500, 1000, 1500, ...]:
    #   Run repair for X seconds.
    #   If success, return.
    
    # If all fail, return a graph from the seed algorithm?
    # We can copy the seed algorithm as a fallback.
    
    # For now, let's just return None? No, must return valid.
    # Let's implement the seed algorithm as a fallback.
    
    # Actually, the prompt says "Keep the function names...".
    # If I return an invalid graph, score is 0.
    # So I MUST return a valid graph.
    
    # Let's include the seed code as a fallback.
    
    pass # Placeholder for actual code

# Let's write the actual code now.

