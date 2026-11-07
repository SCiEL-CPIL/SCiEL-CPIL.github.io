"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    t0 = time.time()
    deadline = t0 + 80.0  # safety margin before 90s wall-clock kill
    n = 15
    
    def is_graphical(deg):
        """Check if degree sequence is graphical (Havel-Hakimi)."""
        d = sorted(deg, reverse=True)
        while d:
            if d[0] < 0:
                return False
            k = d.pop(0)
            if k > len(d):
                return False
            for i in range(k):
                d[i] -= 1
            d.sort(reverse=True)
        return True
    
    def realize_hh(deg):
        """Realize degree sequence via Havel-Hakimi, return edge list or None."""
        d = list(deg)
        edges = []
        # We'll track which vertex has which remaining degree
        # Use a priority approach: repeatedly pick highest-degree vertex
        # and connect to next-highest available vertices
        remaining = list(range(n))
        rem_deg = list(deg)
        
        for _ in range(n):
            # Find vertex with highest remaining degree
            if not remaining:
                break
            max_idx = max(range(len(remaining)), key=lambda i: rem_deg[remaining[i]])
            v = remaining[max_idx]
            kv = rem_deg[v]
            if kv == 0:
                remaining.pop(max_idx)
                continue
            if kv > len(remaining) - 1:
                return None
            # Connect to next kv vertices with highest degrees
            others = [i for i in range(len(remaining)) if i != max_idx]
            others.sort(key=lambda i: rem_deg[remaining[i]], reverse=True)
            for j in range(kv):
                u = remaining[others[j]]
                edges.append([v, u])
                rem_deg[u] -= 1
            rem_deg[v] = 0
            remaining.pop(max_idx)
        
        # Verify
        if len(edges) != sum(deg) // 2:
            return None
        # Check no duplicates
        edge_set = set()
        for e in edges:
            key = (min(e[0], e[1]), max(e[0], e[1]))
            if key in edge_set:
                return None
            edge_set.add(key)
        return edges
    
    def check_valid(edges):
        """Check if graph has no 3-edge path with equal-degree endpoints."""
        if not edges:
            return True
        deg = [0] * n
        adj = [set() for _ in range(n)]
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
        
        # Check every 3-edge path s-u-v-t
        for u, v in edges:
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
        return True
    
    def count_edges(edges):
        return len(edges)
    
    # Strategy: try degree sequences with distinct values, maximize sum
    # Degrees must be 0..14, distinct, graphical
    # Start with high-degree targets and work down
    
    best_edges = None
    best_count = 0
    
    # Try permutations of high-degree subsets
    # A degree sequence with all distinct values: pick 15 distinct values from 0..14
    # That's just a permutation of 0..14, but not all are graphical
    # Actually we can have repeated... wait, no - constraint requires distinct degrees
    
    # Wait - re-reading: "Reject if any simple three-edge path has endpoints of equal degree"
    # This means: for ALL 3-edge paths, endpoints must have DIFFERENT degrees
    # If all vertices have distinct degrees, this is automatically satisfied
    
    # So we need a graphical degree sequence with all distinct values
    # Maximize sum of degrees (which equals 2 * edges)
    
    # Try: start with [14,13,12,11,10,9,8,7,6,5,4,3,2,1,0] and see if graphical
    # If not, reduce some values
    
    # Greedy: try to make degrees as high as possible with all distinct
    # Use backtracking on degree assignments
    
    def search_degrees():
        """Find densest graphical degree sequence with distinct values."""
        # Degrees must be distinct, each in [0, 14]
        # So it's a permutation of 15 distinct values from 0..14
        # But we want to maximize sum, so prefer high values
        # The maximum possible distinct degrees summing to max: 14+13+12+...+0 = 105
        # But 105 is odd, can't be sum of degrees (must be even)
        # So max sum is 104 or 103... let's try
        
        # Actually for distinct degrees on n vertices, the sequence must be a permutation
        # of some 15 distinct values from 0 to 14. Since there are exactly 15 values 0..14,
        # it must be exactly {0,1,2,...,14}, sum = 105 (odd) - not graphical!
        
        # So we can't use all distinct values from 0..14. We need to skip one value.
        # Remove value k: sum = 105 - k, must be even, so k must be odd.
        # To maximize sum, minimize k (odd): k=1, sum=104
        # Try degrees: {0,2,3,4,5,6,7,8,9,10,11,12,13,14} (removed 1)
        # Or k=3: sum=102, etc.
        
        # Actually wait - we don't have to use values from 0..14 exclusively.
        # We just need 15 distinct non-negative integers, each <= 14.
        # The only set of 15 distinct values in [0,14] is {0,1,...,14}.
        # So we MUST use exactly {0,1,...,14}, but sum=105 is odd - impossible!
        
        # Therefore, we CANNOT have all 15 vertices with distinct degrees!
        # We need at least two vertices to share a degree.
        
        # Revised approach: allow some degree collisions, but ensure that
        # for any 3-edge path, the two endpoints don't have the same degree.
        
        return None
    
    # New approach: allow degree repeats but carefully check the path constraint.
    # Greedy construction: start with empty graph, add edges that don't violate constraint.
    
    def greedy_build():
        """Build graph greedily, adding edges that maintain validity."""
        deg = [0] * n
        adj = [set() for _ in range(n)]
        edges = []
        
        def would_violate(u, v):
            """Check if adding edge (u,v) would create a violating 3-edge path."""
            # New paths involving this edge: s-u-v-t and s-v-u-t
            # For each such path, check if deg[s] == deg[t] (in the graph WITH this edge)
            # But degrees change when we add the edge...
            # Actually, the constraint is on the FINAL graph. 
            # So we need to check: after adding (u,v), does any 3-edge path have equal-degree endpoints?
            
            # This is expensive. Instead, let's just build and check at the end.
            # Or: only add edge if it doesn't create a NEW violation.
            # A new violation must involve the new edge (u,v).
            # Paths: s-u-v-t where s in adj[u]\{v}, t in adj[v]\{u}, s != t
            # In the new graph, deg[s] and deg[t] are unchanged (only u and v get +1)
            # So check: for all s in adj[u]\{v}, t in adj[v]\{u}, s!=t: deg[s] != deg[t]
            # Wait, but u and v degrees change, but s and t degrees don't.
            # So we just need: no s in adj[u]\{v} and t in adj[v]\{u} with s!=t and deg[s]==deg[t]
            
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return True
            return False
        
        # Try all edges in random order, add if valid
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if u in adj[v]:
                continue
            if would_violate(u, v):
                continue
            # Add edge
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
            edges.append([u, v])
        
        return edges
    
    # Run greedy with multiple random seeds
    for trial in range(500):
        if time.time() > deadline:
            break
        random.seed(trial)
        edges = greedy_build()
        if edges and count_edges(edges) > best_count:
            best_count = count_edges(edges)
            best_edges = edges
            # Verify
            if not check_valid(best_edges):
                best_edges = None
                best_count = 0
    
    # Also try: start from a high-degree base and remove violating edges
    if time.time() < deadline:
        # Start with a dense graph and remove edges to fix violations
        def dense_start():
            """Create a dense starting graph."""
            deg = [0] * n
            adj = [set() for _ in range(n)]
            edges = []
            # Try to make degrees roughly equal first, then adjust
            # Or just start with a random dense graph
            all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(all_edges)
            for u, v in all_edges[:best_count if best_count else 50]:
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1
                edges.append([u, v])
            return edges, deg, adj
        
        for trial in range(100):
            if time.time() > deadline:
                break
            random.seed(1000 + trial)
            edges, deg, adj = dense_start()
            # Remove edges that cause violations
            changed = True
            while changed and time.time() < deadline:
                changed = False
                for idx in range(len(edges) - 1, -1, -1):
                    if time.time() > deadline:
                        break
                    u, v = edges[idx]
                    # Check if removing this edge helps
                    # Actually, let's check if this edge is part of a violation
                    # Find a violation involving this edge
                    for s in adj[u]:
                        if s == v:
                            continue
                        for t in adj[v]:
                            if t == u or t == s:
                                continue
                            if deg[s] == deg[t]:
                                # This edge is in a violating path
                                # Remove it
                                adj[u].discard(v)
                                adj[v].discard(u)
                                deg[u] -= 1
                                deg[v] -= 1
                                edges.pop(idx)
                                changed = True
                                break
                        if changed:
                            break
                    if changed:
                        break
            
            if len(edges) > best_count and check_valid(edges):
                best_count = len(edges)
                best_edges = edges
    
    if best_edges is None:
        # Fallback: star graph (always valid)
        best_edges = [[0, v] for v in range(1, n)]
    
    # Convert to proper types
    result = []
    for e in best_edges:
        result.append([int(e[0]), int(e[1])])
    return result

# EVOLVE-BLOCK-END
