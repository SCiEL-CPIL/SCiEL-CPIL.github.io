"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _has_c5(n, adj):
    """Check if graph has a 5-cycle using adjacency sets."""
    # A 5-cycle: v0-v1-v2-v3-v4-v0 with all distinct
    # Use: for each edge (u,v), check if there's a path of length 3 from u to v avoiding the direct edge
    # More efficiently: count triangles in the "square" or use a direct method
    # Direct: for each 4-tuple (a,b,c,d) with a<b<c<d, check if it forms a 5-cycle with some 5th vertex
    # Better: iterate over all 5-subsets and check if any induces a C5 (cycle or subgraph)
    # That's O(C(n,5) * 120) which for n=33 is C(33,5)*120 ~ 237336*120 ~ 28M - too slow per check
    
    # Faster: for each pair (u,v), compute number of common neighbors, etc.
    # A 5-cycle exists iff there exist distinct u,v,w,x,y with edges uv, vw, wx, xy, yu
    # Equivalent: there exist distinct u,v with a path of length 4 between them that doesn't use edge uv directly
    # OR: there exist distinct u,v,w with u~w and a path of length 3 from u to w not using edge uw
    
    # Method: for each edge (u,v), check if there's a path u-a-b-c-v where a,b,c distinct from u,v and from each other
    # This is O(E * n^3) worst case but with early termination
    
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            # Look for path u - a - b - c - v with a,b,c distinct, a!=v, c!=u, a!=b, b!=c
            # a must be neighbor of u, c must be neighbor of v, b must be common neighbor of a and c
            # with a != c, a != v, c != u, b != u, b != v, b != a, b != c
            for a in adj[u]:
                if a == v:
                    continue
                for c in adj[v]:
                    if c == u or c == a:
                        continue
                    # Check if a and c share a common neighbor b (b != u, v, a, c)
                    common = adj[a] & adj[c]
                    common.discard(u)
                    common.discard(v)
                    common.discard(a)
                    common.discard(c)
                    if common:
                        return True
    return False

def _alpha(n, adj):
    """Compute independence number using branch and bound."""
    best = [0]
    
    def branch(candidates, current_size):
        if not candidates:
            if current_size > best[0]:
                best[0] = current_size
            return
        if current_size + len(candidates) <= best[0]:
            return
        # Pick vertex with fewest neighbors in candidates
        v = min(candidates, key=lambda x: len(adj[x] & candidates))
        # Branch 1: include v
        new_candidates = candidates - {v} - adj[v]
        branch(new_candidates, current_size + 1)
        # Branch 2: exclude v
        branch(candidates - {v}, current_size)
    
    branch(set(range(n)), 0)
    return best[0]

def _has_independent_set_9(n, adj):
    """Check if there's an independent set of size 9."""
    # Quick check: if alpha < 9, return False
    # Use a faster bounded search
    found = [False]
    
    def search(candidates, current, size):
        if found[0]:
            return
        if size == 9:
            found[0] = True
            return
        if not candidates:
            return
        if size + len(candidates) < 9:
            return
        v = min(candidates, key=lambda x: len(adj[x] & candidates))
        # Include v
        new_candidates = candidates - {v} - adj[v]
        search(new_candidates, current | {v}, size + 1)
        # Exclude v
        search(candidates - {v}, current, size)
    
    search(set(range(n)), set(), 0)
    return found[0]

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def construct():
    deadline = time.time() + 140  # safety margin
    
    # Start with 8 disjoint K4 (n=32, alpha=8, C5-free)
    n = 32
    edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    
    best_n = n
    best_edges = list(edges)
    
    adj = _build_adj(n, edges)
    
    # Try to add vertices one by one, finding good edge sets
    # For each new vertex, try all subsets of edges to existing vertices
    # that maintain C5-freeness and alpha < 9
    
    random.seed(42)
    
    for target_n in range(33, 36):
        if time.time() > deadline:
            break
        
        # Start from current best and try to add (target_n - n) vertices
        cur_n = n
        cur_edges = list(best_edges)
        cur_adj = _build_adj(cur_n, cur_edges)
        
        for new_v in range(cur_n, target_n):
            if time.time() > deadline:
                break
            
            # Try random edge patterns for new vertex
            # The new vertex can connect to any subset of existing vertices
            # We need: no C5 formed, and alpha remains < 9
            
            # Heuristic: connect to a random subset, check constraints
            found = False
            for attempt in range(500):
                if time.time() > deadline:
                    break
                
                # Random subset of edges from new_v to existing vertices
                # Bias toward connecting to many vertices to keep alpha low
                # but not too many to avoid C5
                k = random.randint(cur_n // 3, 2 * cur_n // 3)
                neighbors = random.sample(range(cur_n), k)
                
                new_adj = [set(s) for s in cur_adj]
                new_adj.append(set(neighbors))
                for nb in neighbors:
                    new_adj[nb].add(cur_n)
                
                # Check C5-free
                if _has_c5(cur_n + 1, new_adj):
                    continue
                
                # Check alpha < 9
                if _has_independent_set_9(cur_n + 1, new_adj):
                    continue
                
                # Valid!
                cur_edges = list(cur_edges)
                for nb in neighbors:
                    cur_edges.append([new_v, nb])
                cur_n = new_v + 1
                cur_adj = new_adj
                found = True
                break
            
            if not found:
                # Try more targeted: connect to complement of a small set
                # to minimize alpha increase
                found = False
                for attempt in range(200):
                    if time.time() > deadline:
                        break
                    
                    # Connect to all except a random small set
                    skip_size = random.randint(0, 3)
                    skip = random.sample(range(cur_n), min(skip_size, cur_n))
                    neighbors = [v for v in range(cur_n) if v not in skip]
                    
                    new_adj = [set(s) for s in cur_adj]
                    new_adj.append(set(neighbors))
                    for nb in neighbors:
                        new_adj[nb].add(cur_n)
                    
                    if _has_c5(cur_n + 1, new_adj):
                        continue
                    
                    if _has_independent_set_9(cur_n + 1, new_adj):
                        continue
                    
                    cur_edges = list(cur_edges)
                    for nb in neighbors:
                        cur_edges.append([new_v, nb])
                    cur_n = new_v + 1
                    cur_adj = new_adj
                    found = True
                    break
                
                if not found:
                    break  # Can't add this vertex, stop
        
        if cur_n > best_n:
            best_n = cur_n
            best_edges = cur_edges
            n = cur_n  # Continue from here
    
    # Final validation
    final_adj = _build_adj(best_n, best_edges)
    if _has_c5(best_n, final_adj) or _has_independent_set_9(best_n, final_adj):
        # Fallback to safe 32-vertex construction
        edges = []
        for b in range(8):
            base = 4 * b
            for i in range(4):
                for j in range(i + 1, 4):
                    edges.append([base + i, base + j])
        return {'n': 32, 'edges': edges}
    
    return {'n': best_n, 'edges': best_edges}

# EVOLVE-BLOCK-END
