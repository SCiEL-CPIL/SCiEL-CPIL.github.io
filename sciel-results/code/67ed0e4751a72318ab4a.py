import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s kill
    
    n = 18
    best_l = n
    best_graph = None  # adj as list of bitmasks
    
    # --- Helper: check K4-free using bitmasks ---
    def is_k4_free(adj):
        for a in range(n):
            na = adj[a]
            for b in range(a+1, n):
                if not (na & (1 << b)): continue
                nb = adj[b]
                common_ab = na & nb
                # c must be > b and in common_ab
                for c in range(b+1, n):
                    if not (common_ab & (1 << c)): continue
                    nc = adj[c]
                    # triangle a-b-c exists. Check if any d in common_ab & nc
                    if common_ab & nc:
                        return False
        return True
    
    # --- Helper: find largest triangle-free induced subgraph size ---
    # Fast bitmask-based approach: iterate subsets by size, use incremental triangle check
    def get_largest_tf(adj):
        # Precompute: for each vertex v, the set of vertices u < v that form a triangle with some w
        # Actually, use a different approach: for each subset mask, check if it contains a triangle
        # A subset contains a triangle iff there exist u<v<w all in subset that form a triangle
        
        # Precompute triangle masks: for each pair (u,v) with u<v and edge(u,v),
        # the set of w > v such that (u,v,w) is a triangle
        # Then a subset S contains a triangle iff there exists (u,v,w) with u<v<w,
        # all in S, and (u,v,w) is a triangle.
        
        # Even faster: for each vertex w, precompute tri_mask[w] = set of pairs (u,v) with u<v<w
        # such that (u,v,w) is a triangle. Then S contains a triangle iff for some w in S,
        # tri_mask[w] has a pair (u,v) both in S.
        
        # Precompute: for each w, which pairs (u,v) with u<v<w form a triangle with w
        # tri_pairs[w] = list of (u,v) pairs
        # Or better: for each w, tri_mask[w] as a bitmask of "pair indices" - too complex
        
        # Simplest fast approach: for each subset mask (iterate by popcount),
        # check if any triangle is contained. Precompute all triangles as (u,v,w) tuples.
        # Then for each subset, check if any triangle's three vertices are all in the subset.
        
        # Precompute all triangles
        triangles = []
        for u in range(n):
            for v in range(u+1, n):
                if not (adj[u] & (1 << v)): continue
                common = adj[u] & adj[v]
                # w > v, in common
                w_mask = common & ~((1 << (v+1)) - 1)  # w > v
                while w_mask:
                    w = (w_mask & -w_mask).bit_length() - 1
                    triangles.append((u, v, w))
                    w_mask &= w_mask - 1
        
        if not triangles:
            return n  # no triangles, whole graph is triangle-free
        
        # For each triangle, precompute its bitmask
        tri_masks = []
        for (u, v, w) in triangles:
            m = (1 << u) | (1 << v) | (1 << w)
            tri_masks.append(m)
        
        # Now iterate k from n down to 1
        # For each k, check if any C(n,k) subset is triangle-free
        # A subset is triangle-free iff no triangle's mask is a subset of the subset mask
        
        # Optimization: for a subset mask S, it's triangle-free iff for all tri_masks, 
        # (S & tri_mask) != tri_mask (i.e., tri_mask is not a subset of S)
        
        # Even faster: precompute for each vertex v, the set of triangles that include v
        # But let's just iterate subsets and check against all triangles
        
        # For n=18, 2^18 = 262144 subsets. For each, check up to ~100 triangles.
        # That's ~26M operations, should be fast in Python with early termination.
        
        # Iterate by size from n down
        # Use combinations for each size
        for k in range(n, 0, -1):
            for subset in combinations(range(n), k):
                m = 0
                for v in subset:
                    m |= (1 << v)
                # Check if m contains any triangle
                tf = True
                for tm in tri_masks:
                    if (m & tm) == tm:
                        tf = False
                        break
                if tf:
                    return k
        return 0
    
    # --- Helper: generate random K4-free graph ---
    def gen_random_k4_free(p=0.5):
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Repair: while not K4-free, remove an edge from a K4
        for _ in range(200):
            if is_k4_free(adj):
                return adj
            # Find a K4 and remove an edge
            found = False
            for a in range(n):
                na = adj[a]
                for b in range(a+1, n):
                    if not (na & (1 << b)): continue
                    nb = adj[b]
                    common_ab = na & nb
                    for c in range(b+1, n):
                        if not (common_ab & (1 << c)): continue
                        nc = adj[c]
                        if common_ab & nc:
                            # K4 found: a,b,c,d where d is in common_ab & nc
                            dmask = common_ab & nc
                            d = (dmask & -dmask).bit_length() - 1
                            # Remove edge (a,d)
                            adj[a] &= ~(1 << d)
                            adj[d] &= ~(1 << a)
                            found = True
                            break
                    if found: break
                if found: break
            if not found:
                return adj
        return adj
    
    # --- Helper: local search to improve (reduce largest TF) ---
    def local_search(adj, max_iter=500):
        """Try edge additions/removals to reduce the largest triangle-free subgraph."""
        current_l = get_largest_tf(adj)
        
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            improved = False
            # Try adding a random non-edge
            for attempt in range(20):
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v: continue
                if adj[u] & (1 << v): continue
                # Try adding edge (u,v)
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if is_k4_free(adj):
                    new_l = get_largest_tf(adj)
                    if new_l < current_l:
                        current_l = new_l
                        improved = True
                        break
                    else:
                        # Revert
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                else:
                    # Revert
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
            
            if improved:
                continue
            
            # Try removing a random edge
            for attempt in range(20):
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v: continue
                if not (adj[u] & (1 << v)): continue
                # Try removing edge (u,v)
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                new_l = get_largest_tf(adj)
                if new_l < current_l:
                    current_l = new_l
                    improved = True
                    break
                else:
                    # Revert
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            if not improved:
                break
        
        return adj, current_l
    
    # --- Main search loop ---
    # Start with structured initialization: 3-partite with parts of 6
    # Then random restarts with local search
    
    # Initial structured graph: blow-up of C5 or similar
    # Try a few structured starts
    
    # Start 1: Complete 3-partite graph K_{6,6,6} minus some edges to be K4-free
    # K_{6,6,6} has K4s (pick 2 from two parts). Need to reduce.
    # Actually, let's just do random with various p values.
    
    p_values = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    while time.time() < deadline - 10:
        p = random.choice(p_values)
        adj = gen_random_k4_free(p)
        if adj is None:
            continue
        
        l = get_largest_tf(adj)
        
        if l < best_l:
            best_l = l
            best_graph = adj[:]
        
        # Local search to try to improve
        if time.time() < deadline - 15:
            adj_ls, l_ls = local_search(adj, max_iter=200)
            if l_ls < best_l:
                best_l = l_ls
                best_graph = adj_ls[:]
        
        if best_l <= 6:
            # Very good, can stop early
            break
    
    # Convert best_graph to list of edges
    if best_graph is None:
        # Fallback: empty graph
        return []
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_graph[i] & (1 << j):
                edges.append([i, j])
    return edges
