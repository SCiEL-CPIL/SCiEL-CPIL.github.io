import time
import random
import itertools
import numpy as np

def construct():
    # Exoo's 42-vertex (5,5)-free graph
    # This is a known valid graph.
    # We represent it as a list of bitmasks for speed.
    
    # The adjacency matrix of Exoo's graph (42x42)
    # I will use a specific known instance.
    # If I don't have the exact one, I will try to generate one via fast search.
    
    # Let's try a fast local search first, and if it fails, fallback to Paley(37).
    # But the seed already did that and got 1.0.
    # The seed's search was too slow.
    
    # Strategy: Use a much faster local search using NumPy or optimized Python bitmasks.
    # Or, simply hardcode the 42-vertex graph.
    
    # I will hardcode a known valid 42-vertex (5,5)-free graph.
    # Source: Exoo (1997) or similar.
    
    # Since I cannot browse the web, I will use a generic high-quality random search 
    # that is optimized to finish within time limits, aiming for 42.
    
    # However, the most robust "improvement" over a failing seed is to provide 
    # a known valid larger graph.
    
    # Let's try to construct a 42-vertex graph using a greedy method + local search 
    # with a very tight time limit, and if it finds one, return it.
    # If not, return 37.
    
    # Optimized Local Search:
    # 1. Start with random graph on 42 vertices.
    # 2. Count bad sets.
    # 3. Flip edges that reduce bad sets.
    
    # To make it fast, we precompute the "impact" of each edge.
    # But calculating impact requires iterating 9880 triples.
    # 42*41/2 = 861 edges.
    # 861 * 9880 ~ 8.5M operations per full scan.
    # In Python, this is ~1-2 seconds per full scan.
    # We have 300 seconds. So we can do ~100-200 full scans.
    # This is likely enough to find a solution if one exists nearby.
    
    # Let's implement this optimized search.
    
    n_target = 42
    deadline = time.time() + 290  # Safety margin
    
    # Initialize random graph
    adj = [0] * n_target
    for i in range(n_target):
        for j in range(i + 1, n_target):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # Precompute all triples for each pair (u, v)
    # This is the bottleneck of the seed, but we do it once.
    # C(40, 3) = 9880 triples per pair.
    # 861 pairs.
    # Total triples stored: 8.5M.
    # Memory: 8.5M * 3 bytes (if packed) or tuples.
    # Tuples are heavy. Let's use a list of lists of ints?
    # Or just generate on the fly but optimized?
    
    # Actually, we can precompute the list of (w, x, y) for each (u, v) once.
    # Let's try to do it efficiently.
    
    all_vertices = list(range(n_target))
    pair_triples = {}
    
    # Optimization: Generate triples only for the pairs we visit?
    # No, we need them for delta calculation.
    
    # Let's try a different approach: 
    # Instead of checking all 5-sets, we can check for K5 and Independent 5-sets separately.
    # A 5-set is a K5 if all 10 edges are present.
    # A 5-set is Independent if all 10 edges are absent.
    
    # Let's use a faster delta calculation.
    # When flipping (u, v), the edge count in a 5-set {u, v, w, x, y} changes by +1 or -1.
    # The set becomes bad if it was "almost" bad.
    # Specifically:
    # - If adding edge (u, v) makes it K5: it must have had 9 edges before.
    # - If removing edge (u, v) makes it K5: it must have had 10 edges before (impossible, it wasn't bad? No, if it was K5, it was bad. Removing edge makes it not K5. So delta is -1 for K5 count).
    # - If adding edge (u, v) makes it Independent: impossible (adding edge breaks independence).
    # - If removing edge (u, v) makes it Independent: it must have had 1 edge before (the one we are removing).
    
    # So, we only need to check sets that are "close" to being bad.
    # This is a huge optimization!
    
    # For a flip (u, v):
    # 1. If we are ADDING edge (u, v):
    #    - It creates a K5 if there exists {w, x, y} such that the subgraph on {u, v, w, x, y} has 9 edges (and the missing one is uv).
    #    - It does NOT create an Independent set.
    # 2. If we are REMOVING edge (u, v):
    #    - It removes a K5 if there exists {w, x, y} such that the subgraph on {u, v, w, x, y} has 10 edges (i.e., it was a K5).
    #    - It creates an Independent set if there exists {w, x, y} such that the subgraph on {u, v, w, x, y} has 1 edge (uv) and no other edges.
    
    # This reduces the check from "is it 0 or 10?" to "is it 9 or 10?" (for K5) and "is it 1?" (for Ind).
    # Still requires iterating triples.
    
    # Let's implement the fast search with this insight.
    
    def count_bad_sets(adj):
        bad = 0
        for u in range(n_target):
            for v in range(u + 1, n_target):
                if not ((adj[u] >> v) & 1):
                    continue
                # Check K5 containing u, v
                # We need 3 more vertices w, x, y connected to u and v and each other.
                # Common neighbors of u and v:
                common = adj[u] & adj[v]
                # Remove u, v
                common &= ~(1 << u)
                common &= ~(1 << v)
                # Get list of common neighbors
                cn_list = [i for i in range(n_target) if (common >> i) & 1]
                # We need to choose 3 from cn_list that form a triangle.
                # C(k, 3) checks.
                # If k is small, this is fast.
                # In a random graph, k is ~20. C(20, 3) = 1140.
                # This is much faster than 9880.
                
                for w, x, y in itertools.combinations(cn_list, 3):
                    if ((adj[w] >> x) & 1) and ((adj[w] >> y) & 1) and ((adj[x] >> y) & 1):
                        bad += 1
                        
                # Check Independent 5-set containing u, v?
                # No, if (u, v) is an edge, it can't be in an independent set.
                # So we only check K5 for existing edges.
                
        # Now check Independent 5-sets.
        # Iterate over non-edges (u, v).
        for u in range(n_target):
            for v in range(u + 1, n_target):
                if (adj[u] >> v) & 1:
                    continue
                # Common non-neighbors of u and v
                # Non-neighbors of u: ~adj[u] (masked)
                # Non-neighbors of v: ~adj[v] (masked)
                # We want vertices w such that w is not connected to u AND w is not connected to v.
                # And w != u, v.
                full_mask = (1 << n_target) - 1
                non_u = full_mask ^ adj[u]
                non_u &= ~(1 << u)
                non_v = full_mask ^ adj[v]
                non_v &= ~(1 << v)
                common_non = non_u & non_v
                
                cn_list = [i for i in range(n_target) if (common_non >> i) & 1]
                # We need to choose 3 from cn_list that form an independent set (no edges).
                for w, x, y in itertools.combinations(cn_list, 3):
                    if not ((adj[w] >> x) & 1) and not ((adj[w] >> y) & 1) and not ((adj[x] >> y) & 1):
                        bad += 1
                            
        return bad

    current_bad = count_bad_sets(adj)
    best_bad = current_bad
    best_adj = adj[:]
    
    # Local Search
    while time.time() < deadline:
        u = random.randint(0, n_target - 1)
        v = random.randint(0, n_target - 1)
        if u == v: continue
        if u > v: u, v = v, u
        
        edge_exists = (adj[u] >> v) & 1
        
        # Calculate delta
        delta = 0
        
        if edge_exists:
            # Removing edge (u, v)
            # 1. Might destroy a K5 (if {u,v,w,x,y} was K5)
            #    Check if there are w,x,y in common neighbors forming triangle.
            common = adj[u] & adj[v]
            common &= ~(1 << u)
            common &= ~(1 << v)
            cn_list = [i for i in range(n_target) if (common >> i) & 1]
            for w, x, y in itertools.combinations(cn_list, 3):
                if ((adj[w] >> x) & 1) and ((adj[w] >> y) & 1) and ((adj[x] >> y) & 1):
                    delta -= 1 # Bad set removed
            
            # 2. Might create an Independent set (if {u,v,w,x,y} becomes independent)
            #    This requires that w,x,y are non-neighbors of u and v, and w,x,y are independent.
            full_mask = (1 << n_target) - 1
            non_u = full_mask ^ adj[u]
            non_u &= ~(1 << u)
            non_v = full_mask ^ adj[v]
            non_v &= ~(1 << v)
            common_non = non_u & non_v
            cn_list = [i for i in range(n_target) if (common_non >> i) & 1]
            for w, x, y in itertools.combinations(cn_list, 3):
                if not ((adj[w] >> x) & 1) and not ((adj[w] >> y) & 1) and not ((adj[x] >> y) & 1):
                    delta += 1 # Bad set created
        
        else:
            # Adding edge (u, v)
            # 1. Might create a K5
            common = adj[u] & adj[v]
            common &= ~(1 << u)
            common &= ~(1 << v)
            cn_list = [i for i in range(n_target) if (common >> i) & 1]
            for w, x, y in itertools.combinations(cn_list, 3):
                if ((adj[w] >> x) & 1) and ((adj[w] >> y) & 1) and ((adj[x] >> y) & 1):
                    delta += 1 # Bad set created
            
            # 2. Might destroy an Independent set (if {u,v,w,x,y} was independent)
            #    This requires w,x,y non-neighbors of u,v and independent.
            full_mask = (1 << n_target) - 1
            non_u = full_mask ^ adj[u]
            non_u &= ~(1 << u)
            non_v = full_mask ^ adj[v]
            non_v &= ~(1 << v)
            common_non = non_u & non_v
            cn_list = [i for i in range(n_target) if (common_non >> i) & 1]
            for w, x, y in itertools.combinations(cn_list, 3):
                if not ((adj[w] >> x) & 1) and not ((adj[w] >> y) & 1) and not ((adj[x] >> y) & 1):
                    delta -= 1 # Bad set removed

        if delta < 0:
            # Accept
            if edge_exists:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            current_bad += delta
        else:
            # Simple hill climbing: only accept if delta < 0?
            # Or simulated annealing?
            # Let's stick to strict improvement for now to ensure convergence.
            # If stuck, restart?
            pass

        if current_bad < best_bad:
            best_bad = current_bad
            best_adj = adj[:]
            
        if best_bad == 0:
            break
            
    if best_bad == 0:
        edges = []
        for i in range(n_target):
            for j in range(i + 1, n_target):
                if (best_adj[i] >> j) & 1:
                    edges.append((i, j))
        return n_target, edges
    else:
        # Fallback to Paley(37)
        q = 37
        residues = {(x * x) % q for x in range(1, q)}
        seed_edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
        return q, seed_edges
