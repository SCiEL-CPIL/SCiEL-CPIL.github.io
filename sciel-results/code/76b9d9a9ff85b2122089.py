"""Fewest four-cycles in graphs with 31 vertices and 91 edges."""
import time
import random
import itertools
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    t_start = time.time()
    deadline = t_start + 115.0  # 5s safety margin before 120s limit
    
    n = 31
    target_edges = 91
    c4_free_target = 90
    
    # Helper: count C4s in a graph given by edge list
    def count_c4s(edges, n=31):
        if not edges:
            return 0
        adj = np.zeros((n, n), dtype=np.uint8)
        for u, v in edges:
            adj[u, v] = 1
            adj[v, u] = 1
        # Number of C4s = (1/8) * sum over all pairs (u,v) of [common neighbors of u,v]^2
        # Actually: number of 4-cycles = (1/8) * trace((A^2)^2) - correction
        # More precisely: for each pair (u,v), let c(u,v) = number of common neighbors.
        # Each C4 has 4 pairs of opposite vertices, so count = (1/4) * sum over unordered pairs (u,v) of C(c(u,v), 2)
        # Wait, let me think again.
        # A C4 is u-v-w-x-u. The pairs of opposite vertices are (u,w) and (v,x).
        # For a given C4, there are 2 pairs of opposite vertices.
        # For each pair of opposite vertices (u,w), the number of common neighbors is 2 (v and x).
        # So C(2,2) = 1 for that pair.
        # But we need to count each C4 exactly once.
        # Number of C4s = (1/8) * sum_{u < v} [A^2]_{uv} * ([A^2]_{uv} - 1) / 1? 
        # Let's use: C4_count = (1/8) * (trace((A^2)^2) - 2*trace(A^2) - 4*sum(A) + ...)
        # Simpler: for each unordered pair {u,v}, let c = common_neighbors(u,v).
        # Each C4 contributes to exactly 2 such pairs (the two pairs of opposite vertices).
        # For a C4 u-v-w-x-u, opposite pairs are {u,w} and {v,x}.
        # For pair {u,w}, common neighbors are v and x, so c=2, C(2,2)=1.
        # For pair {v,x}, common neighbors are u and w, so c=2, C(2,2)=1.
        # So each C4 is counted 2 times by sum over pairs of C(c(u,v),2).
        # Therefore C4_count = (1/2) * sum_{u<v} C(common_neighbors(u,v), 2)
        A2 = adj @ adj
        c4_count = 0
        for u in range(n):
            for v in range(u + 1, n):
                c = A2[u, v]
                if c >= 2:
                    c4_count += c * (c - 1) // 2
        return c4_count // 2
    
    # Helper: check if adding edge (u,v) to graph creates a C4
    # Adding (u,v) creates a C4 iff common_neighbors(u,v) >= 1
    # But wait, that's not quite right. Adding (u,v) creates a C4 if there exists
    # a path of length 3 between u and v. That is, common neighbors >= 1.
    # Actually, a C4 is u-v-w-x-u. Adding edge (u,v) to a graph that has path u-w-x-v
    # creates C4 u-w-x-v-u. So we need common neighbors >= 1.
    # But actually we need a path of length 3: u-a-b-v where a,b are distinct and not u,v.
    # common_neighbors counts a such that u-a and a-v exist. For a path of length 3,
    # we need u-a, a-b, b-v with a != b.
    # So adding (u,v) creates a C4 iff there exist distinct a,b such that u-a, a-b, b-v all exist.
    # This is NOT the same as common_neighbors >= 1.
    # common_neighbors >= 1 means there's a path of length 2, which with the new edge gives a triangle, not a C4.
    # For C4: we need a path of length 3.
    
    # Let me reconsider. When we add edge (u,v), new C4s are those that use edge (u,v).
    # A C4 using (u,v) has the form u-v-a-b-u or u-a-b-v-u (same thing).
    # So we need u-a, a-b, b-v with a,b distinct, a!=u,v, b!=u,v, a!=b.
    # Equivalently: there exist distinct a,b such that adj[u,a]=1, adj[a,b]=1, adj[b,v]=1.
    # This is the number of length-3 paths from u to v.
    
    def count_new_c4s(adj, u, v, n=31):
        """Count C4s that would be created by adding edge (u,v)."""
        # Count paths of length 3 from u to v: u-a-b-v with a!=b, a!=u,v, b!=u,v
        count = 0
        neighbors_u = np.where(adj[u] == 1)[0]
        neighbors_v = np.where(adj[v] == 1)[0]
        # For each a in neighbors_u, for each b in neighbors_v, if a!=b and adj[a,b]=1:
        # But we also need a!=v and b!=u (which is guaranteed since u,v not adjacent yet)
        # Actually a could equal v? No, because u-v is not an edge, so v is not in neighbors_u.
        # Similarly u is not in neighbors_v.
        # But a and b could be the same vertex.
        for a in neighbors_u:
            for b in neighbors_v:
                if a != b and adj[a, b] == 1:
                    count += 1
        return count
    
    # Strategy: Build a C4-free graph with 90 edges greedily, then add best 91st edge.
    
    all_edges = list(itertools.combinations(range(n), 2))
    
    best_edges = None
    best_c4_count = float('inf')
    
    max_attempts = 50
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        random.seed(attempt)
        adj = np.zeros((n, n), dtype=np.uint8)
        edges = []
        
        shuffled = all_edges[:]
        random.shuffle(shuffled)
        
        for u, v in shuffled:
            if len(edges) >= c4_free_target:
                break
            if adj[u, v] == 1:
                continue
            # Check if adding (u,v) creates any C4
            # A C4 is created iff there's a path of length 3 from u to v
            # i.e., exists a,b distinct: u-a, a-b, b-v
            neighbors_u = np.where(adj[u] == 1)[0]
            if len(neighbors_u) == 0:
                # Safe to add
                adj[u, v] = 1
                adj[v, u] = 1
                edges.append((u, v))
                continue
            
            neighbors_v = np.where(adj[v] == 1)[0]
            if len(neighbors_v) == 0:
                adj[u, v] = 1
                adj[v, u] = 1
                edges.append((u, v))
                continue
            
            # Check for length-3 paths
            creates_c4 = False
            for a in neighbors_u:
                for b in neighbors_v:
                    if a != b and adj[a, b] == 1:
                        creates_c4 = True
                        break
                if creates_c4:
                    break
            
            if not creates_c4:
                adj[u, v] = 1
                adj[v, u] = 1
                edges.append((u, v))
        
        if len(edges) < c4_free_target:
            continue
        
        # Now we have a C4-free graph with 90 edges.
        # Find the best edge to add to minimize C4s.
        best_add = None
        best_new_c4s = float('inf')
        
        for u, v in all_edges:
            if adj[u, v] == 1:
                continue
            new_c4s = count_new_c4s(adj, u, v, n)
            if new_c4s < best_new_c4s:
                best_new_c4s = new_c4s
                best_add = (u, v)
            if best_new_c4s == 0:
                break
        
        if best_add is None:
            continue
        
        final_edges = edges + [best_add]
        
        # Verify count
        c4_count = count_c4s(final_edges, n)
        
        if c4_count < best_c4_count:
            best_c4_count = c4_count
            best_edges = final_edges
        
        if best_c4_count <= 1:
            break
    
    if best_edges is None:
        # Fallback: return the seed
        pairs = [[i, j] for i in range(31) for j in range(i + 1, 31)]
        return pairs[:91]
    
    return [list(e) for e in best_edges]
# EVOLVE-BLOCK-END
