import time
import random
import heapq
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    # Try n=47 first, then 46
    for target_n in [47, 46]:
        min_deg_needed = target_n - 39
        
        # Multiple restarts
        for restart in range(200):
            if time.time() > deadline:
                break
            
            random.seed(restart * 7919 + target_n)
            
            # Initialize with a random graph
            # Density p=0.2 gives approx 47*46*0.2/2 = 216 edges
            # Min deg needed is 8. Avg deg 0.2*46 = 9.2. Good start.
            adj_mask = [0] * target_n
            deg = [0] * target_n
            edges = set()
            
            for u in range(target_n):
                for v in range(u + 1, target_n):
                    if random.random() < 0.2:
                        adj_mask[u] |= (1 << v)
                        adj_mask[v] |= (1 << u)
                        deg[u] += 1
                        deg[v] += 1
                        edges.add((u, v))
                        edges.add((v, u))
            
            # Local Search Loop
            success = False
            # Limit iterations per restart to avoid infinite loops
            max_iterations = 5000
            for _ in range(max_iterations):
                if time.time() > deadline:
                    break
                
                # 1. Check for C4s
                # We need to find an edge that is part of a C4.
                # An edge (u, v) is in a C4 if |N(u) & N(v)| >= 2.
                # Actually, a C4 is a cycle u-w-v-x-u.
                # The edges are (u,w), (w,v), (v,x), (x,u).
                # If we just check pairs (u,v) that are edges, and count common neighbors,
                # if count >= 2, then (u,v) is in at least one C4?
                # No. If (u,v) is an edge, and they have 2 common neighbors w,x,
                # then u-w-v-x-u is a C4. Yes.
                
                # Find the "worst" edge (in most C4s)
                max_c4s = 0
                worst_edge = None
                # We only care about edges that exist
                # Iterating all pairs is O(n^2). n=47, 1128 pairs. Fast.
                for u in range(target_n):
                    for v in range(u + 1, target_n):
                        if (u, v) not in edges:
                            continue
                        common = adj_mask[u] & adj_mask[v]
                        k = bin(common).count('1')
                        if k >= 2:
                            # Number of C4s containing (u,v) is C(k, 2)
                            c4_count = k * (k - 1) // 2
                            if c4_count > max_c4s:
                                max_c4s = c4_count
                                worst_edge = (u, v)
                
                if worst_edge is not None:
                    # We have a C4. Try to remove an edge to break it.
                    u, v = worst_edge
                    if deg[u] > min_deg_needed and deg[v] > min_deg_needed:
                        # Remove (u, v)
                        adj_mask[u] &= ~(1 << v)
                        adj_mask[v] &= ~(1 << u)
                        deg[u] -= 1
                        deg[v] -= 1
                        edges.remove((u, v))
                        edges.remove((v, u))
                    else:
                        # Cannot remove worst edge. Try to find ANY removable edge in a C4.
                        found_removable = False
                        for u2 in range(target_n):
                            for v2 in range(u2 + 1, target_n):
                                if (u2, v2) not in edges:
                                    continue
                                common = adj_mask[u2] & adj_mask[v2]
                                k = bin(common).count('1')
                                if k >= 2:
                                    if deg[u2] > min_deg_needed and deg[v2] > min_deg_needed:
                                        # Remove (u2, v2)
                                        adj_mask[u2] &= ~(1 << v2)
                                        adj_mask[v2] &= ~(1 << u2)
                                        deg[u2] -= 1
                                        deg[v2] -= 1
                                        edges.remove((u2, v2))
                                        edges.remove((v2, u2))
                                        found_removable = True
                                        break
                            if found_removable:
                                break
                        if not found_removable:
                            # Stuck: C4s exist but all critical edges are at min degree.
                            # This restart fails.
                            break
                else:
                    # No C4s found. Check degree.
                    if min(deg) < min_deg_needed:
                        # Try to add an edge to increase min degree.
                        # Find vertex with min degree
                        min_deg_vertex = min(range(target_n), key=lambda x: deg[x])
                        added = False
                        # Try to add edge (min_deg_vertex, u)
                        # Shuffle order to avoid bias
                        candidates = list(range(target_n))
                        random.shuffle(candidates)
                        for u in candidates:
                            if u == min_deg_vertex:
                                continue
                            if (min_deg_vertex, u) in edges:
                                continue
                            # Check if adding (min_deg_vertex, u) creates a C4
                            # This happens if |N(min_deg_vertex) & N(u)| >= 1
                            # Because if w is common neighbor, min_deg_vertex-w-u-min_deg_vertex is a triangle? No.
                            # C4 is length 4.
                            # Adding (v, u) creates a C4 if there is a path of length 3 between v and u.
                            # Path v-w-x-u.
                            # This is equivalent to checking if there is a common neighbor in the "square" of the adjacency?
                            # Simpler:
                            # A C4 is formed if we add (v, u) and there exist w, x such that
                            # (v, w), (w, x), (x, u) are edges.
                            # This is checking if dist(v, u) == 3 in the current graph.
                            # Alternatively:
                            # The new edge (v, u) creates a C4 if and only if
                            # there is a vertex w adjacent to v and a vertex x adjacent to u
                            # such that (w, x) is an edge.
                            # So: Does N(v) have an edge to N(u)?
                            # i.e., is there w in N(v) and x in N(u) such that (w, x) in edges?
                            
                            # Efficient check:
                            # Iterate w in N(v):
                            #   If N(w) intersects N(u) (excluding v, u? No, w is neighbor of v, x is neighbor of u)
                            #   Wait, if w is in N(v) and x is in N(u) and (w, x) is an edge,
                            #   then v-w-x-u-v is a C4.
                            #   Note: w != u and x != v is guaranteed by simple graph and no triangles?
                            #   If graph is triangle-free, yes. But we are not ensuring triangle-free.
                            #   However, if (v, u) is added, and v-w-x-u exists, it's a C4.
                            #   We must ensure w != u and x != v.
                            #   Since (v, u) is not an edge yet, u is not in N(v) and v is not in N(u).
                            #   So w is in N(v) implies w != u.
                            #   x is in N(u) implies x != v.
                            #   So we just check: Is there an edge between N(v) and N(u)?
                            
                            # Optimization:
                            # Precompute neighbors list?
                            # Or just bitmasks.
                            # N(v) is adj_mask[v].
                            # N(u) is adj_mask[u].
                            # We need to know if any w in N(v) has an edge to any x in N(u).
                            # This is: Does union of N(w) for w in N(v) intersect N(u)?
                            # Or: For each w in N(v), check if (adj_mask[w] & adj_mask[u]) != 0.
                            
                            # Since n=47, N(v) size is ~9.
                            # Check 9 vertices. Fast.
                            
                            creates_c4 = False
                            # Get neighbors of min_deg_vertex
                            # We can iterate bits of adj_mask[min_deg_vertex]
                            mask_v = adj_mask[min_deg_vertex]
                            # Iterate through set bits in mask_v
                            m = mask_v
                            while m:
                                # Get lowest set bit
                                lsb = m & -m
                                w = lsb.bit_length() - 1
                                # Check if w has any neighbor in N(u)
                                # N(u) is adj_mask[u]
                                if adj_mask[w] & adj_mask[u]:
                                    creates_c4 = True
                                    break
                                m ^= lsb
                            
                            if not creates_c4:
                                # Add edge
                                adj_mask[min_deg_vertex] |= (1 << u)
                                adj_mask[u] |= (1 << min_deg_vertex)
                                deg[min_deg_vertex] += 1
                                deg[u] += 1
                                edges.add((min_deg_vertex, u))
                                edges.add((u, min_deg_vertex))
                                added = True
                                break
                                
                        if not added:
                            # Cannot add any edge to min_deg_vertex without creating C4
                            # Try other low degree vertices?
                            # For simplicity, if min vertex is stuck, restart might be better.
                            # Or try to add to any vertex with deg < min_deg_needed.
                            # Let's just break and restart.
                            break
                    else:
                        # Min degree OK and no C4s. Success!
                        success = True
                        break
            
            if success:
                # Construct final edge list
                final_edges = []
                for u in range(target_n):
                    for v in range(u + 1, target_n):
                        if (u, v) in edges:
                            final_edges.append([u, v])
                
                # Verify
                # 1. Min degree
                if min(deg) < min_deg_needed:
                    continue
                # 2. C4 free
                is_c4_free = True
                for u in range(target_n):
                    for v in range(u + 1, target_n):
                        common = adj_mask[u] & adj_mask[v]
                        if common & (common - 1): # > 1 common neighbor
                            is_c4_free = False
                            break
                    if not is_c4_free:
                        break
                
                if is_c4_free:
                    return {'n': target_n, 'edges': final_edges}
    
    # Fallback
    return {'n': 39, 'edges': []}
