import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin

    def instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def strong_conflict_graph(edges, adj):
        m = len(edges)
        # Map edge to index
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        # Neighbor bitmasks for strong conflict graph
        # conflict_mask[i] is a bitmask of indices j such that edge i and edge j conflict
        conflict_mask = [0] * m
        
        # Precompute adjacency sets for vertices for fast lookup
        # adj is a list of sets
        
        for i, e1 in enumerate(edges):
            u1, v1 = e1
            # Edges sharing u1 or v1
            for v in adj[u1]:
                e2 = (min(u1, v), max(u1, v))
                if e2 in edge_to_idx:
                    j = edge_to_idx[e2]
                    conflict_mask[i] |= (1 << j)
            for v in adj[v1]:
                e2 = (min(v1, v), max(v1, v))
                if e2 in edge_to_idx:
                    j = edge_to_idx[e2]
                    conflict_mask[i] |= (1 << j)
            # Edges at distance 2 (connected by an edge)
            # If edge e1=(u1,v1) and edge e2=(u2,v2) are connected by an edge,
            # then one of u1,v1 is adjacent to one of u2,v2, but they don't share vertices.
            # Actually, the definition of strong conflict is:
            # Two edges e1, e2 conflict if they share a vertex OR there is an edge in G 
            # connecting a vertex of e1 to a vertex of e2.
            # The loop above covers "share a vertex".
            # The "connected by an edge" part means:
            # For each neighbor v of u1, if there is an edge (v, w) where w is v1? No, that's sharing u1.
            # If there is an edge (v, w) where w is not u1 or v1?
            # Let's re-read carefully.
            # "no two edges share a vertex or are joined by an edge"
            # So if e1=(u,v) and e2=(x,y), they are joined by an edge if there is an edge (u,x), (u,y), (v,x), or (v,y) in G.
            # The previous loop iterates over neighbors of u1. If neighbor is v1, it's the edge itself (excluded by set logic or self-loop check, but u!=v).
            # If neighbor v is not v1, then edge (u1, v) shares u1. This is covered.
            # What about edge (v, w) where v is neighbor of u1, and w is v1? That's edge (u1, v1) itself.
            # What about edge (v, w) where v is neighbor of u1, and w is NOT v1?
            # Then e2=(v,w) shares vertex v with e1=(u1,v1)? No, e1 is (u1,v1). e2 is (v,w).
            # They share vertex v only if v is u1 or v1.
            # If v is not u1 or v1, they don't share a vertex.
            # But they are joined by edge (u1, v).
            # So we need to check all edges incident to neighbors of u1 and v1.
            
            # Let's restart the conflict check logic clearly.
            # e1 = (u1, v1)
            # Conflicts with e2 = (u2, v2) if:
            # 1. u1==u2 or u1==v2 or v1==u2 or v1==v2 (share vertex)
            # 2. (u1,u2) in G, (u1,v2) in G, (v1,u2) in G, or (v1,v2) in G (joined by edge)
            
            # We can iterate over all edges e2 and check these conditions.
            # With m <= 240, m^2 = 57600 checks per graph. Very fast.
            
        # Let's just use the O(m^2) check for simplicity and correctness, it's fast enough.
        for i in range(m):
            u1, v1 = edges[i]
            for j in range(i + 1, m):
                u2, v2 = edges[j]
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    conflict_mask[i] |= (1 << j)
                    conflict_mask[j] |= (1 << i)
                else:
                    # Check if joined by an edge
                    if (u1, u2) in adj or (u1, v2) in adj or (v1, u2) in adj or (v1, v2) in adj:
                        # adj is a list of sets, so we need to check membership
                        # Wait, adj is list of sets. (u1,u2) in adj is wrong.
                        # Should be u2 in adj[u1] or v2 in adj[u1] etc.
                        pass # Handled below properly
        
        # Re-do conflict mask construction correctly
        conflict_mask = [0] * m
        for i in range(m):
            u1, v1 = edges[i]
            for j in range(i + 1, m):
                u2, v2 = edges[j]
                # Share vertex?
                if u1 in (u2, v2) or v1 in (u2, v2):
                    conflict_mask[i] |= (1 << j)
                    conflict_mask[j] |= (1 << i)
                else:
                    # Joined by edge?
                    if u2 in adj[u1] or v2 in adj[u1] or u2 in adj[v1] or v2 in adj[v1]:
                        conflict_mask[i] |= (1 << j)
                        conflict_mask[j] |= (1 << i)
        return conflict_mask

    def dsatur_coloring(conflict_mask, m):
        color = [-1] * m
        # available_colors[i] is a bitmask of colors NOT used by neighbors of i
        # Initially all colors are available. We don't know max colors yet.
        # We can just track used colors per vertex.
        
        # To speed up, we can maintain a list of uncolored vertices and their saturation.
        # But m is small, so simple loop is fine.
        
        current_max_color = -1
        for _ in range(m):
            # Find vertex with max saturation, tie-break by degree
            best_v = -1
            best_sat = -1
            best_deg = -1
            for v in range(m):
                if color[v] != -1:
                    continue
                # Calculate saturation: number of distinct colors in neighbors
                neighbor_colors = set()
                mask = conflict_mask[v]
                # Iterate through set bits in mask
                mm = mask
                while mm:
                    lsb = mm & -mm
                    idx = lsb.bit_length() - 1
                    if color[idx] != -1:
                        neighbor_colors.add(color[idx])
                    mm ^= lsb
                
                sat = len(neighbor_colors)
                deg = bin(mask).count('1')
                
                if sat > best_sat or (sat == best_sat and deg > best_deg):
                    best_sat = sat
                    best_deg = deg
                    best_v = v
            
            if best_v == -1:
                break
                
            # Assign smallest available color
            used_colors = set()
            mask = conflict_mask[best_v]
            mm = mask
            while mm:
                lsb = mm & -mm
                idx = lsb.bit_length() - 1
                if color[idx] != -1:
                    used_colors.add(color[idx])
                mm ^= lsb
            
            c = 0
            while c in used_colors:
                c += 1
            color[best_v] = c
            if c > current_max_color:
                current_max_color = c
                
        return color, current_max_color + 1

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() > deadline:
            # Return best effort so far? If we haven't finished one, we might return invalid.
            # But we should be fast enough.
            pass
            
        edges = instance(*spec)
        adj = [set() for _ in range(spec[0])]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        m = len(edges)
        if m == 0:
            out.append([])
            continue
            
        conflict_mask = strong_conflict_graph(edges, adj)
        
        # Run DSATUR
        color, num_colors = dsatur_coloring(conflict_mask, m)
        
        # Optional: Local search to reduce colors?
        # For now, DSATUR is a strong baseline.
        
        out.append(color)
        
    return out
