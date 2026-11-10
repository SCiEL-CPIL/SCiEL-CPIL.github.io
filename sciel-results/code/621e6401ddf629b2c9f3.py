import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s buffer
    
    q = 5
    n = 31
    
    # 1. Generate 31 points of PG(2,5)
    # Standard representation of points in PG(2, q)
    points = []
    # (1, x, y) for x, y in GF(q)
    for x in range(q):
        for y in range(q):
            points.append((1, x, y))
    # (0, 1, z) for z in GF(q)
    for z in range(q):
        points.append((0, 1, z))
    # (0, 0, 1)
    points.append((0, 0, 1))
    
    # 2. Construct Polarity Graph
    # Edge if dot product is 0 mod q
    # Use bitsets for adjacency to allow fast intersection and bit counting
    adj = [0] * n
    base_edges = []
    
    for i in range(n):
        for j in range(i + 1, n):
            p_i = points[i]
            p_j = points[j]
            # Dot product in GF(5)
            dot = (p_i[0]*p_j[0] + p_i[1]*p_j[1] + p_i[2]*p_j[2]) % q
            if dot == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                base_edges.append((i, j))
                
    target_edges = 91
    current_edges_count = len(base_edges)
    edges_to_add = target_edges - current_edges_count
    
    # For q=5, current_edges_count should be 90, so edges_to_add = 1.
    if edges_to_add == 1:
        # Identify non-edges
        # We need to find the non-edge (u, v) that minimizes the number of new C4s.
        # A new C4 is formed by u-a-b-v-u where a ~ u, b ~ v, a ~ b, a != b.
        # Since (u,v) is not an edge, a != v and b != u are guaranteed if a~u and b~v?
        # No, a is neighbor of u. If a==v, then v~u, which is false. So a!=v.
        # Similarly b!=u.
        # So we just need to count pairs (a,b) such that:
        # 1. a is in N(u)
        # 2. b is in N(v)
        # 3. a != b
        # 4. (a,b) is an edge in the current graph.
        
        # Let M = N(u) & N(v). These are common neighbors.
        # We need to count edges between vertices in M.
        # For each a in M, count neighbors in M. Sum these counts, divide by 2.
        
        best_edge = None
        min_cost = float('inf')
        
        # Iterate over all pairs to find non-edges
        for u in range(n):
            for v in range(u + 1, n):
                # Check if (u,v) is an edge
                if (adj[u] >> v) & 1:
                    continue
                
                # Calculate cost of adding (u,v)
                common_mask = adj[u] & adj[v]
                
                # Count edges within the set of common neighbors
                # Iterate through set bits in common_mask
                cost = 0
                m = common_mask
                while m:
                    # Get lowest set bit
                    lsb = m & (-m)
                    a = lsb.bit_length() - 1
                    
                    # Count neighbors of a within common_mask
                    # adj[a] & common_mask gives neighbors of a that are also common neighbors
                    # bit_count() counts them
                    cost += (adj[a] & common_mask).bit_count()
                    
                    m ^= lsb
                
                # Each edge (a,b) in M is counted twice (once for a, once for b)
                cost //= 2
                
                if cost < min_cost:
                    min_cost = cost
                    best_edge = (u, v)
                    
        if best_edge is None:
            # Fallback if something went wrong (should not happen)
            # Find any non-edge
            for u in range(n):
                for v in range(u + 1, n):
                    if not ((adj[u] >> v) & 1):
                        best_edge = (u, v)
                        break
                if best_edge: break
            
            if best_edge is None:
                # If graph is complete (impossible for 31 vertices with 90 edges), pick arbitrary
                best_edge = (0, 1)

        # Add the best edge
        u, v = best_edge
        # We don't strictly need to update adj for the return value, 
        # but we add it to the edge list.
        base_edges.append((u, v))
        
        final_edges = []
        for u, v in base_edges:
            final_edges.append([u, v])
            
        return final_edges

    else:
        # Fallback for unexpected edge counts: Greedy + SA
        # This path is unlikely to be taken for q=5, n=31, m=91
        
        # Precompute combos_4 for fast counting
        def count_total_c4s(adj_sets):
            total = 0
            for u in range(n):
                for v in range(u+1, n):
                    common = len(adj_sets[u] & adj_sets[v])
                    if common >= 2:
                        total += common * (common - 1) // 2
            return total // 2
        
        # Greedy initialization
        added_edges = []
        non_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if j not in adj[i]:
                    non_edges.append((i, j))
                    
        current_adj = [set(s) for s in adj]
        
        while len(added_edges) < edges_to_add:
            best_edge = None
            min_c = float('inf')
            for e in non_edges:
                if e in added_edges: continue
                u, v = e
                c = len(current_adj[u] & current_adj[v])
                if c < min_c:
                    min_c = c
                    best_edge = e
            if best_edge is None: break
            added_edges.append(best_edge)
            u, v = best_edge
            current_adj[u].add(v)
            current_adj[v].add(u)
            
        # SA Refinement
        current_added_set = set(added_edges)
        current_adj = [set(s) for s in adj]
        for u, v in current_added_set:
            current_adj[u].add(v)
            current_adj[v].add(u)
            
        current_c4s = count_total_c4s(current_adj)
        best_c4s = current_c4s
        best_added_set = set(current_added_set)
        
        T = 50.0
        T_min = 1.0
        alpha = 0.99
        
        while time.time() < deadline:
            if T < T_min: break
            if not current_added_set: break
            
            e_out = random.choice(list(current_added_set))
            e_in = None
            while e_in is None or e_in in current_added_set:
                e_in = random.choice(non_edges)
                
            u_out, v_out = e_out
            u_in, v_in = e_in
            
            current_adj[u_out].remove(v_out)
            current_adj[v_out].remove(u_out)
            current_added_set.remove(e_out)
            
            current_adj[u_in].add(v_in)
            current_adj[v_in].add(u_in)
            current_added_set.add(e_in)
            
            new_c4s = count_total_c4s(current_adj)
            delta = new_c4s - current_c4s
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_c4s = new_c4s
                if new_c4s < best_c4s:
                    best_c4s = new_c4s
                    best_added_set = set(current_added_set)
            else:
                current_adj[u_in].remove(v_in)
                current_adj[v_in].remove(u_in)
                current_added_set.remove(e_in)
                
                current_adj[u_out].add(v_out)
                current_adj[v_out].add(u_out)
                current_added_set.add(e_out)
                
            T *= alpha
            
        final_edges = []
        for u, v in base_edges:
            final_edges.append([u, v])
        for u, v in best_added_set:
            final_edges.append([u, v])
            
        return final_edges
