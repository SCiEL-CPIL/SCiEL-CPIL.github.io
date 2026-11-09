import time
import random

def construct():
    t0 = time.time()
    deadline = t0 + 118.0

    n = 31
    q = 5

    # --- Step 1: Build C4-free core with 90 edges using PG(2,5) polarity ---
    points = []
    seen = set()
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = pow(x, 3, 5)
                    nx, ny, nz = 1, (y * inv_x) % 5, (z * inv_x) % 5
                elif y != 0:
                    inv_y = pow(y, 3, 5)
                    nx, ny, nz = 0, 1, (z * inv_y) % 5
                else:
                    nx, ny, nz = 0, 0, 1
                key = (nx, ny, nz)
                if key not in seen:
                    seen.add(key)
                    points.append(key)
    
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    # Build adjacency as bitmasks
    adj = [0] * n
    for i in range(n):
        x1, y1, z1 = points[i]
        for j in range(i+1, n):
            x2, y2, z2 = points[j]
            if (x1*x2 + y1*y2 - z1*z2) % 5 == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Verify 90 edges
    edge_count = sum(bin(a).count("1") for a in adj) // 2
    if edge_count != 90:
        # Fallback greedy if polarity construction fails (should not happen)
        adj = [0] * n
        random.seed(42)
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(pairs)
        for u, v in pairs:
            if edge_count >= 90:
                break
            # Check if adding (u,v) creates C4
            # Common neighbors
            common = adj[u] & adj[v]
            if common:
                # If there is a common neighbor, adding (u,v) creates a triangle, not C4 directly?
                # C4 is u-a-b-v-u.
                # If u,v share neighbor w, then u-w-v is path of length 2.
                # Adding (u,v) creates triangle u-w-v-u.
                # Does it create C4?
                # C4 requires path of length 3.
                # If we have common neighbor w, we have u-w-v.
                # We need another path of length 3?
                # Actually, if graph is C4-free, adding (u,v) creates C4 iff there is a path of length 3.
                # Path u-x-y-v.
                # If x=w (common neighbor), then u-w-y-v. Need w~y and y~v.
                # If x!=w, u-x-y-v.
                # Greedy check:
                # For each x in N(u), for each y in N(v), if x~y and x!=y, C4 created.
                # Also if x==y, then u-x-v is path length 2. Adding (u,v) makes triangle.
                # Triangle is not C4.
                # So we only care about x != y.
                creates_c4 = False
                Nu = adj[u]
                Nv = adj[v]
                # Iterate x in Nu
                temp_u = Nu
                while temp_u:
                    lsb = temp_u & -temp_u
                    x = lsb.bit_length() - 1
                    temp_u ^= lsb
                    # Iterate y in Nv
                    temp_v = Nv
                    while temp_v:
                        lsb_v = temp_v & -temp_v
                        y = lsb_v.bit_length() - 1
                        temp_v ^= lsb_v
                        if x == y:
                            continue
                        if (adj[x] >> y) & 1:
                            creates_c4 = True
                            break
                    if creates_c4:
                        break
                if not creates_c4:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_count += 1
        # If still < 90, force add
        if edge_count < 90:
            for u, v in pairs:
                if edge_count >= 90:
                    break
                if not ((adj[u] >> v) & 1):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_count += 1

    # Helper to count C4s created by adding edge (u,v) to current graph
    # This is the number of edges between N(u) and N(v) in the current graph.
    # Note: If u,v are already connected, this logic is for non-edges.
    def count_new_c4s(u, v):
        Nu = adj[u]
        Nv = adj[v]
        count = 0
        # Iterate x in Nu
        temp = Nu
        while temp:
            lsb = temp & -temp
            x = lsb.bit_length() - 1
            temp ^= lsb
            # Count neighbors of x that are in Nv
            # But we must exclude v? No, x is in Nu, so x != u.
            # y is in Nv, so y != v.
            # We count edges (x,y).
            # If x==y, it's a self-loop in the intersection, not an edge.
            # adj[x] & Nv gives neighbors of x in Nv.
            # We need to make sure we don't count (x,x).
            # adj[x] does not have bit x.
            # So adj[x] & Nv is correct.
            count += bin(adj[x] & Nv).count("1")
        return count

    # Find best edge to add
    best_edge = None
    best_cost = float('inf')
    
    for u in range(n):
        for v in range(u+1, n):
            if (adj[u] >> v) & 1:
                continue
            cost = count_new_c4s(u, v)
            if cost < best_cost:
                best_cost = cost
                best_edge = (u, v)
    
    # Add the best edge
    u, v = best_edge
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)
    
    # Current total C4s
    current_c4s = best_cost
    
    # Local Search
    # We want to minimize C4s.
    # Try swapping edge (a,b) with (c,d).
    # Remove (a,b), Add (c,d).
    # Delta = C4s(c,d in G-(a,b)) - C4s(a,b in G) + interaction?
    # It's safer to compute exact delta by simulating.
    # Given n=31, simulation is fast.
    
    def get_edges(adj):
        edges = []
        for i in range(n):
            temp = adj[i]
            while temp:
                lsb = temp & -temp
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append((i, j))
                temp ^= lsb
        return edges

    def get_non_edges(adj, edges_set):
        non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (i, j) not in edges_set:
                    non_edges.append((i, j))
        return non_edges

    edges_set = set(get_edges(adj))
    
    improved = True
    stall = 0
    
    while improved and time.time() < deadline - 10:
        improved = False
        edges_list = list(edges_set)
        non_edges_list = get_non_edges(adj, edges_set)
        
        # Shuffle to avoid deterministic bias
        random.shuffle(edges_list)
        random.shuffle(non_edges_list)
        
        # Try a subset of swaps if too many
        # 91 edges, ~389 non-edges. 91*389 ~ 35k swaps.
        # Each swap simulation is O(deg^2) or similar.
        # Let's try all if fast enough, else random sample.
        
        best_swap = None
        best_delta = 0
        
        # Optimization: Pre-compute C4s for each edge?
        # C4s using edge (u,v) is number of paths of length 3 between u and v.
        # Which is count_new_c4s(u,v) if (u,v) was not an edge?
        # Wait. If (u,v) IS an edge, the C4s containing it are those formed by paths of length 3 in G-(u,v).
        # This is exactly count_new_c4s(u,v) calculated on G-(u,v).
        
        # For each edge e=(a,b), let cost_e be C4s using e.
        # For each non-edge f=(c,d), let cost_f be C4s created by adding f.
        # If we swap e for f:
        # Remove e: C4s decrease by cost_e.
        # Add f: C4s increase by cost_f_in_G_minus_e.
        # cost_f_in_G_minus_e might be different from cost_f_in_G if f interacts with e.
        # Interaction: e=(a,b), f=(c,d).
        # If {a,b} intersects {c,d}, or if a,b are neighbors of c,d in a way that affects the count.
        # Given the small size, let's just simulate the swap for the most promising candidates.
        # Or just simulate all if it fits in time.
        
        # Estimate time: 35k swaps * 1ms per swap = 35s. Might be tight.
        # Let's do random sampling first, then polish.
        
        for _ in range(20): # 20 passes
            if time.time() > deadline - 5:
                break
            random.shuffle(edges_list)
            random.shuffle(non_edges_list)
            
            for e in edges_list:
                if time.time() > deadline - 5:
                    break
                a, b = e
                # Temporarily remove e
                adj[a] &= ~(1 << b)
                adj[b] &= ~(1 << a)
                
                # Find best non-edge to add
                local_best_f = None
                local_best_cost = float('inf')
                
                # Only check non-edges that might be good?
                # Just check all non-edges in the list?
                # To save time, check a random subset of non-edges or all?
                # Let's check all non-edges for the first few edges, then sample.
                
                # Actually, count_new_c4s is fast (bit ops).
                # Let's check all non-edges.
                for f in non_edges_list:
                    c, d = f
                    # f must not be e (it's a non-edge, e is edge, so ok)
                    # f must not be an edge in current adj (it's in non_edges_list, so ok)
                    cost = count_new_c4s(c, d)
                    if cost < local_best_cost:
                        local_best_cost = cost
                        local_best_f = f
                
                # Restore e
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
                
                # Calculate delta
                # We need cost of e in original graph
                # This requires removing e and counting.
                # We just did the removal to check f.
                # Let's re-structure to compute cost_e once.
                
                # This loop is O(E * N_non_edges * CostCount).
                # 91 * 389 * ~10 ops = 350k ops. Very fast.
                # Wait, count_new_c4s iterates over neighbors.
                # Avg degree ~ 6.
                # So ~ 350k * 6 = 2M ops. Instant.
                
                # So we can do exact best swap for each edge.
                
                if local_best_f is not None:
                    # We need cost_e
                    # Remove e
                    adj[a] &= ~(1 << b)
                    adj[b] &= ~(1 << a)
                    cost_e = count_new_c4s(a, b)
                    # Restore e
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
                    
                    # The cost of adding f in G-e is local_best_cost.
                    # Is it exactly local_best_cost?
                    # count_new_c4s(c,d) uses adj.
                    # In the loop above, adj had e removed.
                    # So local_best_cost IS the cost of adding f in G-e.
                    
                    delta = local_best_cost - cost_e
                    if delta < best_delta:
                        best_delta = delta
                        best_swap = (e, local_best_f)
                        
        if best_swap is not None and best_delta < 0:
            e, f = best_swap
            a, b = e
            c, d = f
            # Apply swap
            adj[a] &= ~(1 << b)
            adj[b] &= ~(1 << a)
            adj[c] |= (1 << d)
            adj[d] |= (1 << c)
            
            edges_set.remove(e)
            edges_set.add(f)
            
            current_c4s += best_delta
            improved = True
            stall = 0
        else:
            stall += 1
            if stall > 3:
                break

    # Final polish: try all swaps again
    if time.time() < deadline - 2:
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            edges_list = list(edges_set)
            non_edges_list = get_non_edges(adj, edges_set)
            
            best_swap = None
            best_delta = 0
            
            for e in edges_list:
                a, b = e
                adj[a] &= ~(1 << b)
                adj[b] &= ~(1 << a)
                
                local_best_f = None
                local_best_cost = float('inf')
                
                for f in non_edges_list:
                    c, d = f
                    cost = count_new_c4s(c, d)
                    if cost < local_best_cost:
                        local_best_cost = cost
                        local_best_f = f
                
                cost_e = count_new_c4s(a, b)
                
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
                
                if local_best_f is not None:
                    delta = local_best_cost - cost_e
                    if delta < best_delta:
                        best_delta = delta
                        best_swap = (e, local_best_f)
                        
            if best_swap is not None and best_delta < 0:
                e, f = best_swap
                a, b = e
                c, d = f
                adj[a] &= ~(1 << b)
                adj[b] &= ~(1 << a)
                adj[c] |= (1 << d)
                adj[d] |= (1 << c)
                
                edges_set.remove(e)
                edges_set.add(f)
                
                current_c4s += best_delta
                improved = True

    # Build final edge list
    final_edges = [list(e) for e in edges_set]
    final_edges.sort()
    
    assert len(final_edges) == 91, f"Expected 91 edges, got {len(final_edges)}"
    
    return final_edges
