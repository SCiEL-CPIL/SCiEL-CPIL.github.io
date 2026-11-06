import time
import random

def construct():
    n = 31
    target_edges = 91
    start_time = time.time()
    deadline = start_time + 145.0

    # 1. Construct Polarity Graph of PG(2,5)
    q = 5
    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    key = (1, y * pow(x, q-2, q) % q, z * pow(x, q-2, q) % q)
                elif y != 0:
                    key = (0, 1, z * pow(y, q-2, q) % q)
                else:
                    key = (0, 0, 1)
                if key not in seen:
                    seen.add(key)
                    points.append(key)
    
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    def is_on_polar(p, qpt):
        x, y, z = p
        X, Y, Z = qpt
        return (x * Y + y * X + z * Z) % q == 0

    edges = set()
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if is_on_polar(points[i], points[j]):
                edges.add((i, j))
                adj[i].add(j)
                adj[j].add(i)
    
    assert len(edges) == 90, f"Expected 90 edges, got {len(edges)}"

    # 2. Find best edge to add
    def get_c4s_for_edge(u, v, adj_sets, edge_set):
        """Find all C4s containing edge (u,v) in the graph defined by adj_sets and edge_set."""
        c4s = set()
        # Type 1: (u,v) is a side. u-v-x-y-u
        # x in N(v), y in N(u), x~y, x!=y
        for x in adj_sets[v]:
            if x == u: continue
            for y in adj_sets[u]:
                if y == v: continue
                if x == y: continue
                if x in adj_sets[y]: # x~y
                    c4 = frozenset([u, v, x, y])
                    c4s.add(c4)
        # Type 2: (u,v) is a diagonal. u-x-v-y-u
        # x, y in N(u) cap N(v)
        common = adj_sets[u] & adj_sets[v]
        common_list = list(common)
        for i in range(len(common_list)):
            for j in range(i+1, len(common_list)):
                x = common_list[i]
                y = common_list[j]
                if x != y:
                    c4 = frozenset([u, v, x, y])
                    c4s.add(c4)
        return c4s

    best_add = None
    min_c4s_created = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in edges:
                # Temporarily add to check? No, just calculate potential C4s
                # In the current graph (polarity), N(i) and N(j) are fixed.
                # Adding (i,j) creates C4s based on current adj.
                # Note: adj does not include (i,j) yet.
                c4s = get_c4s_for_edge(i, j, adj, edges)
                if len(c4s) < min_c4s_created:
                    min_c4s_created = len(c4s)
                    best_add = (i, j)

    u_add, v_add = best_add
    edges.add((u_add, v_add))
    adj[u_add].add(v_add)
    adj[v_add].add(u_add)
    
    # 3. Initialize C4 set
    all_c4s = set()
    for u, v in edges:
        all_c4s.update(get_c4s_for_edge(u, v, adj, edges))
    
    current_c4_count = len(all_c4s)
    best_c4_count = current_c4_count
    best_edges = set(edges)
    best_adj = [set(s) for s in adj]
    best_c4s = set(all_c4s)

    random.seed(42)
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # 4. Local Search
    iterations = 0
    while time.time() < deadline and iterations < 50000:
        iterations += 1
        
        if not edges:
            break
            
        # Pick random edge to remove
        u1, v1 = random.choice(list(edges))
        
        # Find C4s containing (u1, v1)
        c4s_to_remove = get_c4s_for_edge(u1, v1, adj, edges)
        
        # Try to find a better swap
        improved = False
        # Limit attempts per iteration to keep pace
        for _ in range(30):
            u2, v2 = random.choice(all_pairs)
            if (u2, v2) in edges or (v2, u2) in edges:
                continue
            if (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
                continue
            
            # Calculate C4s that would be added by (u2, v2)
            # Note: adj currently has (u1, v1) present.
            # We are simulating: remove (u1, v1), add (u2, v2).
            # The C4s added by (u2, v2) depend on the graph AFTER removing (u1, v1).
            # However, (u2, v2) is a non-edge, so it's not in adj.
            # The neighbors of u2 and v2 in the new graph might be different if they were adjacent to u1 or v1.
            # But u2, v2 are non-adjacent to each other.
            # Do u2 or v2 share neighbors with u1 or v1?
            # Yes.
            
            # To be safe, we can compute the delta more carefully.
            # New C4s = C4s containing (u2, v2) in G' where G' = G - (u1, v1) + (u2, v2).
            # This is complex. 
            # Approximation: Assume the change in neighbors is small or ignore it for the "add" part?
            # Better: Compute C4s containing (u2, v2) in the graph where (u1, v1) is removed.
            
            # Let's construct a temporary adj view?
            # Or just compute the count.
            
            # Simpler approach for delta:
            # Delta = |C4s_added| - |C4s_removed|
            # C4s_removed are those containing (u1, v1).
            # C4s_added are those containing (u2, v2) in the modified graph.
            
            # Modified graph adj:
            # adj_mod[u1] = adj[u1] - {v1}
            # adj_mod[v1] = adj[v1] - {u1}
            # adj_mod[u2] = adj[u2] + {v2}
            # adj_mod[v2] = adj[v2] + {u2}
            # Others same.
            
            # We need to find C4s containing (u2, v2) in this modified graph.
            # We can write a helper that takes adj and a specific edge, and returns C4s.
            # But adj is global.
            
            # Let's just compute the number of C4s containing (u2, v2) in the CURRENT graph (with u1,v1 present).
            # This is an overestimate if the removal of (u1,v1) breaks some of those C4s?
            # No, C4s containing (u2,v2) do not contain (u1,v1) unless they share vertices?
            # A C4 has 4 vertices. (u2,v2) uses 2.
            # If a C4 contains (u2,v2) and (u1,v1), it must use all 4 vertices u1,v1,u2,v2.
            # This is a specific case.
            
            # For simplicity and speed, let's just compute C4s for (u2,v2) in current adj.
            c4s_to_add = get_c4s_for_edge(u2, v2, adj, edges)
            
            # Intersection: C4s that are both removed and added?
            # If a C4 is in c4s_to_remove and c4s_to_add, it stays.
            # Net change = |c4s_to_add \ c4s_to_remove| - |c4s_to_remove \ c4s_to_add|
            
            new_c4_count = current_c4_count + len(c4s_to_add) - len(c4s_to_remove) + len(c4s_to_add & c4s_to_remove) * 2
            # No.
            # Set difference logic:
            # New Set = (Old Set - Removed) U Added
            # |New| = |Old| - |Removed \ Added| + |Added \ Removed|
            
            removed_only = c4s_to_remove - c4s_to_add
            added_only = c4s_to_add - c4s_to_remove
            
            new_c4_count = current_c4_count - len(removed_only) + len(added_only)
            
            if new_c4_count < best_c4_count:
                # Accept
                # Update global state
                adj[u1].discard(v1)
                adj[v1].discard(u1)
                adj[u2].add(v2)
                adj[v2].add(u2)
                
                edges.discard((u1, v1))
                edges.add((u2, v2))
                
                # Update C4 set
                all_c4s.difference_update(removed_only)
                all_c4s.update(added_only)
                
                current_c4_count = new_c4_count
                best_c4_count = current_c4_count
                best_edges = set(edges)
                best_adj = [set(s) for s in adj]
                best_c4s = set(all_c4s)
                improved = True
                break
                
        if time.time() > deadline - 5:
            break

    # Return best edges found
    # Ensure we return exactly 91 edges
    if len(best_edges) != target_edges:
        # Fallback to current edges if best is invalid
        best_edges = edges
        
    return [[u, v] for u, v in sorted(best_edges)]
