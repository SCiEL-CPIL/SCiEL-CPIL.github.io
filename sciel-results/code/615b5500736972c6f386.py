import time
import sys
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 118.0
    
    n = 16
    
    # Q3 structure: vertices 0-7, edges between vertices differing by 1 bit
    q3_edges_list = []
    for i in range(8):
        for j in range(i+1, 8):
            if bin(i ^ j).count('1') == 1:
                q3_edges_list.append((i, j))
    
    q3_adj_mask = [0] * 8
    for a, b in q3_edges_list:
        q3_adj_mask[a] |= (1 << b)
        q3_adj_mask[b] |= (1 << a)
    
    # Q3 vertex degrees (all 3)
    # Precompute: for each Q3 vertex, its 3 neighbors
    q3_neighbors = []
    for i in range(8):
        neigh = []
        for j in range(8):
            if i != j and (q3_adj_mask[i] >> j) & 1:
                neigh.append(j)
        q3_neighbors.append(neigh)
    
    # For backtracking: order Q3 vertices by a heuristic
    # When we have 2 pre-assigned vertices, the remaining 6 should be ordered
    # by how many of their neighbors are already assigned (more constrained first)
    
    def check_q3_creation(adj_mask, u, v):
        """Check if adding edge (u,v) creates a Q3 subgraph.
        adj_mask is list of 16-bit masks.
        Returns True if Q3 is created, False otherwise."""
        
        # For each Q3 edge (qa, qb), try mapping qa->u, qb->v and qa->v, qb->u
        for qa, qb in q3_edges_list:
            for mapping in [(qa, u, qb, v), (qa, v, qb, u)]:
                qa_m, gu, qb_m, gv = mapping
                
                # Check: gu and gv must be adjacent (they are, we're adding this edge)
                # Now try to extend to full Q3 embedding
                assignment = {}
                assignment[qa_m] = gu
                assignment[qb_m] = gv
                used_mask = (1 << gu) | (1 << gv)
                
                remaining = [x for x in range(8) if x != qa_m and x != qb_m]
                
                # Order remaining by constraint: number of assigned neighbors
                # This is dynamic, so we'll do it in the backtrack
                def backtrack(idx, remaining_list):
                    if idx == len(remaining_list):
                        return True
                    
                    q = remaining_list[idx]
                    q_neigh_mask = q3_adj_mask[q]
                    
                    # Find candidates: vertices not used, adjacent to all assigned neighbors of q
                    candidates = 0
                    # Check each unassigned vertex
                    for g in range(n):
                        if (used_mask >> g) & 1:
                            continue
                        # Check adjacency to all assigned neighbors of q
                        valid = True
                        for qn in range(8):
                            if (q_neigh_mask >> qn) & 1:
                                if qn in assignment:
                                    gn = assignment[qn]
                                    if not (adj_mask[g] >> gn) & 1:
                                        valid = False
                                        break
                        if valid:
                            candidates |= (1 << g)
                    
                    if candidates == 0:
                        return False
                    
                    # Try each candidate
                    c = candidates
                    while c:
                        bit = c & (-c)
                        g = bit.bit_length() - 1
                        c ^= bit
                        
                        used_mask_new = used_mask | (1 << g)
                        assignment[q] = g
                        
                        # Build new remaining list with ordering
                        new_remaining = remaining_list[idx+1:]
                        # Reorder: put vertices with more assigned neighbors first
                        # Count assigned neighbors for each remaining vertex
                        def count_assigned_neigh(qr):
                            cnt = 0
                            for qn in range(8):
                                if (q3_adj_mask[qr] >> qn) & 1:
                                    if qn in assignment:
                                        cnt += 1
                            return cnt
                        new_remaining.sort(key=count_assigned_neigh, reverse=True)
                        
                        if backtrack(idx + 1, new_remaining):
                            return True
                        
                        del assignment[q]
                    
                    return False
                
                if backtrack(0, remaining):
                    return True  # Q3 found
        
        return False  # No Q3 created
    
    def try_configuration(a_vertices, b_vertices):
        """Try a configuration with given partition A and B.
        A: set of vertices that form a clique
        B: remaining vertices
        Returns (edge_count, edges_list)"""
        
        adj_mask = [0] * n
        edges = []
        
        def add_edge(u, v):
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            edges.append([u, v])
        
        # Add all edges between A and B
        for a in a_vertices:
            for b in b_vertices:
                add_edge(a, b)
        
        # Add clique on A
        for i in range(len(a_vertices)):
            for j in range(i+1, len(a_vertices)):
                add_edge(a_vertices[i], a_vertices[j])
        
        # Add matching on B (pair up consecutive vertices)
        b_list = sorted(b_vertices)
        for i in range(0, len(b_list) - 1, 2):
            add_edge(b_list[i], b_list[i+1])
        
        # Now greedily add more edges in B
        # Generate missing edges in B
        missing_edges = []
        for i in range(len(b_list)):
            for j in range(i+1, len(b_list)):
                u, v = b_list[i], b_list[j]
                if not (adj_mask[u] >> v) & 1:
                    missing_edges.append((u, v))
        
        # Sort by heuristic: edges between vertices with lower current degree first
        # (less likely to create Q3)
        b_set = set(b_vertices)
        def edge_priority(e):
            u, v = e
            du = bin(adj_mask[u] & ((1 << 16) - 1)).count('1')
            dv = bin(adj_mask[v] & ((1 << 16) - 1)).count('1')
            return du + dv
        
        missing_edges.sort(key=edge_priority)
        
        for u, v in missing_edges:
            if time.time() > deadline:
                break
            
            # Tentatively add
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            
            if not check_q3_creation(adj_mask, u, v):
                edges.append([u, v])
            else:
                # Revert
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
        
        return len(edges), edges
    
    # Try multiple configurations
    best_count = 0
    best_edges = []
    
    # Configuration 1: A = {0,1,2}, B = {3..15}
    if time.time() < deadline:
        count, edges = try_configuration([0, 1, 2], list(range(3, 16)))
        if count > best_count:
            best_count = count
            best_edges = edges
    
    # Configuration 2: A = {0,1}, B = {2..15} (K_{2,14} + edge in A)
    if time.time() < deadline:
        count, edges = try_configuration([0, 1], list(range(2, 16)))
        if count > best_count:
            best_count = count
            best_edges = edges
    
    # Configuration 3: A = {0,1,2,3}, B = {4..15} (K_{4,12} + K4)
    if time.time() < deadline:
        count, edges = try_configuration([0, 1, 2, 3], list(range(4, 16)))
        if count > best_count:
            best_count = count
            best_edges = edges
    
    # Configuration 4: A = {0,1,2}, B = {3..15} but with different matching
    # Try adding more edges in A-B that might help
    if time.time() < deadline:
        # Start fresh with A={0,1,2}, B={3..15}
        adj_mask = [0] * n
        edges = []
        
        def add_edge(u, v):
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            edges.append([u, v])
        
        for a in range(3):
            for b in range(3, 16):
                add_edge(a, b)
        
        add_edge(0, 1)
        add_edge(1, 2)
        add_edge(0, 2)
        
        # Add more edges in B - try all possible, sorted by degree
        b_list = list(range(3, 16))
        missing_edges = []
        for i in range(len(b_list)):
            for j in range(i+1, len(b_list)):
                u, v = b_list[i], b_list[j]
                if not (adj_mask[u] >> v) & 1:
                    missing_edges.append((u, v))
        
        def edge_priority2(e):
            u, v = e
            du = bin(adj_mask[u]).count('1')
            dv = bin(adj_mask[v]).count('1')
            return du + dv
        
        missing_edges.sort(key=edge_priority2)
        
        for u, v in missing_edges:
            if time.time() > deadline:
                break
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            if not check_q3_creation(adj_mask, u, v):
                edges.append([u, v])
            else:
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
    
    return best_edges
