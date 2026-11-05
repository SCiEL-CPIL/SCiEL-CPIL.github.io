import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s limit
    
    n = 16
    all_vertices = set(range(n))
    
    # Q3 edge list (8 vertices, 12 edges) - standard cube
    # 0-1-3-7-5-4-2-6-0 is not quite right, let's use standard:
    # 000-001-011-010, 100-101-111-110
    # 0-1, 1-3, 3-2, 2-0 (bottom face)
    # 4-5, 5-7, 7-6, 6-4 (top face)
    # 0-4, 1-5, 3-7, 2-6 (verticals)
    q3_edges = [
        (0,1), (1,3), (3,2), (2,0),
        (4,5), (5,7), (7,6), (6,4),
        (0,4), (1,5), (3,7), (2,6)
    ]
    
    def has_q3_with_edge(adj_mask, u, v):
        """Check if adding edge u-v creates a Q3. adj_mask is list of int bitmasks."""
        # Get neighbors of u and v (excluding each other)
        neu = [x for x in range(n) if x != u and x != v and (adj_mask[u] >> x) & 1]
        nev = [x for x in range(n) if x != v and x != u and (adj_mask[v] >> x) & 1]
        
        if len(neu) < 2 or len(nev) < 2:
            return False
        
        # In Q3 with u=0, v=1:
        # pos 2: neighbor of u, must connect to pos 3 and pos 6
        # pos 4: neighbor of u, must connect to pos 5 and pos 6
        # pos 3: neighbor of v, must connect to pos 2 and pos 7
        # pos 5: neighbor of v, must connect to pos 4 and pos 7
        # pos 6: neighbor of pos 2 and pos 4, must connect to pos 7
        # pos 7: neighbor of pos 3 and pos 5, must connect to pos 6
        
        # Try all assignments of 2 vertices from neu to {pos2, pos4}
        # and 2 vertices from nev to {pos3, pos5}
        # Then check if pos6, pos7 exist.
        
        for pos2 in neu:
            for pos4 in neu:
                if pos2 == pos4:
                    continue
                for pos3 in nev:
                    if pos3 == pos2 or pos3 == pos4:
                        continue
                    for pos5 in nev:
                        if pos5 == pos2 or pos5 == pos3 or pos5 == pos4:
                            continue
                        
                        # Check required edges among these 4
                        if not ((adj_mask[pos2] >> pos3) & 1): continue  # 2-3
                        if not ((adj_mask[pos4] >> pos5) & 1): continue  # 4-5
                        
                        # Now find pos6: must be neighbor of pos2 and pos4
                        # and not in {u,v,pos2,pos3,pos4,pos5}
                        excluded = {u, v, pos2, pos3, pos4, pos5}
                        cand_pos6 = []
                        for x in range(n):
                            if x in excluded: continue
                            if ((adj_mask[pos2] >> x) & 1) and ((adj_mask[pos4] >> x) & 1):
                                cand_pos6.append(x)
                        
                        # Find pos7: must be neighbor of pos3 and pos5
                        cand_pos7 = []
                        for x in range(n):
                            if x in excluded: continue
                            if ((adj_mask[pos3] >> x) & 1) and ((adj_mask[pos5] >> x) & 1):
                                cand_pos7.append(x)
                        
                        # Check if any pair (pos6, pos7) works
                        for pos6 in cand_pos6:
                            for pos7 in cand_pos7:
                                if pos6 == pos7: continue
                                # Check remaining edges: pos6-pos7, pos3-pos7, pos5-pos7
                                if ((adj_mask[pos6] >> pos7) & 1) and \
                                   ((adj_mask[pos3] >> pos7) & 1) and \
                                   ((adj_mask[pos5] >> pos7) & 1):
                                    return True
        return False
    
    def count_edges(adj_mask):
        return sum(bin(adj_mask[i]).count('1') for i in range(n)) // 2
    
    def verify_no_q3(adj_mask):
        """Verify the graph has no Q3 subgraph."""
        for u in range(n):
            for v in range(u+1, n):
                if (adj_mask[u] >> v) & 1:
                    if has_q3_with_edge(adj_mask, u, v):
                        return False
        return True
    
    best_edge_count = 0
    best_adj_mask = None
    
    random.seed(42)
    
    num_restarts = 200
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        adj_mask = [0] * n
        
        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        # Different starting structures
        if restart == 0:
            # K3,13
            for u in range(3):
                for v in range(3, 16):
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
        elif restart == 1:
            # K8,8
            for u in range(8):
                for v in range(8, 16):
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
        elif restart == 2:
            # K3,13 + 13-cycle in part B
            for u in range(3):
                for v in range(3, 16):
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
            # Add cycle 3-4-5-6-7-8-9-10-11-12-13-14-15-3
            for i in range(13):
                u = 3 + i
                v = 3 + ((i + 1) % 13)
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
        elif restart == 3:
            # K4,12 + some structure
            for u in range(4):
                for v in range(4, 16):
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
            # Add edges within part A
            adj_mask[0] |= (1 << 1)
            adj_mask[1] |= (1 << 0)
            adj_mask[0] |= (1 << 2)
            adj_mask[2] |= (1 << 0)
            adj_mask[0] |= (1 << 3)
            adj_mask[3] |= (1 << 0)
        else:
            # Random start with varying density
            density = random.uniform(0.2, 0.6)
            for i, j in all_pairs:
                if random.random() < density:
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)
        
        # Clean up: iteratively remove edges that create Q3
        changed = True
        while changed and time.time() < deadline - 3:
            changed = False
            for u in range(n):
                for v in range(u+1, n):
                    if (adj_mask[u] >> v) & 1:
                        # Temporarily remove
                        adj_mask[u] &= ~(1 << v)
                        adj_mask[v] &= ~(1 << u)
                        # Check if without this edge, no Q3 exists
                        # Actually, we need to check if this edge is part of a Q3
                        # Simpler: check if the current graph (without edge) has Q3
                        # But that's expensive. Instead, check if adding this edge to the rest creates Q3
                        # Let's just check if the full graph has Q3, and if so, remove this edge
                        # This is O(E * Q3_check) which might be too slow
                        # Let's do a different approach: for each edge, check if it's in a Q3
                        adj_mask[u] |= (1 << v)
                        adj_mask[v] |= (1 << u)
                        
                        # Check if this specific edge is in a Q3
                        if has_q3_with_edge(adj_mask, u, v):
                            # Remove this edge
                            adj_mask[u] &= ~(1 << v)
                            adj_mask[v] &= ~(1 << u)
                            changed = True
        
        # Now greedily add missing edges
        if time.time() > deadline - 3:
            break
        
        current_edges = count_edges(adj_mask)
        
        # Get missing edges
        missing = []
        for i in range(n):
            for j in range(i+1, n):
                if not ((adj_mask[i] >> j) & 1):
                    missing.append((i, j))
        
        # Sort by degree sum (add low-degree edges first)
        degrees = [bin(adj_mask[i]).count('1') for i in range(n)]
        missing.sort(key=lambda e: degrees[e[0]] + degrees[e[1]])
        
        for u, v in missing:
            if time.time() > deadline - 2:
                break
            
            # Tentatively add
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            
            # Check if Q3 exists
            if has_q3_with_edge(adj_mask, u, v):
                # Revert
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
            else:
                current_edges += 1
        
        # Verify final graph
        if verify_no_q3(adj_mask):
            if current_edges > best_edge_count:
                best_edge_count = current_edges
                best_adj_mask = adj_mask[:]
    
    # If no valid graph found, return empty or a known valid graph
    if best_adj_mask is None:
        # Return a simple valid graph (e.g., empty or sparse)
        adj_mask = [0] * n
        best_adj_mask = adj_mask
        best_edge_count = 0
    
    # Return edges
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj_mask[i] >> j) & 1:
                edges.append([i, j])
    
    return edges
