import time
import sys
import random
import numpy as np

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 115  # 35 second margin under 150s limit
    
    # 1. Generate vertices of PG(2, 5)
    pts = []
    for x in range(5):
        for y in range(5):
            pts.append((x, y, 1))
    for y in range(5):
        pts.append((1, y, 0))
    pts.append((0, 1, 0))
    
    n = len(pts)
    assert n == 31
    
    # 2. Define Polarity: x1*x2 + y1*y2 + z1*z2 = 0 mod 5
    def is_adjacent(p1, p2):
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        return (x1*x2 + y1*y2 + z1*z2) % 5 == 0
    
    # Build adjacency for core graph (90 edges, C4-free)
    adj = [set() for _ in range(31)]
    edges_core = []
    for i in range(31):
        for j in range(i+1, 31):
            if is_adjacent(pts[i], pts[j]):
                adj[i].add(j)
                adj[j].add(i)
                edges_core.append([i, j])
    
    edge_count = sum(len(s) for s in adj) // 2
    assert edge_count == 90, f"Expected 90 core edges, got {edge_count}"
    
    # 3. Find best edge to add (minimize new C4s)
    # For each non-edge (u,v), the number of new C4s created = number of common neighbors of u and v
    # in the current graph (since adding edge uv creates a C4 for each pair of common neighbors)
    best_u = -1
    best_v = -1
    min_common = float('inf')
    
    for u in range(31):
        if time.time() > deadline - 10:
            break
        for v in range(u+1, 31):
            if v in adj[u]:
                continue
            common = len(adj[u] & adj[v])
            if common < min_common:
                min_common = common
                best_u = u
                best_v = v
                if min_common == 0:
                    break
        if min_common == 0:
            break
    
    if best_u == -1:
        for u in range(31):
            for v in range(u+1, 31):
                if v not in adj[u]:
                    best_u = u
                    best_v = v
                    break
            if best_u != -1:
                break
    
    # Add the edge
    adj[best_u].add(best_v)
    adj[best_v].add(best_u)
    edges_current = edges_core + [[best_u, best_v]]
    
    # 4. Fast C4 counting using numpy
    def count_c4(adj_list):
        """Count C4s using common neighbor counts.
        Total C4s = (1/2) * sum_{i<j} C(common(i,j), 2)
        """
        total = 0
        for i in range(31):
            for j in range(i+1, 31):
                c = len(adj_list[i] & adj_list[j])
                if c >= 2:
                    total += c * (c - 1) // 2
        return total // 2
    
    # Even faster: precompute common neighbor matrix
    def compute_common_matrix(adj_list):
        """Return a 31x31 matrix where common[i][j] = |N(i) ∩ N(j)|"""
        common = np.zeros((31, 31), dtype=np.int32)
        for i in range(31):
            for j in range(i+1, 31):
                c = len(adj_list[i] & adj_list[j])
                common[i][j] = c
                common[j][i] = c
        return common
    
    def count_c4_from_common(common):
        """Count C4s from precomputed common neighbor matrix"""
        total = 0
        for i in range(31):
            for j in range(i+1, 31):
                c = common[i][j]
                if c >= 2:
                    total += c * (c - 1) // 2
        return total // 2
    
    current_c4 = count_c4(adj)
    
    # 5. Local search with efficient delta computation
    # For removing edge (u,v) and adding edge (a,b):
    # We need to compute the change in C4 count.
    # 
    # Key insight: C4 count = (1/2) * sum_{i<j} C(c(i,j), 2) where c(i,j) = |N(i)∩N(j)|
    # 
    # When we remove edge (u,v):
    #   - For each vertex w, if w is adjacent to both u and v, then c(u,v) decreases by 1 (w is no longer a common neighbor via the edge... wait, no)
    #   Actually, removing edge (u,v) doesn't change the common neighbor counts directly.
    #   It changes the adjacency, which affects which pairs are considered.
    #
    # Let me think more carefully. The C4 count formula uses common neighbors.
    # When we remove edge (u,v):
    #   - c(u,v) stays the same (common neighbors don't change when you remove an edge between them)
    #   - But for any vertex w, c(u,w) might change if v was a common neighbor of u and w, or vice versa
    #   - Specifically, if v is in N(u) ∩ N(w), then removing edge (u,v) means v is no longer in N(u), so v is no longer a common neighbor of u and w. So c(u,w) decreases by 1 if v ∈ N(w).
    #   - Similarly, c(v,w) decreases by 1 if u ∈ N(w).
    #
    # This is getting complex. Let me use a different approach:
    # Use a bitset representation for fast common neighbor computation.
    
    # Bitset-based approach
    def sets_to_bitsets(adj_list):
        """Convert adjacency sets to integer bitsets"""
        bitsets = [0] * 31
        for i in range(31):
            mask = 0
            for j in adj_list[i]:
                mask |= (1 << j)
            bitsets[i] = mask
        return bitsets
    
    def count_c4_bitset(bitsets):
        """Count C4s using bitset operations"""
        total = 0
        for i in range(31):
            for j in range(i+1, 31):
                common = bitsets[i] & bitsets[j]
                c = bin(common).count('1')
                if c >= 2:
                    total += c * (c - 1) // 2
        return total // 2
    
    bitsets = sets_to_bitsets(adj)
    current_c4 = count_c4_bitset(bitsets)
    
    # Local search: try swaps
    # For each edge to remove and each non-edge to add, compute delta
    edge_set = set()
    for e in edges_current:
        edge_set.add((min(e[0], e[1]), max(e[0], e[1])))
    
    non_edges = []
    for i in range(31):
        for j in range(i+1, 31):
            if (i, j) not in edge_set:
                non_edges.append((i, j))
    
    best_c4 = current_c4
    best_edges = [e[:] for e in edges_current]
    
    # For efficient delta computation, we'll compute the C4 count after each swap
    # using the bitset representation
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Shuffle edges for diversity
        edge_order = list(range(len(edges_current)))
        random.shuffle(edge_order)
        
        for rem_idx in edge_order:
            if time.time() > deadline - 5:
                break
            
            ru, rv = edges_current[rem_idx]
            ru, rv = min(ru, rv), max(ru, rv)
            
            # Remove edge from bitsets
            bitsets[ru] &= ~(1 << rv)
            bitsets[rv] &= ~(1 << ru)
            
            for add_edge in non_edges:
                au, av = add_edge
                au, av = min(au, av), max(au, av)
                
                # Skip if this would create a self-loop or duplicate
                if au == ru and av == rv:
                    continue
                if au == ru or au == rv or av == ru or av == rv:
                    # This is fine as long as it's not the same edge
                    if (au, av) == (ru, rv):
                        continue
                
                # Add edge to bitsets
                bitsets[au] |= (1 << av)
                bitsets[av] |= (1 << au)
                
                new_c4 = count_c4_bitset(bitsets)
                
                if new_c4 < best_c4:
                    best_c4 = new_c4
                    # Update state
                    edges_current[rem_idx] = [au, av]
                    edge_set.discard((ru, rv))
                    edge_set.add((au, av))
                    # Update non_edges
                    non_edges = [(i, j) for i in range(31) for j in range(i+1, 31) if (i, j) not in edge_set]
                    improved = True
                    break
                
                # Undo add
                bitsets[au] &= ~(1 << av)
                bitsets[av] &= ~(1 << au)
            
            # Undo remove
            bitsets[ru] |= (1 << rv)
            bitsets[rv] |= (1 << ru)
            
            if improved:
                break
    
    # Final verification
    final_edges = [e[:] for e in edges_current]
    assert len(final_edges) == 91, f"Expected 91 edges, got {len(final_edges)}"
    assert len(set(tuple(sorted(e)) for e in final_edges)) == 91
    
    return final_edges
