import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    d = 7
    n = 2 ** d
    total_edges = n * d // 2  # 448
    
    # 1. Generate all edges and map to indices
    # Edge index: for x, bit i (x < y), idx = x*d + i (if x < y) or something unique.
    # Better: list of edges, and a dict (u,v) -> index
    all_edges = []
    edge_index = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(all_edges)
                all_edges.append((x, y))
                edge_index[(x, y)] = idx
                # Note: we only store x<y. For lookup, always normalize.
    
    # 2. Precompute for each edge, the "partner" edges that form a 4-cycle with it.
    # An edge (u, v) with bit i. A 4-cycle is formed by (u, v), (u^j, v^j), and the two edges connecting them?
    # No. A 4-cycle in Q7 involving edge (u, v) (diff bit i) and bit j (j!=i) consists of:
    # e1 = (u, v)
    # e2 = (u^j, v^j)
    # e3 = (u, u^j)
    # e4 = (v, v^j)
    # So for each edge e and each bit j != bit(e), there is a specific set of 3 other edges.
    # Let's store for each edge index, a list of (other_edge_indices, bit_j) that form a C4.
    # Actually, we just need the count of C4s.
    # For an edge e (u,v) with bit i, the number of C4s it is in is the number of j != i
    # such that the other 3 edges are present.
    
    # To speed up, for each edge, we can precompute the indices of the "parallel" edge (u^j, v^j)
    # and the two "side" edges (u, u^j) and (v, v^j).
    
    edge_info = [] # list of (u, v, bit)
    for idx, (u, v) in enumerate(all_edges):
        # find bit
        diff = u ^ v
        bit = diff.bit_length() - 1
        edge_info.append((u, v, bit))
        
    # Precompute C4 partners
    # c4_partners[e_idx] = list of tuples (e_parallel, e_side1, e_side2)
    # where e_parallel is the edge (u^j, v^j)
    c4_partners = [[] for _ in range(total_edges)]
    
    for idx, (u, v, bit) in enumerate(edge_info):
        for j in range(d):
            if j == bit:
                continue
            u2 = u ^ (1 << j)
            v2 = v ^ (1 << j)
            # Edge (u2, v2)
            if u2 > v2: u2, v2 = v2, u2
            if (u2, v2) not in edge_index:
                continue # Should not happen in hypercube
            e_par = edge_index[(u2, v2)]
            
            # Edge (u, u2)
            if u > u2: u, u2 = u2, u
            if (u, u2) not in edge_index:
                continue
            e_s1 = edge_index[(u, u2)]
            
            # Edge (v, v2)
            if v > v2: v, v2 = v2, v
            if (v, v2) not in edge_index:
                continue
            e_s2 = edge_index[(v, v2)]
            
            c4_partners[idx].append((e_par, e_s1, e_s2))
            
    # 3. State: boolean array of edge presence
    present = [True] * total_edges
    
    # 4. Count C4s.
    # A C4 is defined by 4 edges. It is "bad" if all 4 are present.
    # We want to minimize the number of bad C4s.
    # Initially, all edges present, so all C4s are bad.
    # Number of C4s in Q7:
    # Each C4 is determined by 2 bits and a vertex (with those bits 0).
    # Number of C4s = C(7,2) * 2^(7-2) = 21 * 32 = 672.
    
    # Let's verify: 21 * 32 = 672.
    # Each edge is in (7-1) = 6 C4s.
    # Total edge-C4 incidences = 448 * 6 = 2688.
    # 672 * 4 = 2688. Correct.
    
    # We need to delete edges to break all 672 C4s.
    # Strategy: Greedy deletion.
    # At each step, find the edge that is in the most *current* bad C4s.
    # Delete it.
    # Repeat until no bad C4s remain.
    
    # To do this efficiently, we maintain for each edge the count of bad C4s it participates in.
    # c4_count[e] = number of bad C4s containing e.
    
    # Initial c4_count: each edge is in 6 C4s, all bad. So c4_count = [6]*448.
    c4_count = [6] * total_edges
    
    # We also need to know, for each C4, which edges are in it, to update counts when an edge is deleted.
    # Let's list all C4s.
    all_c4s = []
    for x in range(n):
        # Only consider x where the two bits are 0 to avoid duplicates?
        # A C4 is a set of 4 vertices.
        # Let's iterate over pairs of bits (i, j) and vertices x where bit i and j are 0.
        for i in range(d):
            for j in range(i+1, d):
                # x must have bit i and j as 0
                if (x & (1 << i)) == 0 and (x & (1 << j)) == 0:
                    v1 = x
                    v2 = x ^ (1 << i)
                    v3 = x ^ (1 << j)
                    v4 = x ^ (1 << i) ^ (1 << j)
                    
                    # Edges:
                    # (v1, v2) bit i
                    # (v3, v4) bit i
                    # (v1, v3) bit j
                    # (v2, v4) bit j
                    
                    e1 = edge_index.get((min(v1, v2), max(v1, v2)))
                    e2 = edge_index.get((min(v3, v4), max(v3, v4)))
                    e3 = edge_index.get((min(v1, v3), max(v1, v3)))
                    e4 = edge_index.get((min(v2, v4), max(v2, v4)))
                    
                    if e1 is not None and e2 is not None and e3 is not None and e4 is not None:
                        all_c4s.append((e1, e2, e3, e4))
                        
    # Verify count
    # assert len(all_c4s) == 672
    
    # Data structure:
    # bad_c4s: set of indices of C4s that are currently bad (all 4 edges present)
    # Initially all are bad.
    bad_c4s = set(range(len(all_c4s)))
    
    # c4_membership: for each edge, list of C4 indices it belongs to.
    c4_membership = [[] for _ in range(total_edges)]
    for c4_idx, (e1, e2, e3, e4) in enumerate(all_c4s):
        for e in (e1, e2, e3, e4):
            c4_membership[e].append(c4_idx)
            
    # Greedy deletion loop
    # While bad_c4s is not empty:
    #   Find edge e with max c4_count[e] (among present edges)
    #   Delete e
    #   Update c4_count and bad_c4s
    
    # Optimization: Use a heap or just scan. Scanning 448 items is fast.
    # But we need to be careful with time.
    
    while bad_c4s:
        if time.time() > deadline - 10:
            break
            
        # Find max c4_count
        max_count = -1
        best_e = -1
        for e in range(total_edges):
            if present[e] and c4_count[e] > max_count:
                max_count = c4_count[e]
                best_e = e
                
        if best_e == -1:
            break # Should not happen if bad_c4s not empty
            
        # Delete best_e
        present[best_e] = False
        
        # Update counts
        # For each C4 containing best_e:
        for c4_idx in c4_membership[best_e]:
            if c4_idx in bad_c4s:
                # This C4 becomes good (not bad)
                bad_c4s.remove(c4_idx)
                # Decrement count for the other 3 edges in this C4
                e1, e2, e3, e4 = all_c4s[c4_idx]
                for other_e in (e1, e2, e3, e4):
                    if other_e != best_e:
                        c4_count[other_e] -= 1
        # c4_count[best_e] becomes 0 effectively, but we mark it not present.
        
    # Now we have a valid set.
    # Count edges
    m = sum(present)
    
    # Try to add back edges
    # We have a list of removed edges.
    removed = [e for e in range(total_edges) if not present[e]]
    random.shuffle(removed)
    
    for e in removed:
        if time.time() > deadline - 5:
            break
            
        # Check if adding e creates any bad C4
        # e is in c4_membership[e] C4s.
        # For each C4, check if the other 3 edges are present.
        can_add = True
        for c4_idx in c4_membership[e]:
            e1, e2, e3, e4 = all_c4s[c4_idx]
            # Check if the other 3 are present
            # Note: e is not present yet.
            others = [x for x in (e1, e2, e3, e4) if x != e]
            if all(present[x] for x in others):
                can_add = False
                break
                
        if can_add:
            present[e] = True
            m += 1
            
    # Return edges
    out = []
    for idx, (u, v) in enumerate(all_edges):
        if present[idx]:
            out.append([u, v])
            
    return out
# EVOLVE-BLOCK-END
