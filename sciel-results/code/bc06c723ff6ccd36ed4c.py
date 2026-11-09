"""Four-cycle-free edge sets of Q7: maximize edges with no 2-face fully selected."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    d = 7
    n = 1 << d  # 128
    total_edges = n * d // 2  # 448

    # 1. Generate all edges and map to indices
    all_edges = []
    edge_index = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(all_edges)
                all_edges.append((x, y))
                edge_index[(x, y)] = idx

    # 2. Precompute edge info (u, v, bit)
    edge_info = []
    for idx, (u, v) in enumerate(all_edges):
        diff = u ^ v
        bit = diff.bit_length() - 1
        edge_info.append((u, v, bit))

    # 3. Precompute all C4s (faces)
    # A C4 is determined by 2 bits (i, j) and a vertex x where bits i, j are 0.
    # Vertices: x, x^i, x^j, x^i^j
    # Edges: (x, x^i), (x^j, x^i^j), (x, x^j), (x^i, x^i^j)
    all_c4s = []
    for i in range(d):
        for j in range(i + 1, d):
            mask = (1 << i) | (1 << j)
            for x in range(n):
                if (x & mask) == 0:
                    v1 = x
                    v2 = x ^ (1 << i)
                    v3 = x ^ (1 << j)
                    v4 = x ^ (1 << i) ^ (1 << j)
                    
                    e1 = edge_index[(min(v1, v2), max(v1, v2))]
                    e2 = edge_index[(min(v3, v4), max(v3, v4))]
                    e3 = edge_index[(min(v1, v3), max(v1, v3))]
                    e4 = edge_index[(min(v2, v4), max(v2, v4))]
                    
                    all_c4s.append((e1, e2, e3, e4))

    num_c4s = len(all_c4s)
    
    # 4. Precompute for each edge, the list of C4 indices it belongs to
    c4_membership = [[] for _ in range(total_edges)]
    for c4_idx, (e1, e2, e3, e4) in enumerate(all_c4s):
        for e in (e1, e2, e3, e4):
            c4_membership[e].append(c4_idx)

    # 5. Greedy Deletion Phase
    # Start with all edges present.
    # Maintain count of "bad" C4s (all 4 edges present) for each edge.
    # Initially, all C4s are bad. Each edge is in 6 C4s.
    present = [True] * total_edges
    c4_count = [6] * total_edges  # Each edge is in exactly 6 C4s in Q7
    
    bad_c4s = set(range(num_c4s))
    
    # To speed up finding the max, we can just scan. 448 items is small.
    # But we need to update counts efficiently.
    
    while bad_c4s:
        if time.time() > deadline - 30:
            break
            
        max_count = -1
        best_e = -1
        for e in range(total_edges):
            if present[e] and c4_count[e] > max_count:
                max_count = c4_count[e]
                best_e = e
                
        if best_e == -1:
            break
            
        # Delete best_e
        present[best_e] = False
        
        # Update counts for C4s containing best_e
        for c4_idx in c4_membership[best_e]:
            if c4_idx in bad_c4s:
                bad_c4s.remove(c4_idx)
                e1, e2, e3, e4 = all_c4s[c4_idx]
                for other_e in (e1, e2, e3, e4):
                    if other_e != best_e:
                        c4_count[other_e] -= 1

    m = sum(present)
    
    # 6. Local Search / Refinement Phase
    # Try to add back edges that were removed.
    # Also try to swap edges if it improves the count.
    
    removed = [e for e in range(total_edges) if not present[e]]
    random.shuffle(removed)
    
    for e in removed:
        if time.time() > deadline - 10:
            break
            
        # Check if adding e creates any bad C4
        # e is in c4_membership[e] C4s.
        # For each C4 containing e, check if the other 3 edges are present.
        ok = True
        for c4_idx in c4_membership[e]:
            e1, e2, e3, e4 = all_c4s[c4_idx]
            # Check other 3
            others = [x for x in (e1, e2, e3, e4) if x != e]
            if all(present[x] for x in others):
                ok = False
                break
        if ok:
            present[e] = True
            m += 1

    # 7. Second Pass: Try to add more edges
    for _ in range(10):
        if time.time() > deadline - 10:
            break
        progress = False
        for e in removed:
            if present[e]:
                continue
            ok = True
            for c4_idx in c4_membership[e]:
                e1, e2, e3, e4 = all_c4s[c4_idx]
                others = [x for x in (e1, e2, e3, e4) if x != e]
                if all(present[x] for x in others):
                    ok = False
                    break
            if ok:
                present[e] = True
                m += 1
                progress = True
        if not progress:
            break

    # 8. Swap-based local search
    # Try to swap a removed edge with a present edge if it improves the count.
    # Actually, since we want to maximize edges, and we are at a valid state,
    # swapping doesn't directly help unless we can add more edges later.
    # However, the directive mentions swap-based local search as an improvement.
    # Let's try a simple version: pick a random present edge, remove it, 
    # then try to add back all removed edges (including the one we just removed).
    # If we can add more edges than we removed, keep the swap.
    
    # We need to be careful with time.
    # Let's do a few iterations of this.
    
    for _ in range(20):
        if time.time() > deadline - 5:
            break
            
        # Pick a random present edge to remove
        present_edges = [e for e in range(total_edges) if present[e]]
        if not present_edges:
            break
        e_remove = random.choice(present_edges)
        
        # Save state
        old_present = present[:]
        old_m = m
        
        # Remove e_remove
        present[e_remove] = False
        m -= 1
        
        # Try to add back all removed edges
        removed_edges = [e for e in range(total_edges) if not present[e]]
        random.shuffle(removed_edges)
        
        for e in removed_edges:
            if time.time() > deadline - 5:
                break
            ok = True
            for c4_idx in c4_membership[e]:
                e1, e2, e3, e4 = all_c4s[c4_idx]
                others = [x for x in (e1, e2, e3, e4) if x != e]
                if all(present[x] for x in others):
                    ok = False
                    break
            if ok:
                present[e] = True
                m += 1
        
        # If we didn't improve, revert
        if m <= old_m:
            present = old_present
            m = old_m
        else:
            # Keep the improvement
            pass

    # 9. Construct Output
    out = []
    for idx, (u, v) in enumerate(all_edges):
        if present[idx]:
            out.append([u, v])
            
    return out

# EVOLVE-BLOCK-END
