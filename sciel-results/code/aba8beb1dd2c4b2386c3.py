"""Four-cycle-free edge sets of Q7: maximize edges with no 2-face fully selected."""
import time
import random

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
        if time.time() > deadline - 20:
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
    # Also try to swap edges if it improves the count (though we are at 0 bad C4s, 
    # adding an edge might create bad C4s, so we just check validity).
    # Actually, since we want to maximize edges, and we are at a valid state,
    # we can try to add any removed edge if it doesn't create a bad C4.
    
    removed = [e for e in range(total_edges) if not present[e]]
    random.shuffle(removed)
    
    for e in removed:
        if time.time() > deadline - 5:
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

    # 7. Second Pass: Try to add more edges (in case the first pass added some,
    #    which might have blocked others, or simply to be thorough)
    #    Actually, if we add an edge, it only creates constraints, it doesn't relax them.
    #    So if an edge couldn't be added, it still can't be added later.
    #    However, the order of checking matters.
    #    Let's do a few passes.
    
    for _ in range(10):
        if time.time() > deadline - 5:
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

    # 8. Construct Output
    out = []
    for idx, (u, v) in enumerate(all_edges):
        if present[idx]:
            out.append([u, v])
            
    return out

# EVOLVE-BLOCK-END
