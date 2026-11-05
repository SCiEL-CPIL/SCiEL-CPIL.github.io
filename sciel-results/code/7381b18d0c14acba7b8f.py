"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    d = 7
    n = 2 ** d
    
    # Precompute all edges and their 4-cycles
    all_edges = []
    edge_to_idx = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(all_edges)
                all_edges.append((x, y))
                edge_to_idx[(x, y)] = idx
    
    m = len(all_edges)  # 448 edges
    
    # Precompute for each edge, the other 3 edges in each 4-face it belongs to
    # An edge (u,v) is in a 4-face with direction i (the bit that differs)
    # The other 3 edges are: (u^j, v^j), (u^j, u), (v^j, v) for j != i
    # Wait, let me think more carefully.
    # Edge (u,v) where v = u ^ (1<<i).
    # A 4-face containing this edge uses another direction j.
    # The 4 vertices are: u, v=u^(1<<i), u^(1<<j), v^(1<<j)=u^(1<<i)^(1<<j)
    # The 4 edges are: (u,v), (u, u^(1<<j)), (v, v^(1<<j)), (u^(1<<j), v^(1<<j))
    # So the other 3 edges are: (u, u^(1<<j)), (v, v^(1<<j)), (u^(1<<j), v^(1<<j))
    
    edge_faces = [[] for _ in range(m)]  # For each edge, list of (other_edge1, other_edge2, other_edge3) for each face
    
    for idx, (x, y) in enumerate(all_edges):
        # Determine which bit differs
        diff = x ^ y
        i = diff.bit_length() - 1
        for j in range(d):
            if j == i:
                continue
            c = x ^ (1 << j)
            e = y ^ (1 << j)
            e1 = (min(x, c), max(x, c))
            e2 = (min(y, e), max(y, e))
            e3 = (min(c, e), max(c, e))
            idx1 = edge_to_idx.get(e1)
            idx2 = edge_to_idx.get(e2)
            idx3 = edge_to_idx.get(e3)
            if idx1 is not None and idx2 is not None and idx3 is not None:
                edge_faces[idx].append((idx1, idx2, idx3))
    
    # Greedy initialization
    chosen = [False] * m
    count = 0
    for idx in range(m):
        x, y = all_edges[idx]
        ok = True
        for (idx1, idx2, idx3) in edge_faces[idx]:
            if chosen[idx1] and chosen[idx2] and chosen[idx3]:
                ok = False
                break
        if ok:
            chosen[idx] = True
            count += 1
    
    best_count = count
    best_chosen = chosen.copy()
    
    # Simulated Annealing
    T = 5.0
    T_min = 0.01
    cooling = 0.9999
    
    while time.time() < deadline and T > T_min:
        # Randomly pick an edge to toggle
        idx = random.randint(0, m - 1)
        
        if chosen[idx]:
            # Try removing edge
            chosen[idx] = False
            count -= 1
            # Always accept removals in SA (or with high probability)
            if count > best_count:
                best_count = count
                best_chosen = chosen.copy()
            # Actually, we should keep removals to allow exploration
            # But we need to be careful - let's use proper SA
            # Undo and use proper SA logic
            chosen[idx] = True
            count += 1
            
            # Proper SA: try to remove
            chosen[idx] = False
            count -= 1
            if count > best_count:
                best_count = count
                best_chosen = chosen.copy()
            # Accept removal with probability based on temperature
            # For removals, delta = -1, so always accept at reasonable T
            # But we want to keep good solutions, so only keep if it leads to improvement
            # Actually, let's use a different approach: 
            # Try adding edges that are currently not chosen
            pass
        else:
            # Try adding edge - check if it creates a 4-cycle
            ok = True
            for (idx1, idx2, idx3) in edge_faces[idx]:
                if chosen[idx1] and chosen[idx2] and chosen[idx3]:
                    ok = False
                    break
            
            if ok:
                # Can add without creating 4-cycle
                chosen[idx] = True
                count += 1
                if count > best_count:
                    best_count = count
                    best_chosen = chosen.copy()
            else:
                # Would create 4-cycle - this is where SA helps
                # We could remove one of the conflicting edges to add this one
                # For now, just skip (this is where the structural change needs to be)
                pass
        
        T *= cooling
    
    # Convert best_chosen to output format
    out = []
    for idx in range(m):
        if best_chosen[idx]:
            x, y = all_edges[idx]
            out.append([x, y])
    
    return out
# EVOLVE-BLOCK-END
