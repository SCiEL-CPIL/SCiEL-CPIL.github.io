import random
import heapq
import numpy as np

def construct():
    d = 7
    n_vertices = 2 ** d
    total_edges = d * (2 ** (d - 1))
    
    # 1. Generate all edges
    edges = []
    edge_to_id = {}
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                eid = len(edges)
                edges.append((x, y))
                edge_to_id[(x, y)] = eid
                edge_to_id[(y, x)] = eid

    # 2. Generate all 4-cycles
    cycles = set()
    for x in range(n_vertices):
        for i in range(d):
            for j in range(i + 1, d):
                v0 = x
                v1 = x ^ (1 << i)
                v2 = x ^ (1 << i) ^ (1 << j)
                v3 = x ^ (1 << j)
                
                e0 = edge_to_id[(min(v0, v1), max(v0, v1))]
                e1 = edge_to_id[(min(v1, v2), max(v1, v2))]
                e2 = edge_to_id[(min(v2, v3), max(v2, v3))]
                e3 = edge_to_id[(min(v3, v0), max(v3, v0))]
                
                cycle = frozenset([e0, e1, e2, e3])
                cycles.add(cycle)
                
    cycles_list = list(cycles)
    num_cycles = len(cycles_list)
    
    # Map edge to list of cycle indices
    edge_to_cycles = {e: [] for e in range(total_edges)}
    for idx, c in enumerate(cycles_list):
        for e in c:
            edge_to_cycles[e].append(idx)
            
    best_removed = None
    best_size = total_edges + 1
    
    # Number of restarts
    num_restarts = 200
    
    for _ in range(num_restarts):
        removed = set()
        uncovered_cycles = set(range(num_cycles))
        
        # Greedy Selection
        while uncovered_cycles:
            candidates = []
            
            # Iterate edges not in removed
            for e in range(total_edges):
                if e in removed:
                    continue
                count = 0
                for c_idx in edge_to_cycles[e]:
                    if c_idx in uncovered_cycles:
                        count += 1
                if count > 0:
                    candidates.append((count, e))
                    
            if not candidates:
                break
                
            # Get top 5 candidates
            top_k = 5
            if len(candidates) < top_k:
                top_k = len(candidates)
            
            # nlargest is faster than sorting full list
            top_candidates = heapq.nlargest(top_k, candidates)
            
            # Random choice from top
            _, best_edge = random.choice(top_candidates)
            
            removed.add(best_edge)
            
            # Update uncovered cycles
            for c_idx in edge_to_cycles[best_edge]:
                if c_idx in uncovered_cycles:
                    uncovered_cycles.remove(c_idx)
                    
        # Redundancy Removal
        # Count how many removed edges are in each cycle
        cycle_cover_count = np.zeros(num_cycles, dtype=int)
        for e in removed:
            for c_idx in edge_to_cycles[e]:
                cycle_cover_count[c_idx] += 1
                
        changed = True
        while changed:
            changed = False
            for e in list(removed):
                is_essential = False
                for c_idx in edge_to_cycles[e]:
                    if cycle_cover_count[c_idx] == 1:
                        is_essential = True
                        break
                if not is_essential:
                    removed.remove(e)
                    for c_idx in edge_to_cycles[e]:
                        cycle_cover_count[c_idx] -= 1
                    changed = True
                    
        if len(removed) < best_size:
            best_size = len(removed)
            best_removed = removed.copy()
            
    # Construct final output
    final_edges = []
    for i, (u, v) in enumerate(edges):
        if i not in best_removed:
            final_edges.append([u, v])
            
    return final_edges
