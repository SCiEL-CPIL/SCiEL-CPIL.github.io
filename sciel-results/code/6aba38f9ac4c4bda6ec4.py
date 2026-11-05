import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n = 20
    num_colours = 20
    
    # Bipartite sets
    U = list(range(10))
    V = list(range(10, 20))
    
    # Use a Latin square coloring: c(u,v) = (u + v) mod 10, mapped to 0..19
    # Actually let's use c(u,v) = (u + v) mod 10 which uses colors 0-9
    # But we need to check if this satisfies the 4-cycle rainbow condition
    # For 4-cycle (u1,v1,u2,v2): colors are (u1+v1)%10, (u2+v1)%10, (u1+v2)%10, (u2+v2)%10
    # These are NOT always distinct (e.g., u1=0,u2=1,v1=0,v2=1 gives 0,1,1,2)
    
    # Better approach: use c(u,v) = u*10 + v, but that's 0..99, too many
    # Use c(u,v) = (u + v) mod 20? For u,v in 0..9, u+v in 0..18, so mod 20 = u+v
    # Then c(u1,v1)=u1+v1, c(u2,v2)=u2+v2. For 4-cycle: u1+v1, u2+v1, u1+v2, u2+v2
    # u1+v1 vs u2+v2: distinct if u1+v1 != u2+v2
    # u1+v2 vs u2+v1: distinct if u1+v2 != u2+v1
    # But u1+v1 and u1+v2 are always distinct (v1!=v2), same for u1+v1 and u2+v1
    # So we only need: u1+v1 != u2+v2 AND u1+v2 != u2+v1
    # i.e., u1-u2 != v2-v1 AND u1-u2 != -(v2-v1), i.e., u1-u2 != ±(v2-v1)
    
    # So with c(u,v) = u+v (no mod needed since max is 18 < 20), a 4-cycle is bad iff
    # |u1-u2| == |v1-v2|
    
    # Strategy: start with all 100 edges, remove edges to break all "bad" 4-cycles
    # A bad 4-cycle is one where |u1-u2| == |v1-v2|
    
    # Build edge list
    edges = []
    edge_set = set()
    for u in U:
        for v in V:
            edge = (u, v)
            edges.append(edge)
            edge_set.add(edge)
    
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    
    # Find all bad 4-cycles: (u1,v1,u2,v2) where u1<u2, v1<v2, |u1-u2|==|v1-v2|
    bad_cycles = []
    for i in range(10):
        for j in range(i+1, 10):
            u1, u2 = U[i], U[j]
            du = abs(u2 - u1)
            for k in range(10):
                for l in range(k+1, 10):
                    v1, v2 = V[k], V[l]
                    dv = abs(v2 - v1)
                    if du == dv:
                        # 4-cycle: (u1,v1), (u2,v1), (u1,v2), (u2,v2)
                        e1 = (u1, v1)
                        e2 = (u2, v1)
                        e3 = (u1, v2)
                        e4 = (u2, v2)
                        bad_cycles.append((edge_to_idx[e1], edge_to_idx[e2], edge_to_idx[e3], edge_to_idx[e4]))
    
    # Greedy removal: remove edge in most bad cycles
    active_edges = set(range(len(edges)))
    active_cycles = set(range(len(bad_cycles)))
    
    # For each edge, list of cycles it's in
    edge_to_cycles = {i: [] for i in range(len(edges))}
    for ci, cycle in enumerate(bad_cycles):
        for ei in cycle:
            edge_to_cycles[ei].append(ci)
    
    removed_count = 0
    while active_cycles:
        if time.time() > deadline:
            break
        
        # Find edge with max active cycle count
        max_count = 0
        target = -1
        for ei in active_edges:
            count = sum(1 for ci in edge_to_cycles[ei] if ci in active_cycles)
            if count > max_count:
                max_count = count
                target = ei
        
        if target == -1 or max_count == 0:
            break
        
        active_edges.remove(target)
        removed_count += 1
        
        # Remove all cycles containing this edge
        cycles_to_remove = [ci for ci in edge_to_cycles[target] if ci in active_cycles]
        for ci in cycles_to_remove:
            active_cycles.remove(ci)
    
    # Build result
    final_edges = []
    for ei in sorted(active_edges):
        u, v = edges[ei]
        c = u + v  # This is in range 0..18, all valid colors 0..19
        final_edges.append([u, v, c])
    
    # Verify
    adj = {}
    for u, v, c in final_edges:
        adj[(u, v)] = c
        adj[(v, u)] = c
    
    valid = True
    for i in range(10):
        for j in range(i+1, 10):
            u1, u2 = U[i], U[j]
            for k in range(10):
                for l in range(k+1, 10):
                    v1, v2 = V[k], V[l]
                    if (u1,v1) in adj and (u2,v1) in adj and (u1,v2) in adj and (u2,v2) in adj:
                        c1 = adj[(u1,v1)]
                        c2 = adj[(u2,v1)]
                        c3 = adj[(u1,v2)]
                        c4 = adj[(u2,v2)]
                        if len({c1,c2,c3,c4}) < 4:
                            valid = False
                            break
                if not valid:
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback: star
        return [[0, v, 0] for v in range(1, 20)]
    
    return final_edges

if __name__ == "__main__":
    res = construct()
    print(f"Edges: {len(res)}")
