"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a C4-free graph on n vertices with minimum degree >= n - 39.
    Uses a greedy set cover approach on the complement graph.
    """
    
    def try_construct(n, seed):
        random.seed(seed)
        
        # Generate all 4-cycles in K_n
        # A 4-cycle is a set of 4 distinct vertices {a,b,c,d} with a specific order?
        # No, a 4-cycle in a simple graph is defined by 4 vertices and the 4 edges connecting them in a cycle.
        # In K_n, every set of 4 vertices forms 3 distinct 4-cycles (up to rotation/reflection)?
        # No, a set of 4 vertices {a,b,c,d} has exactly 3 ways to form a 4-cycle:
        # (a,b,c,d), (a,b,d,c), (a,c,b,d).
        # However, for the condition "G has no C4", we just need to ensure that for ANY 4 vertices,
        # we don't have the specific edge configuration that forms a C4.
        # Actually, it's easier to think of 4-cycles as specific edge sets of size 4.
        # There are 3 edge sets of size 4 that form a C4 on 4 vertices.
        # Let's enumerate all 4-cycles as edge sets.
        
        cycles = []
        edge_to_cycles = defaultdict(list)
        
        # Iterate over all combinations of 4 vertices
        # For each combination, there are 3 possible 4-cycles.
        # Cycle 1: (0,1), (1,2), (2,3), (3,0)
        # Cycle 2: (0,1), (1,3), (3,2), (2,0) -> Edges: {01, 13, 32, 20}
        # Cycle 3: (0,2), (2,1), (1,3), (3,0) -> Edges: {02, 21, 13, 30}
        
        # We can just generate the edge sets.
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        v = [i, j, k, l]
                        # The 3 distinct 4-cycles on these 4 vertices
                        # 1. i-j-k-l-i
                        e1 = {(i,j), (j,k), (k,l), (l,i)}
                        # 2. i-j-l-k-i
                        e2 = {(i,j), (j,l), (l,k), (k,i)}
                        # 3. i-k-j-l-i
                        e3 = {(i,k), (k,j), (j,l), (l,i)}
                        
                        for e_set in [e1, e2, e3]:
                            # Store as frozenset for hashing
                            fs = frozenset(e_set)
                            cycles.append(fs)
                            for e in e_set:
                                # Normalize edge representation (min, max)
                                u, v_edge = e
                                if u > v_edge:
                                    u, v_edge = v_edge, u
                                edge_to_cycles[(u, v_edge)].append(len(cycles) - 1)
        
        # Remove duplicates if any? The sets are distinct for the 3 cycles.
        # Check if any duplicate frozensets were added?
        # The 3 cycles on 4 vertices are distinct edge sets.
        # So no duplicates from the inner loop.
        
        active_cycles = set(range(len(cycles)))
        bar_G_edges = set()
        deg_bar = [0] * n
        
        # Greedy loop
        while active_cycles:
            # Pick a random active cycle to break ties or just the first one?
            # Random selection might help avoid local minima?
            # Let's pick one randomly.
            idx = random.choice(list(active_cycles))
            cycle_edges = cycles[idx]
            
            best_edge = None
            best_count = -1
            best_candidates = []
            
            for e in cycle_edges:
                u, v = e
                if u > v:
                    u, v = v, u
                # Check degree constraint
                if deg_bar[u] < 38 and deg_bar[v] < 38:
                    # Count active cycles covered by this edge
                    count = 0
                    for c_idx in edge_to_cycles[(u, v)]:
                        if c_idx in active_cycles:
                            count += 1
                    if count > best_count:
                        best_count = count
                        best_edge = (u, v)
                        best_candidates = [(u, v)]
                    elif count == best_count:
                        best_candidates.append((u, v))
            
            if best_edge is None:
                return None # Failed to construct
            
            # If multiple candidates, pick one randomly to add diversity
            chosen_edge = random.choice(best_candidates)
            u, v = chosen_edge
            
            bar_G_edges.add(chosen_edge)
            deg_bar[u] += 1
            deg_bar[v] += 1
            
            # Remove all active cycles containing this edge
            for c_idx in edge_to_cycles[chosen_edge]:
                if c_idx in active_cycles:
                    active_cycles.remove(c_idx)
                    
        # Construct the graph G
        # G is K_n minus bar_G_edges
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if (u, v) not in bar_G_edges:
                    edges.append([u, v])
        
        # Verify
        # 1. Min degree
        for u in range(n):
            deg = sum(1 for v in range(n) if v != u and (min(u,v), max(u,v)) not in bar_G_edges)
            if deg < n - 39:
                return None # Should not happen if logic is correct, but safety check
                
        # 2. C4-free check
        # This is expensive, but n is small.
        # For every 4 vertices, check if any of the 3 cycles is fully present in G.
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        # Check 3 cycles
                        # Cycle 1: i-j, j-k, k-l, l-i
                        if ((i,j) not in bar_G_edges and (j,k) not in bar_G_edges and 
                            (k,l) not in bar_G_edges and (l,i) not in bar_G_edges):
                            return None
                        # Cycle 2: i-j, j-l, l-k, k-i
                        if ((i,j) not in bar_G_edges and (j,l) not in bar_G_edges and 
                            (l,k) not in bar_G_edges and (k,i) not in bar_G_edges):
                            return None
                        # Cycle 3: i-k, k-j, j-l, l-i
                        if ((i,k) not in bar_G_edges and (k,j) not in bar_G_edges and 
                            (j,l) not in bar_G_edges and (l,i) not in bar_G_edges):
                            return None
                            
        return {'n': n, 'edges': edges}

    # Try n=46 first, then 45, etc.
    for n in [46, 45, 44, 43, 42]:
        # Try a few random seeds
        for seed in range(20):
            start_time = time.time()
            res = try_construct(n, seed)
            if res is not None:
                return res
            if time.time() - start_time > 50:
                break # Timeout for this seed, try next or next n
                
    # Fallback to known solution if all fail (should not happen)
    # Re-implement the original q=4 solution as fallback
    q = 4
    n_points = q*q + q + 1
    n_lines = q*q + q + 1
    n = n_points + n_lines
    
    add = [
        [0,1,2,3],
        [1,0,3,2],
        [2,3,0,1],
        [3,2,1,0]
    ]
    mul = [
        [0,0,0,0],
        [0,1,2,3],
        [0,2,3,1],
        [0,3,1,2]
    ]
    
    def add_vec(v1, v2):
        return [add[v1[i]][v2[i]] for i in range(3)]
        
    def mul_vec(v, c):
        return [mul[c][v[i]] for i in range(3)]
        
    def is_zero(v):
        return all(x == 0 for x in v)

    points = []
    seen_points = set()
    for x in range(4):
        for y in range(4):
            for z in range(4):
                v = [x,y,z]
                if is_zero(v): continue
                idx = -1
                for i in range(3):
                    if v[i] != 0:
                        idx = i
                        break
                if idx == -1: continue
                inv = [0,1,3,2]
                c = inv[v[idx]]
                canon = mul_vec(v, c)
                key = tuple(canon)
                if key not in seen_points:
                    seen_points.add(key)
                    points.append(canon)
                    
    lines = points
    
    edges = []
    for i, p in enumerate(points):
        for j, l in enumerate(lines):
            dot = 0
            for k in range(3):
                dot = add[dot][mul[p[k]][l[k]]]
            if dot == 0:
                edges.append([i, n_points + j])
                
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
