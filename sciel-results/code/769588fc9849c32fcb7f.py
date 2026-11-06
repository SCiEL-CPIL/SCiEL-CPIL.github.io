import random
import time

def construct():
    n = 20
    num_colors = 20
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Precompute all edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    E = len(all_edges)
    edge_index = {}
    for i, (u, v) in enumerate(all_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
        
    # Precompute all 4-cycles
    # A 4-cycle is a set of 4 vertices {a,b,c,d} with edges (a,b), (b,c), (c,d), (d,a)
    # In a simple graph, for 4 vertices, there are 3 distinct 4-cycles (permutations of the middle 2 vertices).
    # Cycle 1: a-b-c-d-a
    # Cycle 2: a-b-d-c-a
    # Cycle 3: a-c-b-d-a
    # We represent a cycle by the tuple of 4 edge indices.
    
    cycles = []
    edge_to_cycles = [[] for _ in range(E)]
    
    vertices = range(n)
    # Iterate over all combinations of 4 vertices
    import itertools
    for combo in itertools.combinations(vertices, 4):
        a, b, c, d = combo
        # The 3 possible 4-cycles on these vertices
        # 1: a-b, b-c, c-d, d-a
        e1 = edge_index.get((a, b))
        e2 = edge_index.get((b, c))
        e3 = edge_index.get((c, d))
        e4 = edge_index.get((d, a))
        if None not in (e1, e2, e3, e4):
            cyc = (e1, e2, e3, e4)
            cycles.append(cyc)
            for e in cyc:
                edge_to_cycles[e].append(len(cycles)-1)
                
        # 2: a-b, b-d, d-c, c-a
        e1 = edge_index.get((a, b))
        e2 = edge_index.get((b, d))
        e3 = edge_index.get((d, c))
        e4 = edge_index.get((c, a))
        if None not in (e1, e2, e3, e4):
            cyc = (e1, e2, e3, e4)
            cycles.append(cyc)
            for e in cyc:
                edge_to_cycles[e].append(len(cycles)-1)
                
        # 3: a-c, c-b, b-d, d-a
        e1 = edge_index.get((a, c))
        e2 = edge_index.get((c, b))
        e3 = edge_index.get((b, d))
        e4 = edge_index.get((d, a))
        if None not in (e1, e2, e3, e4):
            cyc = (e1, e2, e3, e4)
            cycles.append(cyc)
            for e in cyc:
                edge_to_cycles[e].append(len(cycles)-1)
                
    # Shuffle edges for greedy
    edge_indices = list(range(E))
    
    best_graph = None
    best_edge_count = 0
    
    max_restarts = 5000 # Increase restarts due to speedup
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
            
        random.shuffle(edge_indices)
        
        # edge_colors[i] = color if present, -1 if absent
        edge_colors = [-1] * E
        edge_count = 0
        
        # Greedy addition
        for e_idx in edge_indices:
            # Check if we can add edge e_idx
            forbidden_colors = set()
            impossible = False
            
            # Check all 4-cycles containing e_idx
            for c_idx in edge_to_cycles[e_idx]:
                cycle = cycles[c_idx]
                # Get the other 3 edges
                other_edges = [e for e in cycle if e != e_idx]
                c1, c2, c3 = other_edges
                
                col1 = edge_colors[c1]
                col2 = edge_colors[c2]
                col3 = edge_colors[c3]
                
                # If any of the other edges is missing, this cycle is not complete, so no constraint from it yet
                if col1 == -1 or col2 == -1 or col3 == -1:
                    continue
                    
                # If all present, check if they are distinct
                if col1 == col2 or col2 == col3 or col1 == col3:
                    impossible = True
                    break
                
                # They are distinct, so the new edge cannot have any of these colors
                forbidden_colors.add(col1)
                forbidden_colors.add(col2)
                forbidden_colors.add(col3)
                
            if impossible:
                continue
                
            if len(forbidden_colors) >= num_colors:
                continue
                
            # Pick a random valid color
            allowed = [c for c in range(num_colors) if c not in forbidden_colors]
            chosen_color = random.choice(allowed)
            
            edge_colors[e_idx] = chosen_color
            edge_count += 1
            
        if edge_count > best_edge_count:
            best_edge_count = edge_count
            # Construct result
            best_graph = []
            for i in range(E):
                if edge_colors[i] != -1:
                    u, v = all_edges[i]
                    best_graph.append([u, v, edge_colors[i]])
                    
    if best_graph is None:
        return []
        
    return best_graph

# EVOLVE-BLOCK-START
# (The function above is the implementation)
# EVOLVE-BLOCK-END
