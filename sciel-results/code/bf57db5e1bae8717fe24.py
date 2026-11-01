import time
import random
import itertools

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer for safety
    
    # The task specifies k=5, g=9.
    # We will use a voltage graph lift of K6 (5-regular, 6 vertices).
    # The lift will have 6*m vertices.
    # We need to find m and voltage assignment such that girth >= g.
    
    base_vertices = 6
    base_edges = [(i, j) for i in range(base_vertices) for j in range(i+1, base_vertices)]
    
    # Precompute all cycles of length 3 to 8 in K6
    # A cycle is a sequence of vertices v0, v1, ..., v_{L-1}, v0
    # We represent it by the list of edges (min, max) with signs?
    # Actually, for voltage sum, we just need the set of edges with orientation.
    # Sum = sum( voltage(edge) * sign )
    # Since graph is undirected, voltage(e) = -voltage(rev_e).
    # We can just store the edges and their orientation relative to a canonical direction.
    
    cycles = []
    # Generate all simple cycles of length 3 to 8
    # Using DFS from 0 to avoid duplicates (fixing start vertex)
    # But we need all cycles.
    
    def find_cycles(G, min_len, max_len):
        # G is adjacency list
        # Returns list of cycles, each is a list of (u, v) tuples representing directed edges in the cycle
        # We normalize cycles by rotating them so the smallest vertex is first, and the second smallest is next?
        # Or just store the set of edges? No, orientation matters for sum.
        # But for K6, we can just generate all permutations and filter? Too slow.
        
        # Let's use a simpler approach:
        # Iterate all subsets of vertices? No.
        # Iterate all paths?
        
        # For K6, we can hardcode or generate efficiently.
        pass
        
    # Simpler: Generate all cycles by picking a start vertex 0.
    # Any cycle must contain 0? No.
    # We need ALL cycles.
    # Total cycles in K6:
    # L=3: 20
    # L=4: 45
    # L=5: 120
    # L=6: 60
    # Total 245.
    
    # Let's generate them explicitly.
    # A cycle is defined by a set of edges and a traversal order.
    # We can represent a cycle by the list of vertices (v0, v1, ..., v_{L-1}) with v0 < min(v1..v_{L-1}) and v0 < v_{L-1} to break rotational symmetry?
    # And we consider (v0, ..., v_{L-1}) and (v0, ..., v_{L-1}) reversed as the same cycle for checking purposes?
    # The voltage sum for the reverse cycle is the negative of the forward sum.
    # If sum != 0, then -sum != 0. So we only need to check one direction.
    
    # Let's generate all simple cycles of length L.
    # Fix v0 = 0.
    # Then the cycle must visit 0.
    # But cycles not containing 0?
    # In K6, a cycle of length L uses L vertices.
    # If L=3, it uses 3 vertices. It might not contain 0.
    # So we must generate all cycles.
    
    # We can generate all cycles by iterating all subsets of vertices of size L,
    # and then all Hamiltonian cycles on that subset?
    # For L=3: subsets of 3 vertices. 1 cycle per subset (triangle).
    # For L=4: subsets of 4 vertices. 3 cycles per subset.
    # For L=5: subsets of 5 vertices. 12 cycles per subset.
    # For L=6: subsets of 6 vertices. 60 cycles per subset (1 subset).
    
    # Let's do this.
    all_cycles = []
    
    # L=3
    for subset in itertools.combinations(range(6), 3):
        v0, v1, v2 = subset
        # Cycle: v0-v1-v2-v0
        # Edges: (v0,v1), (v1,v2), (v2,v0)
        # We need to define orientation.
        # Let's store the list of edges as (u, v) where u < v is the canonical edge,
        # and a sign +1 if traversal is u->v, -1 if v->u.
        # Actually, let's just store the sequence of vertices.
        # We will compute the sum on the fly.
        # To avoid duplicates, we can just store the set of edges?
        # No, the sum depends on orientation.
        # But if we fix the starting vertex to be the smallest, and the direction to be such that the second vertex is smaller than the last?
        
        # Let's just store the list of vertices in cyclic order, normalized.
        # Normalize: rotate so smallest vertex is first.
        # Then, if the second vertex is greater than the last, reverse the cycle?
        
        def normalize_cycle(cycle):
            # cycle is a list of vertices
            L = len(cycle)
            min_idx = cycle.index(min(cycle))
            rotated = cycle[min_idx:] + cycle[:min_idx]
            # Now rotated[0] is min.
            # Check direction: rotated[1] < rotated[-1] ?
            if rotated[1] > rotated[-1]:
                # Reverse (keep rotated[0] fixed)
                rotated = [rotated[0]] + rotated[1:][::-1]
            return tuple(rotated)

        # For L=3, the cycle is v0-v1-v2-v0.
        # Vertices: v0, v1, v2.
        # Possible orientations: (v0,v1,v2) and (v0,v2,v1).
        # They are reverses.
        # We just need one.
        all_cycles.append(normalize_cycle([v0, v1, v2]))

    # L=4
    for subset in itertools.combinations(range(6), 4):
        # Hamiltonian cycles on 4 vertices: (a,b,c,d)
        # 3 distinct cycles: (a,b,c,d), (a,b,d,c), (a,c,b,d)
        a,b,c,d = subset
        all_cycles.append(normalize_cycle([a,b,c,d]))
        all_cycles.append(normalize_cycle([a,b,d,c]))
        all_cycles.append(normalize_cycle([a,c,b,d]))

    # L=5
    for subset in itertools.combinations(range(6), 5):
        # Hamiltonian cycles on 5 vertices: (5-1)! / 2 = 12
        # Generate all permutations of the 4 non-start vertices
        a,b,c,d,e = subset
        # Fix a as start.
        for perm in itertools.permutations([b,c,d,e]):
            all_cycles.append(normalize_cycle([a] + list(perm)))

    # L=6
    # Only one subset: (0,1,2,3,4,5)
    a,b,c,d,e,f = range(6)
    # Fix a. Permutations of 5: 120 / 2 = 60
    for perm in itertools.permutations([b,c,d,e,f]):
        all_cycles.append(normalize_cycle([a] + list(perm)))

    # Remove duplicates
    unique_cycles = list(set(all_cycles))
    
    # For each cycle, precompute the edge list with signs.
    # Edge (u, v) in cycle (v0, ..., v_{L-1}, v0)
    # Traversal v_i -> v_{i+1}.
    # Canonical edge is (min(u,v), max(u,v)).
    # Sign is +1 if u < v, -1 if u > v.
    cycle_data = []
    for cyc in unique_cycles:
        L = len(cyc)
        edges_with_signs = []
        for i in range(L):
            u = cyc[i]
            v = cyc[(i+1)%L]
            if u < v:
                edge_id = (u, v)
                sign = 1
            else:
                edge_id = (v, u)
                sign = -1
            edges_with_signs.append((edge_id, sign))
        cycle_data.append(edges_with_signs)

    # Map edge to index for voltage array
    edge_index = {e: i for i, e in enumerate(base_edges)}
    
    # Now search for m and voltages
    # m is the group size.
    # We want 6*m vertices.
    # We want to minimize n = 6*m.
    # So we try small m first.
    
    # However, for small m, it's very likely that some cycle has sum 0.
    # We can try random searches for each m.
    
    # To ensure we finish in time, we limit the number of random tries per m.
    
    best_n = float('inf')
    best_edges = None
    best_m = None
    best_voltages = None
    
    # Moore bound is 428. 6*m >= 428 => m >= 72.
    # Let's try m from 50 to 500.
    
    # Precompute cycle edge indices and signs for faster sum
    cycle_edge_indices = []
    cycle_signs = []
    for edges_with_signs in cycle_data:
        inds = []
        signs = []
        for e, s in edges_with_signs:
            inds.append(edge_index[e])
            signs.append(s)
        cycle_edge_indices.append(inds)
        cycle_signs.append(signs)
        
    # Convert to numpy arrays for speed?
    # import numpy as np
    # c_inds = np.array(cycle_edge_indices)
    # c_signs = np.array(cycle_signs)
    
    # We will use Python lists for simplicity, as 245 cycles is small.
    
    for m in range(50, 500):
        if time.time() > deadline - 5:
            break
            
        # Try to find a valid assignment for this m
        # Random restarts
        max_tries = 100
        found = False
        
        for _ in range(max_tries):
            if time.time() > deadline - 5:
                break
                
            # Random voltages in Z_m
            # To ensure we don't pick all zeros (which would sum to 0),
            # we can just pick random 1..m-1? No, 0 is allowed for individual edges.
            voltages = [random.randint(0, m-1) for _ in range(len(base_edges))]
            
            # Check all cycles
            valid = True
            for i in range(len(cycle_edge_indices)):
                inds = cycle_edge_indices[i]
                signs = cycle_signs[i]
                s = 0
                for j in range(len(inds)):
                    s += signs[j] * voltages[inds[j]]
                if s % m == 0:
                    valid = False
                    break
            
            if valid:
                found = True
                best_m = m
                best_voltages = voltages
                break
        
        if found:
            # Construct the graph
            n = base_vertices * m
            edges = []
            
            # Vertex (v, h) where v in 0..5, h in 0..m-1
            # Edge between (u, h) and (v, h + voltage(u,v))
            # Note: voltage is directed.
            # We defined voltages for (u,v) with u<v.
            # So voltage(u,v) = voltages[edge_index[(u,v)]]
            # voltage(v,u) = -voltages[edge_index[(u,v)]]
            
            for u in range(base_vertices):
                for v in range(u+1, base_vertices):
                    eid = edge_index[(u,v)]
                    vol = best_voltages[eid]
                    for h in range(m):
                        # Edge (u, h) -- (v, h+vol)
                        u_idx = u * m + h
                        v_idx = v * m + (h + vol) % m
                        edges.append((u_idx, v_idx))
                        
                        # Also the reverse direction is covered by the other (v, h') iteration?
                        # No, we only iterate u < v.
                        # The edge (v, h') -- (u, h' - vol) is the same edge.
                        # When we process (v, h'), we don't have an edge definition for (v, u) in our loop.
                        # But the graph is undirected.
                        # The edge (u, h) - (v, h+vol) is the same as (v, h+vol) - (u, h).
                        # So we just add it once.
                        
            # Check if this n is better
            if n < best_n:
                best_n = n
                best_edges = edges
                
            # Since we iterate m upwards, the first found is the smallest m?
            # Not necessarily, because we might fail for m=50 and succeed for m=51.
            # But we break out of the m loop if we find one?
            # No, we want the MINIMUM n.
            # So we should continue searching for smaller m?
            # But we are iterating m from small to large.
            # So the first valid m we find is the smallest m that works (within our search depth).
            # We should return immediately?
            # Yes, because n = 6*m. Smaller m -> smaller n.
            
            return n, edges
            
    # If no m found in range, fall back to LUW or return best found
    if best_edges is not None:
        return best_n, best_edges
        
    # Fallback to LUW if voltage search fails (should not happen for m up to 500)
    # But to be safe, let's implement a simple LUW fallback or just return a large graph?
    # The task requires valid graph.
    # Let's just use the LUW code from the prompt as fallback.
    
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

BET: {mechanism: "Voltage graph lift of K6 with optimized voltage assignment search allows for smaller n than LUW by exploiting the dense base graph and algebraic girth constraints", q: 0.7, delta: [0.1, 0.5]}
