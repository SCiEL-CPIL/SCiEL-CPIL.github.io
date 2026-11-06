"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 140  # safety margin
    
    d = 7
    n = 1 << d
    total_edges = n * d // 2  # 448
    
    # Precompute all edges and their face memberships
    # Edge (x, y) where y = x ^ (1<<i), x < y
    # A face is defined by two vertices at Hamming distance 2
    # For edge (x, x^(1<<i)), the faces containing it are:
    #   for each j != i: face with vertices x, x^(1<<i), x^(1<<j), x^(1<<(i|j))
    
    # Precompute: for each edge, list of faces (each face = frozenset of 4 edges)
    # Actually, let's precompute for each edge, the set of other edges that form a face with it
    # Edge e1=(x,y) with bit i, edge e2=(x',y') with bit j
    # They form a face if {x,y} and {x',y'} are opposite sides of a square
    
    # For edge (x, x^bit_i), partner edges in faces:
    # For each j != i: the edge (x^bit_j, x^bit_i^bit_j) forms a face with (x, x^bit_i)
    # Also edge (x^bit_i, x^bit_i^bit_j) is the same as above just reversed
    
    # Let me precompute face partners for each edge
    # Edge index: map (x,y) with x<y to an index
    edge_list = []
    edge_to_idx = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edge_list)
                edge_list.append((x, y, i))
                edge_to_idx[(x, y)] = idx
    
    num_edges = len(edge_list)  # 448
    
    # For each edge, precompute which other edges form a 4-cycle with it
    # Edge (x, y) with bit i: for each j != i, the edge (x^(1<<j), y^(1<<j)) forms a face
    # But we need to check that this edge exists (it does, since x^(1<<j) and y^(1<<j) differ in bit i)
    face_partners = [[] for _ in range(num_edges)]
    for idx, (x, y, i) in enumerate(edge_list):
        for j in range(d):
            if j == i:
                continue
            c = x ^ (1 << j)
            e = y ^ (1 << j)
            # Edge (min(c,e), max(c,e))
            if c < e:
                key = (c, e)
            else:
                key = (e, c)
            if key in edge_to_idx:
                face_partners[idx].append(edge_to_idx[key])
    
    # Each edge should have exactly 2*(d-1) = 12 face partners? No.
    # For edge (x, y) with bit i, for each j != i, there's one face.
    # In that face, the opposite edge is (x^(1<<j), y^(1<<j)).
    # So each edge has (d-1) = 6 face partners (one per each other bit).
    # But wait, a face has 4 edges. The edge (x,y) is in (d-1) faces.
    # In each face, it's paired with 3 other edges, but the "opposite" edge is unique.
    # Actually for the C4 constraint: we need all 4 edges of the face to NOT be selected.
    # So if we select edge e, we can't select all 3 other edges of any face containing e.
    
    # Better approach: for each edge, store the faces it belongs to (as sets of edge indices)
    # Then when checking if we can add edge e, we check for each face containing e,
    # whether the other 3 edges are already selected.
    
    # Let's precompute faces: each face is a set of 4 edges
    # A face is determined by a vertex x and two distinct bits i, j
    # Vertices: x, x^(1<<i), x^(1<<j), x^(1<<(i|j))
    # Edges: (x, x^(1<<i)), (x, x^(1<<j)), (x^(1<<i), x^(1<<(i|j))), (x^(1<<j), x^(1<<(i|j)))
    
    # But this double-counts. Let's just use the face_partners approach with a different check.
    
    # For greedy: when considering edge e, check if adding it would complete a C4.
    # A C4 is completed if there exists a face containing e where the other 3 edges are all selected.
    # So for edge e with face partners, we need to check: for each face containing e,
    # are the other 3 edges all in selected?
    
    # Let me precompute for each edge, the list of faces (as tuples of 4 edge indices)
    # Actually, let's just precompute: for each edge, the list of "forbidden triples" - 
    # if all 3 edges in a triple are selected, we can't add this edge.
    
    # Simpler: for each edge e, and for each face f containing e, the other 3 edges form a triple.
    # Precompute these triples.
    
    forbidden_triples = [[] for _ in range(num_edges)]
    for idx, (x, y, i) in enumerate(edge_list):
        for j in range(d):
            if j == i:
                continue
            # Face with bits i and j starting from x
            # Vertices: x, x^i, x^j, x^i^j
            v1 = x
            v2 = x ^ (1 << i)
            v3 = x ^ (1 << j)
            v4 = x ^ (1 << i) ^ (1 << j)
            
            # 4 edges of this face
            edges_in_face = []
            for (a, b) in [(v1, v2), (v1, v3), (v2, v4), (v3, v4)]:
                if a < b:
                    key = (a, b)
                else:
                    key = (b, a)
                edges_in_face.append(edge_to_idx[key])
            
            # Find which one is our edge
            # The other 3 form the triple
            others = [e for e in edges_in_face if e != idx]
            forbidden_triples[idx].append(tuple(others))
    
    # Now greedy with random ordering
    def greedy(order):
        selected = [False] * num_edges
        count = 0
        for idx in order:
            # Check if adding idx would complete a C4
            ok = True
            for triple in forbidden_triples[idx]:
                if all(selected[e] for e in triple):
                    ok = False
                    break
            if ok:
                selected[idx] = True
                count += 1
        return selected, count
    
    # Local search: try to add edges that were rejected, or swap
    def local_search(selected, count):
        # Try adding any unselected edge
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for idx in range(num_edges):
                if not selected[idx]:
                    ok = True
                    for triple in forbidden_triples[idx]:
                        if all(selected[e] for e in triple):
                            ok = False
                            break
                    if ok:
                        selected[idx] = True
                        count += 1
                        improved = True
                        # Restart scan
                        break
        return selected, count
    
    best_count = 0
    best_selected = None
    
    # Multiple random greedy restarts
    rng = random.Random(42)
    base_order = list(range(num_edges))
    
    for restart in range(200):
        if time.time() > deadline - 10:
            break
        
        # Random shuffle
        order = base_order[:]
        rng.shuffle(order)
        
        selected, count = greedy(order)
        
        if count > best_count:
            best_count = count
            best_selected = selected[:]
        
        # Local search
        selected, count = local_search(selected, count)
        
        if count > best_count:
            best_count = count
            best_selected = selected[:]
    
    # Convert to output format
    result = []
    for idx in range(num_edges):
        if best_selected[idx]:
            x, y, _ = edge_list[idx]
            result.append([x, y])
    
    return result

# EVOLVE-BLOCK-END
