import itertools

def construct():
    # Generate PG(2,3) incidence graph
    # Points are non-zero vectors in GF(3)^3 up to scalar
    # GF(3) = {0, 1, 2}
    
    points = []
    for x in range(3):
        for y in range(3):
            for z in range(3):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize: make first non-zero component 1
                vec = [x, y, z]
                for i in range(3):
                    if vec[i] != 0:
                        if vec[i] != 1:
                            vec = [(v * 2) % 3 for v in vec] # multiply by 2 (inverse of 2 is 2 in GF(3))
                        break
                points.append(tuple(vec))
    
    # Remove duplicates (shouldn't be any if normalized correctly, but set is safe)
    points = list(set(points))
    # There should be 13 points
    # Map point vector to index
    point_to_idx = {p: i for i, p in enumerate(points)}
    
    # Lines are also points in the dual, or we can generate them
    # A line in PG(2,3) is defined by a normal vector (a,b,c) such that ax+by+cz=0
    # The set of lines is isomorphic to the set of points.
    # Let's just use the same set of vectors for lines, but distinguish them by index offset.
    lines = points # Same structure
    line_to_idx = {l: i for i, l in enumerate(lines)}
    
    # Incidence: point p is on line l if dot product is 0 in GF(3)
    def dot(p, l):
        return (p[0]*l[0] + p[1]*l[1] + p[2]*l[2]) % 3

    incidence_edges = []
    for pi, p in enumerate(points):
        for li, l in enumerate(lines):
            if dot(p, l) == 0:
                # Vertex pi is a point, vertex 13+li is a line
                incidence_edges.append((pi, 13 + li))
                
    # Total vertices in incidence graph: 26
    # We need to choose 20 vertices to maximize edges
    all_vertices = list(range(26))
    
    best_edges_count = -1
    best_subset = None
    
    # Iterate over all subsets of size 20
    # C(26, 20) = C(26, 6) = 230,230. This is fast.
    for subset in itertools.combinations(all_vertices, 20):
        subset_set = set(subset)
        count = 0
        for u, v in incidence_edges:
            if u in subset_set and v in subset_set:
                count += 1
        if count > best_edges_count:
            best_edges_count = count
            best_subset = subset
            
    # Map best_subset vertices to 0..19
    # best_subset is a tuple of 26 original vertex IDs
    # We need to create a mapping from original ID to new ID 0..19
    # The order in best_subset is arbitrary, but we need a consistent mapping.
    # Let's just sort them or use the order in the combination.
    # Actually, the task requires vertices 0..19.
    # We can map the i-th element of best_subset to i.
    
    orig_to_new = {orig: new for new, orig in enumerate(best_subset)}
    
    final_edges = []
    for u, v in incidence_edges:
        if u in orig_to_new and v in orig_to_new:
            final_edges.append([orig_to_new[u], orig_to_new[v]])
            
    return final_edges
