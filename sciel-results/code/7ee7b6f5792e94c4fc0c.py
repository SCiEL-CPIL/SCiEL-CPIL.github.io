"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Construct the incidence graph of the projective plane of order 4.
    # This graph is C4-free and has minimum degree q+1 = 5.
    # Number of vertices n = 2 * (q^2 + q + 1) = 2 * 21 = 42.
    # Minimum degree 5 >= 42 - 39 = 3.
    # This satisfies the condition.
    
    q = 4
    n_points = q*q + q + 1
    n_lines = q*q + q + 1
    n = n_points + n_lines
    
    # Vertices 0..n_points-1 are points, n_points..n-1 are lines.
    # We need to generate the incidence structure.
    # For q=4, we can use the standard construction over GF(4).
    # GF(4) = {0, 1, alpha, alpha^2} with alpha^2 + alpha + 1 = 0.
    # Points are 1D subspaces of GF(4)^3.
    # Lines are 2D subspaces of GF(4)^3.
    
    # Represent elements of GF(4) as integers 0,1,2,3 where 2=alpha, 3=alpha^2.
    # Addition:
    add = [
        [0,1,2,3],
        [1,0,3,2],
        [2,3,0,1],
        [3,2,1,0]
    ]
    # Multiplication:
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
        
    def same_subspace(v1, v2):
        # Check if v1 and v2 are in the same 1D subspace
        if is_zero(v1) or is_zero(v2):
            return False
        # v2 = c * v1
        # c = v2[0] / v1[0] if v1[0] != 0
        # We just check if there exists c such that v2 = c*v1
        for c in range(4):
            if all(mul[c][v1[i]] == v2[i] for i in range(3)):
                return True
        return False

    # Generate all 1D subspaces (points)
    points = []
    seen_points = set()
    for x in range(4):
        for y in range(4):
            for z in range(4):
                v = [x,y,z]
                if is_zero(v): continue
                # Canonical form: make first non-zero element 1
                # Find first non-zero
                idx = -1
                for i in range(3):
                    if v[i] != 0:
                        idx = i
                        break
                if idx == -1: continue
                # Multiply by inverse of v[idx]
                # In GF(4), inverse of 1 is 1, 2 is 3, 3 is 2
                inv = [0,1,3,2]
                c = inv[v[idx]]
                canon = mul_vec(v, c)
                key = tuple(canon)
                if key not in seen_points:
                    seen_points.add(key)
                    points.append(canon)
                    
    # Generate all 2D subspaces (lines)
    # A line is the orthogonal complement of a point (using dot product)
    # Or just generate all 2D subspaces.
    # In PG(2,4), lines correspond to points of the dual plane.
    # So there are 21 lines.
    # Incidence: Point p is on Line L if p is in L.
    # L = { v in GF(4)^3 | dot(p, v) = 0 } for some normal vector p?
    # Yes, each line is defined by a linear equation a x + b y + c z = 0.
    # The normal vector (a,b,c) defines the line.
    # So lines are in 1-1 correspondence with points (normals).
    
    lines = points # Same list, but interpreted as normals
    
    edges = []
    for i, p in enumerate(points):
        for j, l in enumerate(lines):
            # Check if p is on line l (defined by normal l)
            # dot(p, l) = 0
            dot = 0
            for k in range(3):
                dot = add[dot][mul[p[k]][l[k]]]
            if dot == 0:
                edges.append([i, n_points + j])
                
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
