import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Generate PG(2,5)
    # Points in F_5^3
    points = []
    for x in range(5):
        for y in range(5):
            points.append((x, y, 1))
    for x in range(5):
        points.append((x, 1, 0))
    points.append((1, 0, 0))
    
    n_core = 31
    # Map points to indices
    point_to_idx = {p: i for i, p in enumerate(points)}
    
    # Polarity: x*y' + y*x' + z*z' = 0
    # u is adjacent to v if v is on polar of u
    # Polar of u=(x,y,z) is line x*X + y*Y + z*Z = 0?
    # No, bilinear form B(u,v) = x*v_y + y*v_x + z*v_z.
    # v is on polar of u if B(u,v) = 0.
    
    edges = []
    for i, u in enumerate(points):
        for j, v in enumerate(points):
            if i >= j:
                continue
            # Check if v is on polar of u
            # B(u,v) = u[0]*v[1] + u[1]*v[0] + u[2]*v[2]
            b = (u[0]*v[1] + u[1]*v[0] + u[2]*v[2]) % 5
            if b == 0:
                # u is adjacent to v
                # But if u is isotropic, B(u,u)=0, so u is on its own polar.
                # We skip i==j.
                # Is the graph simple? Yes.
                edges.append([i, j])
                
    # Add 5 leaves
    # Connect to first 5 vertices
    for k in range(5):
        new_v = n_core + k
        edges.append([new_v, k])
        
    n = n_core + 5
    return {'n': n, 'edges': edges}
