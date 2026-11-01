import numpy as np
from itertools import combinations

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Lattice generation
    # Triangular lattice with spacing sqrt(3)
    # We generate points in a hexagon of "radius" K (in terms of lattice steps)
    # Side length of this lattice hexagon in steps is K.
    # Number of points is 3*K*(K-1) + 1.
    # For N=11, K=3 gives 19 points. This is sufficient.
    
    def get_lattice_points(K):
        points = []
        # Axial coordinates (q, r)
        # Convert to Cartesian for distance calculation
        # x = sqrt(3) * (q + r/2)
        # y = 3/2 * r
        for q in range(-K, K + 1):
            for r in range(max(-K, -q - K), min(K, -q + K) + 1):
                x = np.sqrt(3) * (q + r / 2.0)
                y = 1.5 * r
                points.append((x, y, q, r))
        return points

    def hex_radius(x, y):
        # Calculate the minimal side length S' of a regular hexagon (aligned with lattice)
        # that contains the point (x,y).
        # The hexagon has vertices at distance S' from origin.
        # Equivalently, it's the smallest k such that (x,y) is in k * H_1.
        # H_1 has apothem sqrt(3)/2.
        # The "radius" in the direction of the vertices is S'.
        # The "radius" in the direction of the side midpoints is sqrt(3)/2 * S'.
        
        # We can compute the norm using the support function.
        # The hexagon is the intersection of 6 half-planes.
        # Norms are max of projections onto the 6 normal vectors.
        # For a hexagon with side S, the distance from center to side is A = sqrt(3)/2 * S.
        # The normals are at angles 0, 60, ...
        # Wait, if the hexagon has flat sides at angles 0, 60...
        # Let's assume the lattice is aligned with the hexagon.
        # The distance from origin to the side in the q-direction is A.
        
        # Projections:
        # d0 = |x| (normal along x-axis? No, x-axis is through a vertex if pointy-top? 
        # My lattice: q-axis is along x. r-axis is 120 deg?
        # x = sqrt(3)(q + r/2). 
        # If r=0, x = sqrt(3)q. Points are (sqrt(3), 0), (2sqrt(3), 0)...
        # These are along the x-axis.
        # In a standard hexagon aligned with this, the x-axis goes through a vertex?
        # If the lattice points are at (sqrt(3), 0), they are vertices of the "first ring".
        # So the hexagon $H_{S'}$ must have vertices at distance at least sqrt(3) to contain (sqrt(3),0).
        # So S' >= sqrt(3) for that point.
        
        # The hex norm $\|v\|_H$ for a point $v$ is the minimum $S'$ such that $v \in H_{S'}$.
        # Since $H_{S'}$ is a scaled version of $H_1$, $\|v\|_H = S'$.
        
        # How to calculate S' for a point (x,y)?
        # The hexagon $H_{S'}$ is defined by $|x| \le S'$? No.
        # Vertices of $H_1$ (S'=1) are at distance 1.
        # My lattice point (sqrt(3), 0) has distance sqrt(3).
        # So for this point, S' = sqrt(3).
        
        # Let's verify the shape of H_1.
        # If vertices are at distance 1, the apothem is sqrt(3)/2.
        # The point (sqrt(3)/2, 0) is on the edge?
        # No, (sqrt(3)/2, 0) is the midpoint of the edge if the edge is vertical?
        
        # Let's just use the max projection method.
        # The hexagon $H_{S'}$ has support function $h(u) = S' \cdot h_{H_1}(u)$.
        # $h_{H_1}(u)$ for a hexagon with vertices at distance 1.
        # Max projection of $H_1$ on axis $u$ is 1 (if u aligns with vertex) or sqrt(3)/2 (if u aligns with side normal).
        
        # Actually, simpler:
        # The condition for a point p to be in $H_{S'}$ is that $p$ can be written as $S' \cdot q$ where $q \in H_1$.
        # Or simply, the "hex distance" from origin.
        # For the triangular lattice aligned with the hexagon, the hex distance of a point is 
        # max( |x| / (sqrt(3)/2) * something? )
        
        # Let's use the 3 main axes of the hexagon.
        # The hexagon sides are perpendicular to vectors at 0, 60, 120 degrees?
        # If vertices are at 0, 60, ...
        # Then sides are at 30, 90, ...
        # The distance from center to side is $A = S' \frac{\sqrt{3}}{2}$.
        # The projection of point (x,y) on the normal to the side (angle 30) must be $\le A$.
        
        # Normals:
        # n0 = (1, 0) ? No, if vertices are at 0, 60, the side normal is at 30.
        # Let's assume the hexagon has flat tops?
        # My lattice: (sqrt(3), 0) is a point.
        # If I put a hexagon with side S' around it, and it's aligned...
        # The distance from center to the vertex of the hexagon is S'.
        # If (sqrt(3), 0) is a vertex of the cluster, it lies on the axis of symmetry.
        # If the hexagon also has a vertex on the x-axis, then the constraint is $S' \ge \sqrt{3}$.
        
        # Let's calculate S' as the maximum of the projections onto the 3 vertex axes.
        # Vertex axes: 0, 60, 120.
        # p0 = x
        # p1 = x*cos(60) + y*sin(60) = 0.5*x + y*sqrt(3)/2
        # p2 = x*cos(120) + y*sin(120) = -0.5*x + y*sqrt(3)/2
        
        # Wait, is the max projection on the vertex axes equal to the hex radius?
        # For a hexagon with vertices at distance R, the max projection on the vertex axes is R.
        # So S' = max( |p0|, |p1|, |p2| )?
        # Let's check a point on the edge. Midpoint of edge between v0 and v1.
        # v0 = (1, 0). v1 = (0.5, sqrt(3)/2).
        # Midpoint = (0.75, sqrt(3)/4).
        # p0 = 0.75.
        # p1 = 0.5(0.75) + sqrt(3)/2 (sqrt(3)/4) = 0.375 + 3/8 = 0.375 + 0.375 = 0.75.
        # p2 = ...
        # Max is 0.75.
        # But the radius S' should be 1?
        # No, the point (0.75, sqrt(3)/4) is INSIDE the hexagon of side 1.
        # It is on the boundary.
        # So the "radius" in terms of side length S' for this point is 1.
        # But max projection was 0.75.
        
        # The correct formula for the hexagonal norm (where H_1 has side 1) is:
        # $\|p\|_H = \max( \frac{|x|}{\sqrt{3}/2} \dots )$? No.
        
        # Let's use the property that $H_1$ is the convex hull of $\pm v_i$.
        # The norm is the smallest $k$ such that $p \in k H_1$.
        # This is equivalent to $\max_{i} \frac{|p \cdot n_i|}{h_i}$ where $n_i$ are normals and $h_i$ is the distance to the face?
        # For a regular hexagon of side 1:
        # Distance to face (apothem) $a = \frac{\sqrt{3}}{2}$.
        # The 6 face normals are at angles $0, 60, \dots$?
        # If the hexagon has a face perpendicular to the x-axis?
        # Then the vertex is at 60?
        
        # Let's stick to the lattice alignment.
        # Lattice points are at $u = \sqrt{3}(q + r/2, \dots)$.
        # The point $(\sqrt{3}, 0)$ corresponds to $q=1, r=0$.
        # This point is a vertex of the "first ring" cluster.
        # In the optimal packing of 7 hexagons, this point is the center of an outer hexagon.
        # The enclosing hexagon must have a side length such that it contains this center + the unit hexagon.
        # We established $S = 1 + S'$.
        # And $S'$ is the side length of the hexagon containing the centers.
        # For the 7-cluster, the centers form a hexagon of side $\sqrt{3}$ (distance between adjacent centers).
        # So the "center hexagon" has side $\sqrt{3}$.
        # So $S' = \sqrt{3}$.
        # Then $S = 1 + \sqrt{3}$.
        
        # So for any point $p$, we need to find the side length $S'$ of the smallest regular hexagon (aligned) containing $p$.
        # This is the "hex radius" of $p$.
        
        # How to calculate $S'$ for a point $(x,y)$?
        # The hexagon $H_{S'}$ has vertices at distance $S'$ from origin.
        # The point $(x,y)$ is inside if its "hex distance" is $\le S'$.
        # The hex distance is $\max( \frac{|x|}{\cos(30^\circ)} \dots )$?
        
        # Let's use the 3 axes of the hexagon vertices.
        # Let the vertices be at angles $0, 60, 120$.
        # Then the sides are at $30, 90, 150$.
        # The distance from origin to the side at angle $30$ is $S' \frac{\sqrt{3}}{2}$.
        # The projection of $p$ on the normal (angle 30) is $p \cdot n_{30}$.
        # Condition: $|p \cdot n_{30}| \le S' \frac{\sqrt{3}}{2}$.
        # So $S' \ge \frac{2}{\sqrt{3}} |p \cdot n_{30}|$.
        
        # Similarly for normals at 90 and 150.
        # $S' = \max( \frac{2}{\sqrt{3}} |p \cdot n_{30}|, \frac{2}{\sqrt{3}} |p \cdot n_{90}|, \frac{2}{\sqrt{3}} |p \cdot n_{150}| )$.
        
        # Let's test this on $p=(\sqrt{3}, 0)$.
        # $n_{30} = (\cos 30, \sin 30) = (\frac{\sqrt{3}}{2}, \frac{1}{2})$.
        # $p \cdot n_{30} = \sqrt{3} \frac{\sqrt{3}}{2} + 0 = \frac{3}{2}$.
        # $S' \ge \frac{2}{\sqrt{3}} \cdot \frac{3}{2} = \sqrt{3}$.
        # $n_{90} = (0, 1)$. $p \cdot n_{90} = 0$.
        # $n_{150} = (-\frac{\sqrt{3}}{2}, \frac{1}{2})$. $p \cdot n_{150} = -\frac{3}{2}$. Abs is $1.5$.
        # So $S' = \sqrt{3}$. Correct.
        
        n30 = np.array([np.sqrt(3)/2, 0.5])
        n90 = np.array([0, 1])
        n150 = np.array([-np.sqrt(3)/2, 0.5])
        
        p = np.array([x, y])
        proj30 = np.abs(np.dot(p, n30))
        proj90 = np.abs(np.dot(p, n90))
        proj150 = np.abs(np.dot(p, n150))
        
        S_prime = (2.0 / np.sqrt(3)) * np.max([proj30, proj90, proj150])
        return S_prime

    # Generate all 11-point subsets of the 19-point lattice (K=3)
    # K=3 lattice has points with max hex distance 3*sqrt(3)? No, 3 steps.
    # Max distance for K=3 is 3*sqrt(3).
    
    pts = get_lattice_points(3)
    # pts is list of (x, y, q, r)
    
    best_S = float('inf')
    best_cluster = None
    
    # There are C(19, 11) combinations.
    # We can optimize by only considering clusters that contain the origin?
    # Not necessarily, but likely.
    # However, 75k is small.
    
    indices = range(len(pts))
    
    for combo in combinations(indices, 11):
        # Calculate max S_prime
        max_sp = 0
        for idx in combo:
            x, y, _, _ = pts[idx]
            sp = hex_radius(x, y)
            if sp > max_sp:
                max_sp = sp
                if max_sp >= best_S: # Pruning
                    break
        if max_sp < best_S:
            best_S = max_sp
            best_cluster = combo
            # Early stop if we found the theoretical minimum?
            # Theoretical min for 11 points?
            # 7 points need sqrt(3).
            # Adding 4 points might increase it to 2*sqrt(3)?
            # Let's just run.
            
    # best_S is S_prime
    S_outer = 1.0 + best_S
    
    # Get the coordinates of the best cluster
    inner = []
    for idx in best_cluster:
        x, y, _, _ = pts[idx]
        inner.append([x, y, 0.0])
        
    # Center is 0,0
    # Side is S_outer
    # Angle is 0.0
    
    return inner, (0.0, 0.0), S_outer, 0.0
# EVOLVE-BLOCK-END
