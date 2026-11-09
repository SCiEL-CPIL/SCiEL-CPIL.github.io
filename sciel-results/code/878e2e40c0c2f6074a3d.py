<<<<<<< SEARCH
def _build_objective(n):
    def objective(p):
        x = p[:2*n]
        angles = p[2*n:3*n]
        t = p[3*n]
        pen = 0.0
        for i in range(n):
            xi, yi = x[2*i], x[2*i+1]
            ai = angles[i]
            for k in range(6):
                a = math.radians(60.0*k) + ai
                vx = xi + math.cos(a)
                vy = yi + math.sin(a)
                if not _point_in_hex(vx, vy, 0.0, 0.0, t, 0.0):
                    pen += 10.0
        for i in range(n):
            vi = _hex_verts(x[2*i], x[2*i+1], 1.0, angles[i])
            for j in range(i+1, n):
                vj = _hex_verts(x[2*j], x[2*j+1], 1.0, angles[j])
                dc = math.sqrt((x[2*i]-x[2*j])**2 + (x[2*i+1]-x[2*j+1])**2)
                if dc < 4.0:
                    for m in range(6):
                        if _point_in_hex(vi[m,0], vi[m,1], x[2*j], x[2*j+1], 1.0, angles[j]):
                            pen += 10.0
                            break
                        if _point_in_hex(vj[m,0], vj[m,1], x[2*i], x[2*i+1], 1.0, angles[i]):
                            pen += 10.0
                            break
        return 0.5*t*t + pen
    return objective
=======
def _build_objective(n):
    def objective(p):
        x = p[:2*n]
        angles = p[2*n:3*n]
        t = p[3*n]
        pen = 0.0
        # Boundary penalty: smooth quadratic penalty if vertices are outside
        for i in range(n):
            xi, yi = x[2*i], x[2*i+1]
            ai = angles[i]
            for k in range(6):
                a = math.radians(60.0*k) + ai
                vx = xi + math.cos(a)
                vy = yi + math.sin(a)
                # Calculate distance to boundary of outer hexagon
                # Transform to local frame of outer hexagon (center 0,0, side t, rot 0)
                # Outer hexagon inequalities:
                # |x| <= t
                # |y| <= t * sqrt(3)/2
                # |sqrt(3)*x + y| <= 2 * t * sqrt(3)/2 = t * sqrt(3)
                # |sqrt(3)*x - y| <= t * sqrt(3)
                # Wait, standard hexagon with flat top (rot=0) vertices at (t,0), (t/2, t*sqrt(3)/2)...
                # The constraints for a hexagon with circumradius t (side t) centered at origin:
                # Let r = t.
                # The apothem is r * sqrt(3)/2.
                # The boundaries are:
                # 1. x <= r
                # 2. x >= -r
                # 3. y <= r * sqrt(3)/2
                # 4. y >= -r * sqrt(3)/2
                # 5. sqrt(3)*x + y <= r * sqrt(3)
                # 6. -sqrt(3)*x + y <= r * sqrt(3)
                # Let's verify. Vertex at (r, 0).
                # sqrt(3)*r + 0 = r*sqrt(3) <= r*sqrt(3) (OK)
                # -sqrt(3)*r + 0 = -r*sqrt(3) <= r*sqrt(3) (OK)
                # r <= r (OK)
                # 0 <= r*sqrt(3)/2 (OK)
                
                # Distance to each boundary line (signed, positive if outside)
                # Line 1: x - r = 0. dist = x - r
                # Line 2: -x - r = 0. dist = -x - r
                # Line 3: y - r*sqrt(3)/2 = 0. dist = y - r*sqrt(3)/2
                # Line 4: -y - r*sqrt(3)/2 = 0. dist = -y - r*sqrt(3)/2
                # Line 5: sqrt(3)*x + y - r*sqrt(3) = 0. Normal (sqrt(3), 1). Norm 2.
                # dist = (sqrt(3)*x + y - r*sqrt(3)) / 2
                # Line 6: -sqrt(3)*x + y - r*sqrt(3) = 0. Normal (-sqrt(3), 1). Norm 2.
                # dist = (-sqrt(3)*x + y - r*sqrt(3)) / 2
                
                d1 = vx - t
                d2 = -vx - t
                d3 = vy - t * math.sqrt(3)/2
                d4 = -vy - t * math.sqrt(3)/2
                d5 = (math.sqrt(3)*vx + vy - t*math.sqrt(3)) / 2.0
                d6 = (-math.sqrt(3)*vx + vy - t*math.sqrt(3)) / 2.0
                
                max_d = max(d1, d2, d3, d4, d5, d6)
                if max_d > 0:
                    pen += max_d * max_d
        # Overlap penalty
        # Approximate overlap using distance between centers and minimum separation
        # Min separation for two unit hexagons is sqrt(3) if aligned.
        # If rotated, it can be different.
        # Using vertex checks is expensive.
        # Let's use a smooth penalty based on distance between closest points.
        # Or simply check if any vertex of i is inside j or vice versa.
        # To make it smooth, we can use the distance from a point to a polygon.
        # But for speed, let's stick to the vertex check but make it smoother?
        # Actually, the vertex check with `if ... pen += 10` is discontinuous.
        # Let's try to compute the penetration depth.
        # For now, let's just increase the weight and use a smoother approximation for overlap:
        # If dist < 2*sqrt(3), assume potential overlap.
        # Calculate min distance between vertices of i and edges of j? Too slow.
        # Let's use the fact that if we place them in a grid, they don't overlap.
        # We will rely on the optimizer to find a non-overlapping config.
        # To help, we add a penalty if centers are too close.
        min_dist = 1.732 # sqrt(3)
        for i in range(n):
            for j in range(i+1, n):
                dc = math.sqrt((x[2*i]-x[2*j])**2 + (x[2*i+1]-x[2*j+1])**2)
                if dc < min_dist:
                    pen += (min_dist - dc) * (min_dist - dc) * 100.0
        return 0.5*t*t + pen
    return objective
>>>>>>> REPLACE

