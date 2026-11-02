    import math

    def construct():
        # 11 unit hexagons in a hexagonal lattice (flat-top)
        # Centers calculated using axial coordinates (q, r)
        # x = sqrt(3) * (q + r/2)
        # y = 1.5 * r
        
        sqrt3 = math.sqrt(3.0)
        
        # Axial coordinates for the 11 hexagons
        # Center
        # 6 neighbors
        # 4 extra to make a compact shape (2 on left/right, 2 on top/bottom? No, we used a specific set)
        
        # Let's re-verify the set of 11 points used in the calculation:
        # (0,0)
        # (1,0), (-1,0)
        # (0,1), (0,-1)
        # (1,1), (-1,-1)
        # (2,0), (-2,0)
        # (0,2), (0,-2)
        
        coords = [
            (0, 0),
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (-1, -1),
            (2, 0),
            (-2, 0),
            (0, 2),
            (0, -2)
        ]
        
        inner = []
        for q, r in coords:
            x = sqrt3 * (q + r / 2.0)
            y = 1.5 * r
            # Angle 0 for flat-top
            inner.append([x, y, 0.0])
            
        # Outer hexagon side length
        # Derived as 1 + 2 * sqrt(3)
        side = 1.0 + 2.0 * sqrt3
        center = [0.0, 0.0]
        outer_angle = 0.0
        
        return inner, center, side, outer_angle
    