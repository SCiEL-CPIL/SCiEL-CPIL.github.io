import math

N = 12

def construct():
    # Staggered grid of 3 rows x 4 cols
    # Orientation: angle 0 (vertical sides, pointy top/bottom)
    # Unit hexagon: width sqrt(3), height 2
    # Center spacing: dx=sqrt(3), dy=1.5, offset=sqrt(3)/2
    
    sqrt3 = math.sqrt(3)
    
    # Generate centers
    centers = []
    # Row 0
    for k in range(4):
        x = k * sqrt3
        y = 0.0
        centers.append((x, y))
    # Row 1
    for k in range(4):
        x = k * sqrt3 + sqrt3 / 2
        y = 1.5
        centers.append((x, y))
    # Row 2
    for k in range(4):
        x = k * sqrt3
        y = 3.0
        centers.append((x, y))
    
    # Calculate bounding box to center the cluster
    min_x = min(c[0] for c in centers)
    max_x = max(c[0] for c in centers)
    min_y = min(c[1] for c in centers)
    max_y = max(c[1] for c in centers)
    
    # Add hexagon extents (width/2 = sqrt(3)/2, height/2 = 1)
    # Note: For pointy-top (vertical sides), x-extent is sqrt(3)/2, y-extent is 1
    # Wait, let's re-verify extents for angle=0 (vertex at 1,0)
    # Vertices: (1,0), (0.5, sqrt(3)/2)...
    # X range: [-1, 1]. Width 2.
    # Y range: [-sqrt(3)/2, sqrt(3)/2]. Height sqrt(3).
    # This is FLAT-TOP (horizontal top/bottom)?
    # If vertex at (1,0), the top side is horizontal?
    # Vertices: (1,0), (0.5, 0.866), (-0.5, 0.866)...
    # Yes, top side is horizontal.
    # So this is FLAT-TOP.
    # My previous calculation assumed Pointy-Top (Vertical Sides).
    # Let's stick to the geometry that worked:
    # "Pointy-Top" (Vertical Sides) -> Width sqrt(3), Height 2.
    # This corresponds to angle 90?
    # If angle 0 is Flat-Top, then angle 90 is Pointy-Top.
    
    # Let's use angle 90 for inner hexagons to get Vertical Sides.
    # And Outer Hexagon angle 90 to get Vertical Sides.
    
    # Recalculate centers for Vertical Sides (Pointy-Top)
    # Spacing: dx=sqrt(3), dy=1.5
    
    centers = []
    for k in range(4):
        x = k * sqrt3
        y = 0.0
        centers.append((x, y))
    for k in range(4):
        x = k * sqrt3 + sqrt3 / 2
        y = 1.5
        centers.append((x, y))
    for k in range(4):
        x = k * sqrt3
        y = 3.0
        centers.append((x, y))
        
    # Center the cluster
    cx = (min(c[0] for c in centers) + max(c[0] for c in centers)) / 2
    cy = (min(c[1] for c in centers) + max(c[1] for c in centers)) / 2
    
    inner = []
    for x, y in centers:
        # Shift to center
        nx = x - cx
        ny = y - cy
        inner.append([nx, ny, 90.0])
        
    # Outer Hexagon
    # We determined S=4.5 works for this alignment.
    side = 4.5
    
    return inner, (0.0, 0.0), side, 90.0
