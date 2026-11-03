import numpy as np

N = 12

def construct():
    # Generate 7 hexagons in a flower pattern (Ring 0 and 1)
    # Pointy-top orientation
    # Axial coordinates (q, r)
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    
    positions = []
    
    # Center
    positions.append((0, 0))
    
    # Ring 1
    ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    for q, r in ring1:
        positions.append((q, r))
        
    # Ring 2 - pick 5
    # (2, 0), (1, 1), (-1, 2), (-2, 0), (0, -2)
    ring2_subset = [(2, 0), (1, 1), (-1, 2), (-2, 0), (0, -2)]
    for q, r in ring2_subset:
        positions.append((q, r))
        
    # Convert to Cartesian
    inner = []
    for q, r in positions:
        x = np.sqrt(3) * (q + r/2.0)
        y = 1.5 * r
        # Angle 90 degrees for pointy-top? 
        # Standard hexagon in many libs is flat-top (vertex at 0).
        # If we want pointy-top, we rotate by 30 or 90?
        # The task says "angle_degrees".
        # If 0 is flat-top, then 30 is pointy-top?
        # Let's assume 0 is flat-top (vertex at right).
        # Then pointy-top is 30 degrees.
        angle = 30.0
        inner.append([x, y, angle])
        
    # Container
    # Side length 5.0
    # Center (0,0)
    # Angle 30.0 to match pointy-top alignment?
    # If inner are pointy-top, container should be pointy-top to align flats.
    # So container angle 30.0.
    
    return inner, (0.0, 0.0), 5.0, 30.0
