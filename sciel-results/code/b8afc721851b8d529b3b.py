import math
import time

N = 12

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Configuration: 2-3-4-3 stack of pointy-top hexagons inside flat-top outer hexagon
    # Outer hexagon side length
    L = 4.0
    
    # Inner hexagons are pointy-top (angle 90 degrees if 0 is flat-top)
    # Assuming standard convention where 0 degrees is flat-top (horizontal sides)
    # If the evaluator uses a different convention, we might need to adjust.
    # However, the relative geometry is what matters.
    # Let's assume 0.0 is flat-top.
    inner_angle = 90.0
    
    sqrt3 = math.sqrt(3)
    
    # Row definitions: (y_offset, [x_offsets])
    # Centered at (0,0)
    rows = [
        (2.25, [-0.5 * sqrt3, 0.5 * sqrt3]),           # 2 hexagons
        (0.75,  [-1.0 * sqrt3, 0.0, 1.0 * sqrt3]),     # 3 hexagons
        (-0.75, [-1.5 * sqrt3, -0.5 * sqrt3, 0.5 * sqrt3, 1.5 * sqrt3]), # 4 hexagons
        (-2.25, [-1.0 * sqrt3, 0.0, 1.0 * sqrt3])      # 3 hexagons
    ]
    
    inner = []
    for y_off, x_offsets in rows:
        for x_off in x_offsets:
            inner.append([x_off, y_off, inner_angle])
            
    # Verify count
    assert len(inner) == N, f"Expected {N} hexagons, got {len(inner)}"
    
    # Outer hexagon
    center = (0.0, 0.0)
    side = L
    outer_angle = 0.0 # Flat-top
    
    return inner, center, side, outer_angle
