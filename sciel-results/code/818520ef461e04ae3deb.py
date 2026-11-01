import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Define lattice vectors for hexagonal tiling
    # Distance between centers is sqrt(3)
    # u is horizontal, v is at 60 degrees
    u = np.array([np.sqrt(3), 0.0])
    v = np.array([np.sqrt(3)/2, 1.5])
    
    inner = []
    # 3 x 4 parallelogram
    # i in 0..2, j in 0..3
    # Center indices: i_c = 1, j_c = 1.5
    for i in range(3):
        for j in range(4):
            ki = i - 1.0
            kj = j - 1.5
            center = ki * u + kj * v
            # Angle is 0 degrees (aligned with outer hexagon)
            inner.append([center[0], center[1], 0.0])
            
    outer_center = [0.0, 0.0]
    outer_side = 5.0
    outer_angle = 0.0
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
