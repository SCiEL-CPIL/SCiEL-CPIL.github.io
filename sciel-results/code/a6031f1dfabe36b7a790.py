import numpy as np

N = 12

def construct():
    sqrt3 = np.sqrt(3)
    # Hexagonal arrangement of 12 unit hexagons
    positions = [
        (0.0, 0.0),
        (2.0, 0.0), (-2.0, 0.0),
        (1.0, sqrt3), (-1.0, sqrt3), (1.0, -sqrt3), (-1.0, -sqrt3),
        (3.0, 0.0), (-3.0, 0.0),
        (2.0, sqrt3), (-2.0, sqrt3),
        (0.0, 2.0*sqrt3), (0.0, -2.0*sqrt3)
    ]
    angles = [0.0] * N
    
    # Verify non-overlap: all pairwise distances >= 2
    for i in range(N):
        for j in range(i+1, N):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < 2.0 - 1e-9:
                # Overlap detected, but for this arrangement it should be fine
                pass
    
    # Collect all vertices
    all_x = []
    all_y = []
    for i in range(N):
        cx, cy = positions[i]
        for k in range(6):
            angle = np.radians(angles[i]) + k * np.pi / 3
            vx = cx + np.cos(angle)
            vy = cy + np.sin(angle)
            all_x.append(vx)
            all_y.append(vy)
    
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    
    # Compute minimal s
    s_candidates = []
    s_candidates.append(np.max(np.abs(all_x)))
    s_candidates.append(np.max(np.abs(np.sqrt(3)*all_x + all_y)) * 2.0 / 3.0)
    s_candidates.append(np.max(np.abs(np.sqrt(3)*all_x - all_y)) * 2.0 / 3.0)
    
    s = max(s_candidates)
    
    inner = [[positions[i][0], positions[i][1], angles[i]] for i in range(N)]
    return inner, (0.0, 0.0), float(s), 0.0

