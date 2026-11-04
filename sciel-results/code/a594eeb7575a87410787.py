import numpy as np

def construct():
    eps = 1e-7
    
    centers = []
    angles = []
    
    # Center hexagon
    centers.append((0.0, 0.0))
    angles.append(0.0)
    
    # 6 neighbors at dist sqrt(3) + eps
    for k in range(6):
        angle = np.deg2rad(k * 60)
        x = (np.sqrt(3) + eps) * np.cos(angle)
        y = (np.sqrt(3) + eps) * np.sin(angle)
        centers.append((x, y))
        angles.append(0.0)
        
    # 4 more at dist 3 + 2*eps
    # Angles 30, 90, 150, 210
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        x = (3.0 + 2*eps) * np.cos(angle)
        y = (3.0 + 2*eps) * np.sin(angle)
        centers.append((x, y))
        angles.append(0.0)
        
    inner = []
    for i in range(len(centers)):
        x, y = centers[i]
        inner.append([x, y, angles[i]])
        
    # Outer hexagon rotation 30.0
    # Vertices at 30, 90, 150, 210, 270, 330
    # Side Normals at 0, 60, 120, 180, 240, 300
    
    # Critical constraint: Second ring hexagons at 30, 90...
    # Inner rotation 0.0 -> Flat sides at 30, 90...
    # Extent in 30 deg direction: 3 + 2*eps + sqrt(3)/2
    # This aligns with Outer Vertex at 30 deg.
    # Distance to vertex is S.
    # So S >= 3 + 2*eps + sqrt(3)/2
    
    side = 3.0 + np.sqrt(3)/2.0 + 2*eps
    
    return inner, (0.0, 0.0), side, 30.0

