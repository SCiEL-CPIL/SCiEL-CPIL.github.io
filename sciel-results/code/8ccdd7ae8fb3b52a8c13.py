import numpy as np

def construct():
    N = 11
    sqrt3 = np.sqrt(3.0)
    
    # Find largest k such that 3*k*(k+1)+1 <= N
    k = 0
    while 3 * (k + 1) * (k + 2) + 1 <= N:
        k += 1
    
    centers = []
    
    # Generate hexagonal cluster
    for ring in range(k + 1):
        if ring == 0:
            centers.append((0.0, 0.0))
        else:
            corners = []
            for c in range(6):
                angle = np.pi / 3.0 * c
                corners.append((ring * sqrt3 * np.cos(angle), ring * sqrt3 * np.sin(angle)))
            for i in range(6 * ring):
                side = i // ring
                pos = i % ring
                start = corners[side]
                end = corners[(side + 1) % 6]
                t = pos / ring
                x = start[0] + t * (end[0] - start[0])
                y = start[1] + t * (end[1] - start[1])
                centers.append((x, y))
    
    placed = len(centers)
    remaining = N - placed
    
    if remaining > 0:
        ring = k + 1
        corners = []
        for c in range(6):
            angle = np.pi / 3.0 * c
            corners.append((ring * sqrt3 * np.cos(angle), ring * sqrt3 * np.sin(angle)))
        added = 0
        for i in range(6 * ring):
            if added >= remaining:
                break
            side = i // ring
            pos = i % ring
            start = corners[side]
            end = corners[(side + 1) % 6]
            t = pos / ring
            x = start[0] + t * (end[0] - start[0])
            y = start[1] + t * (end[1] - start[1])
            centers.append((x, y))
            added += 1
    
    inner = [[cx, cy, 0.0] for (cx, cy) in centers]
    
    # Compute outer side length as max distance from origin to any vertex
    max_dist = 0.0
    for (cx, cy) in centers:
        for i in range(6):
            angle = np.pi / 3.0 * i
            vx = cx + np.cos(angle)
            vy = cy + np.sin(angle)
            d = np.sqrt(vx * vx + vy * vy)
            if d > max_dist:
                max_dist = d
    
    return inner, (0.0, 0.0), max_dist, 0.0
