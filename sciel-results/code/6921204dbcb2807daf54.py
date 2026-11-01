import numpy as np

N = 14
D = 3

def construct():
    # Cuboctahedron vertices: permutations of (±1, ±1, 0)
    cubocta = []
    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [0]:
                cubocta.append([x, y, z])
                cubocta.append([x, z, y])
                cubocta.append([y, x, z])
    # Remove duplicates if any (the above loop generates 12 unique points)
    # Let's just list them explicitly to be safe and clear
    points = [
        [1, 1, 0], [1, 0, 1], [0, 1, 1],
        [-1, 1, 0], [-1, 0, 1], [0, -1, 1],
        [1, -1, 0], [1, 0, -1], [0, 1, -1],
        [-1, -1, 0], [-1, 0, -1], [0, -1, -1],
        [0, 0, 1],
        [0, 0, -1]
    ]
    return np.array(points, dtype=float)
