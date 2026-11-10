import numpy as np

def kissing_number11() -> np.ndarray:
    d = 11
    points = []

    # --- 2*E8 part (first 8 coordinates) ---
    # E8 root system: all vectors in {0, +/-1}^8 with weight 2
    # Scaled by 2 -> weight 2 vectors with entries +/-2, norm^2 = 8
    for i in range(8):
        for j in range(i + 1, 8):
            for s1 in [2, -2]:
                for s2 in [2, -2]:
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = s1
                    p[j] = s2
                    points.append(p)

    # --- 2*D3 part (last 3 coordinates) ---
    # D3 root system: all vectors in {+/-1, +/-1, 0} (weight 2 in 3D)
    # Number: 2 * C(3,2) * 2 = 12 vectors
    # Scaled by 2 -> entries +/-2 in two positions, norm^2 = 8
    for i in range(3):
        for j in range(i + 1, 3):
            for s1 in [2, -2]:
                for s2 in [2, -2]:
                    p = np.zeros(d, dtype=np.int64)
                    p[8 + i] = s1
                    p[8 + j] = s2
                    points.append(p)

    return np.array(points)
