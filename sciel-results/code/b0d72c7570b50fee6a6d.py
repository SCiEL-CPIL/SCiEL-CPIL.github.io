import numpy as np
from scipy.optimize import minimize

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, minimizing the outer hexagon side length.
    Uses hexagonal lattice placement for inner hexagons and optimizes the outer hexagon.
    """
    def axial_to_pixel(q, r, size=1):
        x = size * np.sqrt(3) * (q + r / 2.0)
        y = size * 1.5 * r
        return x, y

    # 12 positions in axial coords for a compact cluster
    axial_positions = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (2, -1), (1, -2), (0, -2), (-1, -2)
    ]

    inner_hex_data = np.zeros((12, 3))
    for i, (q, r) in enumerate(axial_positions):
        x, y = axial_to_pixel(q, r)
        inner_hex_data[i, 0] = x
        inner_hex_data[i, 1] = y
        inner_hex_data[i, 2] = 0.0

    # Center the inner hexagons at origin
    center = np.mean(inner_hex_data[:, :2], axis=0)
    inner_hex_data[:, :2] -= center

    def get_outer_vertices(cx, cy, s, angle_deg):
        angle_rad = np.radians(angle_deg)
        return np.array([
            [cx + s * np.cos(angle_rad + np.radians(60 * i)),
             cy + s * np.sin(angle_rad + np.radians(60 * i))]
            for i in range(6)
        ])

    def get_all_inner_vertices():
        vertices = []
        for i in range(12):
            ix, iy, iangle = inner_hex_data[i]
            iangle_rad = np.radians(iangle)
            for j in range(6):
                x = ix + np.cos(iangle_rad + np.radians(60 * j))
                y = iy + np.sin(iangle_rad + np.radians(60 * j))
                vertices.append([x, y])
        return np.array(vertices)

    all_inner_vertices = get_all_inner_vertices()

    def max_violation(cx, cy, s, angle_deg):
        outer_vertices = get_outer_vertices(cx, cy, s, angle_deg)
        max_viol = 0.0
        for v in all_inner_vertices:
            for k in range(6):
                p1 = outer_vertices[k]
                p2 = outer_vertices[(k + 1) % 6]
                edge_vec = p2 - p1
                cross = edge_vec[0] * (v[1] - p1[1]) - edge_vec[1] * (v[0] - p1[0])
                if cross < max_viol:
                    max_viol = cross
        return max_viol

    def objective(params):
        cx, cy, s, angle_deg = params
        if s < np.sqrt(12) - 1e-6:
            return 1e10
        viol = max_violation(cx, cy, s, angle_deg)
        if viol > 1e-6:
            return s + (viol + 1e-6) * 10.0
        return s

    x0 = np.array([0.0, 0.0, 7.0, 0.0])

    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 50000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})

    cx, cy, s, angle_deg = result.x
    s = max(s, np.sqrt(12) + 1e-6)

    outer_hex_data = np.array([cx, cy, angle_deg])

    return inner_hex_data, outer_hex_data, s
