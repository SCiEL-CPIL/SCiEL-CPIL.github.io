import numpy as np
import time

N = 11

def _hex_vertices(cx, cy, rot_deg, r=1.0):
    a = np.deg2rad(rot_deg)
    angs = a + np.arange(6) * (np.pi / 3.0)
    return np.column_stack([cx + r * np.cos(angs), cy + r * np.sin(angs)])

def _min_outer_side(inner_pts, outer_rot_deg):
    a = np.deg2rad(outer_rot_deg)
    angs = a + np.arange(6) * (np.pi / 3.0)
    normals = np.column_stack([np.cos(angs), np.sin(angs)])
    proj = inner_pts @ normals.T
    max_proj = proj.max()
    return max_proj / (np.sqrt(3.0) / 2.0)

def _build_cluster(theta_deg):
    d = np.sqrt(3.0)
    h = d * np.sqrt(3.0) / 2.0
    offsets = []
    # middle row y=0: 4 hexes
    for x in [-1.5*d, -0.5*d, 0.5*d, 1.5*d]:
        offsets.append((x, 0.0))
    # top row y=+h: 3 hexes
    for x in [-d, 0.0, d]:
        offsets.append((x, h))
    # bottom row y=-h: 3 hexes
    for x in [-d, 0.0, d]:
        offsets.append((x, -h))
    centers = np.array(offsets, dtype=float)
    centers = centers - centers.mean(axis=0)
    verts_list = [_hex_vertices(cx, cy, theta_deg) for (cx, cy) in centers]
    all_pts = np.vstack(verts_list)
    return centers, all_pts

def construct():
    t0 = time.time()
    deadline = t0 + 145.0

    best_s = float('inf')
    best_theta = 0.0
    best_phi = 0.0

    theta_grid = np.arange(0.0, 60.0, 1.0)
    phi_grid = np.arange(0.0, 60.0, 1.0)

    pts_cache = {}
    for th in theta_grid:
        _, all_pts = _build_cluster(float(th))
        pts_cache[float(th)] = all_pts

    for th in theta_grid:
        if time.time() > deadline - 10.0:
            break
        all_pts = pts_cache[float(th)]
        for ph in phi_grid:
            s = _min_outer_side(all_pts, float(ph))
            if s < best_s:
                best_s = s
                best_theta = float(th)
                best_phi = float(ph)

    if time.time() < deadline - 5.0:
        for step in [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]:
            if time.time() > deadline - 5.0:
                break
            improved = True
            while improved:
                improved = False
                for dth in [-step, 0.0, step]:
                    for dph in [-step, 0.0, step]:
                        th2 = (best_theta + dth) % 60.0
                        ph2 = (best_phi + dph) % 60.0
                        _, all_pts = _build_cluster(th2)
                        s = _min_outer_side(all_pts, ph2)
                        if s < best_s - 1e-12:
                            best_s = s
                            best_theta = th2
                            best_phi = ph2
                            improved = True
                        if time.time() > deadline - 5.0:
                            break
                    if time.time() > deadline - 5.0:
                        break

    centers, _ = _build_cluster(best_theta)
    inner = [[float(cx), float(cy), float(best_theta)] for (cx, cy) in centers]
    return inner, (0.0, 0.0), float(best_s), float(best_phi)
