"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    sqrt3 = np.sqrt(3.0)

    # Precompute unit hexagon vertices at angle 0 (flat-top: vertices at 0,60,...)
    # Using pointy-top convention: vertices at 30, 90, 150, ...
    base_angles = np.radians(np.arange(0, 360, 60))
    base_verts = np.column_stack([np.cos(base_angles), np.sin(base_angles)])

    # The 6 candidate SAT normals for a hexagon at orientation theta:
    # These are the outward normals of the 6 edges, i.e., theta + 30 + k*60
    # Actually for a hexagon with vertices at theta + k*60, the edge normals are theta + 30 + k*60
    # But SAT only needs 6 normals per hexagon; total 12 per pair.

    def hex_verts(cx, cy, theta):
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        return base_verts @ R.T + np.array([cx, cy])

    def hex_normals(theta):
        """Return 6 outward edge normals for a unit hexagon with orientation theta."""
        angs = theta + np.radians(30 + np.arange(0, 360, 60))
        return np.column_stack([np.cos(angs), np.sin(angs)])

    def pair_min_gap(v1, v2, n1, n2, c1, c2):
        """Minimum signed gap between two convex hexagons using SAT.
        v1,v2: (6,2) vertices; n1,n2: (6,2) outward normals; c1,c2: centers.
        Returns the minimum separation (positive = separated, negative = overlap).
        """
        all_n = np.vstack([n1, n2])  # (12, 2)
        min_gap = np.inf
        for k in range(12):
            n = all_n[k]
            p1 = v1 @ n
            p2 = v2 @ n
            d = np.dot(c1 - c2, n)
            if d >= 0:
                gap = p2.min() - p1.max()
            else:
                gap = p1.min() - p2.max()
            if gap < min_gap:
                min_gap = gap
        return min_gap

    def outer_side(centers, thetas, outer_theta):
        """Minimum side length of a regular hexagon (centered at origin, orientation outer_theta)
        that contains all N unit hexagons given by (centers, thetas)."""
        # Outer hexagon: contains point p iff for all 6 edge normals n_k, p . n_k <= apothem = S*sqrt(3)/2
        # So S >= max_k (max_i max_v (v . n_k)) * 2 / sqrt(3)
        # where v are vertices of each inner hexagon in world coords.
        outer_normals = hex_normals(outer_theta)  # (6,2)
        max_proj = 0.0
        for i in range(N):
            v = hex_verts(centers[i, 0], centers[i, 1], thetas[i])
            projs = v @ outer_normals.T  # (6, 6)
            m = projs.max()
            if m > max_proj:
                max_proj = m
        return max_proj * 2.0 / sqrt3

    def objective(params):
        """params: 3*N + 1 = [x0,y0,th0, x1,y1,th1, ..., x11,y11,th11, outer_th]
        Returns outer side length + penalty for overlaps.
        """
        centers = params[:3*N].reshape(N, 3)[:, :2]
        thetas = params[:3*N].reshape(N, 3)[:, 2]
        outer_th = params[3*N]

        # Overlap penalty: for each pair, if gap < 0, add (gap)^2 * penalty_weight
        penalty = 0.0
        for i in range(N):
            v1 = hex_verts(centers[i, 0], centers[i, 1], thetas[i])
            n1 = hex_normals(thetas[i])
            c1 = centers[i]
            for j in range(i + 1, N):
                v2 = hex_verts(centers[j, 0], centers[j, 1], thetas[j])
                n2 = hex_normals(thetas[j])
                c2 = centers[j]
                gap = pair_min_gap(v1, v2, n1, n2, c1, c2)
                if gap < 0:
                    penalty += gap * gap

        side = outer_side(centers, thetas, outer_th)
        return side + 100.0 * penalty

    def generate_initial_config(seed):
        """Generate an initial configuration using a honeycomb-like arrangement with jitter."""
        rng = np.random.RandomState(seed)
        # Start with a 3x4 honeycomb
        pos = []
        for r in range(3):
            for c in range(4):
                x = c * sqrt3
                y = r * 1.5
                if r % 2 == 1:
                    x += sqrt3 / 2
                pos.append([x, y, 0.0])
        pos = np.array(pos)
        pos[:, :2] -= pos[:, :2].mean(axis=0)
        # Add jitter
        pos += rng.uniform(-0.3, 0.3, size=pos.shape)
        # Add random outer rotation
        outer_th = rng.uniform(0, np.pi / 3)
        return np.concatenate([pos.ravel(), [outer_th]])

    # Multi-start local optimization
    best_params = None
    best_val = np.inf
    n_starts = 20

    for start_idx in range(n_starts):
        if time.time() > deadline - 15:
            break

        x0 = generate_initial_config(start_idx)

        try:
            res = minimize(
                objective, x0, method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    if best_params is None:
        # Fallback: just use a honeycomb
        pos = []
        for r in range(3):
            for c in range(4):
                x = c * sqrt3
                y = r * 1.5
                if r % 2 == 1:
                    x += sqrt3 / 2
                pos.append([x, y, 0.0])
        pos = np.array(pos)
        pos[:, :2] -= pos[:, :2].mean(axis=0)
        centers = pos[:, :2]
        thetas = pos[:, 2]
        outer_th = 0.0
        side = outer_side(centers, thetas, outer_th)
        inner = [[float(centers[i, 0]), float(centers[i, 1]), float(np.degrees(thetas[i]))] for i in range(N)]
        return inner, (0.0, 0.0), float(side), float(np.degrees(outer_th))

    centers = best_params[:3*N].reshape(N, 3)[:, :2]
    thetas = best_params[:3*N].reshape(N, 3)[:, 2]
    outer_th = best_params[3*N]

    # Verify no overlaps
    valid = True
    for i in range(N):
        v1 = hex_verts(centers[i, 0], centers[i, 1], thetas[i])
        n1 = hex_normals(thetas[i])
        c1 = centers[i]
        for j in range(i + 1, N):
            v2 = hex_verts(centers[j, 0], centers[j, 1], thetas[j])
            n2 = hex_normals(thetas[j])
            c2 = centers[j]
            gap = pair_min_gap(v1, v2, n1, n2, c1, c2)
            if gap < -1e-4:
                valid = False
                break
        if not valid:
            break

    if not valid:
        # Fallback
        inner = [[3.5 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 35.0, 0.0

    side = outer_side(centers, thetas, outer_th)
    inner = [[float(centers[i, 0]), float(centers[i, 1]), float(np.degrees(thetas[i]))] for i in range(N)]
    return inner, (0.0, 0.0), float(side), float(np.degrees(outer_th))
# EVOLVE-BLOCK-END
