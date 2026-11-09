"""Packing 12 unit hexagons in minimal regular hexagon using PSO."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0

    sqrt3 = math.sqrt(3.0)
    rng = np.random.RandomState(42)

    # Unit hexagon vertices (side=1, angle=0)
    unit_hex = np.array([[math.cos(i * math.pi / 3.0), math.sin(i * math.pi / 3.0)] for i in range(6)])

    def get_hex_vertices(cx, cy, angle_rad):
        """Get vertices of a regular hexagon as numpy array."""
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        R = np.array([[c, -s], [s, c]])
        return (R @ unit_hex.T).T + np.array([cx, cy])

    def min_separation(v1, v2):
        """Minimum separation between two convex polygons using SAT.
        Returns positive if separated, negative if overlapping."""
        # Get all edge normals from both polygons
        all_normals = []
        for v in (v1, v2):
            for i in range(len(v)):
                p1 = v[i]
                p2 = v[(i + 1) % len(v)]
                edge = p2 - p1
                length = np.linalg.norm(edge)
                if length > 1e-12:
                    normal = np.array([-edge[1], edge[0]]) / length
                    all_normals.append(normal)

        min_sep = float('inf')
        for normal in all_normals:
            # Project both polygons onto this normal
            proj1 = v1 @ normal
            proj2 = v2 @ normal
            sep = max(proj1.min() - proj2.max(), proj2.min() - proj1.max())
            min_sep = min(min_sep, sep)

        return min_sep

    def check_overlap(centers, angles_rad):
        """Check if any two inner hexagons overlap."""
        for i in range(N):
            vi = get_hex_vertices(centers[i][0], centers[i][1], angles_rad[i])
            for j in range(i + 1, N):
                vj = get_hex_vertices(centers[j][0], centers[j][1], angles_rad[j])
                if min_separation(vi, vj) < -1e-8:
                    return False
        return True

    def enclosing_side_length(centers, angles_rad, outer_angle_rad=0.0):
        """Compute minimal side length of outer hexagon containing all inner hexagons."""
        normals = []
        for k in range(6):
            angle = outer_angle_rad + math.pi / 6 + k * math.pi / 3
            normals.append((math.cos(angle), math.sin(angle)))

        max_extent = 0.0
        for i in range(N):
            cx, cy = centers[i]
            inner_angle_rad = angles_rad[i]

            for nk in range(6):
                nx, ny = normals[nk]
                n_angle = math.atan2(ny, nx)

                max_dot = -2.0
                for j in range(6):
                    v_angle = inner_angle_rad + j * math.pi / 3.0
                    dot_val = math.cos(v_angle - n_angle)
                    if dot_val > max_dot:
                        max_dot = dot_val

                extent = cx * nx + cy * ny + max_dot
                if extent > max_extent:
                    max_extent = extent

        return 2.0 * max_extent / sqrt3

    def objective(params):
        """Penalty-based objective for PSO."""
        if time.time() > deadline:
            return 1e10

        centers = []
        angles_rad = []
        for i in range(N):
            centers.append((params[i * 3], params[i * 3 + 1]))
            angles_rad.append(params[i * 3 + 2])
        outer_angle_rad = params[36]

        # Compute overlap penalty
        overlap_penalty = 0.0
        for i in range(N):
            vi = get_hex_vertices(centers[i][0], centers[i][1], angles_rad[i])
            for j in range(i + 1, N):
                vj = get_hex_vertices(centers[j][0], centers[j][1], angles_rad[j])
                sep = min_separation(vi, vj)
                if sep < 0:
                    overlap_penalty += (-sep) ** 2

        L = enclosing_side_length(centers, angles_rad, outer_angle_rad)
        return L + 1000.0 * overlap_penalty

    def pso_optimize(n_particles=40, max_iter=80):
        """Particle Swarm Optimization."""
        # Initialize particles
        X = np.zeros((n_particles, 37))
        V = np.zeros((n_particles, 37))

        # Initial positions: mix of lattice and random
        for p in range(n_particles):
            if p < n_particles // 3:
                # Hexagonal lattice
                init_grid = [
                    (0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1),
                    (1, -1), (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
                ]
                for i in range(N):
                    q, r = init_grid[i]
                    X[p, i * 3] = sqrt3 * q + sqrt3 / 2 * r
                    X[p, i * 3 + 1] = 1.5 * r
                    X[p, i * 3 + 2] = 0.0
                X[p, 36] = 0.0
            elif p < 2 * n_particles // 3:
                # Perturbed lattice
                init_grid = [
                    (0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1),
                    (1, -1), (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
                ]
                for i in range(N):
                    q, r = init_grid[i]
                    X[p, i * 3] = sqrt3 * q + sqrt3 / 2 * r + rng.uniform(-0.3, 0.3)
                    X[p, i * 3 + 1] = 1.5 * r + rng.uniform(-0.3, 0.3)
                    X[p, i * 3 + 2] = rng.uniform(-0.2, 0.2)
                X[p, 36] = rng.uniform(-0.2, 0.2)
            else:
                # Random
                for i in range(N):
                    X[p, i * 3] = rng.uniform(-3, 3)
                    X[p, i * 3 + 1] = rng.uniform(-3, 3)
                    X[p, i * 3 + 2] = rng.uniform(-math.pi / 3, math.pi / 3)
                X[p, 36] = rng.uniform(-math.pi / 6, math.pi / 6)

        V[:] = rng.uniform(-0.5, 0.5, size=(n_particles, 37))

        # Evaluate initial fitness
        F = np.array([objective(X[p]) for p in range(n_particles)])

        pbest_X = X.copy()
        pbest_F = F.copy()
        gbest_idx = np.argmin(F)
        gbest_X = X[gbest_idx].copy()
        gbest_F = F[gbest_idx]

        w = 0.9
        c1 = 1.5
        c2 = 1.5

        for it in range(max_iter):
            if time.time() > deadline - 10.0:
                break

            # Update particles
            for p in range(n_particles):
                r1 = rng.random(37)
                r2 = rng.random(37)
                V[p] = w * V[p] + c1 * r1 * (pbest_X[p] - X[p]) + c2 * r2 * (gbest_X - X[p])
                # Clamp velocity
                V[p] = np.clip(V[p], -1.0, 1.0)
                X[p] = X[p] + V[p]

                # Evaluate
                F[p] = objective(X[p])

                if F[p] < pbest_F[p]:
                    pbest_F[p] = F[p]
                    pbest_X[p] = X[p].copy()
                    if F[p] < gbest_F:
                        gbest_F = F[p]
                        gbest_X = X[p].copy()
                        gbest_idx = p

            # Decrease inertia
            w = max(0.4, w * 0.995)

        return gbest_X, gbest_F

    def local_refine(params):
        """Local refinement using Nelder-Mead."""
        if time.time() > deadline - 5.0:
            return params, objective(params)

        try:
            result = minimize(
                objective,
                params,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6, 'adaptive': True}
            )
            if result.fun < objective(params):
                return result.x, result.fun
        except Exception:
            pass
        return params, objective(params)

    # Run PSO
    best_params, best_F = pso_optimize(n_particles=30, max_iter=60)

    # Local refinement
    best_params, best_F = local_refine(best_params)

    # Second PSO round with different initial conditions
    if time.time() < deadline - 20.0:
        # Use best from first round as one particle
        X2 = np.zeros((20, 37))
        for p in range(20):
            if p == 0:
                X2[p] = best_params
            elif p < 10:
                # Perturb best
                X2[p] = best_params + rng.uniform(-0.2, 0.2, size=37)
            else:
                # Random
                for i in range(N):
                    X2[p, i * 3] = rng.uniform(-3, 3)
                    X2[p, i * 3 + 1] = rng.uniform(-3, 3)
                    X2[p, i * 3 + 2] = rng.uniform(-math.pi / 3, math.pi / 3)
                X2[p, 36] = rng.uniform(-math.pi / 6, math.pi / 6)

        V2 = np.zeros((20, 37))
        V2[:] = rng.uniform(-0.3, 0.3, size=(20, 37))

        F2 = np.array([objective(X2[p]) for p in range(20)])
        pbest_X2 = X2.copy()
        pbest_F2 = F2.copy()
        gbest_idx2 = np.argmin(F2)
        gbest_X2 = X2[gbest_idx2].copy()
        gbest_F2 = F2[gbest_idx2]

        w2 = 0.85
        for it in range(50):
            if time.time() > deadline - 5.0:
                break

            for p in range(20):
                r1 = rng.random(37)
                r2 = rng.random(37)
                V2[p] = w2 * V2[p] + 1.5 * r1 * (pbest_X2[p] - X2[p]) + 1.5 * r2 * (gbest_X2 - X2[p])
                V2[p] = np.clip(V2[p], -0.5, 0.5)
                X2[p] = X2[p] + V2[p]
                F2[p] = objective(X2[p])

                if F2[p] < pbest_F2[p]:
                    pbest_F2[p] = F2[p]
                    pbest_X2[p] = X2[p].copy()
                    if F2[p] < gbest_F2:
                        gbest_F2 = F2[p]
                        gbest_X2 = X2[p].copy()

            w2 = max(0.4, w2 * 0.995)

        if gbest_F2 < best_F:
            best_params, best_F = gbest_X2, gbest_F2
            best_params, best_F = local_refine(best_params)

    # Extract final configuration
    centers = []
    angles_rad = []
    for i in range(N):
        centers.append((best_params[i * 3], best_params[i * 3 + 1]))
        angles_rad.append(best_params[i * 3 + 2])
    outer_angle_rad = best_params[36]

    # Check validity
    if not check_overlap(centers, angles_rad):
        # Fallback to lattice
        init_grid = [
            (0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1),
            (1, -1), (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
        ]
        centers = []
        for q, r in init_grid:
            centers.append((sqrt3 * q + sqrt3 / 2 * r, 1.5 * r))
        angles_rad = [0.0] * N
        outer_angle_rad = 0.0

    L = enclosing_side_length(centers, angles_rad, outer_angle_rad)

    inner = []
    for i in range(N):
        inner.append([centers[i][0], centers[i][1], math.degrees(angles_rad[i])])

    center = (0.0, 0.0)
    side = L
    angle = math.degrees(outer_angle_rad)

    return inner, center, side, angle
# EVOLVE-BLOCK-END
