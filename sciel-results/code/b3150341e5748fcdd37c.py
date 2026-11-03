"""Packing 12 unit hexagons in a regular hexagon, minimizing outer side length."""
import time
import math
import numpy as np

N = 12
DEADLINE = 145.0  # safety margin before 150s kill

# Precompute unit hexagon vertices at origin, rotation 0
_HEX_VERTS = np.array([[math.cos(math.radians(60.0 * k)),
                        math.sin(math.radians(60.0 * k))] for k in range(6)])

# Inradius of unit hexagon (distance from center to edge)
_INRADIUS = math.sqrt(3.0) / 2.0
# Two unit hexagons overlap iff center distance < 2*inradius = sqrt(3)
_OVERLAP_DIST = 2.0 * _INRADIUS  # = sqrt(3)

# Outer hexagon (rotation 0, center 0) edge normals (unit, pointing inward)
# Vertices at angles 0,60,...,300. Edge normals point toward center at angles 30,90,...,330
_EDGE_NORMALS = np.array([[math.cos(math.radians(60.0 * k + 30.0)),
                           math.sin(math.radians(60.0 * k + 30.0))] for k in range(6)])


def hex_center_positions(params):
    """params: array of shape (3*N,). Returns centers (N,2) and rotations (N,)."""
    x = params[0::3]
    y = params[1::3]
    theta = params[2::3]
    centers = np.column_stack([x, y])
    return centers, theta


def feasibility(centers, theta, S):
    """Return (ok, violation). ok True if no overlap and all contained.
    violation = max over all constraints of (required - actual), <=0 means feasible."""
    n = len(centers)
    # Pairwise overlap: need dist >= _OVERLAP_DIST
    # Compute all pairwise distances
    diff = centers[:, None, :] - centers[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    overlap_viol = _OVERLAP_DIST - min_pair_dist  # >0 means overlap

    # Containment: for each hex center, max over edges of dot(center, normal) <= S - 1
    # (the unit hexagon extends inradius=1 beyond its center toward each edge)
    dots = centers @ _EDGE_NORMALS.T  # (N, 6)
    max_dot = np.max(dots, axis=1)
    contain_viol = np.max(max_dot) - (S - 1.0)  # >0 means outside

    violation = max(overlap_viol, contain_viol)
    ok = violation <= 1e-9
    return ok, violation


def objective(params, S):
    """Objective: return S + penalty if infeasible."""
    centers, theta = hex_center_positions(params)
    ok, viol = feasibility(centers, theta, S)
    if ok:
        return S
    else:
        return S + 1000.0 * viol


def construct():
    t_start = time.time()

    def time_left():
        return DEADLINE - (time.time() - t_start)

    # --- Initial guess: a hexagonal cluster arrangement ---
    # Place hexagons in a hex lattice with spacing slightly > sqrt(3)
    spacing = _OVERLAP_DIST + 0.05
    # positions in a hex grid
    grid = []
    # center + 6 around + 12 around -> but we need 12
    # Use a "flower": 1 center + 6 ring + 5 of the next ring
    positions = [(0.0, 0.0)]
    # first ring at distance = spacing*sqrt(3)/... actually hex lattice neighbor distance = spacing
    # neighbor directions at 0,60,120,180,240,300
    for k in range(6):
        ang = math.radians(60.0 * k)
        positions.append((spacing * math.cos(ang), spacing * math.sin(ang)))
    # second ring: 6 at distance 2*spacing in same directions, plus 6 in between
    for k in range(6):
        ang = math.radians(60.0 * k)
        positions.append((2.0 * spacing * math.cos(ang), 2.0 * spacing * math.sin(ang)))
    # take first 12
    positions = positions[:N]

    # Estimate initial S from bounding of positions
    pts = np.array(positions)
    max_dot = np.max(pts @ _EDGE_NORMALS.T)
    S0 = max_dot + 1.0 + 0.5

    # Parameter vector: 3*N (x, y, theta) + 1 (S)
    params0 = np.zeros(3 * N + 1)
    for i in range(N):
        params0[3 * i] = positions[i][0]
        params0[3 * i + 1] = positions[i][1]
        params0[3 * i + 2] = 0.0
    params0[-1] = S0

    best_params = params0.copy()
    best_S = S0
    best_obj = S0

    # Quick local refinement via Nelder-Mead to get a decent starting point
    try:
        from scipy.optimize import minimize, differential_evolution
        from scipy.spatial import ConvexHull
        have_scipy = True
    except Exception:
        have_scipy = False

    if have_scipy:
        # Local Nelder-Mead on the initial cluster to tighten it
        try:
            res = minimize(objective, params0, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4})
            if res.fun < best_obj:
                best_params = res.x.copy()
                best_obj = res.fun
                best_S = res.x[-1]
        except Exception:
            pass

        # Now global search with differential evolution
        # Bounds: centers within [-S0, S0], theta in [0, 60), S in [lower, upper]
        # Lower bound on S: 12 hexagons, area lower bound.
        # Area of unit hexagon = (3*sqrt(3)/2)*1^2 ≈ 2.598. 12 of them ≈ 31.18.
        # Outer hex area = (3*sqrt(3)/2)*S^2. So S >= sqrt(12) ≈ 3.464 by area.
        # Practical lower bound a bit higher.
        S_lower = 4.0
        S_upper = best_S if best_S > 0 else 8.0
        S_upper = min(S_upper, 10.0)

        # Coordinate bounds: keep centers within a box
        coord_bound = S_upper + 1.0

        bounds = []
        for i in range(N):
            bounds.append((-coord_bound, coord_bound))  # x
            bounds.append((-coord_bound, coord_bound))  # y
            bounds.append((0.0, 60.0))                  # theta deg
        bounds.append((S_lower, S_upper))

        # Convert objective to work with theta in degrees
        def obj_de(p):
            p = p.copy()
            # theta in degrees -> radians
            p[2::3] = np.radians(p[2::3])
            S = p[-1]
            centers, theta = hex_center_positions(p)
            ok, viol = feasibility(centers, theta, S)
            if ok:
                return S
            else:
                return S + 1000.0 * viol

        # Run differential evolution
        de_params0 = best_params.copy()
        de_params0[2::3] = np.degrees(de_params0[2::3])

        # Adaptive population size
        popsize = 20
        maxiter = 200
        tol = 1e-6

        # Time-budgeted DE: run in chunks
        try:
            # First, a moderate DE run
            res = differential_evolution(obj_de, bounds, x0=de_params0,
                                          popsize=popsize, maxiter=maxiter,
                                          tol=tol, seed=42,
                                          mutation=(0.3, 1.0), recombination=0.8,
                                          polish=True)
            if res.fun < best_obj:
                # convert back
                best_params = res.x.copy()
                best_params[2::3] = np.radians(best_params[2::3])
                best_obj = res.fun
                best_S = res.x[-1]
        except Exception:
            pass

        # If we have time, do a second DE run with tighter bounds around best
        if time_left() > 30:
            try:
                cur_S = best_S
                S_upper2 = cur_S + 0.5
                S_lower2 = max(S_lower, cur_S - 1.5)
                coord_bound2 = S_upper2 + 1.0
                bounds2 = []
                for i in range(N):
                    bounds2.append((-coord_bound2, coord_bound2))
                    bounds2.append((-coord_bound2, coord_bound2))
                    bounds2.append((0.0, 60.0))
                bounds2.append((S_lower2, S_upper2))
                de_params0b = best_params.copy()
                de_params0b[2::3] = np.degrees(de_params0b[2::3])
                res2 = differential_evolution(obj_de, bounds2, x0=de_params0b,
                                               popsize=popsize, maxiter=150,
                                               tol=tol, seed=123,
                                               mutation=(0.3, 1.0), recombination=0.8,
                                               polish=True)
                if res2.fun < best_obj:
                    best_params = res2.x.copy()
                    best_params[2::3] = np.radians(best_params[2::3])
                    best_obj = res2.fun
                    best_S = res2.x[-1]
            except Exception:
                pass

        # Final local polish with Nelder-Mead
        if time_left() > 15:
            try:
                def obj_local(p):
                    S = p[-1]
                    centers, theta = hex_center_positions(p)
                    ok, viol = feasibility(centers, theta, S)
                    if ok:
                        return S
                    else:
                        return S + 1000.0 * viol
                res3 = minimize(obj_local, best_params, method='Nelder-Mead',
                                options={'maxiter': 5000, 'xatol': 1e-5, 'fatol': 1e-5})
                if res3.fun < best_obj:
                    best_params = res3.x.copy()
                    best_obj = res3.fun
                    best_S = res3.x[-1]
            except Exception:
                pass

    # Extract final result
    # best_params: (3*N + 1), theta in radians
    x = best_params[0::3]
    y = best_params[1::3]
    theta_rad = best_params[2::3]
    S_final = best_params[-1]

    # Verify feasibility; if infeasible, increase S until feasible (safety)
    centers = np.column_stack([x, y])
    ok, viol = feasibility(centers, theta_rad, S_final)
    if not ok:
        # bump S
        S_final = S_final + viol + 0.1
        ok, viol = feasibility(centers, theta_rad, S_final)
        while not ok:
            S_final += 0.1
            ok, viol = feasibility(centers, theta_rad, S_final)
            if time_left() < 1:
                break

    # Build inner list: (x, y, angle_degrees)
    inner = []
    for i in range(N):
        inner.append([float(x[i]), float(y[i]), float(math.degrees(theta_rad[i]))])

    center = (0.0, 0.0)
    side = float(S_final)
    angle = 0.0

    return inner, center, side, angle
