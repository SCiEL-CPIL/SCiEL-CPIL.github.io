import numpy as np
import time
from scipy.optimize import minimize


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger
    regular hexagon, minimizing the side length of the outer hexagon.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the
            form (x, y, angle_degrees) containing the (x,y) coordinates and
            angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree)
            containing the (x,y) coordinates and angle_degree of the outer
            hexagon.
        outer_hex_side_length: float representing the side length of the outer
            hexagon.
    """
    start_time = time.time()
    timeout = 620  # Safety margin for 630s limit

    n = 12

    # Axes for separation check (normals to edges of unit hexagon with angle 0)
    u1 = np.array([np.sqrt(3) / 2, 0.5])
    u2 = np.array([0, 1])
    u3 = np.array([-np.sqrt(3) / 2, 0.5])
    axes = np.array([u1, u2, u3])

    min_allowed_proj = np.sqrt(3) + 1e-5

    # Containment axes for outer hexagon (angle 0)
    n1 = np.array([np.sqrt(3) / 2, 0.5])
    n2 = np.array([0, 1])
    n3 = np.array([-np.sqrt(3) / 2, 0.5])
    scale = 2.0 / np.sqrt(3)

    # Base honeycomb 3x4 grid positions (rigid cluster)
    e1 = np.array([1.5, np.sqrt(3) / 2])
    e2 = np.array([0, np.sqrt(3)])
    base = np.array([i * e1 + j * e2 for i in range(3) for j in range(4)])
    base -= base.mean(axis=0)

    def compute_s(centers):
        """Compute minimal enclosing outer hexagon side length for aligned
        inner hexagons centered at `centers`."""
        c_n1 = centers @ n1
        c_n2 = centers @ n2
        c_n3 = centers @ n3
        s1 = scale * max(np.max(c_n1), -np.min(c_n1)) + 1
        s2 = scale * max(np.max(c_n2), -np.min(c_n2)) + 1
        s3 = scale * max(np.max(c_n3), -np.min(c_n3)) + 1
        return max(s1, s2, s3)

    def check_separation(centers):
        """Return True if all pairs of aligned unit hexagons are strictly
        separated (no intersection)."""
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                projs = np.abs(diff @ axes.T)
                if np.max(projs) <= min_allowed_proj:
                    return False
        return True

    def objective_24(x):
        """Full 24-DOF objective: centers with separation penalty."""
        centers = x.reshape((n, 2))
        max_viol = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                projs = np.abs(diff @ axes.T)
                mp = np.max(projs)
                if mp < min_allowed_proj:
                    v = min_allowed_proj - mp
                    if v > max_viol:
                        max_viol = v
        if max_viol > 0:
            return 1e10 + max_viol
        return compute_s(centers)

    def objective_3p(p):
        """3-parameter affine transform of the rigid honeycomb cluster.
        p = [angle_deg, shear, scale_factor]."""
        angle = np.radians(p[0])
        shear = p[1]
        sf = p[2]
        ca, sa = np.cos(angle), np.sin(angle)
        R = np.array([[ca, -sa], [sa, ca]])
        S = np.array([[1, shear], [0, 1]])
        M = R @ S * sf
        centers = (base @ M.T)
        return compute_s(centers)

    def objective_2p(p):
        """2-parameter rotation + scale of the rigid honeycomb cluster.
        p = [angle_deg, scale_factor]."""
        angle = np.radians(p[0])
        sf = p[1]
        ca, sa = np.cos(angle), np.sin(angle)
        R = np.array([[ca, -sa], [sa, ca]])
        centers = (base @ R.T) * sf
        return compute_s(centers)

    best_x = None
    best_f = np.inf

    # Phase 1: 3-parameter affine transform search (rotation + shear + scale)
    print("Phase 1: 3-parameter affine transform search")
    rng = np.random.default_rng(42)
    for restart in range(80):
        if time.time() - start_time > timeout - 150:
            break
        if restart == 0:
            p0 = np.array([0.0, 0.0, 1.0])
        elif restart == 1:
            p0 = np.array([30.0, 0.0, 1.0])
        elif restart == 2:
            p0 = np.array([-30.0, 0.0, 1.0])
        elif restart == 3:
            p0 = np.array([15.0, 0.2, 1.0])
        elif restart == 4:
            p0 = np.array([-15.0, -0.2, 1.0])
        else:
            p0 = np.array([
                rng.uniform(-35, 35),
                rng.uniform(-0.5, 0.5),
                rng.uniform(0.85, 1.15),
            ])
        res = minimize(objective_3p, p0, method='Nelder-Mead',
                       options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11})
        p_best = res.x
        sf = p_best[2]
        angle = np.radians(p_best[0])
        shear = p_best[1]
        ca, sa = np.cos(angle), np.sin(angle)
        R = np.array([[ca, -sa], [sa, ca]])
        S = np.array([[1, shear], [0, 1]])
        M = R @ S * sf
        centers = base @ M.T
        # Check if this configuration is valid (strictly separated)
        if check_separation(centers):
            f = compute_s(centers)
            if f < best_f:
                best_f = f
                best_x = centers.flatten()
                print(f"  restart {restart}: s={f:.6f}")

    # Phase 1b: 2-parameter rotation + scale search (broader angle range)
    print("Phase 1b: 2-parameter rotation+scale search")
    for restart in range(40):
        if time.time() - start_time > timeout - 150:
            break
        if restart == 0:
            p0 = np.array([0.0, 1.0])
        elif restart == 1:
            p0 = np.array([30.0, 1.0])
        elif restart == 2:
            p0 = np.array([-30.0, 1.0])
        elif restart == 3:
            p0 = np.array([60.0, 1.0])
        elif restart == 4:
            p0 = np.array([-60.0, 1.0])
        else:
            p0 = np.array([
                rng.uniform(-90, 90),
                rng.uniform(0.85, 1.15),
            ])
        res = minimize(objective_2p, p0, method='Nelder-Mead',
                       options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11})
        p_best = res.x
        sf = p_best[1]
        angle = np.radians(p_best[0])
        ca, sa = np.cos(angle), np.sin(angle)
        R = np.array([[ca, -sa], [sa, ca]])
        centers = (base @ R.T) * sf
        if check_separation(centers):
            f = compute_s(centers)
            if f < best_f:
                best_f = f
                best_x = centers.flatten()
                print(f"  restart {restart}: s={f:.6f}")

    if best_x is None:
        # Fallback: use the rigid 3x4 grid with a tiny expansion
        centers = base * 1.001
        best_x = centers.flatten()
        best_f = compute_s(centers)
        print(f"  fallback rigid grid: s={best_f:.6f}")

    # Phase 2: Full 24-DOF Nelder-Mead polish from the best result
    print("Phase 2: 24-DOF Nelder-Mead polish")
    x0 = best_x.copy()
    for polish_round in range(6):
        if time.time() - start_time > timeout - 60:
            break
        res = minimize(objective_24, x0, method='Nelder-Mead',
                       options={'maxiter': 15000, 'xatol': 1e-9, 'fatol': 1e-12})
        if res.fun < best_f:
            best_f = res.fun
            best_x = res.x
            print(f"  polish {polish_round}: s={best_f:.6f}")
        # Perturb for next restart
        if time.time() - start_time > timeout - 30:
            break
        x0 = best_x + np.random.normal(0, 0.05, size=24)

    # Phase 3: Final high-precision polish pass
    print("Phase 3: high-precision polish")
    if time.time() - start_time < timeout - 30:
        res = minimize(objective_24, best_x, method='Nelder-Mead',
                       options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-13})
        if res.fun < best_f:
            best_f = res.fun
            best_x = res.x
            print(f"  final polish: s={best_f:.6f}")

    # Extract best result
    centers = best_x.reshape((n, 2))

    # Final safety check: if separation is violated, scale up slightly
    if not check_separation(centers):
        # Expand around centroid
        centroid = centers.mean(axis=0)
        centers = (centers - centroid) * 1.002 + centroid
        best_f = compute_s(centers)

    # Compute final s
    s = compute_s(centers)

    # Add a safety margin to ensure strict containment (floating point)
    s = s + 0.005

    # Construct return arrays
    inner_hex_data = np.zeros((n, 3))
    for i in range(n):
        inner_hex_data[i, 0] = centers[i, 0]
        inner_hex_data[i, 1] = centers[i, 1]
        inner_hex_data[i, 2] = 0.0  # Angle 0

    outer_hex_data = np.array([0.0, 0.0, 0.0])

    return inner_hex_data, outer_hex_data, s
