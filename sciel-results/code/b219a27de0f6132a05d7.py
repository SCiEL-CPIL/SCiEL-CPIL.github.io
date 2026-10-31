"""Circle packing in perimeter-4 rectangle: Delaunay-based contact graph + scipy sparse solver."""
import numpy as np
import time
from scipy.optimize import minimize, least_squares
from scipy.spatial import Delaunay

N = 21
_DEADLINE = 145.0
_START = None


def construct():
    global _START
    if _START is None:
        _START = time.time()

    best_value = 0.0
    best_result = None

    def check_validity(x, y, r, w, h, tol=1e-9):
        if w <= 0.05 or h <= 0.05:
            return False, 0.0
        for i in range(N):
            if r[i] <= 0:
                return False, 0.0
            if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
                return False, 0.0
            if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
                return False, 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx * dx + dy * dy
                r_sum = r[i] + r[j]
                if d2 < r_sum ** 2 - tol:
                    return False, 0.0
        val = float(np.sum(r))
        return True, val

    def _build_contact_graph(x, y, r, w, h):
        """Build contact graph: edges where circles touch or are near-touching, plus wall contacts."""
        n = N
        contacts = set()
        # Circle-circle contacts
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                if abs(d - (r[i] + r[j])) < 0.02:
                    contacts.add((i, j))
        # Wall contacts
        for i in range(n):
            if x[i] - r[i] < 0.02:
                contacts.add((i, 'L'))
            if x[i] + r[i] - w < 0.02:
                contacts.add((i, 'R'))
            if y[i] - r[i] < 0.02:
                contacts.add((i, 'B'))
            if y[i] + r[i] - h < 0.02:
                contacts.add((i, 'T'))
        return contacts

    def _delaunay_init(w0, h0):
        """Initialize circles using Delaunay triangulation of random points in rectangle."""
        rng = np.random.default_rng(int(w0 * 10000))
        # Generate more points than needed, use Delaunay to find good positions
        n_pts = N * 3
        pts = rng.uniform(0.1, 0.9, (n_pts, 2))
        pts[:, 0] *= w0
        pts[:, 1] *= h0
        
        try:
            tri = Delaunay(pts)
        except Exception:
            # Fallback to grid
            side = int(np.ceil(np.sqrt(N)))
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            for i in range(N):
                row, col = divmod(i, side)
                x[i] = (col + 0.5) / side * w0
                y[i] = (row + 0.5) / side * h0
                r[i] = min(w0 / side, h0 / side) * 0.4
            return x, y, r

        # Use centroid of triangles to place circles
        centroids = []
        for simplex in tri.simplices:
            c = pts[simplex].mean(axis=0)
            centroids.append(c)
        
        # Select N points that are well-spread
        centroids = np.array(centroids)
        # Sort by distance from center, take outer ones first
        cx, cy = w0 / 2, h0 / 2
        dists = np.sqrt((centroids[:, 0] - cx) ** 2 + (centroids[:, 1] - cy) ** 2)
        order = np.argsort(dists)[::-1]  # farthest from center first
        selected = order[:N]
        
        x = centroids[selected, 0].copy()
        y = centroids[selected, 1].copy()
        
        # Estimate radii based on local spacing
        r = np.zeros(N)
        for i in range(N):
            min_dist = np.inf
            for j in range(N):
                if i == j:
                    continue
                d = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
                min_dist = min(min_dist, d)
            wall_dist = min(x[i], w0 - x[i], y[i], h0 - y[i])
            r[i] = 0.45 * min(min_dist, wall_dist)
            r[i] = max(r[i], 0.02)
        
        return x, y, r

    def _force_relax(x, y, r, w, h, max_iter=300, dt=0.002):
        """Force-directed relaxation to resolve overlaps."""
        n = N
        x = x.copy().astype(float)
        y = y.copy().astype(float)
        r = r.copy().astype(float)

        for _ in range(max_iter):
            fx = np.zeros(n)
            fy = np.zeros(n)

            # Wall repulsion
            for i in range(n):
                if x[i] - r[i] < 0:
                    fx[i] += (x[i] - r[i]) * 50
                if x[i] + r[i] > w:
                    fx[i] -= (x[i] + r[i] - w) * 50
                if y[i] - r[i] < 0:
                    fy[i] += (y[i] - r[i]) * 50
                if y[i] + r[i] > h:
                    fy[i] -= (y[i] + r[i] - h) * 50

            # Pairwise repulsion
            for i in range(n):
                for j in range(i + 1, n):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    min_d = r[i] + r[j]
                    if d < min_d and d > 1e-12:
                        overlap = min_d - d
                        force = overlap * 30
                        fx[i] += (dx / d) * force
                        fy[i] += (dy / d) * force
                        fx[j] -= (dx / d) * force
                        fy[j] -= (dy / d) * force

            x += dt * fx
            y += dt * fy

            # Clamp
            for i in range(n):
                x[i] = np.clip(x[i], r[i], w - r[i])
                y[i] = np.clip(y[i], r[i], h - r[i])

        return x, y, r

    def _polish_radii(x, y, r, w, h, time_budget=10.0):
        """Grow radii where there's room, iteratively."""
        n = N
        x = x.copy().astype(float)
        y = y.copy().astype(float)
        r = r.copy().astype(float)
        start = time.time()
        max_iter = 2000

        for it in range(max_iter):
            if time.time() - start > time_budget:
                break

            grew = False
            for i in range(n):
                # Distance to walls
                dist_wall = min(x[i] - r[i], w - x[i] - r[i],
                                y[i] - r[i], h - y[i] - r[i])
                # Distance to other circles
                min_dist_other = np.inf
                for j in range(n):
                    if i == j:
                        continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    min_dist_other = min(min_dist_other, d - r[j])

                available = min(dist_wall, min_dist_other)
                if available > 1e-7:
                    growth = min(0.0003, available * 0.3)
                    r[i] += growth
                    grew = True

            if not grew:
                break

            # After growing, do a quick force relaxation
            x, y, r = _force_relax(x, y, r, w, h, max_iter=20, dt=0.001)

        return x, y, r

    def _solve_contact_system(x, y, r, w, h, contacts, max_iter=100):
        """Solve for positions/radii given a contact graph using least squares."""
        n = N
        # Variables: x, y, r for each circle (3N variables)
        # Constraints: for each contact, distance = sum of radii (or wall distance = radius)
        
        def residuals(z):
            xz = z[0::3][:n]
            yz = z[1::3][:n]
            rz = z[2::3][:n]
            res = []
            
            for c in contacts:
                if isinstance(c[1], str):
                    i = c[0]
                    wall = c[1]
                    if wall == 'L':
                        res.append(xz[i] - rz[i])
                    elif wall == 'R':
                        res.append(w - xz[i] - rz[i])
                    elif wall == 'B':
                        res.append(yz[i] - rz[i])
                    elif wall == 'T':
                        res.append(h - yz[i] - rz[i])
                else:
                    i, j = c
                    dx = xz[i] - xz[j]
                    dy = yz[i] - yz[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    res.append(d - rz[i] - rz[j])
            
            return np.array(res)
        
        z0 = np.zeros(3 * n)
        z0[0::3][:n] = x
        z0[1::3][:n] = y
        z0[2::3][:n] = r
        
        try:
            sol = least_squares(residuals, z0, method='lm', max_nfev=500)
            if sol.success or sol.cost < 1e-6:
                x_new = sol.x[0::3][:n]
                y_new = sol.x[1::3][:n]
                r_new = sol.x[2::3][:n]
                # Validate
                valid, val = check_validity(x_new, y_new, r_new, w, h)
                if valid:
                    return x_new, y_new, r_new
        except Exception:
            pass
        
        return x, y, r

    def _slsqp_optimize(x, y, r, w, h, time_budget=20.0):
        """SLSQP optimization with contact-aware objective."""
        n = N
        z0 = np.zeros(3 * n + 1)
        z0[0::3][:n] = x
        z0[1::3][:n] = y
        z0[2::3][:n] = r
        z0[-1] = w

        def objective(z):
            xz = z[0::3][:n]
            yz = z[1::3][:n]
            rz = z[2::3][:n]
            wz = z[-1]
            hz = 2.0 - wz

            if wz < 0.3 or hz < 0.3:
                return 1e6

            obj = -np.sum(rz)

            # Overlap penalty
            penalty_weight = 1000.0
            for i in range(n):
                for j in range(i + 1, n):
                    dx = xz[i] - xz[j]
                    dy = yz[i] - yz[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    dist_req = rz[i] + rz[j]
                    if d < dist_req:
                        obj += penalty_weight * (dist_req - d) ** 2

            # Boundary penalty
            for i in range(n):
                if xz[i] < rz[i]:
                    obj += penalty_weight * (rz[i] - xz[i]) ** 2
                elif xz[i] > wz - rz[i]:
                    obj += penalty_weight * (xz[i] - (wz - rz[i])) ** 2
                if yz[i] < rz[i]:
                    obj += penalty_weight * (rz[i] - yz[i]) ** 2
                elif yz[i] > hz - rz[i]:
                    obj += penalty_weight * (yz[i] - (hz - rz[i])) ** 2

            return obj

        elapsed = time.time() - _START
        remaining = _DEADLINE - elapsed
        maxiter = int(min(200, remaining * 25))
        if maxiter < 10:
            return x, y, r, w, h

        try:
            res = minimize(objective, z0, method='SLSQP',
                           bounds=[(0, 2)] * (3 * n) + [(0.3, 1.7)],
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 50})
            z_opt = res.x
            x_opt = z_opt[0::3][:n]
            y_opt = z_opt[1::3][:n]
            r_opt = z_opt[2::3][:n]
            w_opt = z_opt[-1]
            h_opt = 2.0 - w_opt

            valid, val = check_validity(x_opt, y_opt, r_opt, w_opt, h_opt)
            if valid:
                return x_opt, y_opt, r_opt, w_opt, h_opt
        except Exception:
            pass

        return x, y, r, w, h

    # Main search: multiple aspect ratios with Delaunay initialization
    aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 2.0, 0.5, 1.1, 0.9, 1.25]
    
    for w0 in aspect_ratios:
        if time.time() - _START > _DEADLINE - 15:
            break

        h0 = 2.0 - w0
        if h0 < 0.3:
            continue

        # Delaunay-based initialization
        x, y, r = _delaunay_init(w0, h0)

        # Force relaxation to resolve overlaps
        x, y, r = _force_relax(x, y, r, w0, h0, max_iter=300)

        # Check validity
        valid, val = check_validity(x, y, r, w0, h0)
        if valid and val > best_value:
            best_value = val
            best_result = (x.copy(), y.copy(), r.copy(), w0, h0)

        # Build contact graph and solve
        contacts = _build_contact_graph(x, y, r, w0, h0)
        if len(contacts) > 10:
            x2, y2, r2 = _solve_contact_system(x, y, r, w0, h0, contacts)
            valid, val = check_validity(x2, y2, r2, w0, h0)
            if valid and val > best_value:
                best_value = val
                best_result = (x2.copy(), y2.copy(), r2.copy(), w0, h0)

        # SLSQP optimization
        if time.time() - _START < _DEADLINE - 20:
            x3, y3, r3, w3, h3 = _slsqp_optimize(x, y, r, w0, h0, time_budget=15.0)
            valid, val = check_validity(x3, y3, r3, w3, h3)
            if valid and val > best_value:
                best_value = val
                best_result = (x3.copy(), y3.copy(), r3.copy(), w3, h3)

        # Radius polish
        if time.time() - _START < _DEADLINE - 10:
            x4, y4, r4 = _polish_radii(x, y, r, w0, h0, time_budget=8.0)
            valid, val = check_validity(x4, y4, r4, w0, h0)
            if valid and val > best_value:
                best_value = val
                best_result = (x4.copy(), y4.copy(), r4.copy(), w0, h0)

    # Final polish on best result
    if best_result is not None and time.time() - _START < _DEADLINE - 5:
        x_best, y_best, r_best, w_best, h_best = best_result
        x_final, y_final, r_final = _polish_radii(x_best, y_best, r_best, w_best, h_best, time_budget=5.0)
        valid, val = check_validity(x_final, y_final, r_final, w_best, h_best)
        if valid and val > best_value:
            best_value = val
            best_result = (x_final.copy(), y_final.copy(), r_final.copy(), w_best, h_best)

    if best_result is None:
        # Fallback: simple grid in unit square
        side = int(np.ceil(np.sqrt(N)))
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        for i in range(N):
            row, col = divmod(i, side)
            x[i] = (col + 0.5) / side
            y[i] = (row + 0.5) / side
            r[i] = 0.5 / side - 1e-3
        return np.column_stack((x, y, r))

    x_best, y_best, r_best, w_best, h_best = best_result

    # Safety shrink
    r_best = r_best * (1 - 1e-6)

    return np.column_stack((x_best, y_best, r_best))
