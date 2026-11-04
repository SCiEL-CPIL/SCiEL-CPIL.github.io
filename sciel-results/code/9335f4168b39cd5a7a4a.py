import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay, Voronoi

N = 26

def construct():
    def compute_max_radius_at_point(px, py, centers, radii):
        """Compute max radius of circle centered at (px,py) given existing circles."""
        r = min(px, 1.0 - px, py, 1.0 - py)
        for i in range(len(centers)):
            d = np.sqrt((px - centers[i][0])**2 + (py - centers[i][1])**2)
            r = min(r, d - radii[i])
        return max(r, 0.0)

    def find_largest_empty_circle(centers, radii):
        """Find the center and radius of the largest circle that fits in the voids."""
        if len(centers) == 0:
            return 0.5, 0.5, 0.5

        centers = np.array(centers, dtype=float)
        radii = np.array(radii, dtype=float)
        n = len(centers)

        best_r = 0.0
        best_c = None

        # Candidate 1: Voronoi vertices of centers
        if n >= 3:
            try:
                vor = Voronoi(centers)
                for v in vor.vertices:
                    if np.any(np.isnan(v)):
                        continue
                    if v[0] < 0 or v[0] > 1 or v[1] < 0 or v[1] > 1:
                        continue
                    r = compute_max_radius_at_point(v[0], v[1], centers, radii)
                    if r > best_r:
                        best_r = r
                        best_c = np.array([v[0], v[1]])
            except Exception:
                pass

        # Candidate 2: Boundary points (dense sampling)
        num_samples = 300
        for side in range(4):
            for t in np.linspace(0.001, 0.999, num_samples):
                if side == 0:
                    x, y = t, 0.0
                elif side == 1:
                    x, y = t, 1.0
                elif side == 2:
                    x, y = 0.0, t
                else:
                    x, y = 1.0, t
                r = compute_max_radius_at_point(x, y, centers, radii)
                if r > best_r:
                    best_r = r
                    best_c = np.array([x, y])

        # Candidate 3: Circle-Circle intersection points
        for i in range(n):
            for j in range(i + 1, n):
                c1 = centers[i]
                c2 = centers[j]
                r1 = radii[i]
                r2 = radii[j]
                d = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
                if d < 1e-9 or d > r1 + r2 or d < abs(r1 - r2):
                    continue
                a = (d**2 + r1**2 - r2**2) / (2 * d)
                h2 = r1**2 - a**2
                if h2 < 0:
                    continue
                h = np.sqrt(h2)
                mid_x = c1[0] + a * (c2[0] - c1[0]) / d
                mid_y = c1[1] + a * (c2[1] - c1[1]) / d
                off_x = h * (c2[1] - c1[1]) / d
                off_y = h * (c1[0] - c2[0]) / d
                for p in [np.array([mid_x + off_x, mid_y + off_y]),
                          np.array([mid_x - off_x, mid_y - off_y])]:
                    if p[0] < 0 or p[0] > 1 or p[1] < 0 or p[1] > 1:
                        continue
                    r = compute_max_radius_at_point(p[0], p[1], centers, radii)
                    if r > best_r:
                        best_r = r
                        best_c = p

        # Candidate 4: Delaunay triangle circumcenters (interstitial voids)
        if n >= 3:
            try:
                tri = Delaunay(centers)
                for simplex in tri.simplices:
                    pts = centers[simplex]
                    ax, ay = pts[0]
                    bx, by = pts[1]
                    cx, cy = pts[2]
                    d_val = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
                    if abs(d_val) < 1e-12:
                        continue
                    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d_val
                    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d_val
                    if ux < 0 or ux > 1 or uy < 0 or uy > 1:
                        continue
                    r = compute_max_radius_at_point(ux, uy, centers, radii)
                    if r > best_r:
                        best_r = r
                        best_c = np.array([ux, uy])
            except Exception:
                pass

        if best_c is None:
            best_c = np.array([0.5, 0.5])
            best_r = 0.01

        return best_c[0], best_c[1], max(best_r, 1e-8)

    def greedy_pack(n_circles):
        """Greedy packing: iteratively place largest possible circle in interstitial voids."""
        centers = []
        radii = []

        # Seed with 4 corner circles and 1 center circle
        corner_r = 0.07
        corners = [(corner_r, corner_r), (1 - corner_r, corner_r),
                   (corner_r, 1 - corner_r), (1 - corner_r, 1 - corner_r)]
        for cx, cy in corners:
            centers.append([cx, cy])
            radii.append(corner_r)

        centers.append([0.5, 0.5])
        radii.append(0.10)

        # Greedily add remaining circles by finding largest empty circle
        for _ in range(n_circles - 5):
            cx, cy, r = find_largest_empty_circle(centers, radii)
            if r < 1e-6:
                break
            centers.append([cx, cy])
            radii.append(r)

        return np.array(centers), np.array(radii)

    def energy(params):
        x_ = params[0::3]
        y_ = params[1::3]
        r_ = params[2::3]

        pen_bound = 0.0
        for i in range(N):
            pen_bound += max(0.0, r_[i] - x_[i])**2
            pen_bound += max(0.0, r_[i] - (1.0 - x_[i]))**2
            pen_bound += max(0.0, r_[i] - y_[i])**2
            pen_bound += max(0.0, r_[i] - (1.0 - y_[i]))**2

        pen_overlap = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dist_sq = (x_[i] - x_[j])**2 + (y_[i] - y_[j])**2
                min_dist = r_[i] + r_[j]
                if dist_sq < min_dist**2:
                    overlap = min_dist - np.sqrt(dist_sq)
                    pen_overlap += overlap**2

        lam = 100.0
        return -np.sum(r_) + lam * (pen_bound + pen_overlap)

    def optimize(c_init, r_init):
        params0 = np.zeros(3 * N)
        params0[0::3] = c_init[:, 0]
        params0[1::3] = c_init[:, 1]
        params0[2::3] = r_init

        bounds = []
        for i in range(N):
            bounds.append((0, 1))
            bounds.append((0, 1))
            bounds.append((0, 0.5))

        res = minimize(energy, params0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})

        params_opt = res.x
        x_opt = params_opt[0::3]
        y_opt = params_opt[1::3]
        r_opt = params_opt[2::3]

        # Post-processing: Ensure strict validity
        for _ in range(30):
            changed = False
            for i in range(N):
                r_max_bound = min(x_opt[i], 1 - x_opt[i], y_opt[i], 1 - y_opt[i])
                if r_opt[i] > r_max_bound:
                    r_opt[i] = r_max_bound
                    changed = True
            for i in range(N):
                for j in range(i + 1, N):
                    dist = np.sqrt((x_opt[i] - x_opt[j])**2 + (y_opt[i] - y_opt[j])**2)
                    min_dist = r_opt[i] + r_opt[j]
                    if dist < min_dist:
                        if min_dist > 1e-9:
                            scale = dist / min_dist
                            r_opt[i] *= scale
                            r_opt[j] *= scale
                            changed = True
            if not changed:
                break

        return x_opt, y_opt, r_opt

    # Step 1: Greedy packing using interstitial void identification
    c_init, r_init = greedy_pack(N)

    # Step 2: Optimize from greedy start
    x_opt, y_opt, r_opt = optimize(c_init, r_init)
    best_score = np.sum(r_opt)
    best_x, best_y, best_r = x_opt, y_opt, r_opt

    # Step 3: Multi-start with perturbations
    np.random.seed(42)

    for i in range(15):
        c_pert = c_init + np.random.normal(0, 0.03, (N, 2))
        c_pert = np.clip(c_pert, 0.01, 0.99)
        r_pert = np.clip(r_init * 0.9, 0.01, 0.4)

        x_p, y_p, r_p = optimize(c_pert, r_pert)
        score = np.sum(r_p)
        if score > best_score:
            best_score = score
            best_x, best_y, best_r = x_p, y_p, r_p

    # Step 4: Try different initial configurations with varying corner radii
    for trial in range(6):
        corner_r = 0.04 + trial * 0.015
        centers = []
        radii = []
        corners = [(corner_r, corner_r), (1 - corner_r, corner_r),
                   (corner_r, 1 - corner_r), (1 - corner_r, 1 - corner_r)]
        for cx, cy in corners:
            centers.append([cx, cy])
            radii.append(corner_r)

        # Add center
        centers.append([0.5, 0.5])
        radii.append(0.08)

        # Greedily fill rest
        while len(centers) < N:
            cx, cy, r = find_largest_empty_circle(centers, radii)
            if r < 1e-6:
                break
            centers.append([cx, cy])
            radii.append(r)

        if len(centers) < N:
            continue

        c_init2 = np.array(centers[:N])
        r_init2 = np.array(radii[:N])

        x_p, y_p, r_p = optimize(c_init2, r_init2)
        score = np.sum(r_p)
        if score > best_score:
            best_score = score
            best_x, best_y, best_r = x_p, y_p, r_p

    result = np.zeros((N, 3))
    result[:, 0] = best_x
    result[:, 1] = best_y
    result[:, 2] = best_r

    return result
