import math
import time
import random
import numpy as np
from itertools import combinations, product

N = 11

def construct():
    deadline = time.time() + 148.0
    best_S = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    best_valid = False

    # Precompute hex vertices for side 1, angle 0
    base_verts = np.array([[math.cos(math.radians(60 * i)), math.sin(math.radians(60 * i))] for i in range(6)])

    def get_hex_vertices(cx, cy, angle_deg):
        a = math.radians(angle_deg)
        ca, sa = math.cos(a), math.sin(a)
        R = np.array([[ca, -sa], [sa, ca]])
        return base_verts @ R.T + np.array([cx, cy])

    def compute_required_side(inner, ocx, ocy, oangle):
        """Compute the minimum outer hexagon side length needed to contain all inner hexagons."""
        inner_verts = []
        for (ix, iy, iangle) in inner:
            inner_verts.append(get_hex_vertices(ix, iy, iangle))
        all_verts = np.vstack(inner_verts)
        
        a = math.radians(oangle)
        apothem_factor = math.sqrt(3) / 2
        max_S = 0.0
        for i in range(6):
            theta = a + math.radians(30 + 60 * i)
            nx, ny = math.cos(theta), math.sin(theta)
            n_vec = np.array([nx, ny])
            max_proj = np.max((all_verts - np.array([ocx, ocy])) @ n_vec)
            if max_proj > 0:
                S_needed = max_proj / apothem_factor
                if S_needed > max_S:
                    max_S = S_needed
        return max_S

    def compute_required_side_smooth(inner, ocx, ocy, oangle, beta=100.0):
        """Smooth surrogate using log-sum-exp for the max projection."""
        inner_verts = []
        for (ix, iy, iangle) in inner:
            inner_verts.append(get_hex_vertices(ix, iy, iangle))
        all_verts = np.vstack(inner_verts)
        
        a = math.radians(oangle)
        apothem_factor = math.sqrt(3) / 2
        max_S = 0.0
        for i in range(6):
            theta = a + math.radians(30 + 60 * i)
            nx, ny = math.cos(theta), math.sin(theta)
            n_vec = np.array([nx, ny])
            projs = (all_verts - np.array([ocx, ocy])) @ n_vec
            # Log-sum-exp for smooth max
            max_p = np.max(projs)
            if max_p > -1e-10:
                lse = max_p + math.log(np.sum(np.exp((projs - max_p) * beta)) / len(projs)) / beta
                S_needed = lse / apothem_factor
                if S_needed > max_S:
                    max_S = S_needed
        return max_S

    def sat_overlap_check(verts_a, verts_b):
        """Check if two convex polygons overlap using SAT."""
        for verts in [verts_a, verts_b]:
            for i in range(len(verts)):
                p1 = verts[i]
                p2 = verts[(i + 1) % len(verts)]
                # Normal to edge
                dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                nx, ny = -dy, dx
                # Project both polygons onto this normal
                proj_a = (verts_a - p1) @ np.array([nx, ny])
                proj_b = (verts_b - p1) @ np.array([nx, ny])
                if np.max(proj_a) < np.min(proj_b) or np.max(proj_b) < np.min(proj_a):
                    return False
        return True

    def check_all_disjoint(inner):
        """Check if all inner hexagons are pairwise disjoint."""
        verts_list = []
        for (ix, iy, iangle) in inner:
            verts_list.append(get_hex_vertices(ix, iy, iangle))
        for i in range(N):
            for j in range(i + 1, N):
                if sat_overlap_check(verts_list[i], verts_list[j]):
                    return False
        return True

    def check_inside(inner, ocx, ocy, S, oangle):
        """Check if all inner hexagons are inside the outer hexagon."""
        # Get outer hexagon vertices
        outer_verts = get_hex_vertices(ocx, ocy, oangle) * S
        
        # For each inner hexagon vertex, check if it's inside the outer hexagon
        for (ix, iy, iangle) in inner:
            inner_verts = get_hex_vertices(ix, iy, iangle)
            for v in inner_verts:
                # Point in convex polygon check using cross products
                inside = True
                for i in range(6):
                    p1 = outer_verts[i]
                    p2 = outer_verts[(i + 1) % 6]
                    cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                    if cross < -1e-9:
                        inside = False
                        break
                if not inside:
                    return False
        return True

    def check_validity(inner, ocx, ocy, S, oangle, tol=1e-6):
        """Full validity check: disjoint and inside."""
        if not check_all_disjoint(inner):
            return False
        # Use slightly larger S to account for numerical tolerance
        if not check_inside(inner, ocx, ocy, S * (1 + tol), oangle):
            return False
        return True

    def objective(params, beta=50.0):
        """
        Objective function for optimization.
        params: [x1, y1, a1, x2, y2, a2, ..., x11, y11, a11, ocx, ocy, oangle]
        Returns smooth outer side length + penalty for overlaps.
        """
        inner = []
        for i in range(N):
            inner.append([params[3*i], params[3*i+1], params[3*i+2]])
        ocx, ocy, oangle = params[3*N], params[3*N+1], params[3*N+2]
        
        S = compute_required_side_smooth(inner, ocx, ocy, oangle, beta)
        
        # Penalty for overlaps
        penalty = 0.0
        verts_list = []
        for (ix, iy, iangle) in inner:
            verts_list.append(get_hex_vertices(ix, iy, iangle))
        for i in range(N):
            for j in range(i + 1, N):
                if sat_overlap_check(verts_list[i], verts_list[j]):
                    # Compute minimum translation distance (approximate)
                    # Use center-to-center distance minus minimum distance
                    ca = np.array([inner[i][0], inner[i][1]])
                    cb = np.array([inner[j][0], inner[j][1]])
                    dist = np.linalg.norm(ca - cb)
                    min_dist = math.sqrt(3)  # minimum center distance for touching hexagons
                    if dist < min_dist:
                        penalty += (min_dist - dist) ** 2
        return S + 10.0 * penalty

    def nelder_mead_smooth(objective, x0, max_iter=500, tol=1e-10):
        """Nelder-Mead optimization with smooth objective."""
        n = len(x0)
        simplex = [x0.copy()]
        for i in range(n):
            trial = x0.copy()
            trial[i] += 0.1 if trial[i] == 0 else 0.1 * trial[i]
            simplex.append(trial)
        
        fvals = [objective(simplex[i]) for i in range(n + 1)]
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            # Sort by function value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            if abs(fvals[-1] - fvals[0]) < tol:
                break
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + 1.0 * (centroid - simplex[-1])
            fr = objective(xr)
            
            if fr < fvals[0]:
                # Expansion
                xe = centroid + 2.0 * (centroid - simplex[-1])
                fe = objective(xe)
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                xc = centroid + 0.5 * (simplex[-1] - centroid)
                fc = objective(xc)
                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        fvals[i] = objective(simplex[i])
        
        order = np.argsort(fvals)
        return simplex[order[0]], fvals[order[0]]

    def local_search(inner, ocx, ocy, oangle, seed):
        nonlocal best_S, best_inner, best_center, best_angle, best_valid
        rng = random.Random(seed)
        cur_inner = [list(x) for x in inner]
        cur_S = compute_required_side(cur_inner, ocx, ocy, oangle)
        cur_valid = check_validity(cur_inner, ocx, ocy, cur_S, oangle)
        
        if not cur_valid:
            for _ in range(50):
                idx = rng.randrange(N)
                old_x, old_y, old_a = cur_inner[idx]
                cur_inner[idx][0] = old_x + rng.gauss(0, 0.5)
                cur_inner[idx][1] = old_y + rng.gauss(0, 0.5)
                cur_S = compute_required_side(cur_inner, ocx, ocy, oangle)
                cur_valid = check_validity(cur_inner, ocx, ocy, cur_S, oangle)
                if cur_valid:
                    break
            if not cur_valid:
                return
        
        if cur_valid and cur_S < best_S:
            best_S = cur_S
            best_inner = [list(x) for x in cur_inner]
            best_center = (ocx, ocy)
            best_angle = oangle
            best_valid = True
        
        for it in range(2000):
            if time.time() > deadline:
                break
            old = [list(x) for x in cur_inner]
            idx = rng.randrange(N)
            cur_inner[idx][0] += rng.gauss(0, 0.15)
            cur_inner[idx][1] += rng.gauss(0, 0.15)
            if rng.random() < 0.03:
                cur_inner[idx][2] = rng.choice([0, 30, 60, 90, 120, 150])
            
            new_S = compute_required_side(cur_inner, ocx, ocy, oangle)
            new_valid = check_validity(cur_inner, ocx, ocy, new_S, oangle)
            
            if new_valid and new_S < cur_S:
                cur_S = new_S
                if new_S < best_S:
                    best_S = new_S
                    best_inner = [list(x) for x in cur_inner]
                    best_center = (ocx, ocy)
                    best_angle = oangle
                    best_valid = True
            elif new_valid:
                if rng.random() < 0.1:
                    cur_S = new_S
            else:
                cur_inner = old

    def smooth_optimization(inner_init, ocx_init, ocy_init, oangle_init, seed, beta_start=100.0, beta_end=200.0):
        """Run smooth optimization with annealing beta."""
        nonlocal best_S, best_inner, best_center, best_angle, best_valid
        rng = random.Random(seed)
        
        # Build initial parameter vector
        params = []
        for (ix, iy, iangle) in inner_init:
            params.extend([ix, iy, iangle])
        params.extend([ocx_init, ocy_init, oangle_init])
        
        # Anneal beta
        for beta in [beta_start, beta_start * 1.5, beta_end]:
            if time.time() > deadline:
                break
            x_opt, f_opt = nelder_mead_smooth(lambda p: objective(p, beta), params, max_iter=300)
            params = x_opt
            
            # Extract and check
            inner = []
            for i in range(N):
                inner.append([params[3*i], params[3*i+1], params[3*i+2]])
            ocx, ocy, oangle = params[3*N], params[3*N+1], params[3*N+2]
            
            S = compute_required_side(inner, ocx, ocy, oangle)
            if check_validity(inner, ocx, ocy, S, oangle):
                if S < best_S:
                    best_S = S
                    best_inner = [list(x) for x in inner]
                    best_center = (ocx, ocy)
                    best_angle = oangle
                    best_valid = True
                    # Run local search to refine
                    local_search(inner, ocx, ocy, oangle, seed)
                # Continue from this point
            else:
                # Try to repair
                for _ in range(20):
                    if time.time() > deadline:
                        break
                    idx = rng.randrange(N)
                    inner[idx][0] += rng.gauss(0, 0.3)
                    inner[idx][1] += rng.gauss(0, 0.3)
                    S = compute_required_side(inner, ocx, ocy, oangle)
                    if check_validity(inner, ocx, ocy, S, oangle):
                        if S < best_S:
                            best_S = S
                            best_inner = [list(x) for x in inner]
                            best_center = (ocx, ocy)
                            best_angle = oangle
                            best_valid = True
                        break

    # Generate initial configurations
    def generate_initial_configs():
        configs = []
        sp = math.sqrt(3)
        
        # Config 1: Central hex + 6 ring + 4 more
        cfg1 = [[0.0, 0.0, 0.0]]
        for i in range(6):
            theta = math.radians(60 * i)
            cfg1.append([sp * math.cos(theta), sp * math.sin(theta), 0.0])
        for i in [0, 2, 4]:
            theta = math.radians(60 * i + 30)
            cfg1.append([2 * sp * math.cos(theta), 2 * sp * math.sin(theta), 0.0])
        for i in [1, 3, 5]:
            theta = math.radians(60 * i)
            cfg1.append([2 * sp * math.cos(theta), 2 * sp * math.sin(theta), 0.0])
        configs.append(cfg1[:N])
        
        # Config 2: Linear arrangement
        cfg2 = []
        for i in range(N):
            cfg2.append([(i - N // 2) * sp, 0.0, 0.0])
        configs.append(cfg2)
        
        # Config 3: 2D grid-like
        cfg3 = []
        positions = [(-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1), (0, 2), (1, 2)]
        for (i, j) in positions[:N]:
            x = i * sp
            y = j * sp * math.sqrt(3)
            cfg3.append([x, y, 0.0])
        configs.append(cfg3)
        
        # Config 4: Compact hexagonal cluster
        cfg4 = [[0.0, 0.0, 0.0]]
        for i in range(6):
            theta = math.radians(60 * i)
            cfg4.append([sp * math.cos(theta), sp * math.sin(theta), 0.0])
        for i in range(4):
            theta = math.radians(60 * i + 30)
            cfg4.append([sp * math.sqrt(3) * math.cos(theta), sp * math.sqrt(3) * math.sin(theta), 0.0])
        configs.append(cfg4[:N])
        
        # Config 5: Asymmetric
        rng = random.Random(42)
        cfg5 = [[0.0, 0.0, 0.0]]
        for i in range(N - 1):
            x = rng.gauss(0, 1.5)
            y = rng.gauss(0, 1.5)
            a = rng.choice([0, 30, 60])
            cfg5.append([x, y, a])
        configs.append(cfg5)
        
        return configs

    initial_configs = generate_initial_configs()
    
    # Run smooth optimization from each initial config
    seed_counter = 100
    for cfg in initial_configs:
        if time.time() > deadline:
            break
        # Center the configuration
        cx = sum(p[0] for p in cfg) / N
        cy = sum(p[1] for p in cfg) / N
        centered = [[p[0] - cx, p[1] - cy, p[2]] for p in cfg]
        
        for oangle in [0, 30, 60]:
            if time.time() > deadline:
                break
            smooth_optimization(centered, 0.0, 0.0, oangle, seed_counter)
            seed_counter += 1

    # Also run local search from initial configs as fallback
    for cfg in initial_configs:
        if time.time() > deadline:
            break
        cx = sum(p[0] for p in cfg) / N
        cy = sum(p[1] for p in cfg) / N
        centered = [[p[0] - cx, p[1] - cy, p[2]] for p in cfg]
        for oangle in [0, 30, 60]:
            if time.time() > deadline:
                break
            local_search(centered, 0.0, 0.0, oangle, seed_counter)
            seed_counter += 1

    # Final polish: high-precision local search on best solution
    if best_valid and best_inner is not None:
        if time.time() < deadline - 5.0:
            local_search(best_inner, best_center[0], best_center[1], best_angle, 99999)

    if not best_valid or best_inner is None:
        sp = math.sqrt(3)
        fallback_inner = []
        for k in range(N):
            fallback_inner.append([(k - N // 2) * sp, 0.0, 0.0])
        S = compute_required_side(fallback_inner, 0.0, 0.0, 0.0)
        for _ in range(100):
            if check_validity(fallback_inner, 0.0, 0.0, S, 0.0):
                return fallback_inner, (0.0, 0.0), S, 0.0
            for k in range(N):
                fallback_inner[k][0] *= 1.05
            S = compute_required_side(fallback_inner, 0.0, 0.0, 0.0)
        return fallback_inner, (0.0, 0.0), S, 0.0

    return best_inner, best_center, best_S, best_angle
