# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

def heilbronn_convex13():
    """
    Maximize the ratio of the smallest triangle area to the convex hull area
    for 13 points using differential evolution.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s kill

    # Precompute triangle indices
    tri_indices = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                tri_indices.append((i, j, k))
    tri_indices = np.array(tri_indices)
    n_tri = len(tri_indices)

    def calc_areas(points):
        p1 = points[tri_indices[:, 0]]
        p2 = points[tri_indices[:, 1]]
        p3 = points[tri_indices[:, 2]]
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        return np.abs(cross) * 0.5

    def calc_hull_area(points):
        try:
            hull = ConvexHull(points)
            return hull.volume
        except Exception:
            return 0.0

    def evaluate(points):
        if not np.all(np.isfinite(points)):
            return 0.0
        areas = calc_areas(points)
        min_area = np.min(areas)
        if min_area <= 0:
            return 0.0
        hull_area = calc_hull_area(points)
        if hull_area <= 0:
            return 0.0
        return min_area / hull_area

    def neg_score(points):
        """Objective for minimization (DE minimizes)."""
        return -evaluate(points)

    def init_hex_grid():
        """Generate a hexagonal lattice initialization."""
        points = []
        s = 0.22
        for row in range(-3, 4):
            for col in range(-3, 4):
                x = 0.5 + col * s + (row % 2) * s / 2
                y = 0.5 + row * s * np.sqrt(3) / 2
                points.append([x, y])
        points = np.array(points)
        mask = (points[:, 0] - 0.5)**2 + (points[:, 1] - 0.5)**2 <= 0.48**2
        points = points[mask]
        if len(points) > n:
            # Select n most evenly spaced points
            # Use a simple greedy approach
            selected = [0]
            for _ in range(n - 1):
                best_idx = -1
                best_min_dist = -1
                for idx in range(len(points)):
                    if idx in selected:
                        continue
                    min_dist = min(np.linalg.norm(points[idx] - points[si]) for si in selected)
                    if min_dist > best_min_dist:
                        best_min_dist = min_dist
                        best_idx = idx
                selected.append(best_idx)
            points = points[selected]
        while len(points) < n:
            p = np.random.random(2) * 0.8 + 0.1
            points = np.vstack([points, p])
        return points[:n]

    def init_circle():
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        x = 0.5 + 0.4 * np.cos(angles)
        y = 0.5 + 0.4 * np.sin(angles)
        return np.column_stack((x, y))

    def init_random():
        return np.random.random((n, 2)) * 0.8 + 0.1

    # --- Differential Evolution ---
    def differential_evolution(initial_points, pop_size=40, max_gen=200,
                                F=0.8, CR=0.9, mutation='rand1bin'):
        """
        Run differential evolution on the 26-dimensional space (13 points x 2 coords).
        """
        dim = n * 2
        # Flatten initial points
        x0 = initial_points.flatten()
        
        # Create initial population centered around x0
        pop = np.zeros((pop_size, dim))
        pop[0] = x0
        for i in range(1, pop_size):
            pop[i] = x0 + (np.random.random(dim) - 0.5) * 0.3
            # Clip to [0, 1]
            np.clip(pop[i], 0, 1, out=pop[i])
        
        # Evaluate initial population
        pop_scores = np.array([neg_score(pop[i].reshape(n, 2)) for i in range(pop_size)])
        best_idx = np.argmin(pop_scores)
        best_x = pop[best_idx].copy()
        best_score = pop_scores[best_idx]
        
        for gen in range(max_gen):
            if time.time() > deadline - 10:
                break
            
            # Adaptive F and CR
            F_gen = F * (0.5 + 0.5 * np.random.random())
            CR_gen = CR * (0.5 + 0.5 * np.random.random())
            
            new_pop = pop.copy()
            new_pop_scores = pop_scores.copy()
            
            for i in range(pop_size):
                if time.time() > deadline - 5:
                    break
                
                # Select three distinct random indices different from i
                r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
                while r1 == i or r2 == i or r3 == i:
                    r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
                
                # Mutation: v = x_r1 + F * (x_r2 - x_r3)
                v = pop[r1] + F_gen * (pop[r2] - pop[r3])
                
                # Crossover
                trial = pop[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.random() < CR_gen or j == j_rand:
                        trial[j] = v[j]
                
                # Clip to [0, 1]
                np.clip(trial, 0, 1, out=trial)
                
                # Evaluate
                trial_score = neg_score(trial.reshape(n, 2))
                
                # Selection
                if trial_score <= pop_scores[i]:
                    new_pop[i] = trial
                    new_pop_scores[i] = trial_score
                    if trial_score < best_score:
                        best_score = trial_score
                        best_x = trial.copy()
            
            pop = new_pop
            pop_scores = new_pop_scores
            
            # Optional: Add some random restarts if stagnating
            if gen % 20 == 0 and gen > 0:
                # Replace worst individual with a random one
                worst_idx = np.argmax(pop_scores)
                pop[worst_idx] = np.random.random(dim) * 0.8 + 0.1
                pop_scores[worst_idx] = neg_score(pop[worst_idx].reshape(n, 2))
        
        return best_x.reshape(n, 2), -best_score

    # --- Local Search (coordinate-wise, for final polishing) ---
    def local_search(points, max_iters=500):
        current_points = points.copy()
        current_score = evaluate(current_points)
        step = 0.01
        min_step = 1e-6
        stall = 0
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            improved = False
            order = np.random.permutation(n)
            for i in order:
                for axis in [0, 1]:
                    for direction in [1, -1]:
                        new_point = current_points.copy()
                        new_point[i, axis] += direction * step
                        np.clip(new_point, 0, 1, out=new_point)
                        new_score = evaluate(new_point)
                        if new_score > current_score:
                            current_points = new_point
                            current_score = new_score
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                stall += 1
                if stall > 5:
                    step *= 0.7
                    stall = 0
                    if step < min_step:
                        break
        
        return current_points, current_score

    # --- Main optimization loop ---
    best_points = None
    best_score = 0.0

    # Generate diverse initial candidates
    np.random.seed(42)
    candidates = [
        init_hex_grid(),
        init_circle(),
        init_random(),
        init_random(),
        init_random()
    ]

    # Evaluate initial candidates
    scored_cands = []
    for cand in candidates:
        if not np.all(np.isfinite(cand)):
            continue
        s = evaluate(cand)
        scored_cands.append((s, cand))
        if s > best_score:
            best_score = s
            best_points = cand.copy()

    # Sort by score descending
    scored_cands.sort(key=lambda x: -x[0])

    # Run DE on top candidates
    for s, cand in scored_cands[:3]:
        if time.time() > deadline - 30:
            break
        de_points, de_score = differential_evolution(cand, pop_size=30, max_gen=100)
        if de_score > best_score:
            best_score = de_score
            best_points = de_points.copy()

    # Final local search polishing
    if best_points is not None and time.time() < deadline - 10:
        final_pts, final_score = local_search(best_points, max_iters=300)
        if final_score > best_score:
            best_points = final_pts
            best_score = final_score

    if best_points is None:
        best_points = init_circle()

    return best_points
# EVOLVE-BLOCK-END
