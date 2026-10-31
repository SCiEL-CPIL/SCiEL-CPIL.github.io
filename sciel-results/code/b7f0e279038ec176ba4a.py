"""Circle packing in perimeter-4 rectangle using multi-start simulated annealing."""
import numpy as np
import time
import random

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def objective(circles, W, H):
        """
        Objective function to maximize: sum of radii - penalty for violations.
        Violations: circles outside rectangle, overlapping circles.
        """
        n = len(circles)
        sum_r = 0.0
        penalty = 0.0

        for i in range(n):
            x, y, r = circles[i]
            if r < 0:
                r = 0
            sum_r += r

            # Wall violations
            if x < r:
                penalty += (r - x) ** 2
            if x > W - r:
                penalty += (x - (W - r)) ** 2
            if y < r:
                penalty += (r - y) ** 2
            if y > H - r:
                penalty += (y - (H - r)) ** 2

        # Overlap violations
        for i in range(n):
            x1, y1, r1 = circles[i]
            if r1 < 0:
                r1 = 0
            for j in range(i + 1, n):
                x2, y2, r2 = circles[j]
                if r2 < 0:
                    r2 = 0
                dx = x1 - x2
                dy = y1 - y2
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r1 + r2
                if dist < min_dist:
                    overlap = min_dist - dist
                    penalty += overlap ** 2

        return sum_r - penalty

    def is_valid(circles, W, H, tol=1e-7):
        """Check if packing is valid: no overlaps, all inside rectangle."""
        n = len(circles)
        for i in range(n):
            x, y, r = circles[i]
            if r < -tol:
                return False
            if x < r - tol or x > W - r + tol:
                return False
            if y < r - tol or y > H - r + tol:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                dx = circles[i][0] - circles[j][0]
                dy = circles[i][1] - circles[j][1]
                dist_sq = dx * dx + dy * dy
                min_dist = circles[i][2] + circles[j][2]
                if dist_sq < min_dist * min_dist - tol:
                    return False
        return True

    def random_initial(W, H):
        """Generate a random initial configuration."""
        circles = []
        for i in range(N):
            r = random.uniform(0.01, 0.1)
            x = random.uniform(r, W - r)
            y = random.uniform(r, H - r)
            circles.append([x, y, r])
        return np.array(circles)

    def simulated_annealing(circles, W, H, time_limit):
        """
        Simulated annealing to optimize the packing.
        """
        circles = np.array(circles, dtype=float)
        n = len(circles)
        current_obj = objective(circles, W, H)
        best_circles = circles.copy()
        best_obj = current_obj

        T = 0.1  # Initial temperature
        T_min = 0.0001
        alpha = 0.995  # Cooling rate
        max_steps = 50000

        t0 = time.time()
        step = 0

        while step < max_steps:
            if time.time() > time_limit:
                break

            # Choose a random circle to perturb
            i = random.randint(0, n - 1)

            # Perturb position and radius
            dx = random.gauss(0, 0.02)
            dy = random.gauss(0, 0.02)
            dr = random.gauss(0, 0.01)

            new_x = circles[i, 0] + dx
            new_y = circles[i, 1] + dy
            new_r = circles[i, 2] + dr

            # Clamp to valid bounds
            new_x = np.clip(new_x, 0, W)
            new_y = np.clip(new_y, 0, H)
            new_r = max(0.001, new_r)

            # Save old values
            old_x, old_y, old_r = circles[i, 0], circles[i, 1], circles[i, 2]

            # Apply perturbation
            circles[i, 0] = new_x
            circles[i, 1] = new_y
            circles[i, 2] = new_r

            new_obj = objective(circles, W, H)
            delta_obj = new_obj - current_obj

            # Acceptance criterion
            if delta_obj > 0 or random.random() < np.exp(delta_obj / T):
                current_obj = new_obj
                if current_obj > best_obj:
                    best_obj = current_obj
                    best_circles = circles.copy()
            else:
                # Reject: revert
                circles[i, 0] = old_x
                circles[i, 1] = old_y
                circles[i, 2] = old_r

            # Cool down
            T *= alpha
            if T < T_min:
                T = T_min

            step += 1

        return best_circles, best_obj

    def refine_valid(circles, W, H, time_limit):
        """
        Refine a valid packing by growing circles as much as possible.
        """
        circles = list(circles)
        n = len(circles)
        t0 = time.time()

        for _ in range(100):
            if time.time() > time_limit:
                break

            improved = False
            for i in range(n):
                if time.time() > time_limit:
                    break

                x, y, r = circles[i]
                old_r = r
                r_new = r * 1.01

                # Check if we can grow without moving
                valid = True
                for j in range(n):
                    if i == j:
                        continue
                    dx = x - circles[j][0]
                    dy = y - circles[j][1]
                    dist_sq = dx * dx + dy * dy
                    min_dist = r_new + circles[j][2]
                    if dist_sq < min_dist * min_dist - 1e-8:
                        valid = False
                        break
                if x < r_new - 1e-8 or x > W - r_new + 1e-8:
                    valid = False
                if y < r_new - 1e-8 or y > H - r_new + 1e-8:
                    valid = False

                if valid:
                    circles[i] = [x, y, r_new]
                    improved = True

            if not improved:
                break

        return circles

    # Main optimization loop
    random.seed(42)
    np.random.seed(42)

    best_sum_r = 0.0
    best_config = None
    best_W = 1.0
    best_H = 1.0

    # Try different aspect ratios
    aspect_ratios = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]

    for W_ratio in aspect_ratios:
        if time.time() > deadline - 30:
            break

        W = W_ratio
        H = 2.0 - W
        if W <= 0.3 or H <= 0.3:
            continue

        # Multiple restarts for simulated annealing
        for trial in range(5):
            if time.time() > deadline - 10:
                break

            # Generate random initial configuration
            circles = random_initial(W, H)

            # Run simulated annealing
            circles, obj = simulated_annealing(circles, W, H, time_limit=deadline - 5)

            # Check if the result is valid
            if not is_valid(circles, W, H, tol=1e-6):
                continue

            # Refine the valid packing
            circles = refine_valid(circles, W, H, time_limit=deadline - 5)

            # Validate again
            if not is_valid(circles, W, H, tol=1e-6):
                continue

            total_r = sum(c[2] for c in circles)
            if total_r > best_sum_r:
                best_sum_r = total_r
                best_config = np.array(circles)
                best_W = W
                best_H = H

    # If we haven't found a good solution, try a simple grid fallback
    if best_config is None or len(best_config) < N:
        # Simple 5x5 grid in square
        W = 1.0
        H = 1.0
        cells = []
        for i in range(5):
            for j in range(5):
                if len(cells) >= N:
                    break
                x = (i + 0.5) / 5
                y = (j + 0.5) / 5
                r = 0.5 / 5 * 0.8
                cells.append([x, y, r])
        best_config = np.array(cells[:N])
        best_W = W
        best_H = H

    return best_config

# EVOLVE-BLOCK-END
