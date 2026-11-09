"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Surrogate-guided multi-start optimizer with coordinate line search and adaptive CMA-ES."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 310.0

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        eval_count = 0

        def check_time():
            return time.time() < deadline - 3.0

        def try_update(x, fx):
            nonlocal best_x, best_f, eval_count
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        def clip_box(x):
            return np.clip(x, -5.0, 5.0)

        # ===== Phase 1: Structured initialization =====
        n_init = min(15, budget - 20)
        init_points = []
        init_points.append(np.zeros(n))
        for _ in range(max(1, n_init // 3)):
            init_points.append(rng.uniform(-4.0, 4.0, n))
        for _ in range(max(1, n_init // 3)):
            init_points.append(rng.uniform(-2.0, 2.0, n))
        for axis in range(min(n, 3)):
            pt = np.zeros(n)
            pt[axis] = rng.choice([-3.0, 3.0])
            init_points.append(pt)

        for pt in init_points[:n_init]:
            if not check_time() or eval_count >= budget:
                break
            pt = clip_box(pt)
            fv = func(pt)
            try_update(pt, fv)

        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            try_update(best_x, best_f)

        # ===== Phase 2: Coordinate line search from best =====
        if check_time() and eval_count < budget:
            step_sizes = [2.0, 1.0, 0.5, 0.25, 0.125]
            for coord in range(n):
                for step in step_sizes:
                    if not check_time() or eval_count >= budget:
                        break
                    # Try positive and negative directions
                    for direction in [-1, 1]:
                        if not check_time() or eval_count >= budget:
                            break
                        x_try = best_x.copy()
                        x_try[coord] = clip_box(np.array(best_x)[coord] + direction * step)
                        f_try = func(x_try)
                        try_update(x_try, f_try)
                        if f_try < best_f - 1e-10:
                            best_x = x_try.copy()
                            best_f = f_try

        # ===== Phase 3: Adaptive (1+lambda)-ES with restarts =====
        lambda_pop = 6
        sigma = 0.5
        no_improve = 0
        max_no_improve = 40
        sigma_min = 1e-6
        sigma_max = 5.0

        # Population: list of (x, f)
        population = [(best_x.copy(), best_f)]
        for _ in range(min(4, budget - eval_count - 5)):
            if not check_time() or eval_count >= budget:
                break
            x_div = clip_box(best_x + rng.standard_normal(n) * 1.0)
            f_div = func(x_div)
            try_update(x_div, f_div)
            population.append((x_div, f_div))
        population.sort(key=lambda t: t[1])

        while check_time() and eval_count < budget:
            n_offspring = min(lambda_pop, budget - eval_count)
            offspring = []
            for _ in range(n_offspring):
                if not check_time() or eval_count >= budget:
                    break
                if rng.random() < 0.5 and len(population) > 0:
                    parent = population[0][0]
                elif len(population) > 1:
                    parent = population[rng.integers(0, min(3, len(population)))][0]
                else:
                    parent = best_x
                child = clip_box(parent + sigma * rng.standard_normal(n))
                f_child = func(child)
                try_update(child, f_child)
                offspring.append((child, f_child))

            if not offspring:
                break

            population.extend(offspring)
            population.sort(key=lambda t: t[1])
            max_pop_size = min(12, budget - eval_count - lambda_pop)
            if max_pop_size < 1:
                max_pop_size = 1
            population = population[:max_pop_size]

            if population[0][1] < best_f - 1e-10:
                best_f = population[0][1]
                best_x = population[0][0].copy()
                no_improve = 0
                sigma = min(sigma * 1.1, sigma_max)
            else:
                no_improve += 1
                sigma = max(sigma * 0.93, sigma_min)

            # Diversity injection
            if no_improve > 25 and eval_count < budget - 10:
                if not check_time():
                    break
                x_inject = clip_box(rng.uniform(-4.0, 4.0, n))
                f_inject = func(x_inject)
                try_update(x_inject, f_inject)
                population.append((x_inject, f_inject))
                population.sort(key=lambda t: t[1])
                population = population[:max_pop_size]
                no_improve = 0
                sigma = max(sigma * 0.7, sigma_min)

            # Large jump on severe stagnation
            if no_improve > 35 and sigma < 0.05 and eval_count < budget - 15:
                if not check_time():
                    break
                x_jump = clip_box(best_x + rng.standard_normal(n) * 2.5)
                f_jump = func(x_jump)
                try_update(x_jump, f_jump)
                if f_jump < population[0][1]:
                    population[0] = (x_jump, f_jump)
                    best_f = f_jump
                    best_x = x_jump.copy()
                    no_improve = 0
                    sigma = 1.0
                else:
                    x_rand = clip_box(rng.uniform(-4.0, 4.0, n))
                    f_rand = func(x_rand)
                    try_update(x_rand, f_rand)
                    if f_rand < population[0][1]:
                        population[0] = (x_rand, f_rand)
                        best_f = f_rand
                        best_x = x_rand.copy()
                        no_improve = 0
                        sigma = 1.0
                    else:
                        no_improve = 0
                        sigma = max(sigma * 1.5, 0.5)

        # ===== Phase 4: Fine-tuning =====
        if check_time() and eval_count < budget:
            sigma_fine = max(sigma * 0.1, 0.005)
            x_fine = best_x.copy()
            f_fine = best_f
            no_improve_fine = 0
            max_fine_steps = min(25, budget - eval_count)
            for _ in range(max_fine_steps):
                if not check_time() or eval_count >= budget:
                    break
                y = clip_box(x_fine + sigma_fine * rng.standard_normal(n))
                fy = func(y)
                try_update(y, fy)
                if fy < f_fine - 1e-12:
                    x_fine = y
                    f_fine = fy
                    no_improve_fine = 0
                    sigma_fine = min(sigma_fine * 1.05, 0.5)
                else:
                    sigma_fine *= 0.92
                    no_improve_fine += 1
                if sigma_fine < sigma_min or no_improve_fine > 15:
                    break
            best_x = x_fine
            best_f = f_fine

        return best_x, best_f
# EVOLVE-BLOCK-END
