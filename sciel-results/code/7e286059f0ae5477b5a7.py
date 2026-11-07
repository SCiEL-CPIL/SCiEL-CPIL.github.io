import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0

        try:
            # CMA-ES Parameters for dim=5
            lam = 4 + int(3 * np.log(dim))  # ~10 for dim=5
            mu = lam // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1.0 / np.sum(weights**2)

            c_sigma = (mueff + 2) / (dim + mueff + 5)
            c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
            dampsigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma

            # Reserve budget for a final local polish
            polish_budget = 50
            search_budget = budget - polish_budget

            # Island-based CMA-ES: maintain 3 independent CMA-ES instances
            num_islands = 3
            islands = []
            for i in range(num_islands):
                mean = rng.uniform(-5, 5, dim)
                C = np.eye(dim)
                C_sqrt = np.linalg.cholesky(C)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                sigma = 2.0
                # Evaluate the initial mean
                fx = func(mean)
                calls_made += 1
                if fx < best_f:
                    best_f = fx
                    best_x = mean.copy()
                islands.append({
                    'mean': mean,
                    'C': C,
                    'C_sqrt': C_sqrt,
                    'p_sigma': p_sigma,
                    'p_c': p_c,
                    'sigma': sigma,
                    'best_f': fx,
                    'no_improve': 0
                })

            gen = 0
            while calls_made < search_budget and time.time() < deadline:
                # Select island with the best recent progress (or round-robin)
                # For simplicity, we iterate through islands and let each take one generation
                # if budget permits.
                # To balance, we pick the island with the lowest 'no_improve' count or just cycle.
                # Let's cycle: island_index = gen % num_islands
                island_idx = gen % num_islands
                isl = islands[island_idx]

                # Check if we have enough budget for a full generation
                if calls_made + lam > search_budget:
                    break

                # Generate population for this island
                offspring = []
                for i in range(lam):
                    if calls_made >= search_budget or time.time() >= deadline:
                        break
                    z = rng.standard_normal(dim)
                    x_off = isl['mean'] + isl['sigma'] * (isl['C_sqrt'] @ z)
                    x_off = np.clip(x_off, -5, 5)
                    fx_off = func(x_off)
                    calls_made += 1
                    offspring.append((x_off, fx_off))

                if len(offspring) < mu:
                    break

                # Sort offspring
                offspring.sort(key=lambda x: x[1])
                best_offspring_x, best_offspring_f = offspring[0]

                # Update global best
                if best_offspring_f < best_f:
                    best_f = best_offspring_f
                    best_x = best_offspring_x.copy()

                # Update island best
                if best_offspring_f < isl['best_f']:
                    isl['best_f'] = best_offspring_f
                    isl['no_improve'] = 0
                else:
                    isl['no_improve'] += 1

                # Check for stagnation in this island
                if isl['no_improve'] > 15:
                    # Restart this island around the global best or a random point
                    # 50% chance to restart around global best, 50% random
                    if rng.random() < 0.5 and best_x is not None:
                        isl['mean'] = best_x + rng.normal(0, 1.0, dim)
                        isl['mean'] = np.clip(isl['mean'], -5, 5)
                    else:
                        isl['mean'] = rng.uniform(-5, 5, dim)
                    isl['C'] = np.eye(dim)
                    isl['C_sqrt'] = np.linalg.cholesky(isl['C'])
                    isl['p_sigma'] = np.zeros(dim)
                    isl['p_c'] = np.zeros(dim)
                    isl['sigma'] = 1.5
                    isl['no_improve'] = 0
                    # Evaluate the new mean to update island best
                    if calls_made < search_budget:
                        fx_new = func(isl['mean'])
                        calls_made += 1
                        isl['best_f'] = fx_new
                        if fx_new < best_f:
                            best_f = fx_new
                            best_x = isl['mean'].copy()
                    continue  # Skip CMA-ES update for this generation

                # Standard CMA-ES update
                x_w = np.zeros(dim)
                for i in range(mu):
                    x_w += weights[i] * offspring[i][0]

                # Update evolution paths
                try:
                    C_sqrt_inv = np.linalg.inv(isl['C_sqrt'])
                except:
                    break

                y_w = (x_w - isl['mean']) / isl['sigma']
                isl['p_sigma'] = (1 - c_sigma) * isl['p_sigma'] + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_sqrt_inv @ y_w)

                hsig = 1.0 if np.linalg.norm(isl['p_sigma']) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1))) < (1.4 + 2 / (dim + 1)) * np.sqrt(dim + 1 - 1) else 0.0
                isl['p_c'] = (1 - c_c) * isl['p_c'] + hsig * np.sqrt(c_c * (2 - c_c) * mueff) * (C_sqrt_inv @ y_w)

                # Update mean
                isl['mean'] = x_w

                # Update covariance
                rank_one = np.outer(isl['p_sigma'], isl['p_sigma'])
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    y_i = C_sqrt_inv @ (offspring[i][0] - isl['mean']) / isl['sigma']
                    rank_mu += weights[i] * np.outer(y_i, y_i)

                isl['C'] = (1 + c_1 * (2 - c_1) - c_mu) * isl['C'] + c_1 * rank_one + c_mu * rank_mu
                isl['C'] = (isl['C'] + isl['C'].T) / 2

                # Ensure PSD
                eigvals, eigvecs = np.linalg.eigh(isl['C'])
                eigvals = np.maximum(eigvals, 1e-12)
                isl['C'] = eigvecs @ np.diag(eigvals) @ eigvecs.T

                # Update sigma
                isl['sigma'] = isl['sigma'] * np.exp(c_sigma / dampsigma * (np.linalg.norm(isl['p_sigma']) / np.sqrt(dim + 1 - 1) - 1))

                # Update Cholesky
                try:
                    isl['C_sqrt'] = np.linalg.cholesky(isl['C'])
                except:
                    # Fallback if Cholesky fails
                    isl['C'] = np.eye(dim)
                    isl['C_sqrt'] = np.linalg.cholesky(isl['C'])
                    isl['sigma'] = 1.0

                gen += 1

            # Final Polish: Use remaining budget to refine the global best
            if calls_made < budget and time.time() < deadline:
                x = best_x.copy()
                fx = best_f
                sigma = 0.5
                no_improve = 0
                while calls_made < budget and time.time() < deadline:
                    y = x + sigma * rng.standard_normal(dim)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls_made += 1

                    if fy < fx:
                        x = y
                        fx = fy
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                        no_improve = 0
                        # Slightly increase sigma if improving rapidly? No, keep it stable
                    else:
                        sigma = max(sigma * 0.9, 1e-4)
                        no_improve += 1
                        if no_improve > 20:
                            break

        except Exception:
            pass

        return best_x, best_f
