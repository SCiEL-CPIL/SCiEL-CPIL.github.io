"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restart and deadline guard for Büche-Rastrigin f4."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s kill

        rng = np.random.default_rng(12345)
        dim = self.dim
        total_budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        # CMA-ES Parameters
        lambda_pop = 9
        mu = lambda_pop // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = (weights.sum() ** 2) / np.sum(weights ** 2)

        cc = (4 + mu_eff / lambda_pop) / (lambda_pop + 4 + 2 * mu_eff / lambda_pop)
        cs = (mu_eff + 2) / (lambda_pop + mu_eff + 5)
        c1 = 2 / ((lambda_pop + 2) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((lambda_pop + 2) ** 2 + mu_eff))

        def check_deadline():
            return time.time() >= deadline or calls_made >= total_budget

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def run_cmaes(x_init, sigma_init, max_calls):
            nonlocal calls_made, best_x, best_f

            x_mean = x_init.copy()
            sigma = sigma_init
            C = np.eye(dim)
            B = np.eye(dim)
            D = np.ones(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)

            local_best_f = evaluate(x_mean)
            local_best_x = x_mean.copy()

            gen = 0
            no_improve_gen = 0
            last_improvement = local_best_f

            while not check_deadline() and calls_made < max_calls:
                # Sample lambda points
                offspring = []
                for i in range(lambda_pop):
                    if check_deadline() or calls_made >= max_calls:
                        break
                    z = rng.standard_normal(dim)
                    y = B @ (D * z)
                    x_new = x_mean + sigma * y
                    x_new = np.clip(x_new, -5, 5)
                    f_new = evaluate(x_new)
                    offspring.append((x_new, f_new))

                if len(offspring) == 0:
                    break

                # Sort by fitness
                offspring.sort(key=lambda x: x[1])

                # Update mean
                x_new_mean = np.sum([weights[i] * offspring[i][0] for i in range(min(mu, len(offspring)))], axis=0)

                # Update step size
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / sigma * (B @ (D * (B.T @ (x_new_mean - x_mean))))

                hsig = (np.linalg.norm(ps) < (1.4 + 2 / (dim + 1)) * np.sqrt(lambda_pop + 1 - mu_eff))

                if hsig:
                    sigma *= np.exp(cs / dim * (np.linalg.norm(ps) / np.sqrt(lambda_pop) - 1))

                # Update covariance
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * (x_new_mean - x_mean) / sigma

                rank1 = np.outer(pc, pc)
                rankmu = np.zeros((dim, dim))
                for i in range(min(mu, len(offspring))):
                    y_i = (offspring[i][0] - x_mean) / sigma
                    rankmu += weights[i] * np.outer(y_i, y_i)

                C = (1 - c1 - cmu) * C + c1 * rank1 + cmu * rankmu
                C = (C + C.T) / 2

                # Update mean
                x_mean = x_new_mean
                gen += 1

                # Update best
                if offspring[0][1] < local_best_f:
                    local_best_f = offspring[0][1]
                    local_best_x = offspring[0][0].copy()
                    last_improvement = local_best_f
                    no_improve_gen = 0
                else:
                    no_improve_gen += 1

                # Update basis
                try:
                    D2, B = np.linalg.eigh(C)
                    D = np.sqrt(np.maximum(D2, 1e-12))
                except:
                    D2, B = np.linalg.eigh(C + 1e-10 * np.eye(dim))
                    D = np.sqrt(np.maximum(D2, 1e-12))

                # Check for stall - restart if no improvement for 12 generations
                if no_improve_gen >= 12 and calls_made < max_calls - 10:
                    break

                # Check convergence
                if np.linalg.norm(D) < 1e-12 or sigma < 1e-12:
                    break

            return local_best_f, local_best_x, calls_made

        # Multi-start CMA-ES with diverse initializations
        # Büche-Rastrigin is highly multi-modal, so we need broad exploration
        
        # Generate diverse starting points: mix of random and structured
        n_starts = 5
        starts = []
        
        # Start 1: origin
        starts.append(np.zeros(dim))
        
        # Start 2-4: random points in [-3, 3]^5
        for _ in range(3):
            starts.append(rng.uniform(-3, 3, size=dim))
        
        # Start 5: structured point
        starts.append(rng.uniform(-4, 4, size=dim))
        
        # Budget allocation: give more budget to later starts if earlier ones stagnate
        # Use adaptive allocation based on progress
        base_budget_per_start = max(50, total_budget // (n_starts * 2))
        
        for start_idx, x_init in enumerate(starts):
            if check_deadline():
                break
            
            # Adapt sigma based on start index - later starts use smaller sigma
            # since we've already explored broadly
            if start_idx == 0:
                sigma_init = 2.5
            elif start_idx == 1:
                sigma_init = 1.5
            else:
                sigma_init = 0.8
            
            # Allocate budget: first start gets more, subsequent ones get less
            # but can steal budget if making progress
            if start_idx == 0:
                max_calls_for_start = min(base_budget_per_start * 3, total_budget - calls_made)
            else:
                max_calls_for_start = min(base_budget_per_start, total_budget - calls_made)
            
            if max_calls_for_start < 20:
                break
            
            local_best_f, local_best_x, calls_before = run_cmaes(
                x_init.copy(), sigma_init, calls_made + max_calls_for_start
            )
            
            # If this start made significant progress, give it more budget
            progress = best_f - local_best_f if best_f != np.inf else np.inf
            if progress > 0 and not check_deadline():
                # Continue with next start but with smaller budget
                continue
            elif progress <= 0 and start_idx < n_starts - 1:
                # No progress, move to next start
                continue
        
        # Final refinement: run CMA-ES from the overall best with remaining budget
        if not check_deadline() and best_x is not None:
            remaining = total_budget - calls_made
            if remaining > 30:
                # Use small sigma for final refinement
                run_cmaes(best_x.copy(), 0.3, total_budget)

        return best_x, best_f
# EVOLVE-BLOCK-END
