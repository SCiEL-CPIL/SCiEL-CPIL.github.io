"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Decomposition-based multi-objective optimizer using Tchebycheff scalarization."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        mid = (lo + hi) / 2.0
        span = hi - lo
        rng = np.random.default_rng(0)
        
        # Generate 125 uniformly distributed weight vectors in the simplex
        n_weights = 125
        weights = self._generate_weight_vectors(n_weights, self.n_obj, rng)
        
        # Budget allocation: 50 for initialization, rest for local search
        n_init = 50
        remaining = self.budget - n_init
        evals_per_weight = max(1, remaining // n_weights)
        
        # Initialize with diverse random points
        X_init = lo + span * rng.random((n_init, self.dim))
        F_init = np.array([problem(x) for x in X_init])
        used = n_init
        
        # Track best point per weight vector
        best_X = [None] * n_weights
        best_F = [None] * n_weights
        best_tcheb = [np.inf] * n_weights
        
        # Assign initial points to nearest weight vectors
        for i in range(n_init):
            idx = self._assign_weight(F_init[i], weights, lo, hi)
            tcheb_val = self._tchebycheff_value(F_init[i], weights[idx], lo, hi)
            if tcheb_val < best_tcheb[idx]:
                best_tcheb[idx] = tcheb_val
                best_X[idx] = X_init[i].copy()
                best_F[idx] = F_init[i].copy()
        
        # For weights without initial points, use mid-point
        for wi in range(n_weights):
            if best_X[wi] is None:
                x = mid.copy()
                f = problem(x)
                used += 1
                best_X[wi] = x.copy()
                best_F[wi] = f.copy()
                best_tcheb[wi] = self._tchebycheff_value(f, weights[wi], lo, hi)
        
        # Main loop: Nelder-Mead local search on each subproblem
        for wi in range(n_weights):
            if used >= self.budget or time.time() > deadline:
                break
            
            x_best = best_X[wi].copy()
            f_best = best_F[wi].copy()
            tcheb_best = best_tcheb[wi]
            w = weights[wi]
            
            # Nelder-Mead simplex initialization
            # Start from x_best, create simplex with small perturbations
            alpha = 0.1  # perturbation size
            simplex_X = [x_best.copy()]
            simplex_F = [f_best.copy()]
            simplex_T = [tcheb_best]
            
            # Add dim more vertices
            for d in range(self.dim):
                x_new = x_best.copy()
                perturbation = span[d] * alpha
                if x_best[d] + perturbation <= hi[d]:
                    x_new[d] += perturbation
                elif x_best[d] - perturbation >= lo[d]:
                    x_new[d] -= perturbation
                else:
                    x_new[d] = mid[d]
                x_new = np.clip(x_new, lo, hi)
                
                if used >= self.budget or time.time() > deadline:
                    break
                
                f_new = problem(x_new)
                used += 1
                t_new = self._tchebycheff_value(f_new, w, lo, hi)
                simplex_X.append(x_new)
                simplex_F.append(f_new)
                simplex_T.append(t_new)
            
            # Nelder-Mead iterations
            n_iter = evals_per_weight - (len(simplex_X) - 1)  # remaining evals for this weight
            n_iter = max(1, n_iter)
            
            for it in range(n_iter):
                if used >= self.budget or time.time() > deadline:
                    break
                
                # Sort simplex by Tchebycheff value
                order = np.argsort(simplex_T)
                simplex_X = [simplex_X[i] for i in order]
                simplex_F = [simplex_F[i] for i in order]
                simplex_T = [simplex_T[i] for i in order]
                
                # Update best if improved
                if simplex_T[0] < tcheb_best:
                    tcheb_best = simplex_T[0]
                    best_X[wi] = simplex_X[0].copy()
                    best_F[wi] = simplex_F[0].copy()
                    best_tcheb[wi] = tcheb_best
                
                # Centroid of all but worst
                centroid = np.mean(simplex_X[:-1], axis=0)
                
                # Reflection
                gamma = 1.0
                x_ref = centroid + gamma * (centroid - simplex_X[-1])
                x_ref = np.clip(x_ref, lo, hi)
                
                if used >= self.budget or time.time() > deadline:
                    break
                
                f_ref = problem(x_ref)
                used += 1
                t_ref = self._tchebycheff_value(f_ref, w, lo, hi)
                
                if t_ref < simplex_T[0]:
                    # Expansion
                    if used >= self.budget or time.time() > deadline:
                        break
                    delta = 2.0
                    x_exp = centroid + delta * (x_ref - centroid)
                    x_exp = np.clip(x_exp, lo, hi)
                    f_exp = problem(x_exp)
                    used += 1
                    t_exp = self._tchebycheff_value(f_exp, w, lo, hi)
                    if t_exp < t_ref:
                        x_new, f_new, t_new = x_exp, f_exp, t_exp
                    else:
                        x_new, f_new, t_new = x_ref, f_ref, t_ref
                elif t_ref < simplex_T[-2]:
                    x_new, f_new, t_new = x_ref, f_ref, t_ref
                else:
                    # Contraction
                    rho = 0.5
                    x_con = centroid + rho * (simplex_X[-1] - centroid)
                    x_con = np.clip(x_con, lo, hi)
                    if used >= self.budget or time.time() > deadline:
                        break
                    f_con = problem(x_con)
                    used += 1
                    t_con = self._tchebycheff_value(f_con, w, lo, hi)
                    if t_con < simplex_T[-1]:
                        x_new, f_new, t_new = x_con, f_con, t_con
                    else:
                        # Shrink
                        x_new_list = [simplex_X[0] + 0.5 * (simplex_X[i] - simplex_X[0]) for i in range(1, len(simplex_X))]
                        for i in range(1, len(simplex_X)):
                            if used >= self.budget or time.time() > deadline:
                                break
                            f_sh = problem(x_new_list[i-1])
                            used += 1
                            t_sh = self._tchebycheff_value(f_sh, w, lo, hi)
                            simplex_X[i] = x_new_list[i-1]
                            simplex_F[i] = f_sh
                            simplex_T[i] = t_sh
                        continue
                
                # Replace worst point
                simplex_X[-1] = x_new
                simplex_F[-1] = f_new
                simplex_T[-1] = t_new
        
        # Collect all non-dominated points
        all_X = [x for x in best_X if x is not None]
        all_F = [f for f in best_F if f is not None]
        
        if len(all_X) > 0:
            all_F_arr = np.array(all_F)
            # Keep only non-dominated points
            nondom_mask = np.array([not self._is_dominated(all_F_arr[i], all_F_arr) for i in range(len(all_F_arr))])
            all_X = np.array([all_X[i] for i in range(len(all_X)) if nondom_mask[i]])
        
        return all_X if len(all_X) > 0 else np.array([mid])
    
    def _generate_weight_vectors(self, n_weights, n_obj, rng):
        """Generate uniformly distributed weight vectors in the (n_obj-1)-simplex."""
        # Use Dirichlet distribution for uniform distribution on simplex
        weights = rng.dirichlet(np.ones(n_obj), size=n_weights)
        # Add extreme points for better coverage
        extreme_weights = []
        for i in range(n_obj):
            w = np.zeros(n_obj)
            w[i] = 1.0
            extreme_weights.append(w)
        # Combine and deduplicate
        all_weights = list(weights) + extreme_weights
        # Select n_weights distinct ones
        selected = []
        for w in all_weights:
            if len(selected) >= n_weights:
                break
            if not any(np.allclose(w, s) for s in selected):
                selected.append(w)
        # Fill remaining with random
        while len(selected) < n_weights:
            selected.append(rng.dirichlet(np.ones(n_obj)))
        return np.array(selected[:n_weights])
    
    def _tchebycheff_value(self, f, weight, lo, hi):
        """Compute Tchebycheff scalarization value."""
        # Normalize objectives using the reference point
        ref = np.array([3.0, 5.0, 7.0])
        f_norm = (f - lo) / (hi - lo + 1e-10)
        return np.max(weight * f_norm)
    
    def _assign_weight(self, f, weights, lo, hi):
        """Assign objective vector to nearest weight vector."""
        f_norm = (f - lo) / (hi - lo + 1e-10)
        # Find weight vector that minimizes Tchebycheff distance
        best_idx = 0
        best_val = np.inf
        for i, w in enumerate(weights):
            val = np.max(w * f_norm)
            if val < best_val:
                best_val = val
                best_idx = i
        return best_idx
    
    def _is_dominated(self, f, all_f):
        """Check if f is dominated by any point in all_f."""
        for other_f in all_f:
            if np.all(other_f <= f) and np.any(other_f < f):
                return True
        return False
# EVOLVE-BLOCK-END
