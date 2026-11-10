"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    """
    Multi-objective optimizer using a two-phase strategy:
    1. Low-discrepancy (Sobol) sampling to get a broad initial spread.
    2. Iterative refinement using a greedy non-dominated set update.
       Instead of complex decomposition (MOEA/D) or full NSGA-II sorting,
       we maintain a small set of non-dominated points and generate offspring
       by perturbing the 'most useful' points (those with high hypervolume contribution
       or simply random non-dominated points) with adaptive step sizes.
       
       This 'error_repair' style approach focuses on fixing the 'gaps' in the Pareto front
       by targeting specific regions, rather than evolving a large population with complex selection.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 315.0  # Safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Ensure we don't exceed budget
        if budget <= 0:
            return None

        # Phase 1: Sobol Sampling
        # Generate a set of points using Sobol sequence.
        # We use a portion of the budget for initial exploration.
        # Let's use 50% of the budget for initial sampling if budget is large enough,
        # otherwise use a smaller fixed number.
        
        # Sobol sequence dimension must be power of 2 for best performance, but scipy handles any.
        # However, generating too many points at once might be slow? No, it's vectorized.
        
        # Let's allocate 40% of budget to Sobol, 60% to refinement.
        n_sobol = max(10, int(budget * 0.4))
        if n_sobol >= budget:
            n_sobol = budget - 10 # Leave some for refinement
        
        # Generate Sobol points in [0, 1]^dim
        # Note: scipy.stats.qmc.Sobol requires dimension.
        try:
            sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
            # Generate n_sobol points
            # Note: Sobol generates points in [0, 1].
            # We need to ensure we don't generate more than budget allows.
            points_sobol = sampler.random(n_sobol)
            # Map to bounds
            points_sobol = points_sobol * (hi - lo) + lo
        except Exception:
            # Fallback to random if Sobol fails
            rng = np.random.default_rng(42)
            points_sobol = rng.random((n_sobol, dim)) * (hi - lo) + lo

        # Evaluate all Sobol points
        # We must be careful: problem(x) is called for each point.
        # The evaluator records every call.
        
        F_sobol = np.zeros((n_sobol, n_obj))
        for i in range(n_sobol):
            F_sobol[i] = problem(points_sobol[i])
            if time.time() > deadline:
                break
        
        # Check if we exceeded budget or time
        evals_used = n_sobol
        if time.time() > deadline or evals_used >= budget:
            return None

        # Initialize non-dominated set
        # We will maintain a list of (x, f) pairs that are non-dominated.
        # For efficiency, we'll keep the set small (e.g., max 50-100 points) to allow fast dominance checks.
        # Actually, since we record ALL evaluations, the hypervolume is calculated on the union of all recorded points.
        # But for our internal search, we only need to keep a representative non-dominated set to generate good offspring.
        
        # Let's extract the non-dominated points from the initial population.
        # Dominance check: a dominates b if a_i <= b_i for all i and a_j < b_j for some j.
        
        def is_dominated(f, F_set):
            # F_set is (N, n_obj)
            # Check if any row in F_set dominates f
            # f is (n_obj,)
            # a_i <= f_i for all i AND a_j < f_j for at least one j
            # Vectorized:
            # less_or_equal = (F_set <= f).all(axis=1)
            # strictly_less = (F_set < f).any(axis=1)
            # return np.any(less_or_equal & strictly_less)
            
            # More efficient:
            # If any point is <= f and at least one is < f
            # Note: if F_set contains f itself, it doesn't strictly dominate.
            # But we check strictness.
            
            # Let's use a simpler loop for clarity and correctness first, as N is small.
            for i in range(len(F_set)):
                if np.all(F_set[i] <= f) and np.any(F_set[i] < f):
                    return True
            return False

        # Build initial non-dominated set
        # Sort by first objective to speed up? No, just iterate.
        # Since n_sobol is ~80, O(N^2) is fine.
        
        non_dominated_X = []
        non_dominated_F = []
        
        # Sort by f[0] to help with dominance checks?
        # Let's just do a simple filter.
        # Optimization: If a point is dominated by any already accepted point, skip it.
        
        # To make it faster, we can sort by hypervolume contribution or just f[0].
        # Let's sort by f[0] ascending.
        indices = np.argsort(F_sobol[:, 0])
        
        for idx in indices:
            f = F_sobol[idx]
            x = points_sobol[idx]
            
            # Check if dominated by current non_dominated_F
            if len(non_dominated_F) == 0:
                non_dominated_X.append(x)
                non_dominated_F.append(f)
            else:
                # Vectorized check against current set
                F_arr = np.array(non_dominated_F)
                # Does any in F_arr dominate f?
                # (F_arr <= f).all(axis=1) & (F_arr < f).any(axis=1)
                dominated = np.any((F_arr <= f).all(axis=1) & (F_arr < f).any(axis=1))
                if not dominated:
                    # Does f dominate any in non_dominated_F?
                    # If so, remove those from the set.
                    # (f <= F_arr).all(axis=1) & (f < F_arr).any(axis=1)
                    dominates_mask = (f <= F_arr).all(axis=1) & (f < F_arr).any(axis=1)
                    
                    # Remove dominated points
                    new_X = []
                    new_F = []
                    for j in range(len(non_dominated_X)):
                        if not dominates_mask[j]:
                            new_X.append(non_dominated_X[j])
                            new_F.append(non_dominated_F[j])
                    
                    non_dominated_X = new_X
                    non_dominated_F = new_F
                    
                    # Add f
                    non_dominated_X.append(x)
                    non_dominated_F.append(f)

        # Phase 2: Refinement Loop
        # We have budget - evals_used evaluations left.
        # Strategy: 
        # 1. Pick a point from the non-dominated set.
        # 2. Generate offspring via Gaussian perturbation.
        # 3. Evaluate offspring.
        # 4. If offspring is non-dominated, add to set (and remove dominated).
        # 5. If offspring is dominated, we might still try to "repair" it by adjusting towards a better region?
        #    The "error_repair" idea: if an offspring is dominated, identify which objective is worst 
        #    and perturb specifically to improve it? 
        #    Or simply: use the offspring as a starting point for a local search if it's close?
        #    Let's stick to: if dominated, discard. If non-dominated, accept.
        #    To improve efficiency, we can adapt the step size.
        
        # We also need to ensure we don't get stuck.
        # Let's use a cooling schedule for step size.
        
        remaining_budget = budget - evals_used
        if remaining_budget <= 0:
            return None

        # Initial step size
        step_size = 0.1
        # Decay rate
        decay = 0.999
        
        rng = np.random.default_rng(123)
        
        # To make the "repair" more effective, we can prioritize perturbing points that are 
        # "extreme" in the objective space, as they define the boundary of the hypervolume.
        # Or simply random.
        
        # Let's try a hybrid: 50% random from ND set, 50% from the point with max "spread" 
        # (e.g., max distance to centroid of ND set).
        
        while evals_used < budget:
            if time.time() > deadline:
                break
                
            # Select parent
            if len(non_dominated_X) == 0:
                # Fallback to random if set is empty (shouldn't happen)
                x_parent = lo + (hi - lo) * rng.random(dim)
            else:
                # Choose parent
                # Simple: random
                idx_parent = rng.integers(0, len(non_dominated_X))
                x_parent = non_dominated_X[idx_parent]
                
            # Generate offspring
            # Gaussian noise
            noise = rng.normal(0, step_size, dim) * (hi - lo)
            x_offspring = x_parent + noise
            x_offspring = np.clip(x_offspring, lo, hi)
            
            # Evaluate
            f_offspring = problem(x_offspring)
            evals_used += 1
            
            # Check if non-dominated
            if len(non_dominated_F) == 0:
                non_dominated_X.append(x_offspring)
                non_dominated_F.append(f_offspring)
            else:
                F_arr = np.array(non_dominated_F)
                
                # Is it dominated?
                is_dom = np.any((F_arr <= f_offspring).all(axis=1) & (F_arr < f_offspring).any(axis=1))
                
                if not is_dom:
                    # Does it dominate others?
                    dom_mask = (f_offspring <= F_arr).all(axis=1) & (f_offspring < F_arr).any(axis=1)
                    
                    new_X = []
                    new_F = []
                    for j in range(len(non_dominated_X)):
                        if not dom_mask[j]:
                            new_X.append(non_dominated_X[j])
                            new_F.append(non_dominated_F[j])
                    
                    non_dominated_X = new_X
                    non_dominated_F = new_F
                    
                    non_dominated_X.append(x_offspring)
                    non_dominated_F.append(f_offspring)
                    
                    # If we found a new non-dominated point, we can reset or slightly increase step size?
                    # Or keep decaying. Let's keep decaying to converge.
                    # But to answer the "error_repair" question: 
                    # If it was dominated, we could try to "repair" it by moving it towards the ideal point 
                    # in the direction of the worst objective.
                    # Let's implement a simple repair: if dominated, try 1 extra evaluation 
                    # (if budget allows) to fix the worst objective.
                    # This uses an extra evaluation, so we need to be careful with budget.
                    # Given the tight budget, maybe skip complex repair and rely on the 
                    # diversity of the initial Sobol + local search.
                    
                else:
                    # It was dominated. 
                    # "Error Repair" strategy: 
                    # Identify the objective that is worst (max value relative to ideal).
                    # Perturb x_offspring specifically to improve that objective.
                    # This is a targeted local search.
                    
                    # We have budget left?
                    if evals_used < budget and time.time() < deadline:
                        # Find the worst objective index
                        # We don't know the true ideal, but we can use the min of the ND set.
                        ideal = np.min(np.array(non_dominated_F), axis=0)
                        # f_offspring - ideal gives the "error"
                        errors = f_offspring - ideal
                        worst_obj_idx = np.argmax(errors)
                        
                        # Perturb x_offspring to try to reduce f[worst_obj_idx]
                        # How? Random perturbation, but biased?
                        # Or just a smaller step?
                        # Let's try a small step in a random direction, but evaluate if it improves the worst obj.
                        # This is effectively a 1-step local search.
                        
                        # To save budget, we only do this if we have enough time/budget.
                        # Let's limit repairs to avoid wasting budget.
                        # Only repair if step_size is small enough (later stages).
                        if step_size < 0.05:
                            x_repaired = x_offspring.copy()
                            # Small noise
                            noise_rep = rng.normal(0, step_size * 0.5, dim) * (hi - lo)
                            x_repaired = x_repaired + noise_rep
                            x_repaired = np.clip(x_repaired, lo, hi)
                            
                            f_repaired = problem(x_repaired)
                            evals_used += 1
                            
                            # Check if repaired is non-dominated
                            if len(non_dominated_F) == 0:
                                non_dominated_X.append(x_repaired)
                                non_dominated_F.append(f_repaired)
                            else:
                                F_arr = np.array(non_dominated_F)
                                is_dom_rep = np.any((F_arr <= f_repaired).all(axis=1) & (F_arr < f_repaired).any(axis=1))
                                
                                if not is_dom_rep:
                                    dom_mask_rep = (f_repaired <= F_arr).all(axis=1) & (f_repaired < F_arr).any(axis=1)
                                    
                                    new_X = []
                                    new_F = []
                                    for j in range(len(non_dominated_X)):
                                        if not dom_mask_rep[j]:
                                            new_X.append(non_dominated_X[j])
                                            new_F.append(non_dominated_F[j])
                                    
                                    non_dominated_X = new_X
                                    non_dominated_F = new_F
                                    
                                    non_dominated_X.append(x_repaired)
                                    non_dominated_F.append(f_repaired)

            # Decay step size
            step_size *= decay

        return None

# EVOLVE-BLOCK-END
