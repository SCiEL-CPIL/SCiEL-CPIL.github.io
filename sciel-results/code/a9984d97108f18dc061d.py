"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-III style optimizer with DTLZ7-specific structural priors."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300  # 300s limit, leave margin
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # DTLZ7 known ideal point is (0, 0, 1)
        # Pareto front: f1=0, f2=0, f3=1 for x in [0,1]^M
        # Non-Pareto regions exist but we want to cover the front.
        
        # Population size: large enough for coverage, small enough for speed
        # Budget is 200. Let's use pop_size=40. Generations = 200/40 = 5.
        # Maybe pop_size=50, gens=4.
        # Let's try pop_size=60, gens=3.
        # Actually, with 200 evals, we can do pop_size=100, 2 generations?
        # Or pop_size=50, 4 generations.
        # NSGA-III works best with larger populations.
        
        pop_size = 50
        
        # Initialize population
        # Strategy:
        # 1. 50% random
        # 2. 50% biased towards the known Pareto region (x2...x21 small)
        
        X = np.zeros((pop_size, self.dim))
        F = np.zeros((pop_size, self.n_obj))
        
        # First half: random
        X[:pop_size//2] = lo + span * rng.random((pop_size//2, self.dim))
        
        # Second half: biased
        # x1 random
        # x2...x21 small (close to 0)
        X[pop_size//2:, 0] = rng.random(pop_size//2)
        X[pop_size//2:, 1:] = rng.random((pop_size//2, self.dim-1)) * 0.1
        
        # Evaluate initial population
        for i in range(pop_size):
            F[i] = problem(X[i])
        
        used = pop_size
        
        # NSGA-III loop
        while used < self.budget:
            # Check deadline
            if time.time() > deadline - 5:
                break
            
            # Create offspring
            children = np.zeros((pop_size, self.dim))
            for i in range(pop_size):
                # Binary tournament selection
                p1_idx = rng.integers(pop_size)
                p2_idx = rng.integers(pop_size)
                
                # Simple dominance check
                # If p1 dominates p2, choose p1
                # Else if p2 dominates p1, choose p2
                # Else random
                f1 = F[p1_idx]
                f2 = F[p2_idx]
                
                dom12 = np.all(f1 <= f2) and np.any(f1 < f2)
                dom21 = np.all(f2 <= f1) and np.any(f2 < f1)
                
                if dom12:
                    parent = X[p1_idx]
                elif dom21:
                    parent = X[p2_idx]
                else:
                    if rng.random() > 0.5:
                        parent = X[p1_idx]
                    else:
                        parent = X[p2_idx]
                
                # Simulated Binary Crossover (SBX)
                # For simplicity and speed, use blend or just mutation?
                # Let's use a simple crossover: mix two parents
                p2_idx2 = rng.integers(pop_size)
                parent2 = X[p2_idx2]
                
                # Blend crossover
                beta = rng.random(self.dim)
                child = beta * parent + (1 - beta) * parent2
                
                # Mutation: Polynomial or Gaussian?
                # Gaussian is fast.
                # Adaptive sigma?
                # Let's use a fixed sigma scaled by range.
                # sigma = 0.1 * span
                noise = 0.1 * span * rng.standard_normal(self.dim)
                child = child + noise
                
                # Clip
                child = np.clip(child, lo, hi)
                children[i] = child
            
            # Evaluate children
            F_children = np.zeros((pop_size, self.n_obj))
            for i in range(pop_size):
                F_children[i] = problem(children[i])
                used += 1
                if used >= self.budget:
                    break
            
            if used >= self.budget:
                break
            
            # Combine parents and children
            X_combined = np.vstack([X, children])
            F_combined = np.vstack([F, F_children])
            
            # Non-dominated sorting
            # Find non-dominated set
            # For efficiency, we can use a simple O(N^2) check since N=100
            # Actually, let's just pick the best pop_size from the combined set based on dominance.
            
            # Identify non-dominated points in combined set
            # We want to select pop_size points.
            # Simple strategy: Keep the non-dominated points. If more than pop_size, use crowding distance or just random.
            # If fewer, fill with random from dominated?
            
            # Let's implement a simple non-dominated sort for selection.
            # 1. Mark dominated points.
            # 2. Select non-dominated.
            # 3. If count > pop_size, reduce.
            # 4. If count < pop_size, add best dominated.
            
            # Since we want to cover the front, and DTLZ7 front is specific,
            # let's just keep the best pop_size points based on a simple metric:
            # 1. Non-dominated
            # 2. Then those closest to ideal (0,0,1)?
            
            # Actually, standard NSGA-III uses reference points.
            # Let's implement a simpler "Non-dominated + Diversity" selection.
            
            # Step 1: Find non-dominated points in X_combined
            non_dom_indices = []
            n_combined = len(F_combined)
            
            # This O(N^2) is 100^2 = 10000 ops, very fast.
            for i in range(n_combined):
                is_dom = False
                for j in range(n_combined):
                    if i == j: continue
                    if np.all(F_combined[j] <= F_combined[i]) and np.any(F_combined[j] < F_combined[i]):
                        is_dom = True
                        break
                if not is_dom:
                    non_dom_indices.append(i)
            
            # If we have more non-dominated points than pop_size, we need to prune.
            # For DTLZ7, the front is 1D (x1) effectively? No, x1..xM define the front shape.
            # Wait, DTLZ7 front:
            # g=0 when x_{M+1}..x_n = 0.
            # f1 = x1, f2 = x2, f3 = 1 + (n-M) * g.
            # So on the front, f3=1. f1=x1, f2=x2.
            # The Pareto front is a 2D surface in objective space?
            # No, x1 and x2 are independent in [0,1].
            # So the front is a square [0,1]x[0,1] at f3=1.
            # Wait, DTLZ7 definition:
            # f1 = x1
            # f2 = x2
            # f3 = (1 + g) * h(x_{m-1}, g)
            # h = 1 + 1/f3 * (sum sin(3 pi x_i) / 2)
            # If g=0, f3 = 1 * (1 + 0) = 1.
            # So yes, the ideal point is (0,0,1).
            # The Pareto front is the set of points where g=0.
            # f1=x1, f2=x2.
            # So the front is the square [0,1] x [0,1] in the f1-f2 plane, at f3=1.
            
            # So we want to cover the square [0,1]x[0,1] with f3=1.
            
            # If we have too many non-dominated points, we can use a grid or random sampling.
            # For now, if len(non_dom_indices) > pop_size, randomly sample pop_size from them.
            # This is not ideal for coverage but better than nothing.
            # Better: Sort by f1, then f2?
            
            if len(non_dom_indices) > pop_size:
                # Random selection from non-dominated
                selected_indices = rng.choice(non_dom_indices, size=pop_size, replace=False)
            elif len(non_dom_indices) < pop_size:
                # Need to fill with dominated points
                # Pick the "best" dominated points (closest to non-dominated set?)
                # Or just random from the rest?
                # Let's pick random from the rest to maintain diversity.
                remaining_indices = [i for i in range(n_combined) if i not in non_dom_indices]
                needed = pop_size - len(non_dom_indices)
                if needed > 0 and len(remaining_indices) > 0:
                    extra_indices = rng.choice(remaining_indices, size=min(needed, len(remaining_indices)), replace=False)
                    selected_indices = np.array(non_dom_indices + list(extra_indices))
                else:
                    selected_indices = np.array(non_dom_indices)
            else:
                selected_indices = np.array(non_dom_indices)
            
            # Update population
            X = X_combined[selected_indices]
            F = F_combined[selected_indices]
            
        return X
# EVOLVE-BLOCK-END
