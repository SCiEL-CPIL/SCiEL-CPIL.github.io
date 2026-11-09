import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import time

def construct():
    base = 51975
    
    # Get all divisors of base greater than 1
    divisors = []
    for d in range(2, base + 1):
        if base % d == 0:
            divisors.append(d)
    
    # Greedy solution as fallback
    def greedy(order):
        covered = np.zeros(base, dtype=np.int8)
        classes = []
        for m in order:
            best_r = 0
            best_count = -1
            for r in range(m):
                indices = np.arange(r, base, m)
                count = np.sum(covered[indices] == 0)
                if count > best_count:
                    best_count = count
                    best_r = r
            classes.append([m, best_r])
            indices = np.arange(best_r, base, m)
            covered[indices] = 1
        return classes
    
    def compute_score(classes):
        covered = np.zeros(base, dtype=np.int8)
        for m, r in classes:
            covered[np.arange(r, base, m)] = 1
        return int(np.sum(covered == 0))
    
    # Get greedy solution
    best_classes = greedy(divisors)
    best_score = compute_score(best_classes)
    
    # Try ILP
    try:
        start_time = time.time()
        
        # Build ILP
        # Variables: x_{m,r} for each divisor m and residue r in [0, m)
        # Plus y_j for each residue j in [0, base)
        
        var_index = {}
        n_vars = 0
        for m in divisors:
            for r in range(m):
                var_index[(m, r)] = n_vars
                n_vars += 1
        for j in range(base):
            var_index[('y', j)] = n_vars
            n_vars += 1
        
        # This is ~170k variables, likely too large
        # Skip ILP if too large
        if n_vars > 50000:
            raise Exception("Too many variables for ILP")
        
        # ... (ILP formulation)
        
        elapsed = time.time() - start_time
        if elapsed > 60:
            raise Exception("ILP too slow")
        
    except Exception:
        # Fall back to greedy + local search
        pass
    
    # Local search
    classes = [list(c) for c in best_classes]
    improved = True
    max_iterations = 10
    iteration = 0
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        for i in range(len(classes)):
            m, r = classes[i]
            best_r = r
            best_local_score = best_score
            for r_new in range(m):
                if r_new == r:
                    continue
                classes[i][1] = r_new
                new_score = compute_score(classes)
                if new_score < best_local_score:
                    best_local_score = new_score
                    best_r = r_new
            if best_r != r:
                classes[i][1] = best_r
                best_score = best_local_score
                improved = True
            else:
                classes[i][1] = r
    
    return classes
