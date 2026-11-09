import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix
import time

def construct():
    base = 51975
    
    # Get all divisors of base greater than 1
    divisors = []
    for d in range(2, base + 1):
        if base % d == 0:
            divisors.append(d)
            
    def compute_score(classes):
        covered = np.zeros(base, dtype=np.int8)
        for m, r in classes:
            covered[np.arange(r, base, m)] = 1
        return int(np.sum(covered == 0))
        
    def fast_greedy():
        covered = np.zeros(base, dtype=np.int8)
        classes = []
        available_moduli = divisors.copy()
        
        while available_moduli:
            best_m = None
            best_r = None
            best_gain = -1
            
            for m in available_moduli:
                reshaped = covered.reshape(-1, m)
                gains = np.sum(reshaped == 0, axis=0)
                max_gain = np.max(gains)
                if max_gain > best_gain:
                    best_gain = max_gain
                    best_m = m
                    best_r = np.argmax(gains)
            
            if best_gain <= 0:
                break
                
            classes.append([best_m, best_r])
            available_moduli.remove(best_m)
            covered[np.arange(best_r, base, best_m)] = 1
            
        return classes

    best_classes = fast_greedy()
    best_score = compute_score(best_classes)
    
    if best_score == 0:
        return best_classes
        
    # Try ILP
    try:
        start_time = time.time()
        
        var_index = {}
        n_vars = 0
        for m in divisors:
            for r in range(m):
                var_index[(m, r)] = n_vars
                n_vars += 1
        for j in range(base):
            var_index[('z', j)] = n_vars
            n_vars += 1
            
        c = np.zeros(n_vars)
        for j in range(base):
            c[var_index[('z', j)]] = -1.0
            
        n_rows = base + len(divisors)
        
        # Use COO format for speed
        row_idx = []
        col_idx = []
        data = []
        
        # Constraint 1: z_j - sum x <= 0
        for j in range(base):
            row = j
            # z_j coeff 1
            row_idx.append(row)
            col_idx.append(var_index[('z', j)])
            data.append(1.0)
            
            for m in divisors:
                r = j % m
                row_idx.append(row)
                col_idx.append(var_index[(m, r)])
                data.append(-1.0)
                
        # Constraint 2: sum x_{m,r} <= 1
        for i, m in enumerate(divisors):
            row = base + i
            for r in range(m):
                row_idx.append(row)
                col_idx.append(var_index[(m, r)])
                data.append(1.0)
                
        A = coo_matrix((data, (row_idx, col_idx)), shape=(n_rows, n_vars))
        b_ub = np.zeros(n_rows)
        
        bounds = Bounds(0, 1)
        integrality = np.ones(n_vars)
        constraints = LinearConstraint(A, -np.inf, b_ub)
        
        res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.status == 0:
            x = res.x
            new_classes = []
            for m in divisors:
                selected_r = None
                for r in range(m):
                    if x[var_index[(m, r)]] > 0.5:
                        selected_r = r
                        break
                if selected_r is not None:
                    new_classes.append([m, selected_r])
            
            new_score = compute_score(new_classes)
            if new_score < best_score:
                best_classes = new_classes
                best_score = new_score
                
    except Exception:
        pass
        
    return best_classes
