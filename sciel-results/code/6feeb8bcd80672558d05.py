import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import time

def construct():
    base = 51975
    start_time = time.time()
    
    # Get all divisors > 1 of base
    divisors = [d for d in range(2, base + 1) if base % d == 0]
    # Sort by size - smaller moduli cover more residues on average
    divisors.sort()
    
    # Strategy: try several subsets of moduli, optimize residues via ILP
    best_classes = None
    best_uncovered = base  # worst case
    
    # Precompute for each divisor, the residues it covers for each choice
    # For ILP, we need to know which (m, r) pairs cover which residues
    
    # Try different subset sizes
    # We'll try a few curated subsets
    
    # Subset 1: smallest divisors (they cover more residues each)
    # Subset 2: mix of small and large
    # Subset 3: greedy selected
    
    def evaluate_classes(classes):
        """Count uncovered residues."""
        covered = np.zeros(base, dtype=bool)
        for m, r in classes:
            covered[r::m] = True
        return (~covered).sum()
    
    def solve_ilp(moduli):
        """Solve ILP to find best residues for given moduli.
        Returns list of [m, r] or None if failed."""
        if not moduli:
            return []
        
        n_mod = len(moduli)
        # For each modulus, possible residues
        residue_counts = [m for m in moduli]
        total_vars = sum(residue_counts)
        
        if total_vars > 5000:
            return None  # Too large
        
        # Variable indexing: for modulus i, residues 0..m_i-1
        # var_index[i][r] = global index
        var_index = []
        idx = 0
        for i, m in enumerate(moduli):
            var_index.append([idx + r for r in range(m)])
            idx += m
        
        # Objective: maximize coverage = sum of y_x for x in 0..base-1
        # But y_x are auxiliary. Instead, we can use a different approach.
        
        # Alternative: use a heuristic ILP where we directly maximize 
        # the number of covered residues by introducing y variables.
        
        # Total variables: total_vars (x) + base (y)
        n_y = base
        total_all_vars = total_vars + n_y
        
        if total_all_vars > 10000:
            return None
        
        # Objective: maximize sum y_x -> minimize -sum y_x
        c = np.zeros(total_all_vars)
        c[total_vars:] = -1.0  # y variables: minimize -y = maximize y
        
        # Constraints:
        # 1. For each modulus i: sum_r x_{i,r} = 1
        # 2. For each residue x: y_x <= sum_{i,r: x≡r mod m_i} x_{i,r}
        
        # Build constraint matrix sparsely
        # Use scipy sparse
        from scipy.sparse import lil_matrix, vstack
        
        # Constraint 1: one residue per modulus
        # n_mod rows, each has 1s at the residue positions for that modulus
        A1 = lil_matrix((n_mod, total_vars))
        for i, m in enumerate(moduli):
            for r in range(m):
                A1[i, var_index[i][r]] = 1
        
        # Constraint 2: for each x, y_x - sum_{i,r: x≡r mod m_i} x_{i,r} <= 0
        # This is: -sum x_{i,r} + y_x <= 0
        A2 = lil_matrix((n_y, total_vars + n_y))
        for x in range(n_y):
            # Find all (i, r) such that x % m_i == r
            for i, m in enumerate(moduli):
                r = x % m
                A2[x, var_index[i][r]] = -1
            A2[x, total_vars + x] = 1
        
        A = vstack([A1, A2])
        b = np.concatenate([np.ones(n_mod), np.zeros(n_y)])
        
        # Bounds: all variables in [0, 1]
        lb = np.zeros(total_all_vars)
        ub = np.ones(total_all_vars)
        # x variables should be integer (0 or 1), y can be continuous [0,1]
        # For milp, integrality: 1 for integer, 0 for continuous
        integrality = np.zeros(total_all_vars)
        integrality[:total_vars] = 1  # x variables are binary
        
        # Linear constraints: A @ vars <= b for A2, A1 @ vars = 1
        # milp handles equality via lb=ub in LinearConstraint
        # Let's use LinearConstraint with separate bounds
        
        # Actually, let's just use the A, b form with milp's constraints parameter
        # milp accepts LinearConstraint or (A, b) pairs
        
        # For simplicity, let's use the direct form
        constraints = LinearConstraint(A, -np.inf, b)
        # But A1 should be equality: A1 @ x = 1
        # Let me restructure: use two separate constraint groups
        
        # Actually scipy milp LinearConstraint takes (A, lb, ub) where lb <= A@x <= ub
        # For A1: lb=1, ub=1
        # For A2: lb=-inf, ub=0
        
        # Combine: we need separate constraints
        # Let's just use the A, b <= form by converting A1 to A1@x <= 1 and -A1@x <= -1
        # That doubles constraints. Let me just try with LinearConstraint
        
        constraints_list = []
        # A1 constraints: equality
        A1_csr = A1.tocsr()
        constraints_list.append(LinearConstraint(A1_csr, 1.0, 1.0))
        # A2 constraints: <= 0
        A2_csr = A2.tocsr()
        constraints_list.append(LinearConstraint(A2_csr, -np.inf, 0.0))
        
        try:
            res = milp(
                c=c,
                constraints=constraints_list,
                integrality=integrality,
                bounds=Bounds(lb, ub)
            )
        except Exception:
            return None
        
        if not res.success:
            return None
        
        # Extract solution
        classes = []
        for i, m in enumerate(moduli):
            # Find which residue is 1
            best_r = 0
            best_val = -1
            for r in range(m):
                v = res.x[var_index[i][r]]
                if v > best_val:
                    best_val = v
                    best_r = r
            classes.append([m, int(best_r)])
        
        return classes
    
    # Try several modulus subsets
    
    # Subset A: 10 smallest divisors
    subset_A = divisors[:10]
    # Subset B: 15 smallest
    subset_B = divisors[:15]
    # Subset C: 8 smallest
    subset_C = divisors[:8]
    # Subset D: mix - smallest 5 + some medium
    subset_D = divisors[:5] + divisors[10:15]
    # Subset E: 20 smallest
    subset_E = divisors[:20]
    
    subsets = [subset_C, subset_A, subset_B, subset_D, subset_E]
    
    for subset in subsets:
        if time.time() - start_time > 100:
            break
        
        classes = solve_ilp(subset)
        if classes:
            uncovered = evaluate_classes(classes)
            if uncovered < best_uncovered:
                best_uncovered = uncovered
                best_classes = classes
                # Print for debugging
                print(f"Subset size {len(subset)}, uncovered: {uncovered}", flush=True)
    
    # Also try a pure greedy as fallback
    if best_classes is None or time.time() - start_time < 50:
        covered = np.zeros(base, dtype=bool)
        classes_greedy = []
        for m in divisors:
            if time.time() - start_time > 110:
                break
            best_r = 0
            best_count = -1
            for r in range(m):
                count = (~covered[r::m]).sum()
                if count > best_count:
                    best_count = count
                    best_r = r
            classes_greedy.append([m, best_r])
            covered[best_r::m] = True
        
        uncovered_greedy = (~covered).sum()
        if uncovered_greedy < best_uncovered:
            best_uncovered = uncovered_greedy
            best_classes = classes_greedy
    
    if best_classes is None:
        # Ultimate fallback: single class
        best_classes = [[3, 0]]
    
    return best_classes
