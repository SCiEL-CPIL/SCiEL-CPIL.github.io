import time
from math import gcd
from functools import lru_cache
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

_START = time.time()
_DEADLINE = 80.0

def _time_ok():
    return (time.time() - _START) < _DEADLINE

def construct():
    LCM = 232792560
    weights = [LCM // k for k in range(1, 23)]
    n_elem = 22
    indices = list(range(n_elem))
    
    # Helper to verify dissociation
    def verify_dissociated(selected_indices):
        w = [weights[i] for i in selected_indices]
        n = len(w)
        sums = set()
        for mask in range(1 << n):
            s = 0
            for b in range(n):
                if mask & (1 << b):
                    s += w[b]
            if s in sums:
                return False
            sums.add(s)
        return True

    # Find a conflict (D, E) if S is not dissociated
    # Returns (D, E) as sets of indices in the original [1..22] space, or None
    def find_conflict(selected_indices):
        w = [weights[i] for i in selected_indices]
        n = len(w)
        sums = {}
        for mask in range(1 << n):
            s = 0
            for b in range(n):
                if mask & (1 << b):
                    s += w[b]
            if s in sums:
                # Conflict found between mask and sums[s]
                mask1 = mask
                mask2 = sums[s]
                # Convert to sets of original indices
                set1 = set()
                set2 = set()
                for b in range(n):
                    if mask1 & (1 << b):
                        set1.add(selected_indices[b])
                    if mask2 & (1 << b):
                        set2.add(selected_indices[b])
                # D = set1 - set2, E = set2 - set1
                D = set1 - set2
                E = set2 - set1
                if D and E:
                    return (D, E)
            else:
                sums[s] = mask
        return None

    # Precompute small conflicts for initial MILP constraints
    # D, E disjoint, sum(D) == sum(E)
    small_conflicts = set()
    # Check pairs of small sets
    # We iterate over all pairs of disjoint subsets A, B with |A|<=2, |B|<=2 initially
    # Actually, let's just rely on the lazy constraint approach starting with empty or very few constraints
    # But starting with no constraints might yield a large invalid set, requiring many iterations.
    # Let's add constraints for |D|<=1, |E|<=2 and vice versa.
    
    # Generate all subsets of size 1 and 2
    subsets_1 = [frozenset([i]) for i in range(n_elem)]
    subsets_2 = [frozenset([i, j]) for i in range(n_elem) for j in range(i+1, n_elem)]
    
    # Check w_i == w_j + w_k
    for i in range(n_elem):
        for j in range(n_elem):
            if i == j: continue
            for k in range(n_elem):
                if i == k or j == k: continue
                if weights[i] == weights[j] + weights[k]:
                    D = frozenset([i])
                    E = frozenset([j, k])
                    small_conflicts.add((D, E))
                    small_conflicts.add((E, D)) # Symmetry, though D,E disjoint
    
    # Check w_i + w_j == w_k + w_l
    for i in range(n_elem):
        for j in range(i+1, n_elem):
            s1 = weights[i] + weights[j]
            for k in range(n_elem):
                if k == i or k == j: continue
                for l in range(k+1, n_elem):
                    if l == i or l == j: continue
                    s2 = weights[k] + weights[l]
                    if s1 == s2:
                        D = frozenset([i, j])
                        E = frozenset([k, l])
                        if D.isdisjoint(E):
                            small_conflicts.add((D, E))
                            small_conflicts.add((E, D))

    # MILP Solver
    def solve_milp(conflicts):
        # Variables: x_0..x_21 (binary), y_0..y_m (binary)
        # m = number of conflicts
        # Objective: max sum(x) -> min -sum(x)
        
        if not conflicts:
            # No conflicts, return all 22? 
            # But we know 22 is not dissociated.
            # We will find conflicts in verification.
            pass
            
        n_vars_x = n_elem
        n_vars_y = len(conflicts)
        n_vars = n_vars_x + n_vars_y
        
        c = np.zeros(n_vars)
        c[:n_vars_x] = -1.0 # Maximize sum of x
        
        A = []
        lb = []
        ub = []
        
        # Constraints for each conflict (D, E)
        # y_D + y_E <= 1
        # y_D <= x_i for i in D
        # y_D >= sum(x_i for i in D) - (|D|-1)
        # Similarly for y_E
        
        for idx, (D, E) in enumerate(conflicts):
            y_D_idx = n_vars_x + idx
            y_E_idx = n_vars_x + idx + 1 # Wait, I need 2 y vars per conflict?
            # Yes, one for D, one for E.
            # So n_vars_y should be 2 * len(conflicts)
            pass
        
        # Recalculate n_vars_y
        n_vars_y = 2 * len(conflicts)
        n_vars = n_vars_x + n_vars_y
        c = np.zeros(n_vars)
        c[:n_vars_x] = -1.0
        
        A_rows = []
        lb_list = []
        ub_list = []
        
        for idx, (D, E) in enumerate(conflicts):
            y_D_idx = n_vars_x + 2*idx
            y_E_idx = n_vars_x + 2*idx + 1
            
            # y_D + y_E <= 1
            row = np.zeros(n_vars)
            row[y_D_idx] = 1
            row[y_E_idx] = 1
            A_rows.append(row)
            lb_list.append(-np.inf)
            ub_list.append(1)
            
            # y_D <= x_i for i in D
            for i in D:
                row = np.zeros(n_vars)
                row[i] = -1
                row[y_D_idx] = 1
                A_rows.append(row)
                lb_list.append(-np.inf)
                ub_list.append(0) # y_D - x_i <= 0
                
            # y_D >= sum(x_i) - (|D|-1)
            # sum(x_i) - y_D <= |D|-1
            row = np.zeros(n_vars)
            for i in D:
                row[i] = 1
            row[y_D_idx] = -1
            A_rows.append(row)
            lb_list.append(-np.inf)
            ub_list.append(len(D) - 1)
            
            # y_E <= x_j for j in E
            for j in E:
                row = np.zeros(n_vars)
                row[j] = -1
                row[y_E_idx] = 1
                A_rows.append(row)
                lb_list.append(-np.inf)
                ub_list.append(0)
                
            # y_E >= sum(x_j) - (|E|-1)
            row = np.zeros(n_vars)
            for j in E:
                row[j] = 1
            row[y_E_idx] = -1
            A_rows.append(row)
            lb_list.append(-np.inf)
            ub_list.append(len(E) - 1)
            
        if not A_rows:
            # No constraints, trivial solution
            return list(range(n_elem))
            
        A = np.array(A_rows)
        lb = np.array(lb_list)
        ub = np.array(ub_list)
        
        constraint = LinearConstraint(A, lb, ub)
        integrality = np.zeros(n_vars)
        integrality[:] = 1
        bounds = Bounds(0, 1)
        
        res = milp(c, constraints=constraint, integrality=integrality, bounds=bounds)
        
        if res.status == 0:
            x = res.x[:n_vars_x]
            selected = [i for i in range(n_elem) if x[i] > 0.5]
            return selected
        else:
            return None

    best_set = None
    best_size = 0
    
    # Initial Greedy to get a lower bound
    # Sort by weight desc
    order = sorted(indices, key=lambda i: weights[i], reverse=True)
    current_set = []
    current_sums = {0}
    for i in order:
        if not _time_ok(): break
        w = weights[i]
        new_sums = set()
        valid = True
        for s in current_sums:
            if s + w in current_sums:
                valid = False
                break
            new_sums.add(s + w)
        if valid:
            current_sums = current_sums | new_sums
            current_set.append(i)
            
    if current_set:
        best_set = current_set
        best_size = len(current_set)
        
    # MILP Iterative Improvement
    conflicts = list(small_conflicts)
    
    for iter in range(10):
        if not _time_ok(): break
        
        sol = solve_milp(conflicts)
        if sol is None:
            break
            
        if len(sol) <= best_size:
            # No improvement, but maybe valid?
            if verify_dissociated(sol):
                # If valid but not larger, we can stop if we are confident?
                # Or just continue to find larger?
                # MILP maximizes size, so if it returns size <= best, it's likely stuck or optimal for current constraints.
                # But if it's valid, we can accept it.
                pass
                
        if verify_dissociated(sol):
            if len(sol) > best_size:
                best_set = sol
                best_size = len(sol)
            break # Found a valid solution with max size for current constraints
            
        # If invalid, find conflict and add
        conflict = find_conflict(sol)
        if conflict is None:
            break # Should not happen if verify is correct
            
        D, E = conflict
        # Convert to frozensets of indices
        D_f = frozenset(D)
        E_f = frozenset(E)
        
        # Add to conflicts if not present
        pair = (D_f, E_f)
        if pair not in conflicts:
            conflicts.append(pair)
            # Also add reverse? The constraint is symmetric in effect if we handle D,E correctly.
            # y_D + y_E <= 1 is symmetric.
            # So (D,E) and (E,D) are redundant if we just use one.
            # But my solver creates y_D and y_E for the pair.
            # So just adding (D,E) is enough.
            
    # Final Fallback if MILP didn't improve
    if not best_set:
        best_set = [0, 1, 2, 4, 6, 10, 12, 16, 18] # Fallback
        
    result = sorted([i + 1 for i in best_set])
    if not result:
        result = [1]
        
    return result
