import time
from fractions import Fraction
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

DEADLINE = 85.0
_START = None

def _start_clock():
    global _START
    _START = time.time()

def _deadline_hit():
    global _START
    if _START is None:
        return False
    return (time.time() - _START) > DEADLINE

def _is_reciprocal_subsum_free(S):
    """
    Checks if S is reciprocal-subsum-free.
    """
    if not S:
        return False
    
    S = list(set(S))
    n = len(S)
    recips = {v: Fraction(1, v) for v in S}
    
    for a in S:
        target = recips[a]
        others = [recips[b] for b in S if b != a]
        
        m = len(others)
        mid = m // 2
        left, right = others[:mid], others[mid:]
        
        L = {Fraction(0)}
        for v in left:
            L = L | {x + v for x in L}
        
        R = {Fraction(0)}
        for v in right:
            R = R | {x + v for x in R}
            
        R_set = set(R)
        for l in L:
            if (target - l) in R_set:
                return False
    return True

def _find_all_bad_configs():
    """
    Finds all (a, T) such that a in [1, 24] and T is a subset of [1, 24]\{a}
    where sum(1/b for b in T) == 1/a.
    Returns a list of (a, frozenset(T))
    """
    all_numbers = list(range(1, 25))
    bad_configs = []
    
    for a in all_numbers:
        if _deadline_hit():
            break
        target = Fraction(1, a)
        others = [v for v in all_numbers if v != a]
        
        m = len(others)
        mid = m // 2
        left_vals, right_vals = others[:mid], others[mid:]
        
        # Generate all subset sums for left half, keeping all subsets for each sum
        # Using a dict of Fraction -> list of frozensets
        left_sums = {Fraction(0): [frozenset()]}
        for v in left_vals:
            new_sums = {}
            # Copy existing
            for s, subsets in left_sums.items():
                if s not in new_sums:
                    new_sums[s] = []
                new_sums[s].extend(subsets)
            # Add new subsets including v
            for s, subsets in left_sums.items():
                new_sum = s + Fraction(1, v)
                if new_sum <= target:
                    if new_sum not in new_sums:
                        new_sums[new_sum] = []
                    for sub in subsets:
                        new_sums[new_sum].append(sub | frozenset([v]))
            left_sums = new_sums
            
        # Generate all subset sums for right half
        right_sums = {Fraction(0): [frozenset()]}
        for v in right_vals:
            new_sums = {}
            for s, subsets in right_sums.items():
                if s not in new_sums:
                    new_sums[s] = []
                new_sums[s].extend(subsets)
            for s, subsets in right_sums.items():
                new_sum = s + Fraction(1, v)
                if new_sum <= target:
                    if new_sum not in new_sums:
                        new_sums[new_sum] = []
                    for sub in subsets:
                        new_sums[new_sum].append(sub | frozenset([v]))
            right_sums = new_sums
        
        # Combine
        for l_sum, l_subsets in left_sums.items():
            needed = target - l_sum
            if needed in right_sums:
                r_subsets = right_sums[needed]
                for l_sub in l_subsets:
                    for r_sub in r_subsets:
                        T = l_sub | r_sub
                        if T: # Exclude empty set if possible, though sum 0 != 1/a
                            bad_configs.append((a, T))
                            
    return bad_configs

def _solve_ilp(bad_configs):
    n = 24
    # Objective: maximize sum(x_i) -> minimize -sum(x_i)
    c = -np.ones(n)
    
    A_rows = []
    b_ub = []
    b_lb = []
    
    # Add constraints
    # For each (a, T), x_a + sum(x_t for t in T) <= len(T)
    # Note: We might have duplicate constraints. We can deduplicate by the set of indices.
    seen_constraints = set()
    
    for a, T in bad_configs:
        idx_a = a - 1
        idxs_T = [t - 1 for t in T]
        
        # Create a canonical key for the constraint
        # The constraint is defined by the set of variables involved and the bound.
        # Since the structure is fixed (coeff 1 for a and T, bound |T|),
        # the key is just the sorted tuple of all involved indices.
        involved = sorted([idx_a] + idxs_T)
        key = tuple(involved)
        
        if key in seen_constraints:
            continue
        seen_constraints.add(key)
        
        row = np.zeros(n)
        row[idx_a] = 1.0
        for idx in idxs_T:
            row[idx] += 1.0
            
        A_rows.append(row)
        b_ub.append(len(T))
        b_lb.append(0)
        
    if not A_rows:
        # No constraints, max size is 24
        return set(range(1, 25))
        
    A = np.array(A_rows)
    lb = np.array(b_lb)
    ub_arr = np.array(b_ub)
    
    constraints = LinearConstraint(A, lb, ub_arr)
    integrality = np.ones(n)
    bounds = Bounds(0, 1)
    
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        if res.status == 0:
            sol = res.x
            ilp_set = set([i + 1 for i in range(n) if sol[i] > 0.5])
            return ilp_set
    except Exception:
        pass
        
    return set()

def _local_search_refinement(best_set, all_numbers, deadline_check):
    """
    Try to add elements to best_set using local search.
    """
    final_set = list(best_set)
    remaining = [v for v in all_numbers if v not in best_set]
    remaining.sort(reverse=True)
    
    improved = True
    while improved and not deadline_check():
        improved = False
        for v in remaining:
            if deadline_check():
                break
            trial = final_set + [v]
            if _is_reciprocal_subsum_free(trial):
                final_set.append(v)
                remaining.remove(v)
                improved = True
                break
    
    return set(final_set)

def construct():
    _start_clock()
    
    # 1. Find all bad configs
    bad_configs = _find_all_bad_configs()
    
    # 2. Solve ILP
    best_set = _solve_ilp(bad_configs)
    
    # 3. If ILP fails or returns empty, use greedy
    if not best_set:
        base_set = []
        for v in range(24, 0, -1):
            if _is_reciprocal_subsum_free([v] + base_set):
                base_set.append(v)
        best_set = set(base_set)
        
    # 4. Refinement
    if not _deadline_hit():
        all_numbers = list(range(1, 25))
        best_set = _local_search_refinement(best_set, all_numbers, _deadline_hit)
        
    # 5. Ensure non-empty
    if not best_set:
        best_set = {24}
        
    return list(best_set)

if __name__ == "__main__":
    print(construct())
