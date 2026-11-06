import time
import random
from fractions import Fraction
from itertools import combinations
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

DEADLINE = 85.0  # Safety margin under 90s
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
    Condition: For every a in S, 1/a != sum(1/b for b in T) for any subset T of S\{a}.
    """
    if not S:
        return False
    
    S = list(set(S))
    n = len(S)
    recips = {v: Fraction(1, v) for v in S}
    
    # For each element a in S, check if 1/a is a subset sum of reciprocals of other elements
    for a in S:
        target = recips[a]
        others = [recips[b] for b in S if b != a]
        
        # Meet-in-the-middle subset sum check for small sets
        m = len(others)
        if m > 16:
            # Fallback for large sets, though unlikely to be called in the final validation of optimal sets
            pass
            
        mid = m // 2
        left, right = others[:mid], others[mid:]
        
        L = {Fraction(0)}
        for v in left:
            L = L | {x + v for x in L}
        
        R = {Fraction(0)}
        for v in right:
            R = R | {x + v for x in R}
            
        # Check if target - l == r for some l in L, r in R
        R_set = set(R)
        for l in L:
            if (target - l) in R_set:
                return False
    return True

def _find_bad_configs():
    """
    Finds all (a, T) such that a in [1, 24] and T is a subset of [1, 24]\{a}
    where sum(1/b for b in T) == 1/a.
    Returns a list of (a, tuple(T))
    """
    all_numbers = list(range(1, 25))
    bad_configs = []
    
    for a in all_numbers:
        target = Fraction(1, a)
        others = [v for v in all_numbers if v != a]
        
        # We need to find subsets T of 'others' such that sum(1/b) = target.
        # Meet in the middle is efficient here.
        m = len(others)
        mid = m // 2
        left_vals, right_vals = others[:mid], others[mid:]
        
        # Generate all subset sums for left half
        left_sums = {Fraction(0): frozenset()}
        for v in left_vals:
            new_sums = {}
            for s, subset in left_sums.items():
                new_sum = s + Fraction(1, v)
                # We only care about sums <= target to prune, but Fraction comparison is cheap
                # However, subset sums can be many. Let's just generate all.
                # Actually, we can prune sums > target since all reciprocals are positive.
                if new_sum <= target:
                    new_sums[new_sum] = subset | frozenset([v])
            # Keep the existing sums too
            for s, subset in left_sums.items():
                if s not in new_sums:
                    new_sums[s] = subset
            left_sums = new_sums
            
        # Generate all subset sums for right half
        right_sums = {Fraction(0): frozenset()}
        for v in right_vals:
            new_sums = {}
            for s, subset in right_sums.items():
                new_sum = s + Fraction(1, v)
                if new_sum <= target:
                    new_sums[new_sum] = subset | frozenset([v])
            for s, subset in right_sums.items():
                if s not in new_sums:
                    new_sums[s] = subset
            right_sums = new_sums
        
        # Combine
        for l_sum, l_subset in left_sums.items():
            needed = target - l_sum
            if needed in right_sums:
                r_subset = right_sums[needed]
                T = l_subset | r_subset
                if T: # Non-empty subset
                    bad_configs.append((a, tuple(sorted(T))))
                    
    return bad_configs

def construct():
    _start_clock()
    
    # 1. Generate a strong baseline using greedy descending order
    base_set = []
    recips_in_set = []
    for v in range(24, 0, -1):
        trial_recips = recips_in_set + [Fraction(1, v)]
        if _is_reciprocal_subsum_free([v] + base_set):
            base_set.append(v)
            recips_in_set.append(Fraction(1, v))
    
    best_set = set(base_set)
    
    # 2. Try ILP for global optimization
    try:
        if not _deadline_hit():
            bad_configs = _find_bad_configs()
            
            n = 24
            # x_i = 1 if i+1 is in the set
            # Maximize sum(x_i)
            c = -np.ones(n) # Minimize -sum(x_i)
            
            # Constraints:
            # For each bad config (a, T): x_a + sum_{b in T} x_b <= |T|
            # Note: Indices are 0-based in numpy, values are 1-based.
            A_rows = []
            b_ub = []
            b_lb = []
            
            for a, T in bad_configs:
                idx_a = a - 1
                idxs_T = [t - 1 for t in T]
                
                row = np.zeros(n)
                row[idx_a] = 1.0
                for idx in idxs_T:
                    row[idx] += 1.0
                    
                A_rows.append(row)
                b_ub.append(len(T))
                b_lb.append(0)
                
            if A_rows:
                A = np.array(A_rows)
                lb = np.array(b_lb)
                ub = np.array(b_ub)
            else:
                A = np.empty((0, n))
                lb = np.empty(0)
                ub = np.empty(0)
                
            constraints = LinearConstraint(A, lb, ub)
            integrality = np.ones(n)
            bounds = Bounds(0, 1)
            
            res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
            
            if res.status == 0:
                sol = res.x
                # Threshold to handle floating point precision
                ilp_set = set([i + 1 for i in range(n) if sol[i] > 0.5])
                
                # Validate the ILP solution just in case
                if _is_reciprocal_subsum_free(list(ilp_set)):
                    if len(ilp_set) > len(best_set):
                        best_set = ilp_set
                    # If ILP finds a larger set, it's likely optimal.
                    # We can stop here if we have a very large set, e.g., > 20?
                    # But let's keep the heuristic fallback if ILP fails.
                        
    except Exception as e:
        pass # Fallback to heuristic if ILP fails or times out
    
    # 3. Refinement of the best set found (Heuristic)
    # If ILP didn't run or didn't find anything better, try local search on best_set
    if not _deadline_hit():
        all_numbers = list(range(1, 25))
        remaining = [v for v in all_numbers if v not in best_set]
        remaining.sort(reverse=True)
        
        final_set = list(best_set)
        for v in remaining:
            if _deadline_hit():
                break
            trial = final_set + [v]
            if _is_reciprocal_subsum_free(trial):
                final_set.append(v)
                
        if len(final_set) > len(best_set):
            best_set = set(final_set)
            
    # 4. Ensure non-empty
    if not best_set:
        best_set = {24}
        
    return list(best_set)

if __name__ == "__main__":
    print(construct())
