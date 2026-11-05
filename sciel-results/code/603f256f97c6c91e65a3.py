"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds, linprog
from scipy import sparse

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    N = 256
    
    # Precompute all 3-APs: (a, b, c) where a + c = 2b, a < b < c
    # For each element x, precompute which pairs (a,c) would form an AP with x as middle,
    # and which pairs would form AP with x as end
    # For fast conflict checking: for each x, store the set of elements y such that
    # x and y are in some 3-AP together (i.e., there exists z with {x,y,z} being a 3-AP)
    
    # ap_conflicts[x] = set of y such that x, y are part of some 3-AP
    # Actually, we need: can we add x to set S? 
    # x conflicts with S if there exist a, b in S such that {a, b, x} is a 3-AP
    # This happens if:
    #   - x is the middle: a + c = 2x for some a, c in S (a < x < c)
    #   - x is the end: 2b - a = x for some a, b in S (a < b < x) or 2a - b = x (b < a < x, but x smallest)
    
    # For efficiency, precompute for each x:
    #   middle_pairs[x] = list of (a, c) where a + c = 2x, a < x < c
    #   end_as_largest[x] = list of (a, b) where 2b - a = x, a < b < x  
    #   end_as_smallest[x] = list of (b, c) where 2b - c = x, x < b < c  => c = 2b - x, need x < b < c
    
    # Actually simpler: for checking if x can be added to S:
    # For each y in S:
    #   Check if x, y are in a 3-AP with some third element z in S
    #   The third element z is determined by x, y:
    #     If x < y: z = 2y - x (x is smallest) or z = x + (y-x) = 2y-x... wait
    #     The three elements in AP: p, q, r with p+r=2q
    #     Given x and y, possible third elements:
    #       If x < y: z = 2y - x (AP: x, y, z) or z = 2x - y (AP: z, x, y) - but 2x-y < x so z < x
    #       If x > y: z = 2x - y (AP: y, x, z) or z = 2y - x (AP: z, y, x) - but 2y-x < y so z < y
    #     So for pair (x, y) with x < y, the possible third elements are:
    #       z1 = 2y - x (x is first, y is middle, z1 is last) - valid if z1 <= N
    #       z2 = 2x - y (z2 is first, x is middle, y is last) - valid if z2 >= 1
    #     So x and y conflict if z1 in S or z2 in S (and z1 != x, z1 != y, z2 != x, z2 != y)
    
    # For fast checking: when adding x to S, for each y in S:
    #   z1 = 2*max(x,y) - min(x,y)  # the "forward" extension
    #   z2 = 2*min(x,y) - max(x,y)  # the "backward" extension  
    #   If z1 in S and z1 != x and z1 != y: conflict
    #   If z2 in S and z2 != x and z2 != y: conflict
    # But this is O(|S|) per check, which is fine.
    
    def can_add(x, S_set):
        """Check if x can be added to S_set without creating a 3-AP."""
        for y in S_set:
            if y == x:
                continue
            lo = min(x, y)
            hi = max(x, y)
            # Forward: lo, mid, hi -> next = 2*hi - lo
            z1 = 2 * hi - lo
            if z1 != x and z1 != y and z1 in S_set:
                return False
            # Backward: lo, mid, hi -> prev = 2*lo - hi
            z2 = 2 * lo - hi
            if z2 != x and z2 != y and z2 in S_set:
                return False
        return True
    
    # Greedy initialization: add elements in order of 1/x (smallest x first for max reciprocal)
    selected = set()
    for x in range(1, N + 1):
        if can_add(x, selected):
            selected.add(x)
    
    def compute_weight(sol):
        return sum(Fraction(1, x) for x in sol)
    
    best_set = set(selected)
    best_weight = compute_weight(best_set)
    
    # Local search: try swaps (remove one, add one) and additions
    improved = True
    while improved and time.time() < deadline - 40:
        improved = False
        sol_list = sorted(best_set)
        
        # Try additions first (cheap)
        for y in range(1, N + 1):
            if y in best_set:
                continue
            if can_add(y, best_set):
                new_weight = best_weight + Fraction(1, y)
                if new_weight > best_weight:
                    best_set.add(y)
                    best_weight = new_weight
                    improved = True
        
        # Try swaps
        for idx in range(len(sol_list)):
            if time.time() > deadline - 40:
                break
            removed = sol_list[idx]
            reduced = best_set - {removed}
            reduced_weight = best_weight - Fraction(1, removed)
            
            best_add = None
            best_add_weight = reduced_weight
            for y in range(1, N + 1):
                if y in reduced:
                    continue
                if can_add(y, reduced):
                    w = reduced_weight + Fraction(1, y)
                    if w > best_add_weight:
                        best_add_weight = w
                        best_add = y
            
            if best_add is not None and best_add_weight > best_weight:
                best_set.remove(removed)
                best_set.add(best_add)
                best_weight = best_add_weight
                improved = True
    
    # Now try MILP with the current best as warm start
    if time.time() < deadline - 5:
        try:
            # Build constraint matrix for all 3-APs
            rows = []
            cols = []
            data = []
            
            for d in range(1, N // 2 + 1):
                max_a = N - 2 * d
                for a in range(1, max_a + 1):
                    b = a + d
                    c = a + 2 * d
                    idx_a = a - 1
                    idx_b = b - 1
                    idx_c = c - 1
                    
                    row_idx = len(data) // 3
                    rows.extend([row_idx, row_idx, row_idx])
                    cols.extend([idx_a, idx_b, idx_c])
                    data.extend([1, 1, 1])
            
            n_constraints = len(data) // 3
            A = sparse.csr_matrix((data, (rows, cols)), shape=(n_constraints, N))
            b_vec = np.full(n_constraints, 2.0)
            
            # Objective: maximize sum(1/i * x_i) => minimize -sum(1/i * x_i)
            c_vec = -np.array([1.0 / i for i in range(1, N + 1)])
            
            bounds = Bounds(lb=0, ub=1)
            integrality = np.ones(N)
            
            constraints = LinearConstraint(A, lb=-np.inf, ub=b_vec)
            
            time_left = deadline - time.time() - 2
            if time_left > 1:
                res = milp(c=c_vec, constraints=constraints, integrality=integrality, bounds=bounds,
                           options={"time_limit": time_left, "mip_rel_gap": 0.005})
                
                if res.status == 0 or res.status == 2:  # Optimal or best found
                    x_sol = res.x
                    milp_solution = set(i + 1 for i in range(N) if x_sol[i] > 0.5)
                    if milp_solution:
                        milp_weight = compute_weight(milp_solution)
                        if milp_weight > best_weight:
                            best_weight = milp_weight
                            best_set = milp_solution
                            
        except Exception:
            pass
    
    # Final polish: try more local search on the best solution
    if time.time() < deadline - 2:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            sol_list = sorted(best_set)
            
            # Try 2-for-2 swaps
            for i in range(len(sol_list)):
                if time.time() > deadline - 2:
                    break
                for j in range(i + 1, len(sol_list)):
                    if time.time() > deadline - 2:
                        break
                    x1, x2 = sol_list[i], sol_list[j]
                    reduced = best_set - {x1, x2}
                    reduced_weight = best_weight - Fraction(1, x1) - Fraction(1, x2)
                    
                    # Try adding two new elements
                    best_pair = None
                    best_pair_weight = reduced_weight
                    for y1 in range(1, N + 1):
                        if y1 in reduced:
                            continue
                        if not can_add(y1, reduced):
                            continue
                        temp_set = reduced | {y1}
                        for y2 in range(y1 + 1, N + 1):
                            if y2 in temp_set:
                                continue
                            if can_add(y2, temp_set):
                                w = reduced_weight + Fraction(1, y1) + Fraction(1, y2)
                                if w > best_pair_weight:
                                    best_pair_weight = w
                                    best_pair = (y1, y2)
                    
                    if best_pair is not None and best_pair_weight > best_weight:
                        best_set.discard(x1)
                        best_set.discard(x2)
                        best_set.add(best_pair[0])
                        best_set.add(best_pair[1])
                        best_weight = best_pair_weight
                        improved = True
                        break
                if improved:
                    break
    
    # Final verification
    result = sorted(best_set)
    result_set = set(result)
    valid = True
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            a, b = result[i], result[j]
            lo, hi = min(a, b), max(a, b)
            z1 = 2 * hi - lo
            z2 = 2 * lo - hi
            if z1 != a and z1 != b and z1 in result_set:
                valid = False
                break
            if z2 != a and z2 != b and z2 in result_set:
                valid = False
                break
        if not valid:
            break
    
    if not valid or not result:
        # Fallback to greedy
        result = sorted(selected)
    
    return result
# EVOLVE-BLOCK-END
