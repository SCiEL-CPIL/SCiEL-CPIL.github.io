import math
import time
import random
from itertools import combinations

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    LCM = 30030
    
    # Precompute gcd matrix
    g = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            g[i][j] = math.gcd(divisors[i], divisors[j])

    # Precompute for each divisor d_i, the set of residues r in [0, LCM-1]
    # such that r is a multiple of d_i? No, the condition is on the offset r_i relative to min.
    # Let's stick to the CSP formulation derived:
    # Find distinct r_1..r_n in [0, W] such that r_i ≡ r_j (mod gcd(d_i, d_j)) for all i,j.
    
    # Optimization: 
    # We can precompute for each pair (i, j) the compatibility.
    # Also, we can precompute the "signature" of a value r for each divisor d_i.
    # sig_i(r) = r % gcd(d_i, d_j) ? No, that depends on j.
    
    # Let's use a bitset or a dictionary for domains.
    # Since W is expected to be small (maybe < 100), we can just use lists.
    
    def solve_csp(W):
        # domains[i] is a set of possible values for r_i
        # Initially [0, W]
        # We use MRV heuristic.
        
        # To speed up propagation, we can precompute for each value v in [0, W] and each index i,
        # which other indices j and values u in [0, W] are compatible.
        # compat[i][v] = set of (j, u) such that v ≡ u (mod g[i][j]) and u != v (distinctness handled separately?)
        # Actually, distinctness is a global constraint on the values r_i.
        # r_i must be distinct.
        
        # Let's try a different angle.
        # Since n=10 is small, we can just use backtracking with strong pruning.
        
        domains = [set(range(W + 1)) for _ in range(n)]
        assignment = [None] * n
        
        # Precompute compatibility masks for speed?
        # For a given i and value v, which values u are compatible with j?
        # (v - u) % g[i][j] == 0
        
        def is_compatible(i, v, j, u):
            return (v - u) % g[i][j] == 0

        def backtrack(idx):
            if idx == n:
                return assignment[:]
            
            # MRV: find unassigned variable with smallest domain
            min_size = float('inf')
            min_var = -1
            for k in range(n):
                if assignment[k] is None:
                    size = len(domains[k])
                    if size < min_size:
                        min_size = size
                        min_var = k
                        if size == 0:
                            return None
            
            if min_var == -1:
                return None
            
            # Try values in domain of min_var
            # Order values by least constraining? Or just random/sorted.
            # Let's try sorted for determinism, or random for diversity.
            # For exact search, sorted is fine.
            for v in sorted(domains[min_var]):
                # Check if v is distinct from assigned values
                if any(assignment[k] == v for k in range(n) if assignment[k] is not None):
                    continue
                
                # Check compatibility with assigned variables
                valid = True
                for k in range(n):
                    if assignment[k] is not None and k != min_var:
                        if not is_compatible(min_var, v, k, assignment[k]):
                            valid = False
                            break
                
                if not valid:
                    continue
                
                # Assign
                assignment[min_var] = v
                
                # Propagate: remove v from domains of unassigned variables
                # Also remove values incompatible with v from domains of unassigned variables
                undo_list = []
                feasible = True
                for k in range(n):
                    if assignment[k] is None and k != min_var:
                        # Remove v (distinctness)
                        # Remove incompatible values
                        new_domain = set()
                        for u in domains[k]:
                            if u == v:
                                continue
                            if is_compatible(min_var, v, k, u):
                                new_domain.add(u)
                        
                        if len(new_domain) < len(domains[k]):
                            undo_list.append((k, domains[k].copy()))
                        domains[k] = new_domain
                        
                        if len(domains[k]) == 0:
                            feasible = False
                            break
                
                if feasible:
                    res = backtrack(idx + 1)
                    if res is not None:
                        return res
                
                # Undo
                assignment[min_var] = None
                for k, old_domain in undo_list:
                    domains[k] = old_domain
                    
            return None
        
        return backtrack(0)

    # Search W from 9 upwards
    # 9 is the minimum span for 10 distinct integers.
    
    # To improve chances of finding a solution quickly for small W,
    # we can randomize the order of values in the domain if we want diversity,
    # but for finding the *first* solution with minimal W, deterministic is fine.
    # However, if a particular W has a solution but the search gets stuck in a bad branch,
    # randomized restarts might help. But with n=10 and small W, backtracking should be fast.
    
    # Let's add a time limit to the search for a specific W to avoid getting stuck.
    start_time = time.time()
    deadline = start_time + 55.0
    
    for W in range(9, 1000):
        if time.time() > deadline:
            break
            
        # Clear random state for reproducibility? Or use random for value ordering?
        # Let's try a simple optimization: 
        # If we fail, we might want to retry with different heuristics?
        # For now, let's just run the solver.
        
        res = solve_csp(W)
        
        if res is not None:
            best_r = res
            best_W = W
            break
    else:
        # If no solution found in range, return a valid but non-optimal solution?
        # Or just fail? The task says "Any broken rule makes the whole result invalid".
        # So we must return a valid solution.
        # A trivial valid solution: x_i = i * LCM.
        # Span = 9 * LCM = 270270.
        # This is valid but huge.
        # We should only get here if the search is too slow.
        # Let's try to construct a solution manually if search fails.
        # But search should succeed for W < 100.
        
        # Fallback: Random search for a valid configuration?
        # Or just use the trivial one.
        reps = [int((i + 1) * LCM) for i in range(n)]
        return reps

    # We have best_r. Find m.
    m = None
    # Brute force m in [0, LCM-1]
    for candidate_m in range(LCM):
        valid = True
        for i in range(n):
            if (candidate_m + best_r[i]) % divisors[i] != 0:
                valid = False
                break
        if valid:
            m = candidate_m
            break
            
    if m is None:
        # Should not happen
        m = 0
        
    # Construct x_i
    min_x = min(m + best_r[i] for i in range(n))
    if min_x < 1:
        # Shift m by LCM to make all positive
        # We need m + best_r[i] >= 1 for all i.
        # min(m + best_r[i]) >= 1 => m >= 1 - min(best_r[i])
        # Since m is a solution mod LCM, we can add k*LCM.
        # Let min_r = min(best_r).
        # We need m + min_r >= 1.
        # Current m + min_r = min_x.
        # Add ceil((1 - min_x) / LCM) * LCM to m.
        if min_x < 1:
            shift = (1 - min_x + LCM - 1) // LCM
            if shift < 0:
                shift = 0
            m += shift * LCM
            
    reps = [int(m + best_r[i]) for i in range(n)]
    
    return reps
