"""
Strategy: Iterative Deepening MILP for Prime Residue Class Covering.

This strategy reformulates the problem as a series of Mixed-Integer Linear Programming (MILP) feasibility problems.
For a given target length L, we seek a set of prime-residue classes such that:
1. Every integer n in [1, L] is covered by at least one class.
2. The sum of reciprocals of the selected primes is minimized (or checked against budget 1).
3. Primes are distinct and <= 1000.

We use scipy.optimize.milp with binary variables for each (prime, residue) pair.
The number of variables is sum(p for p in primes <= 1000) ~ 76127.
The number of constraints is L (coverage) + 1 (budget, handled via objective or constraint) + 168 (distinct primes).
For L up to ~5000, this is a large but solvable MILP.

To speed up, we:
1. Pre-compute the incidence matrix: A[n, (p, r)] = 1 if n % p == r.
2. Use a sparse matrix representation.
3. Start with a small L (e.g., 100) and increase it (e.g., +50 or +100) until infeasible or time limit.
4. For each L, solve the MILP. If feasible with sum(1/p) <= 1, update best.
5. To handle the "distinct primes" constraint, we add constraints sum_r x[p, r] <= 1 for each p.
6. Objective: minimize sum (1/p) * x[p, r]. If the minimum objective value is <= 1, it's a valid solution.

This is a structural change from heuristic search to exact optimization for each step.
"""

import time
import numpy as np
from fractions import Fraction
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix, csr_matrix

def _sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    # 1. Setup Primes and Variables
    primes = _sieve(1000)
    num_primes = len(primes)
    
    # Map each variable index to (p, r)
    # Variable index v corresponds to prime p_idx and residue r
    # We create a list of all (p, r) pairs
    var_info = []
    var_to_prime_idx = []
    prime_to_vars = {p_idx: [] for p_idx in range(num_primes)}
    
    v_idx = 0
    for p_idx, p in enumerate(primes):
        for r in range(p):
            var_info.append((p, r))
            var_to_prime_idx.append(p_idx)
            prime_to_vars[p_idx].append(v_idx)
            v_idx += 1
            
    num_vars = v_idx
    # num_vars is approx 76k. This is large but manageable for sparse MILP.
    
    # Reciprocals for objective
    # c[v] = 1/p
    c = np.zeros(num_vars)
    for v, (p, r) in enumerate(var_info):
        c[v] = 1.0 / p
        
    # 2. Pre-compute Coverage Structure
    # We don't store the full matrix A (L x num_vars) in memory for all L.
    # Instead, for a given L, we construct the rows for constraints 1..L.
    # Row n (for integer n) has 1s at variables (p, r) where n % p == r.
    
    # Helper to get the non-zero indices for a specific integer n
    def get_coverage_indices(n):
        indices = []
        for p_idx, p in enumerate(primes):
            r = n % p
            # Find variable index for (p, r)
            # We need a fast lookup from (p_idx, r) to v_idx
            # Since r < p, and we stored them sequentially:
            # offset for p_idx is sum of sizes of previous primes
            # This is slow to compute on the fly. 
            # Let's pre-compute a dictionary or array for faster access?
            # Actually, 76k variables is small enough that we can just loop? 
            # No, L can be 5000. 5000 * 168 primes = 840k ops. That's fine.
            # But constructing the sparse matrix row by row is better.
            
            # Let's create a lookup table: lookup[p_idx][r] = v_idx
            pass
        return indices

    # Let's pre-compute the lookup table for speed
    # lookup[p_idx][r] -> v_idx
    lookup = {}
    v_idx = 0
    for p_idx, p in enumerate(primes):
        lookup[p_idx] = [0] * p
        for r in range(p):
            lookup[p_idx][r] = v_idx
            v_idx += 1
            
    def get_row_indices(n):
        """Returns list of variable indices that cover integer n."""
        idxs = []
        for p_idx in range(num_primes):
            p = primes[p_idx]
            r = n % p
            idxs.append(lookup[p_idx][r])
        return idxs

    # 3. Define MILP Constraints
    # Variables: x[v] in {0, 1}
    bounds = Bounds(lb=0, ub=1)
    
    # Constraint 1: Distinct primes. For each p_idx, sum_{r} x[lookup[p_idx][r]] <= 1
    # This is a set of num_primes constraints.
    # Matrix A_distinct: num_primes x num_vars
    # A_distinct[p_idx, v] = 1 if var v belongs to prime p_idx
    
    # We can define this as a LinearConstraint.
    # It's easier to define the whole constraint matrix at once if possible, 
    # but the coverage constraints depend on L.
    
    # Let's build the constraints for a specific L.
    def solve_for_L(L):
        # Coverage constraints: sum_{v covers n} x[v] >= 1 for n in 1..L
        # Distinct prime constraints: sum_{r} x[p, r] <= 1 for each p
        
        # Total constraints: L + num_primes
        # Total variables: num_vars
        
        # Constructing the sparse matrix A for MILP
        # A is (L + num_primes) x num_vars
        
        # Row 0..L-1: Coverage
        # Row L..L+num_primes-1: Distinct primes
        
        # Using lil_matrix for efficient construction
        A = lil_matrix((L + num_primes, num_vars), dtype=np.float64)
        lb = np.zeros(L + num_primes)
        ub = np.zeros(L + num_primes)
        
        # Coverage rows
        # For each n in 1..L
        # This loop is O(L * num_primes). 
        # For L=5000, num_primes=168, this is ~840k iterations.
        # Each iteration does a dict lookup and array assignment.
        # This might be slow in Python if done naively.
        # Optimization: Pre-compute the column indices for all n?
        # Memory: L * num_primes integers. 5000 * 168 * 8 bytes = 6.7 MB. Acceptable.
        
        # Let's pre-compute the column indices for coverage rows
        # cov_cols[n-1] = list of var indices covering n
        cov_cols = []
        for n in range(1, L + 1):
            idxs = get_row_indices(n)
            cov_cols.append(idxs)
            
        for n in range(L):
            idxs = cov_cols[n]
            A[n, idxs] = 1
            lb[n] = 1
            ub[n] = np.inf
            
        # Distinct prime rows
        # For p_idx in 0..num_primes-1
        row_offset = L
        for p_idx in range(num_primes):
            row = row_offset + p_idx
            # Variables for this prime
            vars_for_p = prime_to_vars[p_idx]
            A[row, vars_for_p] = 1
            lb[row] = 0
            ub[row] = 1
            
        A = A.tocsr()
        
        # Objective: minimize c^T x
        # We want to know if there is a solution with cost <= 1.
        # So we minimize the cost.
        
        try:
            res = milp(c=c, constraints=LinearConstraint(A, lb, ub), integrality=np.ones(num_vars), bounds=bounds)
        except Exception as e:
            # If MILP fails or times out, return infeasible
            return None
            
        if not res.success:
            return None
            
        if res.fun > 1.0 + 1e-9:
            # Minimum cost exceeds budget 1. Infeasible for budget 1.
            return None
            
        # Extract solution
        x = res.x
        # x should be close to 0 or 1. Round it.
        x_rounded = np.round(x).astype(int)
        
        # Verify and build classes
        classes = []
        for v in range(num_vars):
            if x_rounded[v] == 1:
                p, r = var_info[v]
                classes.append((p, r))
                
        # Double check budget with Fractions
        budget = Fraction(0, 1)
        for p, r in classes:
            budget += Fraction(1, p)
            
        if budget > Fraction(1, 1):
            return None
            
        # Double check coverage
        # For n in 1..L, is it covered?
        # We trust the MILP, but a quick check is good for debugging.
        # However, checking L=5000 with 15 primes is fast.
        
        return classes

    # 4. Iterative Deepening
    # Start with a small L and increase.
    # We need to estimate a good step size.
    # Let's start with L=100 and double or add 100?
    # Since the problem is hard, let's try a binary search or linear increase.
    # Linear increase is safer to avoid missing the threshold.
    # But solving MILP is expensive. 
    # Maybe start with a greedy solution to get a lower bound L_greedy.
    
    # Greedy Init for Lower Bound
    def greedy_init():
        classes = []
        budget = Fraction(0, 1)
        # Sort primes by 1/p descending
        for p in primes:
            if budget + Fraction(1, p) <= Fraction(1, 1):
                # Pick best residue for this prime given current coverage?
                # For simplicity, random or 0.
                r = 0 
                classes.append((p, r))
                budget += Fraction(1, p)
        return classes

    # Calculate L for a set of classes
    def calc_L(classes, max_check=10000):
        if not classes:
            return 0
        # Use numpy for speed
        covered = np.zeros(max_check + 1, dtype=bool)
        for p, r in classes:
            if r <= max_check:
                covered[r::p] = True
        gaps = np.where(~covered[1:])[0]
        if len(gaps) > 0:
            return gaps[0]
        return max_check

    best_classes = greedy_init()
    current_L = calc_L(best_classes)
    
    # Try to improve using MILP
    # Start search from current_L + 1
    # If current_L is small, we can jump.
    
    search_L = current_L + 1
    step = 10
    
    # To save time, we can skip small steps if we are far from the limit.
    # But we don't know the limit.
    # Let's just iterate.
    
    # Optimization: If a MILP solve takes too long, we should stop.
    # milp doesn't have a direct time limit parameter in the simple API, 
    # but we can check time between steps.
    
    # Actually, solving a 76k variable MILP might be very slow.
    # Let's restrict the variables.
    # Do we need all primes?
    # Small primes are "expensive" but cover more.
    # Large primes are "cheap" but cover less.
    # If we want to cover L, we likely need small primes.
    # Let's restrict to primes <= 100 initially? 
    # Sum of 1/p for p<=100 is ~1.9. So we can pick a subset.
    # If we restrict to p <= 200, sum is ~2.2.
    # Let's try to dynamically select a subset of primes to consider.
    # Or, better, let the MILP decide, but maybe the solver is slow with 76k vars.
    
    # Alternative: Use a smaller set of candidate primes.
    # Primes > L are useless for covering 1..L? 
    # If p > L, then r must be 0 to cover any multiple, but the first multiple is p > L.
    # So if p > L, it covers nothing in 1..L.
    # Thus, we can restrict variables to p <= L.
    
    # This is a HUGE optimization.
    # For L=100, we only consider p <= 100.
    # For L=500, p <= 500.
    # Number of variables drops significantly.
    
    def solve_for_L_optimized(L):
        # Filter primes <= L
        active_primes = [p for p in primes if p <= L]
        if not active_primes:
            return []
            
        # Rebuild variable info for this L
        # This is expensive to do from scratch every time.
        # But L increases slowly.
        
        # Let's map active_primes to indices
        # We can just use the global var_info but mask out p > L?
        # No, it's easier to just build a new, smaller problem.
        
        # However, rebuilding the sparse matrix is O(L * num_active_primes).
        
        # Let's try this.
        
        num_active = len(active_primes)
        
        # Variable indices for this sub-problem
        # v_idx_local
        local_var_info = []
        local_prime_to_vars = {p: [] for p in active_primes}
        local_lookup = {p: [0]*p for p in active_primes}
        
        v = 0
        for p in active_primes:
            for r in range(p):
                local_var_info.append((p, r))
                local_prime_to_vars[p].append(v)
                local_lookup[p][r] = v
                v += 1
                
        num_local_vars = v
        
        if num_local_vars == 0:
            return []
            
        c_local = np.zeros(num_local_vars)
        for i, (p, r) in enumerate(local_var_info):
            c_local[i] = 1.0 / p
            
        # Constraints
        # Coverage: L rows
        # Distinct: num_active rows
        
        A = lil_matrix((L + num_active, num_local_vars), dtype=np.float64)
        lb = np.zeros(L + num_active)
        ub = np.zeros(L + num_active)
        
        # Coverage
        # For n in 1..L
        # Only primes p <= L are in the set.
        # If p > n, n % p = n. So r must be n.
        # If p <= n, n % p is standard.
        
        for n in range(1, L + 1):
            row = n - 1
            # Find variables covering n
            # For each p in active_primes:
            # r = n % p
            # var_idx = local_lookup[p][r]
            # A[row, var_idx] = 1
            
            # Optimization: Loop over active_primes
            for p in active_primes:
                r = n % p
                idx = local_lookup[p][r]
                A[row, idx] = 1
            lb[row] = 1
            ub[row] = np.inf
            
        # Distinct primes
        row_offset = L
        for i, p in enumerate(active_primes):
            row = row_offset + i
            vars_for_p = local_prime_to_vars[p]
            A[row, vars_for_p] = 1
            lb[row] = 0
            ub[row] = 1
            
        A = A.tocsr()
        
        bounds = Bounds(lb=0, ub=1)
        integrality = np.ones(num_local_vars)
        
        try:
            res = milp(c=c_local, constraints=LinearConstraint(A, lb, ub), integrality=integrality, bounds=bounds)
        except Exception:
            return None
            
        if not res.success:
            return None
            
        if res.fun > 1.0 + 1e-9:
            return None
            
        x = np.round(res.x).astype(int)
        
        classes = []
        for i in range(num_local_vars):
            if x[i] == 1:
                p, r = local_var_info[i]
                classes.append((p, r))
                
        # Exact budget check
        budget = Fraction(0, 1)
        for p, r in classes:
            budget += Fraction(1, p)
        if budget > Fraction(1, 1):
            return None
            
        return classes

    # Iterative loop
    # We start from current_L + 1
    # If current_L is 0, start at 1.
    
    # To avoid getting stuck on small L, we can jump.
    # But let's be safe.
    
    # If time is running out, return best.
    
    # Let's try to jump to a reasonable L first to see if it's feasible.
    # If L=1000 is feasible, we know we can at least get there.
    # But we want to maximize.
    
    # Strategy: 
    # 1. Try to solve for L = current_L + step.
    # 2. If feasible, update current_L, best_classes.
    # 3. If infeasible, reduce step? Or stop?
    #    If infeasible for L, it might be feasible for L-1.
    #    Since we are increasing, the first infeasible L gives us the max L = L-1.
    #    However, the "budget" constraint makes it non-monotonic?
    #    No, if L is feasible, L-1 is feasible (just ignore the last number? No, the set of classes must cover 1..L. If it covers 1..L, it covers 1..L-1).
    #    So feasibility is monotonic with respect to L?
    #    Wait. The set of classes can change.
    #    Is it possible that L=100 is infeasible but L=101 is feasible?
    #    No. If there is a set of classes covering 1..101 with budget <= 1, then the same set covers 1..100 with budget <= 1.
    #    So if L is infeasible, all L' > L are infeasible.
    #    And if L is feasible, all L' < L are feasible.
    #    So we can binary search? 
    #    Or just linearly increase until infeasible.
    
    # Linear increase is safer and simple.
    
    # However, solving MILP for each L is expensive.
    # Let's try to jump by larger steps if we have time.
    
    # Let's start with a modest step.
    
    # Note: The number of variables grows with L.
    # For L=100, vars ~ 200.
    # For L=1000, vars ~ 7000.
    # MILP complexity grows.
    
    # Let's add a timeout check inside the loop.
    
    last_solve_time = time.time()
    
    # We'll try to solve for increasing L.
    # Start L = max(1, current_L + 1)
    
    # If current_L is from greedy, it might be small.
    # Let's try to jump to a target.
    
    # Actually, let's just iterate.
    
    # To prevent infinite loops or slow progress, we limit the number of MILP calls.
    # But we have 145s.
    
    # Let's try to be smart: 
    # If L is small, step is small.
    # If L is large, step is large?
    
    # Let's just use a fixed step of 10 or 20.
    
    # One issue: If the greedy solution is bad (L=10), and the optimal is L=100, 
    # we will do 90 MILP calls.
    # Each call for small L is fast.
    
    # Let's go.
    
    target_L = current_L + 1
    step = 5
    
    while time.time() < deadline:
        # Check if we should increase step
        if target_L > 500:
            step = 20
        elif target_L > 100:
            step = 10
        else:
            step = 5
            
        new_L = target_L
        
        # Solve
        sol_classes = solve_for_L_optimized(new_L)
        
        if sol_classes is not None:
            # Feasible
            # Update best
            # Verify L is actually covered (should be by construction)
            verified_L = calc_L(sol_classes, max_check=new_L + 100)
            if verified_L >= new_L:
                best_classes = sol_classes
                current_L = verified_L
                target_L = new_L + step
            else:
                # Should not happen if MILP is correct
                # Maybe the coverage check in calc_L is buggy or max_check too small?
                # If verified_L < new_L, it means we didn't cover up to new_L.
                # This implies MILP failed or there's a bug.
                # Let's break to be safe.
                break
        else:
            # Infeasible
            # The max L is target_L - 1? 
            # Not necessarily. It might be that target_L is infeasible, but target_L - 1 is feasible.
            # Since we are increasing, and we just updated to target_L - step (roughly),
            # the current best is the previous successful one.
            # We can try to binary search between last_success and new_L?
            # Or just stop, as the gap is small (step size).
            
            # To be precise, we can try new_L - 1, new_L - 2, etc.
            # But that's expensive.
            # Given step is small (5-20), we can just stop.
            # The score will be current_L.
            break
            
    # Final check
    # Ensure best_classes is valid
    if best_classes:
        budget = Fraction(0, 1)
        for p, r in best_classes:
            budget += Fraction(1, p)
        if budget > Fraction(1, 1):
            return []
            
    return [[p, r] for p, r in best_classes]

if __name__ == "__main__":
    classes = construct()
    # Calculate L for reporting
    L = 0
    for i in range(1, 10000):
        covered = False
        for p, r in classes:
            if i % p == r:
                covered = True
                break
        if covered:
            L = i
        else:
            break
    print(f"Covered up to L={L}")
    print(f"Classes: {classes}")
    budget = sum(Fraction(1, p) for p, r in classes)
    print(f"Budget: {budget}")
