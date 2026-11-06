import math
import time
from scipy.optimize import milp, LinearConstraint, Bounds
import numpy as np

# Precompute 12!
T12 = math.factorial(12)  # 479001600

# Generate all divisors of 12!
# Prime factorization: 2^10 * 3^5 * 5^2 * 7^1 * 11^1
primes = [2, 3, 5, 7, 11]
exponents = [10, 5, 2, 1, 1]

def generate_divisors():
    divisors = [1]
    for p, e in zip(primes, exponents):
        new_divisors = []
        p_power = 1
        for _ in range(e + 1):
            for d in divisors:
                new_divisors.append(d * p_power)
            p_power *= p
        divisors = new_divisors
    return divisors

ALL_DIVISORS = generate_divisors()
# Filter divisors > 12
VALID_DIVISORS = sorted([d for d in ALL_DIVISORS if d > 12])

def factorize(n):
    """Return prime factorization of n as a dict {prime: exponent}."""
    factors = {}
    for p in primes:
        while n % p == 0:
            factors[p] = factors.get(p, 0) + 1
            n //= p
    return factors

# Precompute prime factorizations for all valid divisors
DIV_FACTORIZATIONS = [factorize(d) for d in VALID_DIVISORS]
T12_FACTOR = factorize(T12)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin
    
    n_divs = len(VALID_DIVISORS)
    
    # We want to find a subset of distinct divisors > 12 whose product is exactly T12.
    # Minimize the maximum factor.
    # 
    # ILP formulation:
    # Binary variables x_i for each divisor i (i = 0..n_divs-1)
    # 
    # Constraints:
    # 1. For each prime p: sum_i (exp_p(div_i) * x_i) = exp_p(T12)
    # 2. sum_i x_i <= 12 (at most 12 factors)
    # 3. sum_i x_i >= 1 (at least 1 factor)
    # 
    # Objective: minimize max selected value
    # This is non-linear. We can use a binary search on M (the maximum allowed factor),
    # or add a variable t and constraint x_i * div_i <= t for all i.
    # 
    # Actually, since we want to minimize the largest factor, we can:
    # 1. Sort divisors in ascending order
    # 2. Binary search on the index M_idx such that divisors[M_idx] is the max
    # 3. For each candidate max value M, solve ILP with constraint that only divisors <= M can be selected
    # 4. Return the smallest M for which a feasible solution exists
    #
    # But binary search requires monotonicity: if M works, does M' > M work?
    # Not necessarily - the set of available divisors changes, and the ILP might not be feasible for larger M if the constraint structure changes.
    # Actually, if we can decompose T12 using divisors <= M, we can also decompose it using divisors <= M' > M (just use the same subset).
    # So feasibility is monotone in M. Binary search works.
    
    # Alternative: Add variable t, minimize t, with constraints:
    # For each i: x_i * div_i <= t
    # This is still non-linear due to x_i * div_i.
    # But we can linearize: for each i, if x_i = 1 then div_i <= t.
    # This is equivalent to: x_i - (div_i <= t) indicator... 
    # Actually, we can use: for each i, x_i <= 1 if div_i <= t, and x_i = 0 if div_i > t.
    # But t is continuous. We can instead enumerate possible t values from VALID_DIVISORS.
    
    # Best approach: Binary search on M_idx.
    # For a given max_allowed = VALID_DIVISORS[M_idx], solve ILP where:
    # - Only divisors with value <= max_allowed can be selected (x_i = 0 for div_i > max_allowed)
    # - Prime factorization constraints
    # - At most 12 factors
    # - At least 1 factor
    # If feasible, return the solution.
    
    # Precompute the constraint matrix for prime factorizations
    # Number of primes = 5
    n_primes = len(primes)
    
    # A matrix: n_primes x n_divs, where A[p, i] = exp_p(div_i)
    A = np.zeros((n_primes, n_divs), dtype=np.int64)
    for i, div in enumerate(VALID_DIVISORS):
        for p_idx, p in enumerate(primes):
            A[p_idx, i] = DIV_FACTORIZATIONS[i].get(p, 0)
    
    # RHS for prime constraints
    b_prime = np.array([T12_FACTOR.get(p, 0) for p in primes], dtype=np.int64)
    
    # Solve ILP for a given max_index
    def solve_ilp(max_idx):
        """Solve ILP allowing only divisors[0..max_idx] to be selected."""
        # Check time
        if time.time() > deadline:
            return None
        
        # Variables: x_0, ..., x_{max_idx} (binary)
        n_vars = max_idx + 1
        
        # Objective: minimize sum of x_i (just find any feasible solution, or minimize number of factors)
        # Actually, we just need feasibility. Set c = 0 for all.
        # But milp requires an objective. Use c = 0.
        c = np.zeros(n_vars)
        
        # Prime factorization constraints: A_prime @ x = b_prime
        # A_prime is n_primes x n_vars
        A_prime = A[:, :n_vars]
        
        # Equality constraints for primes
        # LinearConstraint(A_prime, lb=b_prime, ub=b_prime)
        
        # Number of factors: 1 <= sum(x_i) <= 12
        ones = np.ones(n_vars)
        
        # Build constraints
        # Constraint 1: A_prime @ x = b_prime (5 equality constraints)
        # Constraint 2: 1 <= ones @ x <= 12
        
        # Combine into single LinearConstraint
        # Total constraints: 5 + 2 = 7
        A_all = np.vstack([
            A_prime,
            ones.reshape(1, -1),
            -ones.reshape(1, -1)
        ])
        lb = np.concatenate([
            b_prime,
            [1],
            [-12]
        ])
        ub = np.concatenate([
            b_prime,
            [12],
            [0]
        ])
        
        constraints = LinearConstraint(A_all, lb, ub)
        bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
        
        try:
            res = milp(
                c=c,
                constraints=constraints,
                integrality=np.ones(n_vars),
                bounds=bounds
            )
        except Exception:
            return None
        
        if res.status != 0:
            return None
        
        # Extract solution
        x = res.x
        factors = []
        for i in range(n_vars):
            if x[i] > 0.5:
                factors.append(VALID_DIVISORS[i])
        
        if not factors:
            return None
        
        # Verify product
        prod = 1
        for f in factors:
            prod *= f
            if prod > T12:
                return None
        
        if prod != T12:
            return None
        
        # Verify all factors > 12 and distinct
        if len(set(factors)) != len(factors):
            return None
        if any(f <= 12 for f in factors):
            return None
        if len(factors) > 12:
            return None
        
        return factors
    
    # Binary search on max_idx
    # We want the smallest max_idx such that solve_ilp(max_idx) is feasible.
    # But we also need to return the actual solution.
    
    # First, check if the full set is feasible
    best_solution = None
    best_max_idx = None
    
    # Binary search: find the smallest max_idx where feasible
    lo = 0
    hi = n_divs - 1
    
    # But we want the minimum largest factor, which corresponds to the smallest max_idx.
    # However, the solution's largest factor might be less than VALID_DIVISORS[max_idx]
    # if the optimal solution doesn't use the largest allowed divisor.
    # So we should: for each candidate max value M (in increasing order), check if there's
    # a decomposition using only divisors <= M. The first such M gives the answer.
    
    # But binary search assumes monotonicity. If feasible for max_idx, then feasible for larger max_idx.
    # So we can binary search for the smallest feasible max_idx.
    
    # However, the actual largest factor in the solution might be smaller than VALID_DIVISORS[max_idx].
    # So after finding the smallest feasible max_idx, we should extract the solution and return it.
    # The largest factor in that solution is the answer.
    
    # But wait: if we allow divisors up to VALID_DIVISORS[max_idx], the solution might not use
    # VALID_DIVISORS[max_idx]. The actual max factor is max(solution).
    # We want to minimize max(solution), not max_idx.
    
    # So the correct approach is: iterate M_idx from 0 to n_divs-1, and for each, solve ILP
    # with divisors <= VALID_DIVISORS[M_idx]. The first feasible solution gives the minimum
    # possible largest factor (which is max of that solution).
    
    # But this is O(n) ILP solves, which might be slow.
    
    # Alternative: Add a variable t and minimize t, with constraints that ensure
    # if x_i = 1 then div_i <= t. This can be linearized using big-M or by enumerating t.
    
    # Since n_divs is manageable (let's check), we can try iterating.
    
    # How many valid divisors are there?
    # 12! has (10+1)*(5+1)*(2+1)*(1+1)*(1+1) = 11*6*3*2*2 = 792 divisors total.
    # Minus those <= 12. So ~790 valid divisors.
    # 790 ILP solves might be too slow.
    
    # Better: Use a single ILP with variable t.
    # Variables: x_0..x_{n-1} (binary), t (continuous)
    # Minimize t
    # Constraints:
    # 1. Prime factorization: A @ x = b_prime
    # 2. 1 <= sum(x) <= 12
    # 3. For each i: x_i * div_i <= t  =>  This is non-linear.
    #     Linearize: for each i, x_i <= 1 and if x_i=1 then div_i <= t.
    #     This is equivalent to: x_i * div_i <= t, which is bilinear.
    #     We can use: x_i <= 1, and for each i, add constraint: div_i * x_i <= t.
    #     But this is still bilinear.
    #
    # Alternative: Enumerate t from VALID_DIVISORS. For each candidate t, set x_i = 0 for
    # div_i > t, and solve feasibility ILP. This is the same as the binary search approach.
    #
    # Given the time constraint, let's try a smarter approach:
    # 1. Start with a loose upper bound on M (e.g., T12 itself)
    # 2. Solve ILP with that bound
    # 3. If feasible, tighten the bound using the actual max factor in the solution
    # 4. Repeat until no improvement
    
    # Actually, the simplest approach that's likely fast enough:
    # Iterate M_idx from 0 upward, but with early termination and efficient ILP.
    # Since the optimal M is likely not too large, this should work.
    
    # Let's try iterating, but with a time check.
    for M_idx in range(n_divs):
        if time.time() > deadline:
            if best_solution is not None:
                return best_solution
            return [T12]
        
        sol = solve_ilp(M_idx)
        if sol is not None:
            best_solution = sol
            best_max_idx = M_idx
            # Return immediately since we're going in increasing order
            return sol
    
    if best_solution is not None:
        return best_solution
    
    return [T12]
