import math
import time

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
    
    # We want to find a subset of distinct divisors > 12 whose product is exactly T12.
    # Minimize the maximum factor.
    
    # Strategy: Iterate through potential maximum factors M in ascending order.
    # For a given M, we check if T12 can be decomposed into a set of distinct factors
    # all <= M, such that the product is T12.
    # To do this efficiently, we use a backtracking search with strong pruning based on
    # prime factorization exponents.
    
    # Pre-compute prefix sums of prime exponents for the divisors sorted in descending order?
    # No, the set of available divisors changes with M.
    
    # Instead, for a fixed M, let S be the set of divisors d in VALID_DIVISORS such that d <= M.
    # We need to find a subset of S such that the sum of prime exponents equals T12_FACTOR.
    # This is a multi-dimensional knapsack / subset sum problem.
    
    # A recursive backtracking approach:
    # State: current remaining exponents for each prime, index in sorted candidate list.
    # Candidates are divisors <= M.
    
    # To make this fast, we sort candidates in descending order? Or ascending?
    # Descending might find large factors first, but we want to ensure we can fill the rest.
    # Actually, standard subset sum backtracking works best if we prune aggressively.
    
    # Let's try iterating M.
    # For each M, we build the list of candidates (divisors <= M).
    # Then we run a solver to see if a valid subset exists.
    
    # Optimization: 
    # If we find a solution for M, that M is the minimal largest factor?
    # Not necessarily. We want to minimize the largest factor in the *solution*.
    # If we iterate M from smallest to largest, the first feasible M gives a solution where
    # all factors are <= M. The actual largest factor in that solution might be less than M.
    # But if we found a solution for M, then max(solution) <= M.
    # If we check M in ascending order, the first M for which a solution exists is the
    # smallest upper bound. The actual max factor of that solution is the answer.
    # Is it possible that a larger M allows a solution with a smaller max factor?
    # No, because if a solution exists with max factor F, then it is a valid solution for any M >= F.
    # So the set of feasible M is [F_min, infinity).
    # Therefore, the smallest feasible M is exactly the minimal possible largest factor.
    # Wait, if the solution for M=100 has factors {20, 50}, max is 50.
    # If we check M=50, is it feasible? Yes, because {20, 50} are both <= 50.
    # So we would have found it at M=50.
    # Thus, iterating M in ascending order and returning the first feasible solution's max factor is correct.
    
    # However, the backtracking search for each M might be slow.
    # We need to make the search for a fixed M very fast.
    
    # Let's implement a recursive solver for a fixed set of candidates.
    
    def solve_subset(candidates, target_exponents, current_exponents, start_idx, path, remaining_count):
        """
        candidates: list of (divisor, factorization_dict)
        target_exponents: dict of target prime exponents
        current_exponents: dict of current accumulated prime exponents
        start_idx: index to start searching from (to ensure distinctness and order)
        path: list of selected divisors
        remaining_count: max number of factors left to select
        """
        # Check if current_exponents equals target_exponents
        if all(current_exponents.get(p, 0) == target_exponents.get(p, 0) for p in primes):
            return path
        
        # Pruning: If any current exponent exceeds target, return None
        for p in primes:
            if current_exponents.get(p, 0) > target_exponents.get(p, 0):
                return None
        
        # Pruning: If remaining_count == 0, we can't add more factors
        if remaining_count == 0:
            return None
        
        # Pruning: If we can't reach the target even if we add the largest possible exponents?
        # This is hard to compute quickly.
        
        # Iterate candidates from start_idx to end
        for i in range(start_idx, len(candidates)):
            d, f_dict = candidates[i]
            
            # Check if adding d keeps exponents within bounds
            new_exponents = {}
            valid = True
            for p in primes:
                new_exp = current_exponents.get(p, 0) + f_dict.get(p, 0)
                if new_exp > target_exponents.get(p, 0):
                    valid = False
                    break
                new_exponents[p] = new_exp
            
            if not valid:
                continue
            
            # Recurse
            res = solve_subset(candidates, target_exponents, new_exponents, i + 1, path + [d], remaining_count - 1)
            if res is not None:
                return res
        
        return None

    # To improve performance, we should sort candidates in a way that helps pruning.
    # Sorting by divisor value descending might help find solutions faster if they exist?
    # Or ascending?
    # Let's try sorting by divisor value ascending.
    
    # We also need to limit the number of factors to 12.
    # The search above allows up to len(candidates) factors, but we pass remaining_count.
    
    # Let's refine the search to be more efficient.
    # Instead of passing dicts, use tuples of exponents for speed.
    
    def solve_subset_fast(candidates, target_exp_tuple, current_exp_tuple, start_idx, path, max_factors):
        # candidates: list of (divisor, exp_tuple)
        # exp_tuple: (e2, e3, e5, e7, e11)
        
        if current_exp_tuple == target_exp_tuple:
            return path
        
        if max_factors == 0:
            return None
        
        # Prune if any exponent exceeds target
        for k in range(len(primes)):
            if current_exp_tuple[k] > target_exp_tuple[k]:
                return None
        
        for i in range(start_idx, len(candidates)):
            d, f_tuple = candidates[i]
            
            new_exp = [0] * 5
            valid = True
            for k in range(5):
                ne = current_exp_tuple[k] + f_tuple[k]
                if ne > target_exp_tuple[k]:
                    valid = False
                    break
                new_exp[k] = ne
            
            if not valid:
                continue
            
            res = solve_subset_fast(candidates, target_exp_tuple, tuple(new_exp), i + 1, path + [d], max_factors - 1)
            if res is not None:
                return res
        
        return None

    # Precompute exponent tuples for all valid divisors
    DIV_EXP_TUPLES = []
    for d in VALID_DIVISORS:
        f = DIV_FACTORIZATIONS[VALID_DIVISORS.index(d)] # This is slow, let's do it better
    # Let's just recompute or store alongside
    # Actually, let's just build the list of (divisor, exp_tuple) once.
    
    # Re-do precomputation properly
    DIV_DATA = []
    for d in VALID_DIVISORS:
        f = factorize(d)
        exp_tuple = tuple(f.get(p, 0) for p in primes)
        DIV_DATA.append((d, exp_tuple))
    
    T12_EXP_TUPLE = tuple(T12_FACTOR.get(p, 0) for p in primes)
    
    # Iterate M
    # We can binary search on M if we want, but linear is safer for correctness of "first feasible".
    # However, linear might be slow if the optimal M is large.
    # Let's try binary search on the index of M in VALID_DIVISORS.
    # Feasibility is monotone: if feasible for M, feasible for M' > M.
    # So we can find the smallest feasible M via binary search.
    
    lo = 0
    hi = len(VALID_DIVISORS) - 1
    best_solution = None
    best_M_idx = None
    
    # First, check if full set is feasible
    # If not, return [T12]
    
    # Let's just do linear scan but with a time check.
    # Or binary search.
    
    # Binary search for the smallest feasible M_idx
    # But we need the actual solution for that M_idx.
    
    # Let's try a hybrid:
    # 1. Binary search to find the range.
    # 2. Once we have a feasible M, we can try to optimize the solution?
    # No, the first feasible M in ascending order gives the minimal max factor.
    
    # Let's do linear scan from 0 to n-1, but skip quickly if not feasible.
    # To make it faster, we can use the fact that if M is small, the candidate set is small.
    
    for M_idx in range(len(VALID_DIVISORS)):
        if time.time() > deadline:
            if best_solution is not None:
                return best_solution
            return [T12]
        
        M = VALID_DIVISORS[M_idx]
        
        # Candidates are divisors <= M
        # We need to select a subset of DIV_DATA[0..M_idx]
        
        # If M is T12, and no other factors, it's valid.
        # But we want distinct factors.
        
        # Prepare candidates for the solver
        # We should sort them? They are already sorted by value in DIV_DATA.
        # For backtracking, order matters.
        # Let's try sorting by value descending to pick large factors first?
        # Or ascending?
        # Usually, ascending is fine.
        
        candidates = DIV_DATA[:M_idx + 1]
        
        # Quick check: Is it possible?
        # Sum of all exponents in candidates must be >= target.
        # But we can just run the solver.
        
        # To speed up, we can pass a limit on max_factors = 12.
        # Also, we can prune if the product of the smallest remaining factors cannot reach the target?
        # Hard to do with exponents.
        
        sol = solve_subset_fast(candidates, T12_EXP_TUPLE, (0,0,0,0,0), 0, [], 12)
        
        if sol is not None:
            # Verify
            prod = 1
            for f in sol:
                prod *= f
            if prod == T12 and len(set(sol)) == len(sol) and all(f > 12 for f in sol) and len(sol) <= 12:
                return sol
    
    return [T12]
