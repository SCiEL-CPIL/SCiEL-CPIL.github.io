"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

DEADLINE = 82.0
T_START = time.time()

def _is_subset_sum_free(exponents):
    """Check if multiset of positive exponents is subset-sum-free.
    No element equals the sum of any subset of the other elements."""
    pos = [e for e in exponents if e > 0]
    n = len(pos)
    for i in range(n):
        target = pos[i]
        others = [pos[j] for j in range(n) if j != i]
        m = len(others)
        if m == 0:
            continue
        for mask in range(1, 1 << m):
            s = 0
            for j in range(m):
                if mask & (1 << j):
                    s += others[j]
                    if s > target:
                        break
            if s == target:
                return False
    return True

def _check_validity(numbers):
    """Check if all subset products are distinct using prime exponent vectors."""
    if not numbers:
        return True
    # Get all primes up to max number
    max_val = max(numbers)
    primes = []
    for x in range(2, max_val + 1):
        is_prime = True
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(x)
    
    # For each prime, check that the exponents across all numbers are subset-sum-free
    for p in primes:
        exps = []
        for num in numbers:
            e = 0
            x = num
            while x % p == 0:
                x //= p
                e += 1
            exps.append(e)
        if not _is_subset_sum_free(exps):
            return False
    return True

def _full_subset_product_check(numbers):
    """Brute-force check: all 2^n subset products must be distinct."""
    n = len(numbers)
    if n > 25:
        return None  # too expensive, use exponent check
    prods = set()
    for mask in range(1 << n):
        prod = 1
        for i in range(n):
            if mask & (1 << i):
                prod *= numbers[i]
        if prod in prods:
            return False
        prods.add(prod)
    return True

def _best_sum_free_subset(exponents_available):
    """Find the largest subset of exponents that is subset-sum-free."""
    best = []
    for k in range(len(exponents_available), -1, -1):
        found = False
        for combo in itertools.combinations(exponents_available, k):
            if _is_subset_sum_free(list(combo)):
                best = list(combo)
                found = True
                break
        if found:
            break
    return best

def construct():
    global DEADLINE, T_START
    deadline = DEADLINE
    if deadline < 0:
        deadline = time.time() + 82.0

    def time_left():
        return deadline - time.time()

    # Primes in [2, 40]
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # For each prime, find the best sum-free subset of exponents
    # and build candidate prime powers
    candidate_items = []
    used_values = set()
    
    for p in primes:
        # Max exponent m with p^m <= 40
        m = 1
        val = p
        while p ** (m + 1) <= 40:
            m += 1
        exponents_available = list(range(1, m + 1))
        
        # Find best sum-free subset
        best_exps = _best_sum_free_subset(exponents_available)
        
        for e in best_exps:
            v = p ** e
            if 2 <= v <= 40 and v not in used_values:
                candidate_items.append(v)
                used_values.add(v)
    
    # Verify the candidate set
    n = len(candidate_items)
    valid = _check_validity(candidate_items)
    if valid is None or valid:
        # Double-check with full subset product enumeration if feasible
        if n <= 25:
            valid = _full_subset_product_check(candidate_items)
    
    if not valid:
        # Fallback: all primes (always valid since each has exponent 1 for its own prime, 0 for others)
        candidate_items = primes[:]
    
    # Try to improve: attempt to add more elements
    # We can try adding composite numbers that are products of primes not already "used up"
    # But this is complex; let's try a greedy improvement pass
    
    # Try adding each unused number from 2..40
    current = list(candidate_items)
    current_set = set(current)
    improved = True
    while improved and time_left() > 5:
        improved = False
        for v in range(2, 41):
            if v in current_set:
                continue
            trial = current + [v]
            if len(trial) <= 25:
                if _full_subset_product_check(trial):
                    current = trial
                    current_set.add(v)
                    improved = True
            else:
                if _check_validity(trial):
                    current = trial
                    current_set.add(v)
                    improved = True
    
    # Final validation
    n = len(current)
    if n <= 25:
        if not _full_subset_product_check(current):
            current = primes[:]
    else:
        if not _check_validity(current):
            current = primes[:]
    
    # Ensure types
    result = [int(x) for x in current]
    assert len(result) == len(set(result)), "duplicates"
    assert all(isinstance(x, int) and not isinstance(x, bool) and 2 <= x <= 40 for x in result)
    
    return list(result)
# EVOLVE-BLOCK-END
