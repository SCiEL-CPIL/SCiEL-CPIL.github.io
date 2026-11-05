import time
from math import isqrt, log
import heapq

def _find_gaussian_prime(p):
    """Find x, y such that x^2 + y^2 = p for prime p = 1 mod 4."""
    # Use Cornacchia's algorithm or simple search for small p
    # Since p is small (< 1000), simple search is fine
    for x in range(1, isqrt(p) + 1):
        y2 = p - x * x
        y = isqrt(y2)
        if y * y == y2:
            return x, y
    raise ValueError(f"No solution for {p}")

def _mul_gaussian(z1, z2):
    a, b = z1
    c, d = z2
    return (a * c - b * d, a * d + b * c)

def _pow_gaussian(z, n):
    if n == 0:
        return (1, 0)
    result = (1, 0)
    base = z
    while n:
        if n % 2 == 1:
            result = _mul_gaussian(result, base)
        base = _mul_gaussian(base, base)
        n //= 2
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    # Primes p = 1 mod 4
    primes = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, 137, 149, 157, 173, 181, 193, 197, 229, 233, 241, 257, 269, 277, 281, 293, 313, 317, 337, 349, 353, 373, 389, 397, 401, 409, 421, 433, 449, 457, 461, 509, 521, 541, 557, 569, 577, 593, 601, 613, 617, 641, 653, 661, 673, 677, 701, 709, 733, 757, 761, 769, 773, 797, 809, 821, 829, 853, 857, 877, 881, 929, 937, 941, 953, 977, 997]
    
    # Find Gaussian primes
    gauss_primes = []
    for p in primes:
        x, y = _find_gaussian_prime(p)
        gauss_primes.append((p, (x, y)))
        
    # Optimize exponents
    # Maximize sum(log(e_i + 1)) subject to sum(e_i * log(p_i)) <= log(10^18)
    # Using greedy with priority queue
    # State: exponents list
    exponents = [0] * len(primes)
    current_log_N = 0.0
    limit_log_N = log(10**18)
    
    # Priority queue: (-gain, index)
    # Gain of increasing e_i by 1 is log(e_i + 2) - log(e_i + 1)
    # Cost is log(p_i)
    # We want to maximize gain/cost? No, we want to fit within budget.
    # Greedy: at each step, pick the index that gives the highest marginal gain per unit cost?
    # Or just highest marginal gain?
    # Since we want to maximize product, we should distribute "budget" to where it yields most log-increase.
    # Actually, simpler: just try to increment the exponent that gives the best "value" increase for the "cost".
    # Value = log(e+1). Cost = log(p).
    # We want to maximize sum(Value) subject to sum(Cost) <= Limit.
    # This is like knapsack.
    # Approximation: sort by marginal value / marginal cost?
    # Marginal value of e->e+1 is log((e+2)/(e+1)).
    # Marginal cost is log(p).
    # Ratio R_i(e) = log((e+2)/(e+1)) / log(p_i).
    # Pick max R_i(e), increment e_i. Repeat until budget exhausted.
    
    heap = []
    for i, p in enumerate(primes):
        # Initial e=0. Next e=1.
        # Gain = log(2) - log(1) = log(2)
        # Cost = log(p)
        gain = log(2)
        cost = log(p)
        heapq.heappush(heap, (-gain/cost, i, 0)) # store current exponent? No, just index. We track exponents array.
        
    while True:
        if time.time() > deadline - 10:
            break
        neg_ratio, i, _ = heap[0]
        # Calculate next cost
        next_e = exponents[i] + 1
        next_cost = log(primes[i])
        if current_log_N + next_cost > limit_log_N:
            # Try next best? Or just break if the best one doesn't fit?
            # If the best one doesn't fit, maybe a smaller one does?
            # But we are picking max ratio. If max ratio doesn't fit, others might.
            # Let's pop and check.
            heapq.heappop(heap)
            if not heap:
                break
            # Re-evaluate? No, just continue loop.
            continue
        
        # Pop the best
        heapq.heappop(heap)
        # Increment exponent
        exponents[i] += 1
        current_log_N += log(primes[i])
        
        # Push new state for this index
        new_e = exponents[i]
        # Next gain: log(new_e + 2) - log(new_e + 1)
        next_gain = log((new_e + 2) / (new_e + 1))
        next_cost = log(primes[i])
        heapq.heappush(heap, (-next_gain/next_cost, i, new_e))

    # Construct N
    N = 1
    for i, p in enumerate(primes):
        if exponents[i] > 0:
            N *= p ** (2 * exponents[i])
            
    if N > 10**18:
        # Fallback
        N = 5**2 * 13**2
        exponents = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # Just reset to simple case if something went wrong
        # Actually, let's just use a known good N if optimization fails
        pass

    # Generate pairs
    # z = u * prod pi_i^{k_i} * conj(pi_i)^{2*e_i - k_i}
    # u in {1, i, -1, -i}
    # We only need positive a,b.
    # We can generate all z, then filter.
    
    pairs_set = set()
    
    # Precompute powers of pi and conj(pi)
    # pi_powers[i] = [pi^0, pi^1, ..., pi^{2*e_i}]
    # conj_pi_powers[i] = [conj(pi)^0, ..., conj(pi)^{2*e_i}]
    
    pi_powers = []
    conj_pi_powers = []
    for i, (p, pi) in enumerate(gauss_primes):
        e = exponents[i]
        max_k = 2 * e
        if max_k == 0:
            pi_powers.append([(1, 0)])
            conj_pi_powers.append([(1, 0)])
            continue
            
        x, y = pi
        conj_pi = (x, -y)
        
        p_list = [(1, 0)]
        c_list = [(1, 0)]
        for k in range(1, max_k + 1):
            p_list.append(_mul_gaussian(p_list[-1], pi))
            c_list.append(_mul_gaussian(c_list[-1], conj_pi))
            
        pi_powers.append(p_list)
        conj_pi_powers.append(c_list)
        
    # Iterate over all combinations of k_i
    # Number of combinations is prod(2*e_i + 1)
    # This can be large, but hopefully not too large.
    
    # Use recursion or iterative product
    # Let's use itertools.product if available, or manual recursion
    
    import itertools
    
    ranges = [range(2 * exponents[i] + 1) for i in range(len(primes))]
    
    count = 0
    for combo in itertools.product(*ranges):
        if time.time() > deadline - 5:
            break
        z = (1, 0)
        for i, k in enumerate(combo):
            z = _mul_gaussian(z, pi_powers[i][k])
            z = _mul_gaussian(z, conj_pi_powers[i][2 * exponents[i] - k])
            
        a, b = z
        # We want a > 0, b > 0
        # Also a^2 + b^2 = N
        # Since z is a solution, this holds.
        # We need to handle signs.
        # The generated z can be any of the 4 rotations.
        # We just take abs? No, a and b must be positive.
        # If a < 0, take -a? No, that changes the sum of squares? No, (-a)^2 + b^2 = a^2+b^2.
        # So we can just take abs(a), abs(b).
        # But we must ensure a != 0, b != 0.
        # If a == 0 or b == 0, then N is a square.
        # Powerful numbers include squares.
        # But pair (0, N) is not valid if N is not powerful?
        # The problem says a, b are powerful. 0 is not usually considered powerful (factorization undefined or empty).
        # "1 counts as powerful".
        # Let's assume a, b >= 1.
        
        if a == 0 or b == 0:
            continue
            
        a = abs(a)
        b = abs(b)
        
        if a <= b:
            pairs_set.add((a, b))
            
    pairs = [list(p) for p in pairs_set]
    
    if not pairs:
        # Fallback
        return {'N': 5, 'pairs': [[1, 4]]}
        
    return {'N': N, 'pairs': pairs}
