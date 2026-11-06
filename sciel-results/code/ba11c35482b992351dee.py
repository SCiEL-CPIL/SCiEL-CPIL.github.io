import math
import time
import random
from itertools import product

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin for 120s limit
    
    # Primes to consider. Product of first 15 primes is ~6.1e17.
    # Including 16th prime (67) makes product ~4e19 > 1e18.
    # We focus on square-free structures primarily, as they maximize divisor count per bit of budget.
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]
    k = len(primes)
    LIMIT = 10**18

    def calc_score(exps):
        """
        Calculate tau_perp(n) for n = prod p_i^e_i.
        Returns (score, n).
        Optimized: Generate divisors, sort, count coprime adjacencies.
        """
        n = 1
        divisors = [1]
        
        for i, e in enumerate(exps):
            if e == 0:
                continue
            p = primes[i]
            n *= (p ** e)
            if n > LIMIT:
                return -1, n
            
            new_divs = []
            p_pow = 1
            for _ in range(e + 1):
                for d in divisors:
                    new_divs.append(d * p_pow)
                p_pow *= p
            divisors = new_divs
            
        if n < 2:
            return 0, n
            
        divisors.sort()
        
        score = 0
        prev = divisors[0]
        for i in range(1, len(divisors)):
            curr = divisors[i]
            if math.gcd(prev, curr) == 1:
                score += 1
            prev = curr
            
        return score, n

    # Strategy:
    # 1. Greedy Construction: Maximize divisor count tau(n) within budget.
    #    This is known to correlate with high tau_perp.
    #    We start with exponents that maximize tau(n) for n <= 10^18.
    #    Standard approach: assign higher exponents to smaller primes.
    
    # Greedy initialization:
    # Start with all exponents 0.
    # Repeatedly increment the exponent of the prime that gives the best "bang for buck"
    # in terms of divisor count increase per log(n) increase, or simply try to fit
    # the standard "highly composite" like structure.
    
    # A faster heuristic for initialization:
    # Try to fit 2^a * 3^b * 5^c ... such that exponents are non-increasing.
    # Let's use a simple local search from a "dense" starting point.
    
    # Starting point: The number with the maximum number of divisors below 10^18.
    # This is a known problem. Let's approximate it or find it quickly.
    # Max divisors for n < 10^18 is around 103680 (for n = 897612484786617600? No, that's < 10^18).
    # Actually, let's just run a fast greedy to get a good starting point.
    
    current_exps = [0] * k
    current_n = 1
    
    # Greedy: try to add factors that increase tau most efficiently.
    # tau(n) = prod (e_i + 1).
    # Adding p_i^1 changes tau by factor (e_i+2)/(e_i+1) and n by p_i.
    # We want to maximize log(tau) / log(n).
    
    # Simple greedy loop:
    # While we can add a prime power, pick the one that maximizes (new_tau / old_tau) / (new_n / old_n)
    # This is equivalent to maximizing (e_i+2)/(e_i+1) / p_i.
    
    # Let's do a few passes.
    for _ in range(50):
        best_gain = -1
        best_idx = -1
        for i in range(k):
            p = primes[i]
            if current_n * p > LIMIT:
                continue
            old_tau_factor = current_exps[i] + 1
            new_tau_factor = current_exps[i] + 2
            gain = (new_tau_factor / old_tau_factor) / p
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        
        if best_idx == -1:
            break
        
        current_exps[best_idx] += 1
        current_n *= primes[best_idx]
        
    # Now we have a high-divisor-count number. Let's evaluate it.
    best_score, best_n = calc_score(current_exps)
    
    # Local Search:
    # We will perform a structured local search.
    # Moves:
    # 1. Increase exponent of p_i by 1 (if fits).
    # 2. Decrease exponent of p_i by 1.
    # 3. Swap: Decrease exp of p_j by 1, Increase exp of p_i by 1 (if fits).
    
    # To make search faster, we limit the search to exponents 0-10 for small primes, 0-3 for larger.
    
    random.seed(42)
    
    T = 5.0
    T_min = 0.1
    cooling = 0.995
    
    last_improvement = time.time()
    
    while time.time() < DEADLINE:
        move_type = random.random()
        
        new_exps = current_exps[:]
        
        if move_type < 0.4:
            # Single change: increment or decrement
            i = random.randint(0, k - 1)
            delta = random.choice([-1, 1])
            if random.random() < 0.1:
                delta = random.choice([-2, 2])
            new_exps[i] += delta
            if new_exps[i] < 0:
                new_exps[i] = 0
            if new_exps[i] > 12: # Cap to avoid huge numbers
                new_exps[i] = 12
        elif move_type < 0.8:
            # Swap: remove from large, add to small
            candidates_remove = [i for i in range(k) if current_exps[i] > 0]
            if not candidates_remove:
                continue
            # Weight towards larger primes for removal (to free up budget)
            weights = [primes[i] for i in candidates_remove]
            i_rem = random.choices(candidates_remove, weights=weights, k=1)[0]
            
            # Pick a prime to add to (prefer smaller primes)
            candidates_add = [i for i in range(k)]
            weights_add = [1.0 / primes[i] for i in candidates_add]
            i_add = random.choices(candidates_add, weights=weights_add, k=1)[0]
            
            new_exps[i_rem] -= 1
            new_exps[i_add] += 1
            if new_exps[i_rem] < 0:
                new_exps[i_rem] = 0
            if new_exps[i_add] > 12:
                new_exps[i_add] = 12
        else:
            # Random perturbation: change 2-3 exponents
            for _ in range(random.randint(2, 3)):
                i = random.randint(0, k - 1)
                new_exps[i] += random.choice([-1, 1])
                if new_exps[i] < 0:
                    new_exps[i] = 0
                if new_exps[i] > 12:
                    new_exps[i] = 12

        # Check validity and score
        n_new = 1
        valid = True
        for i, e in enumerate(new_exps):
            if e > 0:
                n_new *= (primes[i] ** e)
                if n_new > LIMIT:
                    valid = False
                    break
        
        if not valid:
            continue
            
        score_new, _ = calc_score(new_exps)
        
        if score_new > current_score:
            current_score = score_new
            current_n = n_new
            current_exps = new_exps
            if current_score > best_score:
                best_score = current_score
                best_n = current_n
                best_exps = new_exps[:]
                last_improvement = time.time()
        else:
            # Accept worse with probability
            delta_score = score_new - current_score
            if delta_score == 0:
                if random.random() < 0.5:
                    current_score = score_new
                    current_n = n_new
                    current_exps = new_exps
            else:
                prob = math.exp(delta_score / T)
                if random.random() < prob:
                    current_score = score_new
                    current_n = n_new
                    current_exps = new_exps
        
        T *= cooling
        if T < T_min:
            T = T_min
            
    return best_n

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
