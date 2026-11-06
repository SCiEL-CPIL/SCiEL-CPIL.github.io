"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import factorial

def construct():
    start_time = time.time()
    DEADLINE = start_time + 80  # 80s safety margin within 90s limit
    
    target = factorial(12)  # 479001600
    # Prime factorization of 12!: 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    total_exponents = [10, 5, 2, 1, 1]  # for primes [2,3,5,7,11]
    primes = [2, 3, 5, 7, 11]
    
    best = [target]  # fallback: single factor
    best_max = target
    
    # Generate all divisors of 12! as (exponent_tuple, value)
    divisors = []
    def gen_divisors(idx, current_exp, current_val):
        if idx == len(primes):
            divisors.append((current_exp[:], current_val))
            return
        for e in range(total_exponents[idx] + 1):
            current_exp[idx] = e
            gen_divisors(idx + 1, current_exp, current_val)
            current_val *= primes[idx]
    
    gen_divisors(0, [0]*5, 1)
    
    # Filter divisors > 12
    valid_divs = [(exp, val) for exp, val in divisors if val > 12]
    # Sort by value for better pruning
    valid_divs.sort(key=lambda x: x[1])
    
    # Map value to exponents for lookup
    val_to_exp = {}
    for exp, val in valid_divs:
        val_to_exp[val] = exp
    
    # DFS: pick distinct divisors whose product = target
    # State: remaining exponents, list of chosen factors, current max
    def dfs(rem_exp, chosen, current_max, depth):
        nonlocal best, best_max
        if time.time() > DEADLINE:
            return
        
        # Check if we've used all prime factors
        if all(e == 0 for e in rem_exp):
            if chosen:
                mx = max(chosen)
                if mx < best_max:
                    best_max = mx
                    best = chosen[:]
            return
        
        # Pruning: if current_max >= best_max, no improvement possible
        # (we could still tie, but we want strictly better or equal with more factors? 
        #  Actually we just want min max, so if current_max >= best_max, prune)
        if current_max >= best_max:
            return
        
        # Lower bound: remaining product must be distributed among factors > 12
        # Compute remaining product
        rem_prod = 1
        for i, e in enumerate(rem_exp):
            rem_prod *= primes[i] ** e
        
        # We need to form factors from rem_prod, each > 12, distinct from chosen
        # Minimum number of additional factors: 1 (if rem_prod > 12 and not in chosen)
        # Maximum number: limited by distinctness and each > 12
        
        # Try adding one more factor
        # The next factor must divide rem_prod, be > 12, not in chosen
        # To keep max small, try smaller divisors first
        
        # Find all divisors of rem_prod that are > 12 and not in chosen
        # Generate divisors of rem_exp
        candidates = []
        def gen_cands(idx, cur_exp, cur_val):
            if idx == len(primes):
                if cur_val > 12 and cur_val not in chosen_set:
                    candidates.append((cur_exp[:], cur_val))
                return
            for e in range(rem_exp[idx] + 1):
                cur_exp[idx] = e
                gen_cands(idx + 1, cur_exp, cur_val)
                cur_val *= primes[idx]
        
        chosen_set = set(chosen)
        gen_cands(0, [0]*5, 1)
        
        # Sort candidates by value (smaller first to keep max small)
        candidates.sort(key=lambda x: x[1])
        
        for c_exp, c_val in candidates:
            if time.time() > DEADLINE:
                return
            new_rem = [rem_exp[i] - c_exp[i] for i in range(5)]
            new_max = max(current_max, c_val)
            dfs(new_rem, chosen + [c_val], new_max, depth + 1)
    
    dfs(total_exponents[:], [], 0, 0)
    
    # If no improvement found, return the fallback
    if best_max == target:
        return [target]
    return best

# EVOLVE-BLOCK-END
