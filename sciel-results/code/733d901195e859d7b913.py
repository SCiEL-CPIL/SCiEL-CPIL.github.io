"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 5000

# EVOLVE-BLOCK-START
def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def construct():
    primes = _primes(N)
    num_primes = len(primes)
    
    # res_elements[i][r] is a list of active x such that x % primes[i] == r
    res_elements = []
    for p in primes:
        res_elements.append([[] for _ in range(p)])
        
    active = [True] * (N + 1)
    current_size = N + 1
    
    for x in range(N + 1):
        for i, p in enumerate(primes):
            r = x % p
            res_elements[i][r].append(x)
            
    # violated[i] is True if prime primes[i] is violated (all residues non-empty)
    violated = [True] * num_primes
    
    while True:
        # Check if any violated primes p <= current_size remain
        has_violated = False
        for i in range(num_primes):
            if primes[i] > current_size:
                break
            if violated[i]:
                has_violated = True
                break
        if not has_violated:
            break
            
        # Calculate gain for each x
        gain = [0] * (N + 1)
        
        for i in range(num_primes):
            if primes[i] > current_size:
                break
            if not violated[i]:
                continue
            p = primes[i]
            for r in range(p):
                if len(res_elements[i][r]) == 1:
                    x = res_elements[i][r][0]
                    gain[x] += 1
        
        max_gain = 0
        best_x = -1
        for x in range(N + 1):
            if gain[x] > max_gain:
                max_gain = gain[x]
                best_x = x
                
        if best_x == -1:
            # Fallback: Find min length residue among violated primes p <= current_size
            min_len = float('inf')
            best_i = -1
            best_r = -1
            for i in range(num_primes):
                if primes[i] > current_size:
                    break
                if not violated[i]:
                    continue
                p = primes[i]
                for r in range(p):
                    l = len(res_elements[i][r])
                    if l < min_len:
                        min_len = l
                        best_i = i
                        best_r = r
            best_x = res_elements[best_i][best_r][0]
            
        # Remove best_x
        x = best_x
        active[x] = False
        current_size -= 1
        
        for i in range(num_primes):
            p = primes[i]
            r = x % p
            lst = res_elements[i][r]
            lst.remove(x)
            if len(lst) == 0:
                violated[i] = False
                
    return [x for x in range(N + 1) if active[x]]
# EVOLVE-BLOCK-END
