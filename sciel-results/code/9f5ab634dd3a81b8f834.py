import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    # Primes up to 100
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    def get_divisors(n):
        """Get all divisors of n in sorted order."""
        divs = [1]
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                new_divs = []
                power = 1
                while temp % p == 0:
                    temp //= p
                    power *= p
                    for d in divs:
                        new_divs.append(d * power)
                divs += new_divs
            if temp == 1:
                break
        if temp > 1:
            for d in divs[:]:
                divs.append(d * temp)
        return sorted(divs)

    def calc_score(n):
        """Calculate tau_perp(n): number of adjacent coprime pairs in sorted divisors."""
        if n < 2:
            return 0
        divs = get_divisors(n)
        if len(divs) < 2:
            return 0
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score

    def factorize(n):
        factors = {}
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                factors[p] = count
        if temp > 1:
            factors[temp] = 1
        return factors

    # Strategy: Explore n = 2^a * (product of first k odd primes)^1
    # This creates a nice structure where divisors are 2^i * d for i=0..a, d|odd_part
    # Adjacent coprime pairs occur at transitions between different odd parts
    
    best_n = 2
    best_score = 0
    
    # Precompute products of first k odd primes
    odd_primes = primes[1:]  # 3, 5, 7, ...
    odd_prod = 1
    odd_prods = [1]
    for i, p in enumerate(odd_primes):
        odd_prod *= p
        if odd_prod > 10**18:
            break
        odd_prods.append(odd_prod)
    
    # Evaluate various (a, k) combinations
    candidates = []
    
    # For each number of odd primes k, find max a such that 2^a * odd_prod <= 10^18
    for k in range(1, len(odd_prods)):
        op = odd_prods[k]
        if op > 10**18:
            break
        # Max a: 2^a <= 10^18 / op
        max_a = int(math.log2(10**18 / op))
        for a in range(0, max_a + 1):
            n = (2 ** a) * op
            if n <= 10**18:
                candidates.append(n)
    
    # Also try some non-uniform exponents on odd primes
    # e.g., 2^a * 3^b * 5 * 7 * ... 
    for a in range(0, 15):
        base = 2 ** a
        if base > 10**18:
            break
        # Try different b for prime 3
        for b in range(1, 8):
            n = base * (3 ** b)
            # Multiply by next primes until we exceed 10^18
            for p in odd_primes[1:]:  # 5, 7, 11, ...
                if n * p > 10**18:
                    break
                n *= p
            if n >= 2:
                candidates.append(n)
    
    # Remove duplicates
    candidates = list(set(candidates))
    
    # Evaluate all candidates
    for n in candidates:
        if time.time() > deadline:
            break
        s = calc_score(n)
        if s > best_score:
            best_score = s
            best_n = n
    
    # Local search from best candidate
    if time.time() < deadline - 30:
        visited = set()
        visited.add(best_n)
        current_n = best_n
        current_score = best_score
        max_steps = 200
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
                
            factors = factorize(current_n)
            improved = False
            neighbors = []
            
            # Increase exponent of a prime
            for p in factors.keys():
                new_n = current_n * p
                if new_n <= 10**18 and new_n not in visited:
                    neighbors.append(new_n)
                    
            # Decrease exponent of a prime
            for p in factors.keys():
                e = factors[p]
                if e > 1:
                    new_n = current_n // p
                    if new_n not in visited:
                        neighbors.append(new_n)
                elif e == 1:
                    new_n = current_n // p
                    if new_n >= 2 and new_n not in visited:
                        neighbors.append(new_n)
                        
            # Add new small prime
            for p in primes:
                if p not in factors:
                    new_n = current_n * p
                    if new_n <= 10**18 and new_n not in visited:
                        neighbors.append(new_n)
                    else:
                        break

            # Evaluate neighbors
            best_neighbor_n = current_n
            best_neighbor_score = current_score
            
            for new_n in neighbors:
                if time.time() > deadline - 5:
                    break
                s = calc_score(new_n)
                visited.add(new_n)
                if s > best_neighbor_score:
                    best_neighbor_score = s
                    best_neighbor_n = new_n
                    improved = True
                    
            if not improved:
                break
                
            current_n = best_neighbor_n
            current_score = best_neighbor_score
            
            if current_score > best_score:
                best_score = current_score
                best_n = current_n

    return best_n
