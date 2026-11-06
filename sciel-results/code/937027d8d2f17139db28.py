import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    def get_primes(n):
        sieve = [True] * (n+1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5)+1):
            if sieve[i]:
                for j in range(i*i, n+1, i):
                    sieve[j] = False
        return [i for i in range(2, n+1) if sieve[i]]
    
    def is_valid(seq):
        """Fast check: return True if all consecutive block products are distinct."""
        products = set()
        n = len(seq)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                if prod in products:
                    return False
                products.add(prod)
        return True
    
    def get_score(seq):
        """
        Returns a score where higher is better.
        Penalizes duplicates heavily.
        """
        products = set()
        n = len(seq)
        num_dups = 0
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                if prod in products:
                    num_dups += 1
                else:
                    products.add(prod)
        
        if num_dups == 0:
            return n
        else:
            return -10000 * num_dups + n

    primes = get_primes(128)
    all_nums = list(range(2, 129))
    
    # Greedy initial construction: start with primes, try adding composites
    current_set = set(primes)
    current_seq = sorted(current_set)
    current_score = get_score(current_seq)
    
    # Try to greedily add composites in order of smallest prime factor
    composites = [x for x in all_nums if x not in current_set]
    # Sort by number of prime factors (fewer first) to reduce collisions
    def num_prime_factors(x):
        count = 0
        d = 2
        while d * d <= x:
            while x % d == 0:
                count += 1
                x //= d
            d += 1
        if x > 1:
            count += 1
        return count
    
    composites.sort(key=lambda x: (num_prime_factors(x), x))
    
    for x in composites:
        if time.time() > deadline - 5:
            break
        new_set = current_set | {x}
        new_seq = sorted(new_set)
        if is_valid(new_seq):
            current_set = new_set
            current_seq = new_seq
            current_score = get_score(new_seq)
    
    best_seq = current_seq[:]
    best_score = current_score
    
    # Simulated Annealing
    T = 100.0
    cooling = 0.999
    steps_per_temp = 50
    
    while time.time() < deadline:
        for _ in range(steps_per_temp):
            if time.time() > deadline:
                break
                
            # Generate move - weighted toward add/swap
            move_type = random.choices(['add', 'remove', 'swap'], weights=[0.4, 0.1, 0.5])[0]
            new_set = set(current_set)
            
            if move_type == 'add':
                candidates = [x for x in all_nums if x not in current_set]
                if not candidates: continue
                x = random.choice(candidates)
                new_set.add(x)
            elif move_type == 'remove':
                if not current_set: continue
                x = random.choice(list(current_set))
                new_set.remove(x)
            elif move_type == 'swap':
                candidates = [x for x in all_nums if x not in current_set]
                if not candidates or not current_set: continue
                x_out = random.choice(list(current_set))
                x_in = random.choice(candidates)
                new_set.remove(x_out)
                new_set.add(x_in)
            
            new_seq = sorted(new_set)
            new_score = get_score(new_seq)
            
            # Acceptance criterion
            delta = new_score - current_score
            if delta > 0:
                current_set = new_set
                current_seq = new_seq
                current_score = new_score
            else:
                if T > 0:
                    prob = math.exp(delta / T)
                    if random.random() < prob:
                        current_set = new_set
                        current_seq = new_seq
                        current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_seq = current_seq[:]
                
        T *= cooling
        if T < 1.0:
            T = 1.0
    
    # Final local search polish: try all single additions/removals/swaps
    if time.time() < deadline - 2:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Try adding each missing number
            for x in all_nums:
                if x in current_set:
                    continue
                new_set = current_set | {x}
                new_seq = sorted(new_set)
                if is_valid(new_seq):
                    current_set = new_set
                    current_seq = new_seq
                    current_score = get_score(new_seq)
                    if current_score > best_score:
                        best_score = current_score
                        best_seq = current_seq[:]
                    improved = True
                    break
            
            # Try removing each element
            for x in list(current_set):
                if time.time() > deadline - 2:
                    break
                new_set = current_set - {x}
                new_seq = sorted(new_set)
                new_score = get_score(new_seq)
                if new_score > current_score:
                    current_set = new_set
                    current_seq = new_seq
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_seq = current_seq[:]
                    improved = True
                    break
            
            # Try swaps
            if time.time() < deadline - 2:
                for x_out in list(current_set):
                    if time.time() > deadline - 2:
                        break
                    for x_in in all_nums:
                        if x_in in current_set or x_in == x_out:
                            continue
                        new_set = (current_set - {x_out}) | {x_in}
                        new_seq = sorted(new_set)
                        if is_valid(new_seq):
                            new_score = get_score(new_seq)
                            if new_score > current_score:
                                current_set = new_set
                                current_seq = new_seq
                                current_score = new_score
                                if current_score > best_score:
                                    best_score = current_score
                                    best_seq = current_seq[:]
                                improved = True
                                break
                    if improved:
                        break
    
    return best_seq
