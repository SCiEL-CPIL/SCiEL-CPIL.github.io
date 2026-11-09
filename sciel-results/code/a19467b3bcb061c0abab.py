import time
import random

def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def is_valid(seq):
    n = len(seq)
    if n == 0:
        return True
    products = set()
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= seq[j]
            if prod in products:
                return False
            products.add(prod)
    return True

def construct():
    deadline = time.time() + 85.0
    best_seq = []
    best_len = 0
    
    # Strategy 1: Ascending Greedy (Seed)
    values, products = [], set()
    for x in range(2, 129):
        tails, prod = [x], x
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    if len(values) > best_len:
        best_len = len(values)
        best_seq = values
        
    # Strategy 2: Prime-based
    primes = [x for x in range(2, 129) if is_prime(x)]
    if is_valid(primes):
        if len(primes) > best_len:
            best_len = len(primes)
            best_seq = primes

    # Strategy 3: Completion of Seed and Prime sequences
    # We define a helper to try to extend a sequence
    def extend_sequence(base_seq, iterations):
        current_best = list(base_seq)
        current_best_len = len(current_best)
        missing = [x for x in range(2, 129) if x not in current_best]
        
        for _ in range(iterations):
            if time.time() > deadline:
                break
            random.shuffle(missing)
            temp_seq = list(current_best)
            # Try to add each missing number in the shuffled order
            for x in missing:
                new_seq = sorted(temp_seq + [x])
                if is_valid(new_seq):
                    temp_seq = new_seq
            if len(temp_seq) > current_best_len:
                current_best = temp_seq
                current_best_len = len(current_best)
                missing = [x for x in range(2, 129) if x not in current_best]
                if not missing:
                    break
        return current_best

    # Extend Seed
    seed_extended = extend_sequence(best_seq, 500)
    if len(seed_extended) > best_len:
        best_len = len(seed_extended)
        best_seq = seed_extended
        
    # Extend Primes
    if len(primes) > 0:
        prime_extended = extend_sequence(primes, 500)
        if len(prime_extended) > best_len:
            best_len = len(prime_extended)
            best_seq = prime_extended

    return best_seq
