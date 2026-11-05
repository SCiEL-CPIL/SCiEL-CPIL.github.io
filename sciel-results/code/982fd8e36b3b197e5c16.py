import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 85.0

    # 1. Greedy Baseline
    def greedy_construct(order):
        values, products = [], set()
        for x in order:
            if x < 2 or x > 128:
                continue
            tails, prod = [x], x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
        values.sort()
        return values

    # Standard greedy
    best_seq = greedy_construct(range(2, 129))
    best_len = len(best_seq)

    # Try a few other orderings for greedy
    # Primes first
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(n**0.5)+1):
            if n % i == 0: return False
        return True
        
    primes = [x for x in range(2, 129) if is_prime(x)]
    composites = [x for x in range(2, 129) if not is_prime(x)]
    
    if time.time() - start_time < DEADLINE:
        seq_p = greedy_construct(primes + composites)
        if len(seq_p) > best_len:
            best_seq = seq_p
            best_len = len(seq_p)

    # Helper to check validity and get products
    def get_products(seq):
        prods = set()
        n = len(seq)
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                prods.add(p)
        return prods

    def is_valid(seq):
        n = len(seq)
        return len(get_products(seq)) == n * (n + 1) // 2

    # Local Search: Try to insert elements
    def try_insert(seq, elem):
        # Try inserting elem at every position
        n = len(seq)
        for i in range(n + 1):
            # Check bounds
            if i > 0 and seq[i-1] >= elem:
                continue
            if i < n and seq[i] <= elem:
                continue
            
            new_seq = seq[:i] + [elem] + seq[i:]
            if is_valid(new_seq):
                return new_seq
        return None

    # Local Search: Try to improve by inserting
    improved = True
    while improved and time.time() - start_time < DEADLINE:
        improved = False
        used = set(best_seq)
        remaining = [x for x in range(2, 129) if x not in used]
        # Shuffle remaining to break symmetry? Or just iterate.
        for x in remaining:
            if time.time() - start_time > DEADLINE:
                break
            new_seq = try_insert(best_seq, x)
            if new_seq:
                best_seq = new_seq
                best_len = len(best_seq)
                improved = True
                break # Restart loop with new state

    # Local Search: Try to improve by removing one and adding two?
    # This is harder. Let's try a different strategy:
    # Randomized Restart Greedy
    # Run greedy with random perturbations in the order.
    
    for _ in range(50): # Try 50 random orderings
        if time.time() - start_time > DEADLINE:
            break
        random.shuffle(remaining) # Wait, remaining changes. Use full range.
        order = list(range(2, 129))
        random.shuffle(order)
        seq_rand = greedy_construct(order)
        if len(seq_rand) > best_len:
            best_seq = seq_rand
            best_len = len(best_seq)
            # Try to extend this new best
            improved = True
            while improved and time.time() - start_time < DEADLINE:
                improved = False
                used = set(best_seq)
                rem = [x for x in range(2, 129) if x not in used]
                for x in rem:
                    if time.time() - start_time > DEADLINE:
                        break
                    new_seq = try_insert(best_seq, x)
                    if new_seq:
                        best_seq = new_seq
                        best_len = len(best_seq)
                        improved = True
                        break

    return best_seq
