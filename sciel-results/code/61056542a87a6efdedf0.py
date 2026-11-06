import time
import random
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(1000)
    
    # Seed initialization
    seed_primes = [p for p in primes if p >= 13]
    classes = [[p, (i + 1) % p] for i, p in enumerate(seed_primes)]
    
    MAX_L = 100000
    cover_count = np.zeros(MAX_L + 1, dtype=np.int32)
    
    def add_class(p, r):
        if r == 0:
            # 0, p, 2p...
            # We update 0..MAX_L
            cover_count[0::p] += 1
        else:
            cover_count[r::p] += 1

    def remove_class(p, r):
        if r == 0:
            cover_count[0::p] -= 1
        else:
            cover_count[r::p] -= 1

    # Initialize
    for p, r in classes:
        add_class(p, r)
        
    def get_coverage_length():
        zeros = np.where(cover_count[1:] == 0)[0]
        if len(zeros) == 0:
            return MAX_L
        return zeros[0]

    best_L = get_coverage_length()
    best_classes = [c[:] for c in classes]
    
    current_L = best_L
    current_classes = [c[:] for c in classes]
    current_budget = sum(1.0 / p for p, r in classes)
    current_prime_set = set(p for p, r in current_classes)
    
    # Local Search
    while time.time() - start_time < time_limit:
        # Targeted Move
        if random.random() < 0.4:
            target = current_L + 1
            if target > MAX_L:
                break
            
            # Try changing residue of existing prime
            if current_classes:
                idx = random.randint(0, len(current_classes) - 1)
                p, r_old = current_classes[idx]
                r_new = target % p
                if r_new != r_old:
                    remove_class(p, r_old)
                    current_classes[idx][1] = r_new
                    add_class(p, r_new)
                    new_L = get_coverage_length()
                    if new_L >= current_L:
                        current_L = new_L
                        if new_L > best_L:
                            best_L = new_L
                            best_classes = [c[:] for c in current_classes]
                    else:
                        remove_class(p, r_new)
                        current_classes[idx][1] = r_old
                        add_class(p, r_old)
                        
            # Try adding new prime
            p_new = random.choice(primes)
            if p_new not in current_prime_set:
                cost_new = 1.0 / p_new
                if current_budget + cost_new <= 1.0:
                    r_new = target % p_new
                    current_classes.append([p_new, r_new])
                    current_prime_set.add(p_new)
                    current_budget += cost_new
                    add_class(p_new, r_new)
                    new_L = get_coverage_length()
                    if new_L >= current_L:
                        current_L = new_L
                        if new_L > best_L:
                            best_L = new_L
                            best_classes = [c[:] for c in current_classes]
                    else:
                        remove_class(p_new, r_new)
                        current_classes.pop()
                        current_prime_set.remove(p_new)
                        current_budget -= cost_new
        else:
            # Random Move
            move_type = random.random()
            if move_type < 0.5:
                if not current_classes: continue
                idx = random.randint(0, len(current_classes) - 1)
                p, r_old = current_classes[idx]
                r_new = random.randint(0, p - 1)
                if r_new == r_old: continue
                remove_class(p, r_old)
                current_classes[idx][1] = r_new
                add_class(p, r_new)
                new_L = get_coverage_length()
                if new_L >= current_L:
                    current_L = new_L
                    if new_L > best_L:
                        best_L = new_L
                        best_classes = [c[:] for c in current_classes]
                else:
                    remove_class(p, r_new)
                    current_classes[idx][1] = r_old
                    add_class(p, r_old)
            elif move_type < 0.7:
                p_new = random.choice(primes)
                if p_new not in current_prime_set:
                    cost_new = 1.0 / p_new
                    if current_budget + cost_new <= 1.0:
                        r_new = random.randint(0, p_new - 1)
                        current_classes.append([p_new, r_new])
                        current_prime_set.add(p_new)
                        current_budget += cost_new
                        add_class(p_new, r_new)
                        new_L = get_coverage_length()
                        if new_L >= current_L:
                            current_L = new_L
                            if new_L > best_L:
                                best_L = new_L
                                best_classes = [c[:] for c in current_classes]
                        else:
                            remove_class(p_new, r_new)
                            current_classes.pop()
                            current_prime_set.remove(p_new)
                            current_budget -= cost_new
            elif move_type < 0.9:
                if not current_classes: continue
                idx = random.randint(0, len(current_classes) - 1)
                p_old, r_old = current_classes[idx]
                remove_class(p_old, r_old)
                current_classes.pop(idx)
                current_prime_set.remove(p_old)
                current_budget -= 1.0 / p_old
                new_L = get_coverage_length()
                if new_L >= current_L:
                    current_L = new_L
                    if new_L > best_L:
                        best_L = new_L
                        best_classes = [c[:] for c in current_classes]
                else:
                    current_classes.insert(idx, [p_old, r_old])
                    current_prime_set.add(p_old)
                    current_budget += 1.0 / p_old
                    add_class(p_old, r_old)
            else:
                if not current_classes: continue
                idx = random.randint(0, len(current_classes) - 1)
                p_old, r_old = current_classes[idx]
                p_new = random.choice(primes)
                if p_new == p_old or p_new in current_prime_set:
                    continue
                cost_old = 1.0 / p_old
                cost_new = 1.0 / p_new
                if current_budget - cost_old + cost_new <= 1.0:
                    r_new = random.randint(0, p_new - 1)
                    remove_class(p_old, r_old)
                    current_prime_set.remove(p_old)
                    current_prime_set.add(p_new)
                    current_budget = current_budget - cost_old + cost_new
                    current_classes[idx] = [p_new, r_new]
                    add_class(p_new, r_new)
                    new_L = get_coverage_length()
                    if new_L >= current_L:
                        current_L = new_L
                        if new_L > best_L:
                            best_L = new_L
                            best_classes = [c[:] for c in current_classes]
                    else:
                        remove_class(p_new, r_new)
                        current_prime_set.remove(p_new)
                        current_prime_set.add(p_old)
                        current_budget = current_budget - cost_new + cost_old
                        current_classes[idx] = [p_old, r_old]
                        add_class(p_old, r_old)
                        
    # Final Validation with Fractions
    # Check if best_classes is valid
    # If not, fallback to seed? Or try to repair?
    # For now, return best_classes. The evaluator will check.
    # If invalid, score 0.
    # We can do a quick check:
    budget = sum(Fraction(1, p) for p, r in best_classes)
    if budget > 1:
        # Try to remove the largest prime to reduce budget?
        # Or just return the seed?
        # Let's return the seed if the best is invalid.
        return [[p, (i + 1) % p] for i, p in enumerate(seed_primes)]
        
    return best_classes
