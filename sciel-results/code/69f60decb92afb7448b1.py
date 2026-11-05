import time
from math import gcd
from fractions import Fraction
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    def is_coprime(a, b):
        return gcd(a, b) == 1

    def calc_score(A):
        return sum(1.0 / (512.0 - a) for a in A)

    # Precompute prime factors for each number 1..511
    prime_factors = {}
    for n in range(1, 512):
        factors = set()
        temp = n
        d = 2
        while d * d <= temp:
            while temp % d == 0:
                factors.add(d)
                temp //= d
            d += 1
        if temp > 1:
            factors.add(temp)
        prime_factors[n] = factors

    def get_prime_factors(n):
        return prime_factors[n]

    # --- Phase 1: Initial Construction ---
    A = []
    for a in range(511, 0, -1):
        if all(is_coprime(a, b) for b in A):
            A.append(a)
    
    current_score = calc_score(A)
    
    # --- Phase 2: Maximum Weight Matching Approach ---
    # For each prime p < 512, we can choose at most one number that is a power of p
    # (i.e., p^k for some k >= 1, where p^k < 512)
    # This ensures pairwise coprime since each chosen number uses a distinct prime
    
    # Build mapping: prime -> list of (power_value, score_contribution)
    prime_to_powers = defaultdict(list)
    for p in range(2, 512):
        if len(prime_factors[p]) == 1:  # p is prime
            power = p
            while power < 512:
                score = 1.0 / (512.0 - power)
                prime_to_powers[p].append((power, score))
                power *= p
            # Also include p itself if not already included (it should be)
    
    # For each prime, we want to select the power that gives maximum score
    # But we need to ensure the selected powers are distinct and pairwise coprime
    # Since each prime maps to distinct prime powers, this is automatically satisfied
    
    # However, we might also want to consider composite numbers that are products of multiple primes
    # For now, let's focus on prime powers
    
    # Create a list of (score, prime, power) for all prime powers
    all_prime_powers = []
    for p, powers in prime_to_powers.items():
        for power, score in powers:
            all_prime_powers.append((score, p, power))
    
    # Sort by score descending
    all_prime_powers.sort(key=lambda x: -x[0])
    
    # Greedy: select prime powers in order of score, ensuring no two share a prime
    selected_primes = set()
    A_matching = []
    for score, p, power in all_prime_powers:
        if p in selected_primes:
            continue
        # Check if power is coprime to all already selected
        valid = True
        for a in A_matching:
            if not is_coprime(power, a):
                valid = False
                break
        if valid:
            A_matching.append(power)
            selected_primes.add(p)
    
    matching_score = calc_score(A_matching)
    
    # Use the better of the two approaches
    if matching_score > current_score:
        A = A_matching
        current_score = matching_score
    
    # --- Phase 3: Local Search ---
    improved = True
    in_A = set(A)
    
    while improved and time.time() < deadline:
        improved = False
        n = len(A)
        
        # 1. Try Adds
        for b in range(511, 0, -1):
            if time.time() > deadline: break
            if b in in_A: continue
            if all(is_coprime(b, a) for a in A):
                A.append(b)
                in_A.add(b)
                current_score += 1.0 / (512.0 - b)
                improved = True
                break
        
        if improved: continue
        
        # 2. Try Merges
        merged = False
        for i in range(len(A)):
            if time.time() > deadline: break
            for j in range(i + 1, len(A)):
                p = A[i]
                q = A[j]
                c = p * q
                if c >= 512: continue
                
                old_val = 1.0 / (512.0 - p) + 1.0 / (512.0 - q)
                new_val = 1.0 / (512.0 - c)
                if new_val <= old_val: continue
                
                valid = True
                for k in range(len(A)):
                    if k == i or k == j: continue
                    if not is_coprime(c, A[k]):
                        valid = False
                        break
                if valid:
                    new_A = []
                    for k in range(len(A)):
                        if k != i and k != j:
                            new_A.append(A[k])
                    new_A.append(c)
                    
                    A = new_A
                    in_A = set(A)
                    current_score += (new_val - old_val)
                    improved = True
                    merged = True
                    break
            if merged: break
        
        if improved: continue
        
        # 3. Try Swaps
        swapped = False
        for i in range(len(A)):
            if time.time() > deadline: break
            a = A[i]
            for b in range(511, a, -1):
                if b in in_A: continue
                
                valid = True
                for j in range(len(A)):
                    if j == i: continue
                    if not is_coprime(b, A[j]):
                        valid = False
                        break
                
                if valid:
                    A[i] = b
                    in_A.remove(a)
                    in_A.add(b)
                    current_score += (1.0 / (512.0 - b)) - (1.0 / (512.0 - a))
                    improved = True
                    swapped = True
                    break
            if swapped: break
        
        if improved: continue
        
        # 4. Try Splits
        split = False
        for i in range(len(A)):
            if time.time() > deadline: break
            c = A[i]
            
            found_pair = False
            for p in range(2, int(c**0.5) + 1):
                if c % p != 0: continue
                q = c // p
                if p >= q: break
                if gcd(p, q) != 1: continue
                
                if p in in_A or q in in_A: continue
                
                valid = True
                for j in range(len(A)):
                    if j == i: continue
                    if not is_coprime(p, A[j]) or not is_coprime(q, A[j]):
                        valid = False
                        break
                
                if valid:
                    old_val = 1.0 / (512.0 - c)
                    new_val = 1.0 / (512.0 - p) + 1.0 / (512.0 - q)
                    if new_val > old_val:
                        new_A = []
                        for j in range(len(A)):
                            if j != i:
                                new_A.append(A[j])
                        new_A.append(p)
                        new_A.append(q)
                        
                        A = new_A
                        in_A = set(A)
                        current_score += (new_val - old_val)
                        improved = True
                        split = True
                        found_pair = True
                        break
            if split: break
        
        if improved: continue
        
        # 5. Random Perturbation
        if time.time() < deadline - 10:
            for _ in range(10):
                if time.time() > deadline: break
                
                idx = random.randint(0, len(A) - 1)
                old_a = A[idx]
                
                candidates = [b for b in range(511, 0, -1) if b not in in_A]
                if not candidates: continue
                
                b = random.choice(candidates)
                
                valid = True
                for j in range(len(A)):
                    if j == idx: continue
                    if not is_coprime(b, A[j]):
                        valid = False
                        break
                
                if not valid: continue
                
                A[idx] = b
                in_A.remove(old_a)
                in_A.add(b)
                current_score = calc_score(A)
                
                improved_local = True
                while improved_local and time.time() < deadline - 2:
                    improved_local = False
                    for b_add in range(511, 0, -1):
                        if b_add in in_A: continue
                        if all(is_coprime(b_add, a) for a in A):
                            A.append(b_add)
                            in_A.add(b_add)
                            current_score += 1.0 / (512.0 - b_add)
                            improved_local = True
                            break
                    if improved_local: continue
                    
                    merged = False
                    for i in range(len(A)):
                        for j in range(i + 1, len(A)):
                            p, q = A[i], A[j]
                            c = p * q
                            if c >= 512: continue
                            old_val = 1.0 / (512.0 - p) + 1.0 / (512.0 - q)
                            new_val = 1.0 / (512.0 - c)
                            if new_val <= old_val: continue
                            valid = True
                            for k in range(len(A)):
                                if k == i or k == j: continue
                                if not is_coprime(c, A[k]):
                                    valid = False
                                    break
                            if valid:
                                new_A = [A[k] for k in range(len(A)) if k != i and k != j]
                                new_A.append(c)
                                A = new_A
                                in_A = set(A)
                                current_score += (new_val - old_val)
                                improved_local = True
                                merged = True
                                break
                        if merged: break
                    if improved_local: continue
                    
                    swapped = False
                    for i in range(len(A)):
                        a = A[i]
                        for b_swap in range(511, a, -1):
                            if b_swap in in_A: continue
                            valid = True
                            for j in range(len(A)):
                                if j == i: continue
                                if not is_coprime(b_swap, A[j]):
                                    valid = False
                                    break
                            if valid:
                                A[i] = b_swap
                                in_A.remove(a)
                                in_A.add(b_swap)
                                current_score += (1.0 / (512.0 - b_swap)) - (1.0 / (512.0 - a))
                                improved_local = True
                                swapped = True
                                break
                        if swapped: break
                    if improved_local: continue
                    
                    split = False
                    for i in range(len(A)):
                        c = A[i]
                        for p in range(2, int(c**0.5) + 1):
                            if c % p != 0: continue
                            q = c // p
                            if p >= q: break
                            if gcd(p, q) != 1: continue
                            if p in in_A or q in in_A: continue
                            valid = True
                            for j in range(len(A)):
                                if j == i: continue
                                if not is_coprime(p, A[j]) or not is_coprime(q, A[j]):
                                    valid = False
                                    break
                            if valid:
                                old_val = 1.0 / (512.0 - c)
                                new_val = 1.0 / (512.0 - p) + 1.0 / (512.0 - q)
                                if new_val > old_val:
                                    new_A = [A[j] for j in range(len(A)) if j != i]
                                    new_A.append(p)
                                    new_A.append(q)
                                    A = new_A
                                    in_A = set(A)
                                    current_score += (new_val - old_val)
                                    improved_local = True
                                    split = True
                                    break
                        if split: break
                    if improved_local: continue

    return A
