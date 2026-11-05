import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110 # Safety margin
    
    # Primes to consider
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    # Initial state: First 15 primes with exponent 1
    # This is a good starting point, known to be valid and decent
    current_dict = {p: 1 for p in primes[:15]}
    current_product = 1
    for p in current_dict:
        current_product *= p
        
    def calc_score(d, prod):
        # Generate divisors
        divs = [1]
        for p, e in d.items():
            if e == 0:
                continue
            new_divs = []
            for val in divs:
                v = val
                for _ in range(e):
                    new_divs.append(v)
                    v *= p
            divs = new_divs
        divs.sort()
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score

    best_score = calc_score(current_dict, current_product)
    best_dict = current_dict.copy()
    best_product = current_product
    
    # Local Search
    # We will perform random walks
    # To avoid O(N) product calculation, we update incrementally
    
    while time.time() < deadline:
        # Pick a mutation
        mut_type = random.randint(0, 3)
        new_dict = current_dict.copy()
        new_product = current_product
        
        valid = True
        
        if mut_type == 0:
            # Increase exponent
            if not new_dict:
                valid = False
            else:
                p = random.choice(list(new_dict.keys()))
                if new_product * p <= 10**18:
                    new_dict[p] = new_dict[p] + 1
                    new_product *= p
                else:
                    valid = False
                    
        elif mut_type == 1:
            # Decrease exponent
            candidates = [p for p, e in new_dict.items() if e > 1]
            if not candidates:
                valid = False
            else:
                p = random.choice(candidates)
                new_dict[p] = new_dict[p] - 1
                new_product //= p
                if new_dict[p] == 0:
                    del new_dict[p]
                    
        elif mut_type == 2:
            # Add new prime
            available = [p for p in primes if p not in new_dict]
            if not available:
                valid = False
            else:
                p = random.choice(available)
                if new_product * p <= 10**18:
                    new_dict[p] = 1
                    new_product *= p
                else:
                    valid = False
                    
        else:
            # Remove prime
            if not new_dict:
                valid = False
            else:
                p = random.choice(list(new_dict.keys()))
                new_product //= new_dict[p]
                del new_dict[p]

        if not valid:
            continue
            
        # Evaluate
        new_score = calc_score(new_dict, new_product)
        
        if new_score > best_score:
            best_score = new_score
            best_dict = new_dict.copy()
            best_product = new_product
            current_dict = new_dict.copy()
            current_product = new_product
        else:
            # With some probability, accept worse moves to escape local optima?
            # Or just reject?
            # Given the time limit, greedy is safer.
            # But we can occasionally reset to best.
            pass

    n = 1
    for p, e in best_dict.items():
        n *= (p ** e)
        
    return n
