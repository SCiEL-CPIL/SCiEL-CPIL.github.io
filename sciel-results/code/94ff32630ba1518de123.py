import time
import math
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin (evaluator kills at 150s)
    
    N = 512
    weights = [1.0 / (N - a) for a in range(N)]
    
    # Precompute gcd table for speed
    gcd_table = [[0]*N for _ in range(N)]
    for i in range(1, N):
        for j in range(i+1, N):
            gcd_table[i][j] = math.gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]
    
    # Precompute prime factors for each number
    prime_factors = [set() for _ in range(N)]
    for p in range(2, N):
        # Check if p is prime
        is_prime = True
        for d in range(2, int(math.isqrt(p)) + 1):
            if p % d == 0:
                is_prime = False
                break
        if not is_prime:
            continue
        for multiple in range(p, N, p):
            prime_factors[multiple].add(p)
    
    # Candidates sorted by weight descending (a close to 511 first)
    candidates = list(range(N-1, 0, -1))
    
    def get_score(s):
        return sum(weights[a] for a in s)
    
    def is_coprime_with_all(a, chosen_list):
        for b in chosen_list:
            if gcd_table[a][b] > 1:
                return False
        return True
    
    def greedy_init(order):
        chosen = []
        chosen_set = set()
        for a in order:
            if is_coprime_with_all(a, chosen):
                chosen.append(a)
                chosen_set.add(a)
        return chosen
    
    # Multiple initialization strategies
    random.seed(42)
    
    # Strategy 1: Largest to smallest
    init1 = greedy_init(candidates)
    
    # Strategy 2: Randomized (several tries)
    best_random_init = []
    best_random_score = 0
    for trial in range(10):
        order = list(candidates)
        random.shuffle(order)
        init = greedy_init(order)
        s = get_score(init)
        if s > best_random_score:
            best_random_score = s
            best_random_init = init
    
    # Strategy 3: Perturbed greedy
    order3 = list(candidates)
    for i in range(len(order3)-1, 0, -1):
        if random.random() < 0.15:
            j = random.randint(0, i)
            order3[i], order3[j] = order3[j], order3[i]
    init3 = greedy_init(order3)
    
    # Strategy 4: Prime-focused initialization - prioritize large primes and their powers
    # This helps build a base of large coprime numbers
    primes_under_N = [p for p in range(2, N) if len(prime_factors[p]) == 1 and p in prime_factors[p]]
    # Sort primes by weight descending
    primes_sorted = sorted(primes_under_N, key=lambda p: -weights[p])
    init4 = greedy_init(primes_sorted + [a for a in candidates if a not in primes_under_N])
    
    candidates_list = [init1, best_random_init, init3, init4]
    best_init = max(candidates_list, key=get_score)
    best_score = get_score(best_init)
    best_chosen = list(best_init)
    
    # 2-for-1 swap: replace two smaller elements with one larger element
    def two_for_one_swap(current, current_set, current_score):
        """Try to replace two elements with one larger coprime element."""
        if time.time() > deadline:
            return None, None, None
        
        n = len(current)
        if n < 2:
            return None, None, None
        
        best_gain = 0
        best_replacement = None
        best_removed = None
        
        # Try all pairs of elements to remove
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i+1, n):
                if time.time() > deadline:
                    break
                a, b = current[i], current[j]
                remaining = [x for x in current if x != a and x != b]
                remaining_set = current_set - {a, b}
                
                # Find best replacement c that is coprime to all remaining
                # and has weight > weight[a] + weight[b]
                for c in range(N-1, 0, -1):
                    if c in remaining_set:
                        continue
                    if is_coprime_with_all(c, remaining):
                        gain = weights[c] - weights[a] - weights[b]
                        if gain > best_gain:
                            best_gain = gain
                            best_replacement = c
                            best_removed = (a, b)
        
        if best_replacement is not None:
            a, b = best_removed
            new_current = [x for x in current if x != a and x != b]
            new_current.append(best_replacement)
            new_set = current_set - {a, b}
            new_set.add(best_replacement)
            new_score = current_score + best_gain
            return new_current, new_set, new_score
        
        return None, None, None
    
    # 1-for-1 swap: replace one element with another
    def one_for_one_swap(current, current_set, current_score):
        """Try to replace one element with a better one."""
        if time.time() > deadline:
            return None, None, None
        
        n = len(current)
        best_gain = 0
        best_replacement = None
        best_removed = None
        
        for i in range(n):
            if time.time() > deadline:
                break
            a = current[i]
            remaining = [x for x in current if x != a]
            remaining_set = current_set - {a}
            
            for c in range(N-1, 0, -1):
                if c in remaining_set:
                    continue
                if is_coprime_with_all(c, remaining):
                    gain = weights[c] - weights[a]
                    if gain > best_gain:
                        best_gain = gain
                        best_replacement = c
                        best_removed = a
        
        if best_replacement is not None:
            a = best_removed
            new_current = [x for x in current if x != a]
            new_current.append(best_replacement)
            new_set = current_set - {a}
            new_set.add(best_replacement)
            new_score = current_score + best_gain
            return new_current, new_set, new_score
        
        return None, None, None
    
    # Add element if possible
    def try_add_element(current, current_set, current_score):
        """Try to add a new element."""
        if time.time() > deadline:
            return None, None, None
        
        for c in range(N-1, 0, -1):
            if c in current_set:
                continue
            if is_coprime_with_all(c, current):
                new_current = list(current)
                new_current.append(c)
                new_set = set(current_set)
                new_set.add(c)
                new_score = current_score + weights[c]
                return new_current, new_set, new_score
        
        return None, None, None
    
    # Local search with 2-for-1, 1-for-1, and add moves
    if time.time() < deadline - 5:
        current = list(best_chosen)
        current_set = set(current)
        current_score = best_score
        
        # Iterate through different move types
        for iteration in range(200):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try 2-for-1 swaps (most impactful)
            new_current, new_set, new_score = two_for_one_swap(current, current_set, current_score)
            if new_current is not None and new_score > current_score + 1e-12:
                current = new_current
                current_set = new_set
                current_score = new_score
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_chosen = list(current)
                continue
            
            # Try 1-for-1 swaps
            new_current, new_set, new_score = one_for_one_swap(current, current_set, current_score)
            if new_current is not None and new_score > current_score + 1e-12:
                current = new_current
                current_set = new_set
                current_score = new_score
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_chosen = list(current)
                continue
            
            # Try adding elements
            new_current, new_set, new_score = try_add_element(current, current_set, current_score)
            if new_current is not None:
                current = new_current
                current_set = new_set
                current_score = new_score
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_chosen = list(current)
                continue
            
            if not improved:
                break
    
    # Final polish: try more aggressive 2-for-1 swaps with multiple candidates
    if time.time() < deadline - 2:
        current = list(best_chosen)
        current_set = set(current)
        current_score = best_score
        
        for _ in range(50):
            if time.time() > deadline:
                break
            
            improved = False
            n = len(current)
            
            if n >= 2:
                for i in range(n):
                    if time.time() > deadline:
                        break
                    for j in range(i+1, n):
                        if time.time() > deadline:
                            break
                        a, b = current[i], current[j]
                        remaining = [x for x in current if x != a and x != b]
                        
                        # Try multiple replacement candidates
                        for c in range(N-1, 0, -1):
                            if c in current_set and c not in (a, b):
                                continue
                            if is_coprime_with_all(c, remaining):
                                gain = weights[c] - weights[a] - weights[b]
                                if gain > 1e-12:
                                    new_current = remaining + [c]
                                    new_set = set(remaining)
                                    new_set.add(c)
                                    new_score = current_score + gain
                                    current = new_current
                                    current_set = new_set
                                    current_score = new_score
                                    improved = True
                                    if current_score > best_score:
                                        best_score = current_score
                                        best_chosen = list(current)
                                    break
                        if improved:
                            break
                    if improved:
                        break
            
            if not improved:
                break
    
    return best_chosen
