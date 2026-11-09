import time
import random
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin

    all_numbers = list(range(2, 129))
    
    # Precompute prime factorization for each number in [2, 128]
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    def factorize(n):
        """Return prime exponent vector as a tuple."""
        exponents = [0] * num_primes
        for i, p in enumerate(primes):
            if p * p > n:
                break
            while n % p == 0:
                exponents[i] += 1
                n //= p
        if n > 1:
            exponents[prime_to_idx[n]] += 1
        return tuple(exponents)
    
    # Precompute factorizations
    factorizations = {}
    for n in range(2, 129):
        factorizations[n] = factorize(n)
    
    # Precompute all-pairs sums and differences for speed
    # We'll use a different approach: represent vectors as tuples but cache operations
    
    def check_valid(seq):
        """
        Check if strictly increasing sequence 'seq' has distinct products
        for all non-empty consecutive blocks.
        """
        n = len(seq)
        if n == 0:
            return True
        
        # Compute prefix products as exponent vectors
        S = [tuple([0]*num_primes)] * (n + 1)
        for i in range(n):
            f = factorizations[seq[i]]
            S[i+1] = tuple(S[i][k] + f[k] for k in range(num_primes))
            
        seen_products = set()
        for i in range(n):
            for j in range(i + 1, n + 1):
                prod_vec = tuple(S[j][k] - S[i][k] for k in range(num_primes))
                if prod_vec in seen_products:
                    return False
                seen_products.add(prod_vec)
        return True

    def get_prefix_products(seq):
        n = len(seq)
        S = [tuple([0]*num_primes)] * (n + 1)
        for i in range(n):
            f = factorizations[seq[i]]
            S[i+1] = tuple(S[i][k] + f[k] for k in range(num_primes))
        return S

    def get_product_set(seq):
        S = get_prefix_products(seq)
        n = len(seq)
        seen_products = set()
        for i in range(n):
            for j in range(i + 1, n + 1):
                seen_products.add(tuple(S[j][k] - S[i][k] for k in range(num_primes)))
        return seen_products

    def try_insert_fast(seq, c, S, prod_set):
        """
        Try to insert c into seq.
        Returns (new_seq, new_S, new_prod_set) if valid, else None.
        """
        n = len(seq)
        # Find insertion index
        idx = 0
        for i in range(n):
            if seq[i] < c:
                idx = i + 1
            else:
                break
        
        new_seq = seq[:idx] + [c] + seq[idx:]
        
        c_vec = factorizations[c]
        new_products_to_check = []
        for i in range(idx + 1):
            for j in range(idx, n + 1):
                # Product of seq[i..j-1] is S[j] - S[i]
                # Full product is c_vec + (S[j] - S[i])
                val = tuple(c_vec[k] + S[j][k] - S[i][k] for k in range(num_primes))
                if val in prod_set:
                    return None
                new_products_to_check.append(val)
        
        # If all valid, update state
        new_prod_set = set(prod_set)
        new_prod_set.update(new_products_to_check)
        
        # Update prefix products
        new_S = [tuple([0]*num_primes)] * (n + 2)
        for i in range(idx + 1):
            new_S[i] = S[i]
        for i in range(idx + 1, n + 2):
            new_S[i] = tuple(S[i-1][k] + c_vec[k] for k in range(num_primes))
            
        return new_seq, new_S, new_prod_set

    def greedy_build(order):
        seq = []
        S = get_prefix_products(seq)
        prod_set = get_product_set(seq)
        used = set(seq)
        
        for c in order:
            if c in used:
                continue
            res = try_insert_fast(seq, c, S, prod_set)
            if res is not None:
                seq, S, prod_set = res
                used = set(seq)
        
        return seq

    def local_search_with_sa(seq, max_iter=200):
        """Local search with simulated annealing for better escape from local optima."""
        current_seq = seq[:]
        current_S = get_prefix_products(current_seq)
        current_prod_set = get_product_set(current_seq)
        used = set(current_seq)
        remaining = [x for x in all_numbers if x not in used]
        
        best_seq = current_seq[:]
        best_S = current_S
        best_prod_set = current_prod_set
        
        temperature = 1.0
        cooling_rate = 0.95
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try insertions with heuristic ordering
            scored_insertions = []
            for c in remaining:
                if time.time() > deadline:
                    break
                res = try_insert_fast(current_seq, c, current_S, current_prod_set)
                if res is not None:
                    new_seq, new_S, new_prod_set = res
                    new_count = len(new_prod_set) - len(current_prod_set)
                    scored_insertions.append((new_count, c, new_seq, new_S, new_prod_set))
            
            if scored_insertions:
                scored_insertions.sort(key=lambda x: x[0])
                # With probability based on temperature, accept a non-best insertion
                if random.random() < temperature * 0.3 and len(scored_insertions) > 1:
                    # Pick a random insertion (not necessarily best)
                    choice_idx = random.randint(0, len(scored_insertions) - 1)
                    new_count, c, new_seq, new_S, new_prod_set = scored_insertions[choice_idx]
                else:
                    new_count, c, new_seq, new_S, new_prod_set = scored_insertions[0]
                
                current_seq, current_S, current_prod_set = new_seq, new_S, new_prod_set
                used = set(current_seq)
                remaining = [x for x in all_numbers if x not in used]
                improved = True
                
                if len(current_seq) > len(best_seq):
                    best_seq = current_seq[:]
                    best_S = current_S
                    best_prod_set = current_prod_set
            else:
                # No insertion possible, try swaps
                if remaining and len(current_seq) > 0:
                    for _ in range(5):
                        if time.time() > deadline:
                            break
                        if not remaining:
                            break
                        
                        i = random.randint(0, len(current_seq) - 1)
                        x_to_remove = current_seq[i]
                        c_to_add = random.choice(remaining)
                        
                        new_set = set(current_seq)
                        new_set.remove(x_to_remove)
                        new_set.add(c_to_add)
                        new_seq = sorted(new_set)
                        
                        if check_valid(new_seq):
                            current_seq = new_seq
                            current_S = get_prefix_products(new_seq)
                            current_prod_set = get_product_set(new_seq)
                            used = set(current_seq)
                            remaining = [x for x in all_numbers if x not in used]
                            improved = True
                            
                            if len(current_seq) > len(best_seq):
                                best_seq = current_seq[:]
                                best_S = current_S
                                best_prod_set = current_prod_set
                            break
            
            # Cool down
            temperature *= cooling_rate
            
            if not improved and temperature < 0.01:
                break
        
        return best_seq

    # Backtracking with time-aware pruning
    def backtrack_search(seq, S, prod_set, remaining, best_seq_ref):
        """
        Chronological backtracking search with aggressive time checking.
        """
        if time.time() > deadline:
            return
        
        if len(seq) > len(best_seq_ref[0]):
            best_seq_ref[0] = seq[:]
        
        if not remaining:
            return
        
        # Score candidates by how many new products they create
        scored_candidates = []
        for c in remaining:
            if time.time() > deadline:
                return
            res = try_insert_fast(seq, c, S, prod_set)
            if res is not None:
                new_seq, new_S, new_prod_set = res
                new_count = len(new_prod_set) - len(prod_set)
                scored_candidates.append((new_count, c, new_seq, new_S, new_prod_set))
        
        if not scored_candidates:
            return
        
        # Sort by number of new products (ascending)
        scored_candidates.sort(key=lambda x: x[0])
        
        for new_count, c, new_seq, new_S, new_prod_set in scored_candidates:
            if time.time() > deadline:
                return
            
            new_remaining = [x for x in remaining if x != c]
            backtrack_search(new_seq, new_S, new_prod_set, new_remaining, best_seq_ref)

    best_seq = []
    best_len = 0
    
    # Try multiple orders to find a good starting point
    orders = [
        all_numbers,
        sorted(all_numbers, reverse=True),
        sorted(all_numbers, key=lambda x: math.gcd(x, 128)),
        sorted(all_numbers, key=lambda x: len([i for i in range(2, x) if x % i == 0]))
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        seq = greedy_build(order)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]

    if best_seq:
        best_seq = local_search_with_sa(best_seq)

    # Backtracking search from current best (if time remains)
    if time.time() < deadline - 3:
        best_ref = [best_seq[:]]
        current_S = get_prefix_products(best_seq)
        current_prod_set = get_product_set(best_seq)
        used = set(best_seq)
        remaining = [x for x in all_numbers if x not in used]
        
        # Sort remaining by a heuristic: prefer numbers with fewer prime factors
        remaining.sort(key=lambda x: sum(factorizations[x]))
        
        backtrack_search(best_seq, current_S, current_prod_set, remaining, best_ref)
        if len(best_ref[0]) > len(best_seq):
            best_seq = best_ref[0][:]

    # Final Polish with multiple random restarts and SA local search
    restart_count = 0
    while time.time() < deadline - 2:
        restart_count += 1
        random.shuffle(all_numbers)
        new_seq = greedy_build(all_numbers)
        new_seq = local_search_with_sa(new_seq, max_iter=100)
        
        if len(new_seq) > len(best_seq):
            best_seq = new_seq[:]
            # Run backtracking if we found a significantly better solution
            if time.time() < deadline - 3:
                best_ref = [best_seq[:]]
                current_S = get_prefix_products(best_seq)
                current_prod_set = get_product_set(best_seq)
                used = set(best_seq)
                remaining = [x for x in all_numbers if x not in used]
                remaining.sort(key=lambda x: sum(factorizations[x]))
                backtrack_search(best_seq, current_S, current_prod_set, remaining, best_ref)
                if len(best_ref[0]) > len(best_seq):
                    best_seq = best_ref[0][:]

    # Final Verification
    if not check_valid(best_seq):
        best_seq = [2, 3, 5, 7]
        if not check_valid(best_seq):
            best_seq = [2]

    return best_seq
