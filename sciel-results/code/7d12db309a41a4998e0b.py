"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    # Precompute prime factorizations for numbers 2..128
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    
    def factorize(n):
        """Return tuple of (prime, exponent) for n."""
        factors = []
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                exp = 0
                while n % p == 0:
                    n //= p
                    exp += 1
                factors.append((p, exp))
        if n > 1:
            factors.append((n, 1))
        return tuple(factors)
    
    # Precompute factorizations
    factor_cache = {}
    for i in range(2, 129):
        factor_cache[i] = factorize(i)
    
    # For efficiency, represent factor vectors as sorted tuples for hashing
    
    def add_factor_vectors(v1, v2):
        """Add two factor vectors (tuples of (prime, exp))."""
        if not v1:
            return v2
        if not v2:
            return v1
        result = {}
        for p, e in v1:
            result[p] = e
        for p, e in v2:
            result[p] = result.get(p, 0) + e
        return tuple(sorted(result.items()))
    
    def compute_tails_factors(values, x):
        """Compute all consecutive-block product factor vectors ending at x."""
        tails = []
        prod_vec = factor_cache[x]
        tails.append(prod_vec)
        for a in reversed(values):
            prod_vec = add_factor_vectors(prod_vec, factor_cache[a])
            tails.append(prod_vec)
        return tails
    
    def is_valid_factors(values, product_set, x):
        """Check if x can be appended without product collision using factor vectors."""
        tails = compute_tails_factors(values, x)
        for t in tails:
            if t in product_set:
                return False, tails
        return True, tails
    
    def compute_all_products(values):
        """Compute all consecutive block products for a sequence."""
        products = set()
        n = len(values)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= values[j]
                products.add(prod)
        return products
    
    def compute_tails(values, x):
        """Compute all consecutive-block products ending at x for the sequence values+[x]."""
        tails = []
        prod = x
        tails.append(prod)
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        return tails
    
    def is_valid(values, products, x):
        """Check if x can be appended to values without product collision."""
        tails = compute_tails(values, x)
        for t in tails:
            if t in products:
                return False, tails
        return True, tails
    
    def greedy_from(values, products, start):
        """Continue greedily from current state. Returns (values, products)."""
        for x in range(start, 129):
            if time.time() > deadline:
                break
            valid, tails = is_valid(values, products, x)
            if valid:
                values.append(x)
                products.update(tails)
        return values, products
    
    # Greedy baseline using factor vectors for speed
    def run_greedy_factors():
        values = []
        product_set = set()
        for x in range(2, 129):
            if time.time() > deadline:
                break
            tails = compute_tails_factors(values, x)
            valid = True
            for t in tails:
                if t in product_set:
                    valid = False
                    break
            if valid:
                values.append(x)
                product_set.update(tails)
        return values, product_set
    
    # Run greedy to get initial solution
    best_seq, best_prods = run_greedy_factors()
    best_len = len(best_seq)
    
    # Beam search with factor vectors for efficiency
    beam = {(): set()}
    W = 300  # Beam width
    
    for x in range(2, 129):
        if time.time() > deadline:
            break
        
        next_beam = {}
        
        for seq, prods in beam.items():
            # Option 1: Skip x
            if seq not in next_beam:
                next_beam[seq] = prods
            
            # Option 2: Include x
            # Compute tails using factor vectors
            tails = compute_tails_factors(list(seq), x)
            
            # Check collision
            has_collision = False
            for t in tails:
                if t in prods:
                    has_collision = True
                    break
            
            if not has_collision:
                new_seq = seq + (x,)
                new_prods = set(prods)
                new_prods.update(tails)
                next_beam[new_seq] = new_prods
        
        # Prune beam
        sorted_seqs = sorted(next_beam.keys(), key=lambda s: (-len(s), sum(s) if s else 0))
        
        # Keep top W
        beam = {s: next_beam[s] for s in sorted_seqs[:W]}
        
        # Update best
        if sorted_seqs and len(sorted_seqs[0]) > best_len:
            best_len = len(sorted_seqs[0])
            best_seq = list(sorted_seqs[0])
            best_prods = next_beam[sorted_seqs[0]]
    
    # Now try to improve with local search
    current = list(best_seq)
    current_products = compute_all_products(current)
    current_len = len(current)
    
    # Simulated annealing parameters
    T = 10.0
    cooling = 0.95
    
    random.seed(42)
    
    for iteration in range(3000):
        if time.time() > deadline:
            break
        
        # Generate a neighbor
        neighbor = list(current)
        neighbor_products = set(current_products)
        
        move_type = random.choice(['swap', 'insert', 'remove', 'replace'])
        
        if move_type == 'swap' and len(current) >= 2:
            # Swap two adjacent elements
            i = random.randint(0, len(current) - 2)
            j = i + 1
            # Check if swapping maintains strictly increasing order
            if i > 0 and current[i-1] >= current[j]:
                continue
            if j < len(current) - 1 and current[i] >= current[j+1]:
                continue
            neighbor[i], neighbor[j] = current[j], current[i]
            
            # Recompute products
            neighbor_products = compute_all_products(neighbor)
            # Check for duplicates
            if len(neighbor_products) != sum(len(neighbor) - i for i in range(len(neighbor))):
                continue
            
        elif move_type == 'insert':
            # Try to insert a new element
            pos = random.randint(0, len(current))
            # Find a value to insert
            if pos == 0:
                lo, hi = 2, current[0] - 1 if current else 128
            elif pos == len(current):
                lo, hi = current[-1] + 1, 128
            else:
                lo, hi = current[pos-1] + 1, current[pos] - 1
            
            if lo > hi:
                continue
            
            new_val = random.randint(lo, hi)
            neighbor = current[:pos] + [new_val] + current[pos:]
            
            # Check strictly increasing
            valid_inc = True
            for k in range(len(neighbor) - 1):
                if neighbor[k] >= neighbor[k+1]:
                    valid_inc = False
                    break
            if not valid_inc:
                continue
            
            # Recompute products
            neighbor_products = compute_all_products(neighbor)
            # Check for duplicates
            if len(neighbor_products) != sum(len(neighbor) - i for i in range(len(neighbor))):
                neighbor = list(current)
                neighbor_products = set(current_products)
                
        elif move_type == 'remove' and len(current) > 1:
            # Remove a random element
            pos = random.randint(0, len(current) - 1)
            neighbor = current[:pos] + current[pos+1:]
            neighbor_products = compute_all_products(neighbor)
            
        elif move_type == 'replace' and len(current) >= 1:
            # Replace a random element with another valid value
            pos = random.randint(0, len(current) - 1)
            
            if pos == 0:
                lo, hi = 2, (current[1] - 1 if len(current) > 1 else 128)
            elif pos == len(current) - 1:
                lo, hi = (current[pos-1] + 1 if pos > 0 else 2), 128
            else:
                lo, hi = current[pos-1] + 1, current[pos+1] - 1
            
            if lo > hi:
                continue
            
            new_val = random.randint(lo, hi)
            if new_val == current[pos]:
                continue
            
            neighbor = list(current)
            neighbor[pos] = new_val
            
            # Check strictly increasing
            valid_inc = True
            for k in range(len(neighbor) - 1):
                if neighbor[k] >= neighbor[k+1]:
                    valid_inc = False
                    break
            if not valid_inc:
                continue
            
            # Recompute products
            neighbor_products = compute_all_products(neighbor)
            # Check for duplicates
            if len(neighbor_products) != sum(len(neighbor) - i for i in range(len(neighbor))):
                neighbor = list(current)
                neighbor_products = set(current_products)
        
        # Accept or reject based on simulated annealing
        delta = len(neighbor) - current_len
        
        if delta > 0:
            # Always accept improvements
            current = neighbor
            current_products = neighbor_products
            current_len = len(current)
            
            if current_len > best_len:
                best = list(current)
                best_len = current_len
        
        else:
            # Accept worse solutions with probability exp(delta/T)
            if random.random() < (2.718281828 ** (delta / T)):
                current = neighbor
                current_products = neighbor_products
                current_len = len(current)
        
        T *= cooling
    
    # Final greedy polish: try to extend the best solution found
    if time.time() < deadline - 5:
        polish_values = list(best_seq)
        polish_products = compute_all_products(polish_values)
        polish_values, polish_products = greedy_from(polish_values, polish_products, 
                                                      polish_values[-1] + 1 if polish_values else 2)
        if len(polish_values) > best_len:
            best_seq = polish_values
            best_len = len(best_seq)
    
    return best_seq
# EVOLVE-BLOCK-END
