import time
import sys
import random
from collections import Counter

# Increase recursion limit just in case, though depth is small
sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 90s limit
    
    # 1. Precompute prime factorization for numbers 2 to 128
    # primes up to 128
    primes = []
    is_prime = [True] * 129
    for i in range(2, 129):
        if is_prime[i]:
            primes.append(i)
            for j in range(i*i, 129, i):
                if j < 129:
                    is_prime[j] = False
    
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # factorization: list of dictionaries {prime_idx: exponent} for each number 2..128
    # We only care about numbers 2..128
    max_val = 128
    factorizations = [None] * (max_val + 1)
    for i in range(2, max_val + 1):
        n = i
        factors = {}
        for p_idx, p in enumerate(primes):
            if p * p > n:
                break
            count = 0
            while n % p == 0:
                count += 1
                n //= p
            if count > 0:
                factors[p_idx] = count
        if n > 1:
            p_idx = prime_to_idx[n]
            factors[p_idx] = 1
        factorizations[i] = factors

    def add_factors(f1, f2):
        """Add two factorization dictionaries, return new dict"""
        if not f1: return dict(f2)
        if not f2: return dict(f1)
        res = dict(f1)
        for k, v in f2.items():
            res[k] = res.get(k, 0) + v
        return res

    def get_block_products(seq):
        """
        Given a sequence, return a list of all non-empty consecutive block products.
        Returns a list of tuples representing the prime exponent vectors.
        """
        prods = []
        n = len(seq)
        for i in range(n):
            # Start with empty factorization
            current_factors = {}
            for j in range(i, n):
                # Multiply by seq[j]
                current_factors = add_factors(current_factors, factorizations[seq[j]])
                # Convert to a hashable representation (tuple of sorted items)
                # Since keys are indices and we iterate primes in order, we can just use the dict itself if we freeze it?
                # No, dicts are not hashable. Convert to tuple of (key, value) pairs.
                # To ensure consistent ordering, we can rely on the fact that we only add keys in increasing order?
                # Actually, add_factors preserves insertion order of keys if we are careful, but python dicts preserve insertion order.
                # However, if we merge, the order might change.
                # Better: convert to a canonical form.
                # Since the number of primes is small (31 primes up to 128), we can represent as a tuple of length len(primes)
                # This is O(31) per product. Number of products is O(L^2).
                # This might be slow for L=50 (2500 products * 31 ops = 77k ops per fitness eval).
                # Let's try the tuple of items approach first, it's faster if sparse.
                # But wait, if we use tuple(sorted(factors.items())), it's O(K log K) where K is number of distinct primes in product.
                # K is small.
                prods.append(tuple(sorted(current_factors.items())))
        return prods

    def evaluate(seq):
        """
        Returns a fitness score.
        Higher is better.
        Score = Length - Penalty
        Penalty is proportional to the number of duplicate block products.
        """
        if not seq:
            return 0
        
        # Check strictly increasing
        for i in range(len(seq) - 1):
            if seq[i] >= seq[i+1]:
                return -1000 # Invalid sequence, heavy penalty
        
        # Check range
        if seq[0] < 2 or seq[-1] > 128:
            return -1000
            
        prods = get_block_products(seq)
        
        # Count duplicates
        counts = Counter(prods)
        duplicates = sum(c - 1 for c in counts.values() if c > 1)
        
        # Fitness: We want to maximize length, minimize duplicates.
        # If duplicates > 0, the sequence is invalid.
        # We use a score that prioritizes length but strongly penalizes duplicates.
        # Score = len(seq) * 100 - duplicates * 1000
        # This way, a valid sequence of length 10 scores 1000.
        # An invalid sequence of length 10 with 1 duplicate scores 1000 - 1000 = 0.
        # An invalid sequence of length 20 with 1 duplicate scores 2000 - 1000 = 1000 (tied with valid length 10).
        # To ensure valid sequences always beat invalid ones of any length:
        # If valid: Score = len * 1000
        # If invalid: Score = - (duplicates * 1000 + len)
        # Let's use:
        if duplicates == 0:
            return len(seq) * 1000
        else:
            return - (duplicates * 1000 + len(seq))

    # 2. Initial Solution
    # Greedy construction
    current_seq = []
    used_prods = set()
    for x in range(2, 129):
        # Check if x can be added
        # New products are x, x*last, x*last*prev, ...
        # We need to check against used_prods
        # To do this efficiently, we can maintain the suffix products of the current sequence?
        # Or just recompute for the candidate?
        # Since we are doing this once, recompute is fine.
        # But we need to check if ANY new product is in used_prods.
        
        # Let's just use the full check for the greedy start
        # This is O(L^2) per step, total O(L^3). L is small.
        
        # Actually, let's just use a simpler greedy:
        # Try to add x.
        # Calculate all block products of current_seq + [x]
        # If any product is in used_prods, skip x.
        # Else, add x and update used_prods.
        
        # Optimization: Maintain suffix products of current_seq
        # suffix_prods[i] = product of seq[i:]
        # When adding x, new products are:
        # x
        # x * seq[-1]
        # x * seq[-2] * seq[-1]
        # ...
        # x * seq[0] * ... * seq[-1]
        
        # We don't have suffix_prods stored easily in the simple loop above.
        # Let's just do the full check for the initial greedy.
        
        candidate = current_seq + [x]
        # Check if candidate is valid with respect to previous products?
        # No, we need to check if the NEW products (those involving x) collide with OLD products.
        # And also if the NEW products collide with EACH OTHER? 
        # Wait, if current_seq was valid, its products are unique.
        # New products are suffixes of candidate ending at x.
        # Do any of these new products equal any old product?
        # Do any of these new products equal each other?
        # Since they are suffixes of a strictly increasing sequence of integers >= 2, 
        # p_k = x * prod(seq[-k:])
        # p_{k+1} = x * seq[-k-1] * prod(seq[-k:])
        # Since seq[-k-1] >= 2, p_{k+1} >= 2 * p_k.
        # So new products are distinct from each other.
        # We only need to check if any new product is in used_prods.
        
        new_prods = []
        prod = x
        new_prods.append(prod)
        for i in range(len(current_seq) - 1, -1, -1):
            prod *= current_seq[i]
            new_prods.append(prod)
        
        # Check collision
        valid = True
        for p in new_prods:
            # We need to compare the factorization of p with used_prods?
            # used_prods contains tuples of factorizations.
            # We need to factorize p.
            # p can be very large.
            # But p is a product of numbers <= 128.
            # We can factorize p using the prime factorizations of the components?
            # Yes, we already have factorizations for each number.
            # So we can construct the factorization of the new products directly!
            
            pass # This logic is getting complicated for the greedy part.
            # Let's simplify: Just use the full get_block_products for the greedy step?
            # No, that's O(L^3) total.
            
            # Let's stick to the factorization approach for the SA, 
            # but for the initial greedy, let's just use a simple integer product check?
            # Max product: 128^50 is huge. Python handles big ints.
            # But comparing big ints is slower than comparing tuples of small ints.
            
            # Let's just use the integer product for the initial greedy to keep it simple and fast.
            # Then convert the initial sequence to the factorization-based representation for SA?
            # No, SA needs to evaluate fitness frequently.
            # Let's just use the factorization-based evaluation for everything.
            
            # How to check if new product p (integer) is in used_prods (set of tuples)?
            # We need to factorize p.
            # We can do this by summing the factorizations of the components.
            
            break # Stop this nested loop idea, it's messy.
        
        # Let's restart the greedy logic with a cleaner approach.
        # We will maintain a set of "used factorization tuples".
        # To check if a new integer product P is in the set, we factorize P.
        # Factorizing P (which is a product of known small ints) is easy if we track the components?
        # Or just use the fact that P is built from seq elements.
        
        # Actually, the SA part is the main focus. The initial solution just needs to be "good enough".
        # Let's use a simple greedy with integer products for the initial seed.
        # It will be fast.
        
        pass

    # Revised Plan:
    # 1. Generate initial sequence using simple integer product greedy.
    # 2. Convert this to a format usable by the SA evaluator?
    #    The SA evaluator uses `get_block_products` which returns factorization tuples.
    #    The initial sequence is just a list of ints. `evaluate` will handle it.
    
    # Greedy Initialization with Integer Products (Fast)
    current_seq = []
    used_int_prods = set()
    for x in range(2, 129):
        # Calculate new integer products
        new_int_prods = []
        prod = x
        new_int_prods.append(prod)
        for i in range(len(current_seq) - 1, -1, -1):
            prod *= current_seq[i]
            new_int_prods.append(prod)
            
        if not any(p in used_int_prods for p in new_int_prods):
            current_seq.append(x)
            used_int_prods.update(new_int_prods)
            
    best_seq = current_seq[:]
    best_score = evaluate(best_seq)
    
    # 3. Simulated Annealing
    current_score = best_score
    current_seq = best_seq[:]
    
    # Temperature
    T = 100.0
    T_min = 0.1
    alpha = 0.99
    
    # Number of steps based on time
    steps = 0
    
    while time.time() < deadline:
        steps += 1
        
        # Generate neighbor
        # Randomly modify the sequence
        # Options:
        # 1. Insert a random number from [2,128] not in seq at a random position
        # 2. Remove a random number from seq
        # 3. Swap two numbers? (Violates strictly increasing, so need to sort? No, seq must be strictly increasing. Swapping breaks order unless we sort, but sorting changes the set of consecutive blocks completely. Better to stick to Insert/Delete)
        
        # Let's use Insert and Delete.
        if random.random() < 0.5 and len(current_seq) < 127:
            # Insert
            # Pick a number not in current_seq
            available = [x for x in range(2, 129) if x not in current_seq]
            if not available:
                continue
            x = random.choice(available)
            pos = random.randint(0, len(current_seq))
            
            # Check if inserting x at pos maintains strictly increasing order
            # x must be > current_seq[pos-1] (if pos > 0)
            # x must be < current_seq[pos] (if pos < len)
            if pos > 0 and x <= current_seq[pos-1]:
                continue
            if pos < len(current_seq) and x >= current_seq[pos]:
                continue
                
            neighbor = current_seq[:]
            neighbor.insert(pos, x)
            
        elif random.random() < 0.5 and len(current_seq) > 1:
            # Delete
            pos = random.randint(0, len(current_seq) - 1)
            neighbor = current_seq[:]
            neighbor.pop(pos)
            
        else:
            # Try to replace? Or just skip
            continue
            
        neighbor_score = evaluate(neighbor)
        
        # Acceptance criterion
        delta = neighbor_score - current_score
        if delta > 0:
            # Always accept better
            current_seq = neighbor
            current_score = neighbor_score
            if current_score > best_score:
                best_score = current_score
                best_seq = current_seq[:]
        else:
            # Accept worse with probability exp(delta / T)
            if delta < -1000: # Too bad, skip
                continue
            p = pow(2.71828, delta / T)
            if random.random() < p:
                current_seq = neighbor
                current_score = neighbor_score
                
        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min
            # Optionally restart with higher T if stuck?
            # Let's just keep cooling.

    return best_seq

# Test locally? No, just output the code.
