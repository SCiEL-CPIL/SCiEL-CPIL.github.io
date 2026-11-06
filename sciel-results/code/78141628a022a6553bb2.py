import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    def get_primes(n):
        sieve = [True] * (n+1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5)+1):
            if sieve[i]:
                for j in range(i*i, n+1, i):
                    sieve[j] = False
        return [i for i in range(2, n+1) if sieve[i]]
    
    def compute_all_products(seq):
        """Compute set of all nonempty consecutive block products."""
        products = set()
        n = len(seq)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                products.add(prod)
        return products
    
    def is_valid(seq):
        """Check if all consecutive block products are distinct."""
        n = len(seq)
        products = set()
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                if prod in products:
                    return False
                products.add(prod)
        return True
    
    def try_insert_fast(seq, x, products_set):
        """Try to insert x into seq, checking only new products involving x.
        Returns (new_seq, new_products_set) or (None, None)."""
        n = len(seq)
        # Find valid insertion positions
        valid_positions = []
        if n == 0:
            valid_positions = [0]
        else:
            if x < seq[0]:
                valid_positions.append(0)
            for i in range(n-1):
                if seq[i] < x < seq[i+1]:
                    valid_positions.append(i+1)
            if x > seq[-1]:
                valid_positions.append(n)
        
        if not valid_positions:
            return None, None
        
        # Try each valid position
        for pos in valid_positions:
            if time.time() > deadline:
                return None, None
            
            # Compute new products that involve x at position pos
            # Products starting at pos: x * seq[pos], x * seq[pos] * seq[pos+1], ...
            # Products ending at pos: seq[pos-1] * x, seq[pos-2] * seq[pos-1] * x, ...
            # Products passing through pos: seq[i] * ... * x * ... * seq[j]
            
            new_products = set()
            
            # Products where x is the start
            prod = x
            for j in range(pos, n):
                new_products.add(prod)
                if j + 1 < n:
                    prod *= seq[j + 1]
            
            # Products where x is the end (but not already counted if x is start)
            prod = x
            for i in range(pos - 1, -1, -1):
                prod *= seq[i]
                new_products.add(prod)
            
            # Check if any new product is already in the set
            conflict = False
            for p in new_products:
                if p in products_set:
                    conflict = True
                    break
            
            if not conflict:
                # Check that new products are distinct from each other
                # (they should be since they all involve x and have different ranges)
                # But let's verify
                if len(new_products) != len(new_products):
                    continue
                
                # Build new sequence and update products set
                new_seq = seq[:pos] + [x] + seq[pos:]
                new_products_set = products_set | new_products
                return new_seq, new_products_set
        
        return None, None
    
    def greedy_build(candidates_order, primes):
        """Build sequence by greedily inserting candidates in given order."""
        seq = primes[:]
        # Compute initial products set
        products_set = compute_all_products(seq)
        
        for comp in candidates_order:
            if time.time() > deadline:
                break
            new_seq, new_products = try_insert_fast(seq, comp, products_set)
            if new_seq is not None:
                seq = new_seq
                products_set = new_products
        
        return seq
    
    primes = get_primes(128)
    best_seq = primes[:]
    best_len = len(primes)
    
    # Get all composites in [2, 128]
    prime_set = set(primes)
    composites = [x for x in range(2, 129) if x not in prime_set]
    
    # Strategy 1: Multiple greedy passes with different orders
    orders = []
    orders.append(composites[:])  # ascending
    orders.append(composites[::-1])  # descending
    orders.append(sorted(composites, key=lambda x: x % 7))  # by remainder mod 7
    orders.append(sorted(composites, key=lambda x: x % 5))  # by remainder mod 5
    orders.append(sorted(composites, key=lambda x: x % 3))  # by remainder mod 3
    
    # Add some random orders
    for _ in range(5):
        random_order = composites[:]
        random.shuffle(random_order)
        orders.append(random_order)
    
    for order in orders:
        if time.time() > deadline:
            break
        seq = greedy_build(order, primes)
        if len(seq) > best_len:
            best_seq = seq
            best_len = len(seq)
    
    # Strategy 2: Iterative refinement on best_seq
    if time.time() < deadline:
        remaining = [x for x in range(2, 129) if x not in set(best_seq)]
        # Try multiple times to add remaining elements
        for _ in range(10):
            if time.time() > deadline:
                break
            progress = False
            for comp in remaining:
                if time.time() > deadline:
                    break
                products_set = compute_all_products(best_seq)
                new_seq, new_products = try_insert_fast(best_seq, comp, products_set)
                if new_seq is not None:
                    best_seq = new_seq
                    best_len = len(best_seq)
                    remaining.remove(comp)
                    progress = True
                    break
            if not progress:
                break
    
    # Strategy 3: Local search - swap or remove+add
    if time.time() < deadline:
        for iteration in range(20):
            if time.time() > deadline:
                break
            improved = False
            # Try removing one element and re-adding others
            for i in range(len(best_seq)):
                if time.time() > deadline:
                    break
                reduced = best_seq[:i] + best_seq[i+1:]
                reduced_set = set(reduced)
                remaining_after = [x for x in range(2, 129) if x not in reduced_set]
                
                # Try to build from reduced
                seq2 = reduced[:]
                products_set = compute_all_products(seq2)
                for comp in remaining_after:
                    if time.time() > deadline:
                        break
                    new_seq2, new_products = try_insert_fast(seq2, comp, products_set)
                    if new_seq2 is not None:
                        seq2 = new_seq2
                        products_set = new_products
                
                if len(seq2) > len(best_seq):
                    best_seq = seq2
                    best_len = len(best_seq)
                    improved = True
                    break
            
            if not improved:
                break
    
    # Strategy 4: Random restarts with local optimization
    if time.time() < deadline:
        for restart in range(10):
            if time.time() > deadline:
                break
            
            # Start from a random subset of primes + some composites
            random.seed(restart * 12345 + 67890)
            num_primes = random.randint(len(primes) // 2, len(primes))
            selected_primes = random.sample(primes, num_primes)
            selected_primes.sort()
            
            # Add some random composites
            num_comps = random.randint(0, min(5, len(composites)))
            selected_comps = random.sample(composites, num_comps)
            
            initial_seq = sorted(selected_primes + selected_comps)
            
            # Now greedily add remaining
            seq = initial_seq[:]
            products_set = compute_all_products(seq)
            remaining = [x for x in range(2, 129) if x not in set(seq)]
            
            for comp in remaining:
                if time.time() > deadline:
                    break
                new_seq, new_products = try_insert_fast(seq, comp, products_set)
                if new_seq is not None:
                    seq = new_seq
                    products_set = new_products
            
            if len(seq) > best_len:
                best_seq = seq
                best_len = len(seq)
    
    # Final validation
    if not is_valid(best_seq):
        # Fallback to primes
        best_seq = primes[:]
    
    return best_seq
