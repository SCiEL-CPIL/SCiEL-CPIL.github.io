import time
from functools import lru_cache
from collections import defaultdict

def construct():
    start_time = time.time()
    # Use 55s to be safe within the 60s limit
    deadline = start_time + 55.0

    best_seq = []
    best_len = 0

    # Precompute prime factorization for numbers 2-128
    # Represent each number as a tuple of (prime, exponent) pairs
    def prime_factorize(n):
        factors = []
        d = 2
        while d * d <= n:
            if n % d == 0:
                exp = 0
                while n % d == 0:
                    n //= d
                    exp += 1
                factors.append((d, exp))
            d += 1
        if n > 1:
            factors.append((n, 1))
        return tuple(factors)

    # Precompute factorizations for all numbers 2-128
    factorizations = {}
    for i in range(2, 129):
        factorizations[i] = prime_factorize(i)

    # Helper to check if adding x to current_seq creates unique products
    # and returns the list of new products if valid, else None
    def try_add(current_seq, x, current_products):
        # Calculate new products: x, x*last, x*last*second_last, etc.
        # We check for collisions on the fly to avoid list creation if possible
        # But we need the list to add/remove from the set later.
        # So we compute the list first.
        
        new_prods = []
        prod = x
        new_prods.append(prod)
        
        # Calculate products of x * suffix of current_seq
        for a in reversed(current_seq):
            prod *= a
            new_prods.append(prod)
            
        # Check for collisions with existing products
        for p in new_prods:
            if p in current_products:
                return None # Collision
                
        return new_prods

    # Memoization cache: (last_picked, frozenset_of_products) -> max_len_achievable
    # This is expensive to key, so we'll use a simpler heuristic-based memo
    memo = {}

    def dfs(last_picked, current_seq, current_products):
        nonlocal best_seq, best_len

        if time.time() > deadline:
            return

        current_len = len(current_seq)
        
        # Update best
        if current_len > best_len:
            best_len = current_len
            best_seq = current_seq[:]

        # Pruning: Max possible extension
        # Remaining numbers are last_picked+1 to 128.
        # Count is 128 - last_picked.
        if current_len + (128 - last_picked) <= best_len:
            return

        if last_picked >= 128:
            return

        # Memoization key: use last_picked and a hash of current_products
        # We'll use a tuple of sorted products as key (expensive but effective)
        # For efficiency, use last_picked and len(current_products) as partial key
        # Full memoization would be too expensive, so use heuristic pruning

        # Try adding next numbers, with ordering heuristic
        # Prefer numbers that are prime powers or have simple factorizations
        # to reduce product collision likelihood
        candidates = list(range(last_picked + 1, 129))
        
        # Sort candidates by a heuristic: prefer numbers with fewer/smaller prime factors
        # This tends to produce smaller products and fewer collisions
        def complexity_score(n):
            factors = factorizations[n]
            total_exp = sum(e for _, e in factors)
            max_prime = max(p for p, _ in factors) if factors else 1
            return (total_exp, max_prime, n)
        
        candidates.sort(key=complexity_score)
        
        # But we must maintain strictly increasing order, so we can't reorder arbitrarily
        # Instead, we try in natural order but with early termination if time is tight
        
        for x in range(last_picked + 1, 129):
            if time.time() > deadline:
                return

            new_prods = try_add(current_seq, x, current_products)
            
            if new_prods is not None:
                # Add x
                current_seq.append(x)
                for p in new_prods:
                    current_products.add(p)
                
                dfs(x, current_seq, current_products)
                
                # Undo
                current_seq.pop()
                for p in new_prods:
                    current_products.remove(p)
                    
                if time.time() > deadline:
                    return

    # 1. Greedy initialization to get a good lower bound quickly
    greedy_seq = []
    greedy_prods = set()
    for x in range(2, 129):
        if time.time() > deadline:
            break
        new_prods = try_add(greedy_seq, x, greedy_prods)
        if new_prods is not None:
            greedy_seq.append(x)
            for p in new_prods:
                greedy_prods.add(p)
    
    best_seq = greedy_seq[:]
    best_len = len(greedy_seq)
    
    # 2. Local search: try to improve by swapping or inserting
    def local_improve(seq):
        """Try to improve sequence by local swaps and insertions."""
        nonlocal best_seq, best_len
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_prods = set()
            for a in seq:
                current_prods.add(a)
            # Compute all products
            all_prods = set()
            for i in range(len(seq)):
                prod = 1
                for j in range(i, len(seq)):
                    prod *= seq[j]
                    all_prods.add(prod)
            
            # Try removing each element and see if we can add a better one
            for remove_idx in range(len(seq)):
                if time.time() > deadline:
                    return seq
                test_seq = seq[:remove_idx] + seq[remove_idx+1:]
                test_prods = set()
                # Recompute products for test_seq
                for i in range(len(test_seq)):
                    prod = 1
                    for j in range(i, len(test_seq)):
                        prod *= test_seq[j]
                        test_prods.add(prod)
                
                # Try adding each candidate
                for x in range(2, 129):
                    if x in test_seq:
                        continue
                    # Check if x can be inserted while maintaining increasing order
                    # Find insertion point
                    insert_pos = -1
                    for pos in range(len(test_seq) + 1):
                        if pos == 0:
                            if pos == len(test_seq) or test_seq[0] > x:
                                insert_pos = pos
                                break
                        elif pos == len(test_seq):
                            if test_seq[-1] < x:
                                insert_pos = pos
                                break
                        else:
                            if test_seq[pos-1] < x < test_seq[pos]:
                                insert_pos = pos
                                break
                    
                    if insert_pos == -1:
                        continue
                    
                    # Check products
                    new_prods = []
                    prod = x
                    new_prods.append(prod)
                    # Products with suffix before insert_pos
                    for a in reversed(test_seq[:insert_pos]):
                        prod *= a
                        new_prods.append(prod)
                    
                    valid = True
                    for p in new_prods:
                        if p in test_prods:
                            valid = False
                            break
                    
                    if valid:
                        new_seq = test_seq[:insert_pos] + [x] + test_seq[insert_pos:]
                        # Verify full validity
                        all_p = set()
                        for i in range(len(new_seq)):
                            prod = 1
                            for j in range(i, len(new_seq)):
                                prod *= new_seq[j]
                                if prod in all_p:
                                    valid = False
                                    break
                                all_p.add(prod)
                            if not valid:
                                break
                        
                        if valid and len(new_seq) > len(seq):
                            seq = new_seq
                            improved = True
                            if len(seq) > best_len:
                                best_len = len(seq)
                                best_seq = seq[:]
                            break
                if improved:
                    break
        
        return seq

    # Apply local improvement to greedy result
    improved_seq = local_improve(greedy_seq[:])
    if len(improved_seq) > best_len:
        best_seq = improved_seq
        best_len = len(improved_seq)
    
    # 3. DFS to find optimal solution starting from empty
    dfs(1, [], set())
    
    # 4. Final local improvement on DFS result if it's better
    if len(best_seq) > 0 and time.time() < deadline:
        final_seq = local_improve(best_seq[:])
        if len(final_seq) > best_len:
            best_seq = final_seq
            best_len = len(final_seq)
    
    return best_seq
