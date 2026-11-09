import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    # Primes up to 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # We just need the numbers themselves to compute products.
    candidates = list(range(2, 41))
    
    best_set = []
    best_size = 0
    
    # Sort candidates? 
    # Numbers with fewer prime factors might be "safer" to add?
    # Or maybe numbers that are prime powers of large primes?
    # Let's just sort by value or try a few orderings.
    # A common strategy for distinct subset sums/products is to pick numbers that grow fast.
    # But here we are constrained to [2, 40].
    
    # Heuristic: Start with primes, as they are most independent.
    # Then try to add composites.
    
    # Let's use a recursive backtracking search.
    # To avoid deep recursion limits or overhead, we can iterate.
    # But K is small.
    
    def backtrack(current_set, current_products, remaining_indices):
        nonlocal best_set, best_size, deadline
        
        if time.time() > deadline:
            return
            
        current_size = len(current_set)
        
        # Update best
        if current_size > best_size:
            best_size = current_size
            best_set = current_set[:]
            
        # Pruning: if current_size + len(remaining_indices) <= best_size, no improvement possible
        if current_size + len(remaining_indices) <= best_size:
            return
            
        if not remaining_indices:
            return
            
        # Try adding the first candidate in remaining
        # We can choose which candidate to try. 
        # To find MAX size, we should try to add as many as possible.
        # Greedy choice: try to add the candidate that allows the most future additions?
        # Simple approach: try adding each candidate, then recurse.
        # To reduce branching, we can fix an order and decide include/exclude.
        
        # Let's pick the first candidate from remaining.
        idx = remaining_indices[0]
        val = candidates[idx]
        rest = remaining_indices[1:]
        
        # Option 1: Exclude val
        backtrack(current_set, current_products, rest)
        
        # Option 2: Include val
        # Check if adding val creates a collision
        new_products = set()
        collision = False
        for p in current_products:
            new_p = p * val
            if new_p in current_products or new_p in new_products:
                collision = True
                break
            new_products.add(new_p)
        
        if not collision:
            # Also check if new_products collide with each other? 
            # Since current_products are distinct and val != 1, 
            # p1*val == p2*val implies p1==p2, so no internal collision in new_products.
            # We just need to check against current_products.
            # The loop above checks `if new_p in current_products`.
            # We also need to ensure `new_p` are distinct from each other.
            # As argued, if current_products are distinct, p*val are distinct.
            # So we just need to check intersection with current_products.
            
            # Wait, the loop breaks on first collision. 
            # We need to be sure there are no collisions.
            # The logic `if new_p in current_products` is correct for intersection.
            
            next_products = current_products.union(new_products)
            backtrack(current_set + [val], next_products, rest)

    # Initial state: empty set, product {1}
    # Order of candidates: 
    # If we process in increasing order, we might miss better combinations?
    # No, backtracking explores all subsets if we do Include/Exclude for each.
    # But the order affects pruning efficiency.
    # Let's sort candidates by some heuristic.
    # Primes are good.
    # Let's try sorting by number of prime factors (ascending) to get "simple" numbers first.
    
    def prime_factor_count(n):
        count = 0
        d = 2
        nn = n
        while d * d <= nn:
            while nn % d == 0:
                count += 1
                nn //= d
            d += 1
        if nn > 1:
            count += 1
        return count

    # Actually, just using the natural order 2..40 is fine for correctness.
    # To find the MAXIMUM, we must explore.
    # The search space is $2^{39}$, which is too large for naive backtracking.
    # We need strong pruning.
    # The "distinct subset products" constraint is very tight.
    # The number of valid sets is likely small.
    
    # Let's try a different approach: 
    # The set of primes is a valid set of size 12.
    # Can we add any composite?
    # If we add a composite $c$, it must not collide with any product of primes.
    # Products of primes are all square-free numbers (if we only use primes once).
    # Wait, the set of primes $\{2, 3, \dots\}$ generates all square-free numbers as subset products.
    # If we add $4 = 2^2$, does it collide?
    # Subset products of primes: $1, 2, 3, 6, 5, 10, \dots$ (all square-free).
    # $4$ is not square-free. So $4$ is not in the set of products of primes.
    # But we must check if $4 \times (\text{product of primes})$ collides with another product.
    # $4 \times 1 = 4$. Not in primes products.
    # $4 \times 2 = 8$. Not in primes products (8 is not square-free).
    # $4 \times 3 = 12$. Not in primes products.
    # It seems we can add prime powers!
    # Specifically, if we have the set of primes, we can add $p^k$ if $p^k$ is not a product of other primes in the set.
    # Since primes are distinct, $p^k$ for $k \ge 2$ is never a product of distinct primes.
    # So we can add $4, 8, 9, 25, \dots$?
    # Let's check $\{2, 3, 4\}$.
    # Products: $1, 2, 3, 4, 6, 8, 12, 24$. Distinct.
    # $\{2, 3, 4, 5\}$.
    # Products of $\{2,3,4\}$: $1, 2, 3, 4, 6, 8, 12, 24$.
    # Add 5: $5, 10, 15, 20, 30, 40, 60, 120$.
    # Any overlap?
    # $1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60, 120$.
    # All distinct.
    # So we can add multiple prime powers.
    
    # Strategy: 
    # 1. Start with all primes.
    # 2. Try to add prime powers $p^k$ where $p^k \le 40$.
    # 3. Check for collisions.
    # 4. If collision, try different combinations.
    
    # This is a much smaller search space if we assume the base is primes.
    # But maybe we can drop a prime to add two composites?
    # Dropping a prime reduces dimension by 1. Adding 2 composites increases size by 1.
    # Is it possible to have size 13?
    # Primes: 12.
    # Prime powers $\le 40$: 
    # 2: 4, 8, 16, 32
    # 3: 9, 27
    # 5: 25
    # 7: 49 (>40)
    # Total prime powers (excluding primes themselves): 4 + 2 + 1 = 7.
    # Total candidates: 12 primes + 7 prime powers = 19.
    # Plus other composites like 6, 10, 12, 14, 15, 18, 20, 21, 22, 24, 26, 28, 30, 33, 34, 35, 36, 38, 39.
    
    # Let's run the backtracking search on the full set 2..40, but with a good heuristic.
    # Heuristic: Sort by "prime power-ness" or just try to include numbers that are powers of primes first.
    
    # To ensure we finish in time, we'll add a node limit or time check.
    
    # Let's refine the backtracking to be faster.
    # Use a list for current products, but keep a set for O(1) lookup.
    
    def solve():
        # Candidates 2..40
        # Sort by value? 
        # Let's try sorting by number of prime factors (ascending) to get "atomic" numbers first.
        # Or maybe descending?
        # Let's just use the standard backtracking with time limit.
        
        # To improve pruning, we can calculate an upper bound.
        # Max possible size is 12 + 7 = 19? No, constraints apply.
        
        # Let's try a greedy approach first to get a good lower bound.
        # Greedy: Start with primes. Try to add each prime power. If collision, skip.
        
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        current_set = primes[:]
        current_prods = {1}
        for p in primes:
            new_prods = {x * p for x in current_prods}
            current_prods.update(new_prods)
            
        # Now try to add other numbers
        other_numbers = [n for n in range(2, 41) if n not in primes]
        
        # Sort other numbers by "potential" (e.g. prime powers first)
        def is_prime_power(n):
            if n < 2: return False
            for p in primes:
                if p > n: break
                if n % p == 0:
                    nn = n
                    while nn % p == 0:
                        nn //= p
                    if nn == 1:
                        return True
            return False
            
        other_numbers.sort(key=lambda x: (not is_prime_power(x), x))
        
        for n in other_numbers:
            # Check if n can be added
            new_prods = {x * n for x in current_prods}
            if not (new_prods & current_prods):
                current_set.append(n)
                current_prods.update(new_prods)
                
        return current_set

    greedy_sol = solve()
    best_size = len(greedy_sol)
    best_set = greedy_sol
    
    # Now use backtracking to see if we can do better
    # We can use the greedy solution as a lower bound for pruning.
    
    # To make backtracking fast, we can limit the search to permutations of the "promising" numbers.
    # Or just run it on all 39 numbers with the time limit.
    
    # Given the small size, let's try to run a full search but with aggressive pruning.
    # We will sort candidates by the greedy order.
    
    candidates = list(range(2, 41))
    # Re-order candidates to put greedy solution members first?
    # No, that might bias.
    # Let's just run backtracking.
    
    # To avoid recursion depth issues, we can use a stack.
    # But K is small.
    
    # Let's just run the backtracking defined earlier, but with the initial best set from greedy.
    
    # The previous backtracking was O(2^N). 
    # With N=39, it's too slow if not pruned.
    # The "distinct subset products" constraint is very strong.
    # The branching factor will drop quickly.
    
    # Let's try to run it.
    
    # One optimization: 
    # If we have a set S, and we consider adding n.
    # If n is a multiple of some element in S, it's more likely to collide?
    # Not necessarily.
    
    # Let's stick to the time-limited backtracking.
    
    def backtrack_v2(current_set, current_products, remaining_indices):
        nonlocal best_set, best_size, deadline
        
        if time.time() > deadline:
            return
            
        current_size = len(current_set)
        
        if current_size > best_size:
            best_size = current_size
            best_set = current_set[:]
            
        if current_size + len(remaining_indices) <= best_size:
            return
            
        if not remaining_indices:
            return
            
        idx = remaining_indices[0]
        val = candidates[idx]
        rest = remaining_indices[1:]
        
        # Exclude
        backtrack_v2(current_set, current_products, rest)
        
        # Include
        # Check collision
        new_prods = set()
        collision = False
        for p in current_products:
            np = p * val
            if np in current_products or np in new_prods:
                collision = True
                break
            new_prods.add(np)
            
        if not collision:
            next_prods = current_products.union(new_prods)
            backtrack_v2(current_set + [val], next_prods, rest)

    # Initialize
    # Use the greedy set as a starting point for best_size
    # But backtracking starts from empty.
    # To speed up, we can start backtracking from the greedy set?
    # No, we might need to swap elements.
    
    # Let's just run from empty.
    # To help pruning, we sort candidates by "difficulty" or "value".
    # Let's sort by value.
    
    backtrack_v2([], {1}, list(range(len(candidates))))
    
    return [int(x) for x in best_set]

return construct()
