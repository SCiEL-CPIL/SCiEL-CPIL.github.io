"""Improved program: search for longer sequences via greedy + local optimization."""
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    # First, get the greedy baseline
    def greedy_construct():
        values, products = [], set()
        for x in range(2, 129):
            tails, prod = [x], x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
        return values
    
    best_seq = greedy_construct()
    best_len = len(best_seq)
    
    # Helper to compute all consecutive block products for a sequence
    def get_all_products(seq):
        products = set()
        n = len(seq)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                products.add(prod)
        return products
    
    # Verify the greedy solution is valid
    def is_valid(seq):
        products = get_all_products(seq)
        return len(products) == len(seq) * (len(seq) + 1) // 2
    
    if not is_valid(best_seq):
        # Should not happen, but fallback
        best_seq = list(range(2, 129))
        best_len = len(best_seq)
    
    # Try local search: swap elements to see if we can do better
    # The key insight: the greedy approach takes numbers in order.
    # Maybe we can skip some numbers and include others to get a longer sequence.
    
    # Try different strategies:
    # 1. Greedy with different starting points or orderings
    # 2. Try to insert numbers that were skipped
    
    # Let's try a more sophisticated approach:
    # Build the sequence greedily, but at each step, try all candidates
    # and pick the one that allows the longest future extension.
    
    # Actually, let's try a different greedy: instead of going 2..128 in order,
    # try to build the longest possible sequence using a backtracking approach
    # with pruning, but with a time limit.
    
    # Strategy: Use the greedy result as a starting point, then try to
    # improve by replacing elements or finding better combinations.
    
    # Let's try: for each number in the greedy solution, see if we can
    # replace it with a different number to allow more numbers overall.
    
    # Actually, a simpler approach: try greedy with different orderings.
    # The constraint is that consecutive block products must be distinct.
    # Numbers with many small prime factors create more "conflicting" products.
    
    # Let's try: sort by number of distinct prime factors (ascending) to
    # minimize conflicts early on.
    
    def count_distinct_prime_factors(n):
        count = 0
        d = 2
        temp = n
        while d * d <= temp:
            if temp % d == 0:
                count += 1
                while temp % d == 0:
                    temp //= d
            d += 1
        if temp > 1:
            count += 1
        return count
    
    # Try greedy with different orderings
    candidates = list(range(2, 129))
    
    # Ordering 1: by number of prime factors (ascending)
    order1 = sorted(candidates, key=lambda x: count_distinct_prime_factors(x))
    
    def greedy_with_order(order):
        values, products = [], set()
        for x in order:
            if x < 2 or x > 128:
                continue
            tails, prod = [x], x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
        # Sort the result to make it strictly increasing
        values.sort()
        return values
    
    # Try ordering 1
    if time.time() - start_time < DEADLINE:
        seq1 = greedy_with_order(order1)
        if len(seq1) > best_len and is_valid(seq1):
            best_seq = seq1
            best_len = len(seq1)
    
    # Ordering 2: by value (original greedy) - already done
    
    # Ordering 3: try to prioritize numbers that are "independent" (prime powers, etc.)
    def is_prime_power(n):
        for p in [2, 3, 5, 7, 11, 13]:
            q = n
            while q % p == 0:
                q //= p
            if q == 1:
                return True
        return False
    
    order3 = sorted(candidates, key=lambda x: (not is_prime_power(x), x))
    
    if time.time() - start_time < DEADLINE:
        seq3 = greedy_with_order(order3)
        if len(seq3) > best_len and is_valid(seq3):
            best_seq = seq3
            best_len = len(seq3)
    
    # Try a backtracking approach with limited depth
    # Start with the greedy solution and try to add more elements
    
    # Try: take the greedy solution, and for each position, try inserting
    # a new element from the remaining numbers
    
    used = set(best_seq)
    remaining = [x for x in range(2, 129) if x not in used]
    
    # Try inserting each remaining number at each position
    def try_insert(seq, new_elem, pos):
        """Try inserting new_elem at position pos in seq, check if valid."""
        new_seq = seq[:pos] + [new_elem] + seq[pos:]
        # Check strictly increasing
        for i in range(len(new_seq) - 1):
            if new_seq[i] >= new_seq[i+1]:
                return None
        # Check all products distinct
        products = get_all_products(new_seq)
        if len(products) == len(new_seq) * (len(new_seq) + 1) // 2:
            return new_seq
        return None
    
    # Try to insert one more element
    if time.time() - start_time < DEADLINE:
        improved = False
        for new_elem in remaining:
            if time.time() - start_time > DEADLINE:
                break
            for pos in range(len(best_seq) + 1):
                result = try_insert(best_seq, new_elem, pos)
                if result is not None:
                    best_seq = result
                    best_len = len(result)
                    used.add(new_elem)
                    remaining.remove(new_elem)
                    improved = True
                    break
            if improved:
                break
    
    # Repeat the insertion process
    if time.time() - start_time < DEADLINE:
        for _ in range(10):  # Try up to 10 more insertions
            if time.time() - start_time > DEADLINE:
                break
            improved = False
            for new_elem in remaining[:]:
                if time.time() - start_time > DEADLINE:
                    break
                for pos in range(len(best_seq) + 1):
                    result = try_insert(best_seq, new_elem, pos)
                    if result is not None:
                        best_seq = result
                        best_len = len(result)
                        used.add(new_elem)
                        remaining.remove(new_elem)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
    
    # Final validation
    if not is_valid(best_seq):
        # Fallback to greedy
        best_seq = greedy_construct()
    
    return best_seq
