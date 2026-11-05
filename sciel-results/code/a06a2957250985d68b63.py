"""Stronger program: smarter search for longer dense sequences with distinct block products."""
import time
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    def get_all_products(seq):
        """Compute all consecutive block products for a sequence."""
        products = set()
        n = len(seq)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                products.add(prod)
        return products

    def is_valid(seq):
        """Check if sequence is strictly increasing and all block products are distinct."""
        if len(seq) == 0:
            return True
        for i in range(len(seq) - 1):
            if seq[i] >= seq[i + 1]:
                return False
        if any(x < 2 or x > 128 for x in seq):
            return False
        products = get_all_products(seq)
        return len(products) == len(seq) * (len(seq) + 1) // 2

    def greedy_construct():
        """Standard greedy: try numbers 2..128 in order."""
        values, products = [], set()
        for x in range(2, 129):
            tails = []
            prod = 1
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            tails.append(x)
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
        return values

    def greedy_with_order(order):
        """Greedy with a custom ordering."""
        values, products = [], set()
        for x in order:
            if x < 2 or x > 128:
                continue
            tails = []
            prod = 1
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            tails.append(x)
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
        values.sort()
        return values

    # Initial greedy baseline
    best_seq = greedy_construct()
    best_len = len(best_seq)

    # Try different orderings
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

    def is_prime_power(n):
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23]:
            q = n
            while q % p == 0:
                q //= p
            if q == 1:
                return True
        return False

    # Ordering by number of distinct prime factors (ascending - fewer is better)
    order1 = sorted(range(2, 129), key=lambda x: count_distinct_prime_factors(x))
    if time.time() - start_time < DEADLINE:
        seq1 = greedy_with_order(order1)
        if is_valid(seq1) and len(seq1) > best_len:
            best_seq = seq1
            best_len = len(seq1)

    # Ordering prioritizing prime powers
    order2 = sorted(range(2, 129), key=lambda x: (not is_prime_power(x), x))
    if time.time() - start_time < DEADLINE:
        seq2 = greedy_with_order(order2)
        if is_valid(seq2) and len(seq2) > best_len:
            best_seq = seq2
            best_len = len(seq2)

    # Ordering by total number of prime factors with multiplicity (ascending)
    def total_prime_factors(n):
        count = 0
        d = 2
        temp = n
        while d * d <= temp:
            while temp % d == 0:
                count += 1
                temp //= d
            d += 1
        if temp > 1:
            count += 1
        return count

    order3 = sorted(range(2, 129), key=lambda x: total_prime_factors(x))
    if time.time() - start_time < DEADLINE:
        seq3 = greedy_with_order(order3)
        if is_valid(seq3) and len(seq3) > best_len:
            best_seq = seq3
            best_len = len(seq3)

    # Local search: try to insert elements one at a time
    used = set(best_seq)
    remaining = [x for x in range(2, 129) if x not in used]

    def try_insert_at(seq, new_elem, pos):
        """Try inserting new_elem at position pos in seq."""
        if new_elem in seq:
            return None
        if pos == len(seq):
            test_seq = seq + [new_elem]
        elif pos == 0:
            test_seq = [new_elem] + seq
        else:
            test_seq = seq[:pos] + [new_elem] + seq[pos:]
        if not is_valid(test_seq):
            return None
        return test_seq

    # Iterative insertion
    max_iterations = 30
    for iteration in range(max_iterations):
        if time.time() - start_time > DEADLINE:
            break

        improved = False
        for new_elem in remaining[:]:
            if time.time() - start_time > DEADLINE:
                break
            for pos in range(len(best_seq) + 1):
                result = try_insert_at(best_seq, new_elem, pos)
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
            # Try swapping: replace one element with another, then try to insert the old one
            swapped = False
            for j in range(len(best_seq)):
                if time.time() - start_time > DEADLINE:
                    break
                orig_val = best_seq[j]
                for new_elem in remaining[:]:
                    if time.time() - start_time > DEADLINE:
                        break
                    test_seq = best_seq[:]
                    test_seq[j] = new_elem
                    test_seq.sort()
                    if not is_valid(test_seq):
                        continue
                    # Try to insert orig_val
                    for pos in range(len(test_seq) + 1):
                        result = try_insert_at(test_seq, orig_val, pos)
                        if result is not None:
                            best_seq = result
                            best_len = len(result)
                            used.discard(new_elem)
                            used.add(orig_val)
                            if new_elem not in remaining:
                                remaining.append(new_elem)
                            if orig_val in remaining:
                                remaining.remove(orig_val)
                            swapped = True
                            break
                    if swapped:
                        break
                if swapped:
                    break
            if not swapped:
                break

    # Final validation
    if not is_valid(best_seq):
        best_seq = greedy_construct()

    return best_seq
