"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import combinations

_N = 256

def _sieve_primes(limit):
    """Return list of primes up to limit."""
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    return [i for i, v in enumerate(is_prime) if v]

def _factorize(n, primes):
    """Return prime factorization of n as dict {prime: exponent}."""
    factors = {}
    for p in primes:
        if p * p > n:
            break
        while n % p == 0:
            factors[p] = factors.get(p, 0) + 1
            n //= p
    if n > 1:
        factors[n] = factors.get(n, 1) + 1
    return factors

def _largest_prime_power(p, limit):
    """Return the largest power of p that is <= limit."""
    pp = p
    while pp * p <= limit:
        pp *= p
    return pp

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s limit

    # Get all primes up to 256
    primes = _sieve_primes(_N)

    # Precompute prime factor sets for each number 2..256
    prime_sets = {}
    for i in range(2, _N + 1):
        n = i
        ps = set()
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                ps.add(p)
                while n % p == 0:
                    n //= p
        if n > 1:
            ps.add(n)
        prime_sets[i] = ps

    def _shares_prime(a, b):
        """Check if a and b share any prime factor."""
        if a == 1 or b == 1:
            return False
        return bool(prime_sets[a] & prime_sets[b])

    def _is_coprime_set(num, sel_set):
        """Check if num is coprime to all elements in sel_set."""
        if num == 1:
            return True
        ps = prime_sets[num]
        for s in sel_set:
            if s == 1:
                continue
            if ps & prime_sets[s]:
                return False
        return True

    # Build greedy solution: prefer largest prime powers first
    # For each prime, the best candidate is its largest power <= 256
    # Start with 1, then add numbers in descending order of value
    # But prioritize prime powers over composites with multiple factors
    selected = [1]
    selected_set = {1}

    # Sort candidates: prefer numbers that are prime powers (fewer distinct prime factors)
    # and higher value
    candidates = list(range(_N, 1, -1))
    candidates.sort(key=lambda x: (-x, len(prime_sets[x])))

    for num in candidates:
        if _is_coprime_set(num, selected_set):
            selected.append(num)
            selected_set.add(num)

    current_sum = sum(selected)

    # Local search with multiple move types
    # Move types: add, 1-for-1 swap, 2-for-1 swap, 1-for-2 swap, 2-for-2 swap
    no_improve_count = 0

    while time.time() < deadline:
        improved = False

        # Move 1: Try single additions (in descending order of value)
        for num in range(_N, 1, -1):
            if num not in selected_set and _is_coprime_set(num, selected_set):
                selected.append(num)
                selected_set.add(num)
                current_sum += num
                improved = True
                break

        if improved:
            no_improve_count = 0
            continue

        # Move 2: Try 1-for-1 swaps
        best_swap_11 = None
        best_swap_11_gain = 0
        for rem in selected:
            if rem == 1:
                continue
            sel_without_rem = selected_set - {rem}
            for add in range(_N, 1, -1):
                if add in selected_set or add == rem:
                    continue
                if _is_coprime_set(add, sel_without_rem):
                    gain = add - rem
                    if gain > best_swap_11_gain:
                        best_swap_11_gain = gain
                        best_swap_11 = (rem, add)

        if best_swap_11 is not None:
            rem, add = best_swap_11
            selected.remove(rem)
            selected_set.remove(rem)
            selected.append(add)
            selected_set.add(add)
            current_sum += best_swap_11_gain
            improved = True
            no_improve_count = 0
            continue

        # Move 3: Try 2-for-1 swaps (remove one, add two)
        best_swap_21 = None
        best_swap_21_gain = 0
        for rem in selected:
            if rem == 1:
                continue
            sel_without_rem = selected_set - {rem}
            # Try pairs of additions
            for add1 in range(_N, 1, -1):
                if add1 in selected_set or add1 == rem:
                    continue
                if not _is_coprime_set(add1, sel_without_rem):
                    continue
                temp_set = sel_without_rem | {add1}
                for add2 in range(add1 - 1, 1, -1):
                    if add2 in temp_set or add2 == rem:
                        continue
                    if not _is_coprime_set(add2, temp_set):
                        continue
                    gain = add1 + add2 - rem
                    if gain > best_swap_21_gain:
                        best_swap_21_gain = gain
                        best_swap_21 = (rem, add1, add2)

        if best_swap_21 is not None:
            rem, add1, add2 = best_swap_21
            selected.remove(rem)
            selected_set.remove(rem)
            selected.extend([add1, add2])
            selected_set.update([add1, add2])
            current_sum += best_swap_21_gain
            improved = True
            no_improve_count = 0
            continue

        # Move 4: Try 1-for-2 swaps (remove two, add one)
        best_swap_12 = None
        best_swap_12_gain = 0
        sel_list = [x for x in selected if x != 1]
        for i in range(len(sel_list)):
            for j in range(i + 1, len(sel_list)):
                rem1, rem2 = sel_list[i], sel_list[j]
                sel_without = selected_set - {rem1, rem2}
                for add in range(_N, 1, -1):
                    if add in selected_set or add == rem1 or add == rem2:
                        continue
                    if _is_coprime_set(add, sel_without):
                        gain = add - rem1 - rem2
                        if gain > best_swap_12_gain:
                            best_swap_12_gain = gain
                            best_swap_12 = (rem1, rem2, add)

        if best_swap_12 is not None:
            rem1, rem2, add = best_swap_12
            selected.remove(rem1)
            selected.remove(rem2)
            selected_set.discard(rem1)
            selected_set.discard(rem2)
            selected.append(add)
            selected_set.add(add)
            current_sum += best_swap_12_gain
            improved = True
            no_improve_count = 0
            continue

        # Move 5: Try 2-for-2 swaps (remove two, add two)
        best_swap_22 = None
        best_swap_22_gain = 0
        for i in range(len(sel_list)):
            for j in range(i + 1, len(sel_list)):
                rem1, rem2 = sel_list[i], sel_list[j]
                sel_without = selected_set - {rem1, rem2}
                for add1 in range(_N, 1, -1):
                    if add1 in selected_set or add1 == rem1 or add1 == rem2:
                        continue
                    if not _is_coprime_set(add1, sel_without):
                        continue
                    temp_set = sel_without | {add1}
                    for add2 in range(add1 - 1, 1, -1):
                        if add2 in temp_set or add2 == rem1 or add2 == rem2:
                            continue
                        if not _is_coprime_set(add2, temp_set):
                            continue
                        gain = add1 + add2 - rem1 - rem2
                        if gain > best_swap_22_gain:
                            best_swap_22_gain = gain
                            best_swap_22 = (rem1, rem2, add1, add2)

        if best_swap_22 is not None:
            rem1, rem2, add1, add2 = best_swap_22
            selected.remove(rem1)
            selected.remove(rem2)
            selected_set.discard(rem1)
            selected_set.discard(rem2)
            selected.extend([add1, add2])
            selected_set.update([add1, add2])
            current_sum += best_swap_22_gain
            improved = True
            no_improve_count = 0
            continue

        # No improvement found
        no_improve_count += 1
        if no_improve_count >= 3:
            break

    # Ensure result is a list of Python ints
    result = [int(x) for x in selected]
    return result
# EVOLVE-BLOCK-END
