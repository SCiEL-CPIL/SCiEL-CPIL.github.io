import time
import sys
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    def _all_products(seq):
        """Compute all nonempty consecutive block products."""
        prods = set()
        n = len(seq)
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                prods.add(p)
        return prods

    def _can_append(seq, x, prods):
        """Check if x can be appended to seq without product collision."""
        prod = x
        if prod in prods:
            return False
        for a in reversed(seq):
            prod *= a
            if prod in prods:
                return False
        return True

    def _append_products(seq, x):
        """Return the new products added when appending x."""
        new_prods = []
        prod = x
        new_prods.append(prod)
        for a in reversed(seq):
            prod *= a
            new_prods.append(prod)
        return new_prods

    def _can_insert(seq, idx, x, prods):
        """Check if x can be inserted at position idx in seq without collision."""
        # Products involving x:
        # 1. x itself
        # 2. x * seq[idx], x * seq[idx] * seq[idx+1], ... (suffix starting at idx)
        # 3. ... * x, ... * x * seq[idx], ... (prefix ending at idx-1, then x)
        # Actually, we need to check all consecutive blocks that include x.
        
        # Blocks that include x:
        # - [x]
        # - [x, seq[idx], ..., seq[j]] for j >= idx
        # - [seq[i], ..., seq[idx-1], x] for i <= idx-1
        # - [seq[i], ..., seq[idx-1], x, seq[idx], ..., seq[j]] for i <= idx-1, j >= idx
        
        # Simpler: compute all products that involve x in the new sequence
        # and check they don't collide with existing products (excluding those that
        # would be "split" by the insertion).
        
        # Actually, the cleanest way: build the new sequence and check all products.
        # But that's O(n^2). Let's be smarter.
        
        # Products that change when inserting x at idx:
        # - Any product that spans across the insertion point
        # - Products entirely before idx: unchanged
        # - Products entirely after idx: unchanged
        # - Products that include x: new
        
        # New products to check:
        # 1. x alone
        # 2. x * seq[idx] * ... * seq[j] for j = idx..n-1
        # 3. seq[i] * ... * seq[idx-1] * x for i = 0..idx-1
        # 4. seq[i] * ... * seq[idx-1] * x * seq[idx] * ... * seq[j] for i<=idx-1, j>=idx
        
        # But we also need to ensure these new products don't collide with each other
        # or with existing products that are NOT split by the insertion.
        
        # The existing products that remain valid are those entirely before idx or
        # entirely after idx. Products that span the insertion point are "broken"
        # and replaced by new ones.
        
        # This is complex. Let's just rebuild and check for small sequences.
        # For efficiency, we'll compute the new products incrementally.
        
        n = len(seq)
        new_prods = set()
        
        # x alone
        new_prods.add(x)
        
        # x * suffix starting at idx
        prod = x
        for j in range(idx, n):
            prod *= seq[j]
            new_prods.add(prod)
        
        # prefix ending at idx-1 * x
        prod = x
        for i in range(idx - 1, -1, -1):
            prod *= seq[i]
            new_prods.add(prod)
        
        # prefix ending at idx-1 * x * suffix starting at idx
        for i in range(idx - 1, -1, -1):
            left_prod = 1
            for k in range(i, idx):
                left_prod *= seq[k]
            right_prod = x
            for j in range(idx, n):
                right_prod *= seq[j]
                new_prods.add(left_prod * right_prod)
        
        # Now, existing products that are "broken" (span the insertion point)
        # should be removed from prods. But we're checking against prods, which
        # includes them. So we need to check new_prods against prods minus the
        # broken ones.
        
        # Simpler approach: just check that new_prods don't intersect with prods,
        # but this is conservative (may reject valid insertions).
        # For now, let's use the conservative check.
        
        if new_prods & prods:
            return False
        return True

    def _try_insert(seq, prods, x, best_len_target):
        """Try to insert x into seq at any valid position."""
        n = len(seq)
        # Try positions from 0 to n
        for idx in range(n + 1):
            if idx > 0 and seq[idx - 1] >= x:
                continue
            if idx < n and seq[idx] <= x:
                continue
            if _can_insert(seq, idx, x, prods):
                new_seq = seq[:idx] + [x] + seq[idx:]
                new_prods = _all_products(new_seq)
                return new_seq, new_prods
        return None, None

    def _try_swap(seq, prods, i, j):
        """Try swapping elements at positions i and j."""
        if seq[i] >= seq[j]:
            return None, None
        new_seq = seq[:]
        new_seq[i], new_seq[j] = seq[j], seq[i]
        # Check strictly increasing
        for k in range(len(new_seq) - 1):
            if new_seq[k] >= new_seq[k + 1]:
                return None, None
        new_prods = _all_products(new_seq)
        return new_seq, new_prods

    # Get all primes in [2, 128]
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True

    primes = [p for p in range(2, 129) if is_prime(p)]
    composites = [c for c in range(2, 129) if not is_prime(c)]

    best_seq = primes[:]
    best_prods = _all_products(primes)
    best_len = len(best_seq)

    # Try to insert composites
    def _insertion_search(seq, prods, composites_to_try, time_limit):
        """Greedy/randomized insertion of composites."""
        seq = seq[:]
        prods = prods.copy()
        remaining = list(composites_to_try)
        random.shuffle(remaining)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for x in remaining[:]:
                if time.time() > time_limit:
                    break
                new_seq, new_prods = _try_insert(seq, prods, x, len(seq) + 1)
                if new_seq is not None:
                    seq = new_seq
                    prods = new_prods
                    remaining.remove(x)
                    improved = True
        return seq, prods

    # Multiple random restarts
    num_restarts = 50
    restarts_done = 0
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
        restarts_done += 1
        
        # Start with primes, try to insert composites with random order
        seq = primes[:]
        prods = _all_products(primes)
        
        # Randomized insertion
        comp_list = composites[:]
        random.shuffle(comp_list)
        
        for x in comp_list:
            if time.time() > deadline - 5:
                break
            new_seq, new_prods = _try_insert(seq, prods, x, len(seq) + 1)
            if new_seq is not None:
                seq = new_seq
                prods = new_prods
        
        if len(seq) > best_len:
            best_seq = seq[:]
            best_prods = prods
            best_len = len(seq)
        
        # Local search: try swaps
        if time.time() > deadline - 5:
            break
        swap_improved = True
        while swap_improved and time.time() < deadline - 5:
            swap_improved = False
            n = len(seq)
            for i in range(n - 1):
                if time.time() > deadline - 5:
                    break
                for j in range(i + 1, n):
                    if time.time() > deadline - 5:
                        break
                    new_seq, new_prods = _try_swap(seq, prods, i, j)
                    if new_seq is not None and len(new_seq) >= len(seq):
                        # Check if it allows more insertions later?
                        # For now, accept if valid
                        # Try to insert more composites
                        seq2 = new_seq[:]
                        prods2 = new_prods.copy()
                        comp_list2 = [c for c in composites if c not in seq2]
                        random.shuffle(comp_list2)
                        for x in comp_list2:
                            if time.time() > deadline - 5:
                                break
                            ns, np_ = _try_insert(seq2, prods2, x, len(seq2) + 1)
                            if ns is not None:
                                seq2 = ns
                                prods2 = np_
                        if len(seq2) > len(seq):
                            seq = seq2
                            prods = prods2
                            swap_improved = True
                            break
                if swap_improved:
                    break

        if len(seq) > best_len:
            best_seq = seq[:]
            best_prods = prods
            best_len = len(seq)

    return best_seq
