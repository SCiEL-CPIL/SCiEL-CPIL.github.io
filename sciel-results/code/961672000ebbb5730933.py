"""Optimized: incremental tail-product checking + simulated annealing.
The core speedup: appending x to a valid sequence only requires checking
O(n) tail products (x, x*a[-1], x*a[-1]*a[-2], ...) against the existing
product set, instead of recomputing all O(n^2) block products.
Simulated annealing allows temporary product collisions to escape local
optima, with restarts from diverse bases (primes, odds, composites)."""
import time
import random
import math

def construct():
    deadline = time.time() + 80.0  # safety margin under 90s kill
    random.seed(12345)

    candidates = list(range(2, 129))

    def _is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True

    def _compute_products(seq):
        """Compute all nonempty consecutive block products for a valid sequence."""
        prods = set()
        n = len(seq)
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                prods.add(p)
        return prods

    def _is_valid(seq):
        """Check strict increase and distinct block products."""
        n = len(seq)
        for i in range(1, n):
            if seq[i] <= seq[i - 1]:
                return False
        expected = n * (n + 1) // 2
        return len(_compute_products(seq)) == expected

    def _append_check(seq, prods, x):
        """Check if x can be appended to seq (which is valid).
        Returns (True, updated_prods) or (False, None).
        Only O(n) new products need checking (tails ending at x)."""
        if seq and x <= seq[-1]:
            return False, None
        tails = [x]
        p = x
        for a in reversed(seq):
            p *= a
            tails.append(p)
        # Check for collisions
        for t in tails:
            if t in prods:
                return False, None
        new_prods = prods | set(tails)
        return True, new_prods

    def _greedy_append(cands, order):
        """Greedy: append candidates in given order if valid."""
        values = []
        prods = set()
        for x in cands:
            if order and x not in order:
                continue
            ok, new_prods = _append_check(values, prods, x)
            if ok:
                values.append(x)
                prods = new_prods
        return values

    def _greedy_randomized(cands, n_trials=200):
        """Multiple randomized greedy passes with shuffled order."""
        best = []
        best_len = 0
        for t in range(n_trials):
            shuffled = list(cands)
            random.shuffle(shuffled)
            # Bias toward smaller numbers (they're more flexible)
            shuffled.sort(key=lambda x: (random.random(), x))
            values = []
            prods = set()
            for x in shuffled:
                ok, new_prods = _append_check(values, prods, x)
                if ok:
                    values.append(x)
                    prods = new_prods
            if len(values) > best_len:
                best = values
                best_len = len(values)
        return best

    def _simulated_annealing(base_seq, remaining, time_limit):
        """SA: start from base, try insertions/removals/swaps with cooling."""
        current = list(base_seq)
        current_prods = _compute_products(current)
        current_len = len(current)
        best = list(current)
        best_prods = set(current_prods)
        best_len = current_len

        used = set(current)
        rem = [x for x in remaining if x not in used]

        T = 50.0
        T_min = 0.1
        cooling = 0.995

        while time.time() < time_limit and T > T_min:
            # Pick a random move
            move_type = random.random()
            improved = False

            if move_type < 0.6 and rem:
                # Insert a random remaining element at a random position
                c = random.choice(rem)
                pos = random.randint(0, len(current))
                # Check position validity
                if pos > 0 and current[pos - 1] >= c:
                    pos = -1
                elif pos < len(current) and c >= current[pos]:
                    pos = -1
                if pos >= 0:
                    new_seq = current[:pos] + [c] + current[pos:]
                    new_prods = _compute_products(new_seq)
                    # Count collisions
                    n_blocks = len(new_seq) * (len(new_seq) + 1) // 2
                    collisions = n_blocks - len(new_prods)
                    if collisions == 0:
                        # Valid insertion
                        current = new_seq
                        current_prods = new_prods
                        used.add(c)
                        rem.remove(c)
                        if len(current) > best_len:
                            best = list(current)
                            best_prods = set(current_prods)
                            best_len = len(current)
                        improved = True
                    elif collisions <= 2 and random.random() < math.exp(-collisions / T):
                        # Accept with probability (allowing temporary invalidity)
                        current = new_seq
                        current_prods = new_prods
                        # We'll fix later by removing problematic elements
                        # For now, just track
            elif move_type < 0.8 and len(current) > 3:
                # Remove a random element to make room
                pos = random.randint(0, len(current) - 1)
                c = current[pos]
                new_seq = current[:pos] + current[pos + 1:]
                if new_seq and new_seq[pos - 1] >= (new_seq[pos] if pos < len(new_seq) else 0):
                    pass  # still increasing
                new_prods = _compute_products(new_seq)
                n_blocks = len(new_seq) * (len(new_seq) + 1) // 2
                collisions = n_blocks - len(new_prods)
                if collisions == 0:
                    current = new_seq
                    current_prods = new_prods
                    used.discard(c)
                    rem.append(c)
                    improved = True
            elif move_type < 0.95 and len(current) >= 2:
                # Swap two adjacent elements (if still increasing)
                pos = random.randint(0, len(current) - 2)
                if current[pos] < current[pos + 1]:
                    new_seq = list(current)
                    new_seq[pos], new_seq[pos + 1] = new_seq[pos + 1], new_seq[pos]
                    # Check if still increasing
                    if all(new_seq[i] < new_seq[i + 1] for i in range(len(new_seq) - 1)):
                        new_prods = _compute_products(new_seq)
                        n_blocks = len(new_seq) * (len(new_seq) + 1) // 2
                        collisions = n_blocks - len(new_prods)
                        if collisions == 0:
                            current = new_seq
                            current_prods = new_prods
                            improved = True

            # If no improvement, try to repair: remove elements causing collisions
            if not improved and len(current) > 2:
                n_blocks = len(current) * (len(current) + 1) // 2
                if len(current_prods) < n_blocks:
                    # Find and remove a problematic element
                    for pos in range(len(current)):
                        test_seq = current[:pos] + current[pos + 1:]
                        test_prods = _compute_products(test_seq)
                        test_blocks = len(test_seq) * (len(test_seq) + 1) // 2
                        if len(test_prods) == test_blocks:
                            c = current[pos]
                            current = test_seq
                            current_prods = test_prods
                            used.discard(c)
                            rem.append(c)
                            break

            T *= cooling

        # Ensure best is valid
        if not _is_valid(best):
            # Rebuild from scratch with greedy
            best = _greedy_randomized(candidates, 50)

        return best

    def _insertion_search(base_seq, remaining, time_limit):
        """Greedy insertion: repeatedly find best (c, position) to insert."""
        current = list(base_seq)
        current_prods = _compute_products(current)
        used = set(current)
        rem = list(remaining)

        progress = True
        while progress and time.time() < time_limit:
            progress = False
            best_new = None
            best_c = None
            best_pos = -1
            best_gap = -1

            for c in rem:
                if time.time() > time_limit:
                    break
                n = len(current)
                for pos in range(n + 1):
                    if pos > 0 and current[pos - 1] >= c:
                        continue
                    if pos < n and c >= current[pos]:
                        continue
                    new_seq = current[:pos] + [c] + current[pos:]
                    new_prods = _compute_products(new_seq)
                    n_blocks = len(new_seq) * (len(new_seq) + 1) // 2
                    if len(new_prods) == n_blocks:
                        # Valid! Score by gap
                        gap = float('inf')
                        if pos > 0:
                            gap = min(gap, c - current[pos - 1])
                        if pos < n:
                            gap = min(gap, current[pos] - c)
                        if gap > best_gap:
                            best_gap = gap
                            best_new = new_seq
                            best_c = c
                            best_pos = pos
                            progress = True

            if best_c is not None:
                current = best_new
                current_prods = _compute_products(current)
                used.add(best_c)
                rem.remove(best_c)

        return current

    # === Main search ===
    best = []
    best_len = 0

    def _try_update(seq):
        nonlocal best, best_len
        if len(seq) > best_len and _is_valid(seq):
            best = seq
            best_len = len(seq)

    # Phase 1: Fast greedy baselines
    # Primes first (most flexible)
    primes = [x for x in candidates if _is_prime(x)]
    _try_update(_greedy_randomized(primes, 50))

    # Odds
    odds = list(range(3, 129, 2))
    _try_update(_greedy_randomized(odds, 100))

    # Full range
    _try_update(_greedy_randomized(candidates, 50))

    # Phase 2: Insertion search from best base
    if time.time() < deadline - 20:
        used = set(best)
        remaining = [x for x in candidates if x not in used]
        random.shuffle(remaining)
        cand = _insertion_search(best, remaining, deadline - 15)
        _try_update(cand)

    # Phase 3: Simulated annealing from diverse bases
    if time.time() < deadline - 10:
        # SA from primes base
        remaining = [x for x in candidates if x not in set(primes)]
        random.shuffle(remaining)
        cand = _simulated_annealing(primes, remaining, deadline - 8)
        _try_update(cand)

    if time.time() < deadline - 8:
        # SA from current best
        used = set(best)
        remaining = [x for x in candidates if x not in used]
        random.shuffle(remaining)
        cand = _simulated_annealing(best, remaining, deadline - 6)
        _try_update(cand)

    # Phase 4: More randomized greedy with different biases
    if time.time() < deadline - 5:
        # Bias toward composites (they have more prime factors, might be more flexible)
        composites = [x for x in candidates if not _is_prime(x)]
        _try_update(_greedy_randomized(composites, 100))

        # Mixed: interleave primes and composites
        mixed = list(candidates)
        random.shuffle(mixed)
        for t in range(50):
            shuffled = list(candidates)
            random.shuffle(shuffled)
            values = []
            prods = set()
            for x in shuffled:
                ok, new_prods = _append_check(values, prods, x)
                if ok:
                    values.append(x)
                    prods = new_prods
            _try_update(values)

    # Phase 5: Final insertion polish
    if time.time() < deadline - 3:
        used = set(best)
        remaining = [x for x in candidates if x not in used]
        random.shuffle(remaining)
        cand = _insertion_search(best, remaining, deadline - 2)
        _try_update(cand)

    # Final validation
    if not _is_valid(best):
        # Fallback
        best = _greedy_randomized(odds, 20)

    return best
