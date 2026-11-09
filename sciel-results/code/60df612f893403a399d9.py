import time
import sys
import random
import math


# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 90s kill

    all_numbers = list(range(2, 129))

    # ---------- prime factorization / exponent vectors ----------
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True

    primes = [x for x in all_numbers if is_prime(x)]
    num_primes = len(primes)
    prime_to_idx = {p: i for i, p in enumerate(primes)}

    exp_vecs = {}
    for n in all_numbers:
        vec = [0] * num_primes
        temp = n
        for pi, p in enumerate(primes):
            if p * p > temp:
                break
            while temp % p == 0:
                vec[pi] += 1
                temp //= p
        if temp > 1:
            vec[prime_to_idx[temp]] += 1
        exp_vecs[n] = tuple(vec)

    # ---------- validity ----------
    def check_sequence_valid(seq):
        n = len(seq)
        if n == 0:
            return True
        prods = set()
        for i in range(n):
            vec = [0] * num_primes
            for j in range(i, n):
                ej = exp_vecs[seq[j]]
                for k in range(num_primes):
                    vec[k] += ej[k]
                tv = tuple(vec)
                if tv in prods:
                    return False
                prods.add(tv)
        return True

    def is_valid(seq):
        # strictly increasing + distinct block products
        for i in range(len(seq) - 1):
            if seq[i] >= seq[i + 1]:
                return False
        return check_sequence_valid(seq)

    # ---------- insertion helpers ----------
    def try_insert(seq, c):
        """Insert c (not in seq) at a valid sorted position. Returns new_seq or None."""
        n = len(seq)
        # candidate positions where sorted order holds
        lo = 0
        while lo < n and seq[lo] < c:
            lo += 1
        hi = n
        while hi > 0 and seq[hi - 1] > c:
            hi -= 1
        # valid positions: [lo, hi]
        if lo > hi:
            return None
        # try central first, then outward
        mid = (lo + hi) / 2.0
        order = sorted(range(lo, hi + 1), key=lambda i: (abs(i - mid), i))
        for i in order:
            new_seq = seq[:i] + [c] + seq[i:]
            if check_sequence_valid(new_seq):
                return new_seq
        return None

    def extend_sequence(base_seq, iterations):
        """Repeatedly try to insert missing numbers in shuffled order."""
        current_best = list(base_seq)
        current_best_len = len(current_best)
        used = set(current_best)
        missing = [x for x in all_numbers if x not in used]

        for _ in range(iterations):
            if time.time() > deadline:
                break
            if not missing:
                break
            random.shuffle(missing)
            temp_seq = list(current_best)
            temp_len = current_best_len
            temp_used = set(temp_seq)
            for x in missing:
                if x in temp_used:
                    continue
                new_seq = try_insert(temp_seq, x)
                if new_seq is not None:
                    temp_seq = new_seq
                    temp_len += 1
                    temp_used.add(x)
            if temp_len > current_best_len:
                current_best = temp_seq
                current_best_len = temp_len
                used = set(current_best)
                missing = [x for x in all_numbers if x not in used]
        return current_best

    def local_search(seq, max_iter=200):
        """Improve via insertions and remove-reinsert swaps."""
        current = seq[:]
        current_len = len(current)
        used = set(current)
        remaining = [x for x in all_numbers if x not in used]

        for _ in range(max_iter):
            if time.time() > deadline:
                break
            improved = False

            # insertions
            random.shuffle(remaining)
            for c in remaining:
                new_seq = try_insert(current, c)
                if new_seq is not None:
                    current = new_seq
                    current_len = len(current)
                    used = set(current)
                    remaining = [x for x in all_numbers if x not in used]
                    improved = True
                    break

            # swaps: remove one, add another (keep same or longer length)
            if not improved and remaining:
                for i in range(len(current)):
                    for c in remaining:
                        temp_seq = current[:i] + current[i + 1:]
                        new_seq = try_insert(temp_seq, c)
                        if new_seq is not None and len(new_seq) >= current_len:
                            current = new_seq
                            current_len = len(current)
                            used = set(current)
                            remaining = [x for x in all_numbers if x not in used]
                            improved = True
                            break
                    if improved:
                        break
            if not improved:
                break

        return current

    def simulated_annealing(seq, max_iter=1500):
        current = seq[:]
        best = current[:]
        best_len = len(current)
        temperature = 5.0
        cooling = 0.998

        for _ in range(max_iter):
            if time.time() > deadline:
                break
            neighbor = current[:]
            action = random.random()
            ok = True

            if action < 0.3:
                if len(neighbor) > 0:
                    idx = random.randint(0, len(neighbor) - 1)
                    neighbor.pop(idx)
            elif action < 0.6:
                if len(neighbor) >= 2:
                    i, j = random.sample(range(len(neighbor)), 2)
                    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                    if not all(neighbor[k] < neighbor[k + 1] for k in range(len(neighbor) - 1)):
                        ok = False
            else:
                if len(neighbor) > 0:
                    idx = random.randint(0, len(neighbor) - 1)
                    old_val = neighbor[idx]
                    new_val = random.randint(2, 128)
                    if new_val != old_val:
                        neighbor[idx] = new_val
                        if not all(neighbor[k] < neighbor[k + 1] for k in range(len(neighbor) - 1)):
                            ok = False

            if not ok:
                continue
            if not check_sequence_valid(neighbor):
                continue

            neighbor_len = len(neighbor)
            if neighbor_len > len(current):
                current = neighbor
                if neighbor_len > best_len:
                    best = neighbor[:]
                    best_len = neighbor_len
            else:
                energy_diff = len(current) - neighbor_len
                if random.random() < math.exp(-energy_diff / temperature):
                    current = neighbor
            temperature *= cooling

        return best

    # ---------- build seeds ----------
    best_seq = []
    best_len = 0

    def consider(seq):
        nonlocal best_seq, best_len
        if seq and is_valid(seq) and len(seq) > best_len:
            best_seq = seq[:]
            best_len = len(seq)

    # Seed 1: ascending greedy
    values, products = [], set()
    for x in range(2, 129):
        tails, prod = [x], x
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    consider(values)

    # Seed 2: primes
    consider(primes)

    # Seed 3: small random increasing sets
    for _ in range(6):
        if time.time() > deadline:
            break
        k = random.randint(6, 14)
        cand = random.sample(all_numbers, k)
        cand.sort()
        if check_sequence_valid(cand):
            consider(cand)

    # ---------- refine best seed ----------
    if best_seq:
        extended = extend_sequence(best_seq, 400)
        consider(extended)
        ls = local_search(extended, max_iter=200)
        consider(ls)
        sa = simulated_annealing(ls, max_iter=1500)
        consider(sa)

    # ---------- refine primes seed too ----------
    prime_extended = extend_sequence(primes, 400)
    consider(prime_extended)
    prime_ls = local_search(prime_extended, max_iter=150)
    consider(prime_ls)

    # ---------- final validation / fallback ----------
    if not best_seq or not is_valid(best_seq):
        best_seq = [2]

    return best_seq

# EVOLVE-BLOCK-END
