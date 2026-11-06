"""Combined program: efficient prime selection + diagonal init + greedy prefix extension."""
import time
import math
import random
from fractions import Fraction

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    DEADLINE = 110  # safety margin before 120s/150s limit

    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(math.sqrt(n)) + 1):
            if sieve[i]:
                for j in range(i * i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    all_primes = get_primes(1000)

    # Build prime set: start with primes >= 11, remove largest until sum(1/p) <= 1
    candidates = [p for p in all_primes if p >= 11]
    candidates_desc = sorted(candidates, reverse=True)
    current_sum = sum(Fraction(1, p) for p in candidates)
    primes_set = list(candidates)
    while current_sum > 1:
        if not candidates_desc:
            break
        p_remove = candidates_desc.pop()
        primes_set.remove(p_remove)
        current_sum -= Fraction(1, p_remove)

    primes_set = sorted(primes_set)
    n_primes = len(primes_set)

    # Verify budget
    exact_sum = sum(Fraction(1, p) for p in primes_set)
    if exact_sum > 1:
        # Fallback: use primes >= 13
        primes_set = [p for p in all_primes if p >= 13]
        exact_sum = sum(Fraction(1, p) for p in primes_set)
        n_primes = len(primes_set)

    L_CHECK_LIMIT = 15000
    L_FINAL_LIMIT = 100000

    def evaluate_coverage(residues, L_limit):
        """Return (L, cover_count) where cover_count[n] = number of classes covering n."""
        cover_count = [0] * (L_limit + 1)
        for i in range(n_primes):
            p = primes_set[i]
            r = residues[i]
            start = r
            if start == 0:
                start = p
            if start <= L_limit:
                for n in range(start, L_limit + 1, p):
                    cover_count[n] += 1
        L = 0
        for n in range(1, L_limit + 1):
            if cover_count[n] == 0:
                break
            L = n
        return L, cover_count

    def update_counts(cover_count, p, r_old, r_new, L_limit):
        if r_old == r_new:
            return
        start_old = r_old
        if start_old == 0:
            start_old = p
        if start_old <= L_limit:
            for n in range(start_old, L_limit + 1, p):
                cover_count[n] -= 1
        start_new = r_new
        if start_new == 0:
            start_new = p
        if start_new <= L_limit:
            for n in range(start_new, L_limit + 1, p):
                cover_count[n] += 1

    def calc_L(cover_count, L_limit):
        L = 0
        for n in range(1, L_limit + 1):
            if cover_count[n] == 0:
                break
            L = n
        return L

    def init_residues_diagonal():
        """Diagonal init: residue i covers number i+1."""
        res = [0] * n_primes
        for i in range(n_primes):
            p = primes_set[i]
            res[i] = (i + 1) % p
        return res

    def init_residues_greedy():
        """Greedy init: for each prime, pick residue covering most uncovered numbers in 1..L_CHECK_LIMIT."""
        covered = [0] * (L_CHECK_LIMIT + 1)
        res = [0] * n_primes
        for i in range(n_primes):
            p = primes_set[i]
            best_r = 0
            best_count = -1
            for r in range(p):
                start = r
                if start == 0:
                    start = p
                count = 0
                for n in range(start, L_CHECK_LIMIT + 1, p):
                    if covered[n] == 0:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_r = r
            res[i] = best_r
            start = best_r
            if start == 0:
                start = p
            for n in range(start, L_CHECK_LIMIT + 1, p):
                covered[n] += 1
        return res

    def try_multi_move(cover_count, residues, L, L_limit, max_moves=3):
        """Try up to max_moves random single-residue changes, keeping those that don't reduce L."""
        for _ in range(max_moves):
            idx = random.randint(0, n_primes - 1)
            p = primes_set[idx]
            old_r = residues[idx]
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                continue
            update_counts(cover_count, p, old_r, new_r, L_limit)
            residues[idx] = new_r
            new_L = calc_L(cover_count, L_limit)
            if new_L >= L:
                L = new_L
            else:
                update_counts(cover_count, p, new_r, old_r, L_limit)
                residues[idx] = old_r
        return L

    best_L = 0
    best_classes = None

    # Try diagonal initialization first (known to be strong)
    residues = init_residues_diagonal()
    L, cover_count = evaluate_coverage(residues, L_CHECK_LIMIT)
    if L > best_L:
        best_L = L
        best_classes = [[primes_set[i], residues[i]] for i in range(n_primes)]

    # Greedy prefix-extension local search
    stuck_counter = 0
    max_stuck = 200

    while time.time() - start_time < DEADLINE - 10:
        if L >= L_CHECK_LIMIT:
            break

        hole = L + 1

        best_new_L = L
        best_idx = -1
        best_r_new = -1
        best_r_old = -1

        # Try each prime to cover the hole
        for idx in range(n_primes):
            p = primes_set[idx]
            r_old = residues[idx]
            r_new = hole % p

            if r_old == r_new:
                continue

            update_counts(cover_count, p, r_old, r_new, L_CHECK_LIMIT)
            residues[idx] = r_new
            new_L = calc_L(cover_count, L_CHECK_LIMIT)

            if new_L > best_new_L:
                best_new_L = new_L
                best_idx = idx
                best_r_new = r_new
                best_r_old = r_old
                # Revert to test other primes fairly
                update_counts(cover_count, p, r_new, r_old, L_CHECK_LIMIT)
                residues[idx] = r_old
            else:
                update_counts(cover_count, p, r_new, r_old, L_CHECK_LIMIT)
                residues[idx] = r_old

        if best_idx != -1:
            p = primes_set[best_idx]
            update_counts(cover_count, p, best_r_old, best_r_new, L_CHECK_LIMIT)
            residues[best_idx] = best_r_new
            L = best_new_L
            stuck_counter = 0

            if L > best_L:
                best_L = L
                best_classes = [[primes_set[i], residues[i]] for i in range(n_primes)]
        else:
            stuck_counter += 1
            if stuck_counter > max_stuck:
                # Multi-move random restart with perturbation
                L = try_multi_move(cover_count, residues, L, L_CHECK_LIMIT, max_moves=15)
                if L > best_L:
                    best_L = L
                    best_classes = [[primes_set[i], residues[i]] for i in range(n_primes)]
                stuck_counter = 0

    # Final evaluation up to L_FINAL_LIMIT
    if best_classes:
        final_L = 0
        covered = [False] * (L_FINAL_LIMIT + 1)
        for p, r in best_classes:
            start = r
            if start == 0:
                start = p
            if start <= L_FINAL_LIMIT:
                for n in range(start, L_FINAL_LIMIT + 1, p):
                    covered[n] = True
        for n in range(1, L_FINAL_LIMIT + 1):
            if covered[n]:
                final_L = n
            else:
                break
        # Use the larger of the two
        if final_L > best_L:
            best_L = final_L

    # If we have time, do a second pass with greedy init
    if time.time() - start_time < DEADLINE - 30 and best_L < 999:
        residues2 = init_residues_greedy()
        L2, cover_count2 = evaluate_coverage(residues2, L_CHECK_LIMIT)
        if L2 > best_L:
            best_L = L2
            best_classes = [[primes_set[i], residues2[i]] for i in range(n_primes)]
        
        # Continue local search from this starting point
        L = L2
        cover_count = cover_count2
        stuck_counter = 0
        while time.time() - start_time < DEADLINE - 10:
            if L >= L_CHECK_LIMIT:
                break
            hole = L + 1
            best_new_L = L
            best_idx = -1
            best_r_new = -1
            best_r_old = -1
            for idx in range(n_primes):
                p = primes_set[idx]
                r_old = residues2[idx]
                r_new = hole % p
                if r_old == r_new:
                    continue
                update_counts(cover_count, p, r_old, r_new, L_CHECK_LIMIT)
                residues2[idx] = r_new
                new_L = calc_L(cover_count, L_CHECK_LIMIT)
                if new_L > best_new_L:
                    best_new_L = new_L
                    best_idx = idx
                    best_r_new = r_new
                    best_r_old = r_old
                    update_counts(cover_count, p, r_new, r_old, L_CHECK_LIMIT)
                    residues2[idx] = r_old
                else:
                    update_counts(cover_count, p, r_new, r_old, L_CHECK_LIMIT)
                    residues2[idx] = r_old
            if best_idx != -1:
                p = primes_set[best_idx]
                update_counts(cover_count, p, best_r_old, best_r_new, L_CHECK_LIMIT)
                residues2[best_idx] = best_r_new
                L = best_new_L
                stuck_counter = 0
                if L > best_L:
                    best_L = L
                    best_classes = [[primes_set[i], residues2[i]] for i in range(n_primes)]
            else:
                stuck_counter += 1
                if stuck_counter > max_stuck:
                    L = try_multi_move(cover_count, residues2, L, L_CHECK_LIMIT, max_moves=15)
                    if L > best_L:
                        best_L = L
                        best_classes = [[primes_set[i], residues2[i]] for i in range(n_primes)]
                    stuck_counter = 0

    # Safety: verify budget
    if best_classes:
        total = sum(Fraction(1, p) for p, r in best_classes)
        if total > 1:
            # Fallback to primes >= 13 with diagonal residues
            primes_set = [p for p in all_primes if p >= 13]
            n_primes = len(primes_set)
            residues = init_residues_diagonal()
            best_classes = [[primes_set[i], residues[i]] for i in range(n_primes)]

    if not best_classes:
        # Last resort fallback
        primes_set = [p for p in all_primes if p >= 13]
        n_primes = len(primes_set)
        residues = [(i + 1) % p for i, p in enumerate(primes_set)]
        best_classes = [[primes_set[i], residues[i]] for i in range(n_primes)]

    return best_classes

# EVOLVE-BLOCK-END
