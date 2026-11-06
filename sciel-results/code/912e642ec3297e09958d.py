import time
import math
import random
from fractions import Fraction

def _is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill

    random.seed(42)

    # Generate primes up to 1000
    primes = [p for p in range(2, 1001) if _is_prime(p)]

    # Prime set: primes >= 13. Sum of 1/p for p in [13, 1000] is ~0.92 < 1.
    # This leaves budget headroom and matches the seed's proven structure.
    selected_primes = [p for p in primes if p >= 13]

    # Verify budget with exact rationals
    total_budget = sum(Fraction(1, p) for p in selected_primes)
    assert total_budget <= 1, f"Budget exceeded: {total_budget}"

    # Initial residues: (i+1) % p so that the i-th prime covers integer i+1
    # This gives a reasonable starting coverage.
    classes = [(p, (i + 1) % p) for i, p in enumerate(selected_primes)]

    # Coverage window: we care about the prefix, but need to look far enough
    # to measure L exactly. Cap at a value that is fast but sufficient.
    max_L = 5000

    # Build coverage count array (how many classes cover each n)
    count = [0] * (max_L + 1)
    for p, r in classes:
        for n in range(r, max_L + 1, p):
            if n >= 1:
                count[n] += 1

    def find_prefix(count):
        for n in range(1, max_L + 1):
            if count[n] == 0:
                return n - 1
        return max_L

    best_L = find_prefix(count)
    best_classes = [list(c) for c in classes]

    # Helper: apply a residue change to the count array
    def apply_change(count, p, old_r, new_r):
        for n in range(old_r, max_L + 1, p):
            if n >= 1:
                count[n] -= 1
        for n in range(new_r, max_L + 1, p):
            if n >= 1:
                count[n] += 1

    # Iterated Local Search
    # Phase 1: Single-kick local search (residue flips) until no improvement
    # Phase 2: Multi-kick perturbations to escape local minima, then re-optimize
    improved = True
    kick_round = 0
    while time.time() < deadline - 10:
        if improved:
            # Local search: try flipping each residue to its best alternative
            improved = False
            for i in range(len(classes)):
                if time.time() > deadline - 10:
                    break
                p, r = classes[i]
                best_r = r
                best_L_local = best_L
                # Try all residues
                for new_r in range(p):
                    if new_r == r:
                        continue
                    apply_change(count, p, r, new_r)
                    L_local = find_prefix(count)
                    if L_local > best_L_local:
                        best_L_local = L_local
                        best_r = new_r
                    apply_change(count, p, new_r, r)
                if best_r != r:
                    apply_change(count, p, r, best_r)
                    classes[i] = (p, best_r)
                    best_L = best_L_local
                    improved = True
                    best_classes = [list(c) for c in classes]
        else:
            # No improvement from single kicks: apply a multi-kick perturbation
            kick_round += 1
            # Perturb a random subset of residues (multi-kick)
            # Larger kicks in later rounds to escape deeper minima
            n_kicks = random.randint(2, min(5, len(classes)))
            for _ in range(n_kicks):
                i = random.randrange(len(classes))
                p, r = classes[i]
                new_r = random.randrange(p)
                if new_r != r:
                    apply_change(count, p, r, new_r)
                    classes[i] = (p, new_r)
            # Recompute L after perturbation
            current_L = find_prefix(count)
            if current_L > best_L:
                best_L = current_L
                best_classes = [list(c) for c in classes]
            # If perturbation made things much worse, we still re-optimize
            # (local search will recover)
            # But if we're way below best, consider reverting to best
            if current_L < best_L - 200:
                # Revert to best_classes and rebuild count
                classes = [tuple(c) for c in best_classes]
                count = [0] * (max_L + 1)
                for p, r in classes:
                    for n in range(r, max_L + 1, p):
                        if n >= 1:
                            count[n] += 1
                best_L = find_prefix(count)

    return best_classes
