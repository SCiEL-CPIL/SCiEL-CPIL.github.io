"""Sparse squares of twelve-term integer polynomials."""
import time
import random
import sys
from collections import Counter

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 85.0  # safety margin before 90s kill
    best = None
    best_score = 10**9

    def square_nonzeros(coeffs):
        """Return number of nonzero entries in exact integer square of coeffs."""
        n = len(coeffs)
        # convolution
        sq = [0] * (2 * n - 1)
        for i in range(n):
            ci = coeffs[i]
            if ci == 0:
                continue
            for j in range(n):
                cj = coeffs[j]
                if cj == 0:
                    continue
                sq[i + j] += ci * cj
        return sum(1 for v in sq if v != 0)

    def validate(coeffs):
        """Check constraints: length<=41, nonzero final, entries in [-64,64], exactly 12 nonzero."""
        if len(coeffs) > 41:
            return False
        if len(coeffs) == 0 or coeffs[-1] == 0:
            return False
        if any(c < -64 or c > 64 for c in coeffs):
            return False
        if sum(1 for c in coeffs if c != 0) != 12:
            return False
        return True

    def make_candidate(positions, values):
        """Build coefficient list from sorted positions and values."""
        max_pos = max(positions)
        coeffs = [0] * (max_pos + 1)
        for p, v in zip(positions, values):
            coeffs[p] = v
        # Ensure length <= 41 and nonzero final
        if len(coeffs) > 41:
            return None
        if coeffs[-1] == 0:
            return None
        return coeffs

    def random_initial():
        """Generate a random valid 12-term polynomial."""
        # Choose 12 distinct positions in [0, 40]
        positions = sorted(random.sample(range(41), 12))
        # Ensure last position has nonzero (it will, since we assign values)
        values = [random.choice([-1, 1, -2, 2, -3, 3, -4, 4, -5, 5]) for _ in range(12)]
        coeffs = make_candidate(positions, values)
        if coeffs is None:
            return None
        return coeffs

    # Try some structured initial candidates first
    structured = []
    # Alternating signs at consecutive positions
    for start in range(0, 30):
        positions = list(range(start, start + 12))
        if positions[-1] > 40:
            continue
        for pattern in [
            [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
            [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1],
            [1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1],
        ]:
            coeffs = make_candidate(positions, pattern)
            if coeffs:
                structured.append(coeffs)
    # Symmetric positions
    for base in range(0, 15):
        positions = sorted([base + i for i in range(6)] + [base + 10 + i for i in range(6)])
        if positions[-1] > 40:
            continue
        values = [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1]
        coeffs = make_candidate(positions, values)
        if coeffs:
            structured.append(coeffs)

    candidates = structured + [random_initial() for _ in range(200)]
    candidates = [c for c in candidates if c is not None]

    # Evaluate all initial candidates
    for coeffs in candidates:
        if time.time() > deadline:
            break
        if not validate(coeffs):
            continue
        s = square_nonzeros(coeffs)
        if s < best_score:
            best_score = s
            best = list(coeffs)

    if best is None:
        # Fallback: simple valid candidate
        return [1] * 12

    # Local search / simulated annealing
    def neighbor(coeffs):
        """Generate a random neighbor by modifying one or two coefficients."""
        c = list(coeffs)
        n = len(c)
        op = random.random()
        if op < 0.4:
            # Change value of one term
            nz = [i for i in range(n) if c[i] != 0]
            if nz:
                idx = random.choice(nz)
                new_val = random.choice([-64, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 64])
                if new_val != 0:
                    c[idx] = new_val
        elif op < 0.7:
            # Move a term to a new position
            nz = [i for i in range(n) if c[i] != 0]
            if len(nz) >= 2:
                idx = random.choice(nz)
                val = c[idx]
                c[idx] = 0
                # Try to place at a new position
                for _ in range(10):
                    new_pos = random.randint(0, 40)
                    if c[new_pos] == 0:
                        c[new_pos] = val
                        break
        elif op < 0.85:
            # Add a new term and remove another (keep 12 terms)
            nz = [i for i in range(n) if c[i] != 0]
            if len(nz) == 12:
                remove_idx = random.choice(nz)
                val = c[remove_idx]
                c[remove_idx] = 0
                for _ in range(20):
                    new_pos = random.randint(0, 40)
                    if c[new_pos] == 0:
                        c[new_pos] = random.choice([-3, -2, -1, 1, 2, 3])
                        break
        else:
            # Swap two values
            nz = [i for i in range(n) if c[i] != 0]
            if len(nz) >= 2:
                i, j = random.sample(nz, 2)
                c[i], c[j] = c[j], c[i]

        # Trim trailing zeros
        while len(c) > 0 and c[-1] == 0:
            c.pop()
        if len(c) == 0 or len(c) > 41:
            return None
        if sum(1 for x in c if x != 0) != 12:
            return None
        if any(x < -64 or x > 64 for x in c):
            return None
        if c[-1] == 0:
            return None
        return c

    # Simulated annealing
    temp = 5.0
    cooling = 0.999
    current = list(best)
    current_score = best_score
    iterations = 0
    max_iter = 500000

    while time.time() < deadline - 2.0:
        if iterations % 1000 == 0:
            temp *= cooling
        nb = neighbor(current)
        if nb is None:
            iterations += 1
            continue
        nb_score = square_nonzeros(nb)
        delta = nb_score - current_score
        if delta < 0 or random.random() < (2.718281828 ** (-delta / max(temp, 0.01))):
            current = nb
            current_score = nb_score
            if current_score < best_score:
                best_score = current_score
                best = list(current)
        iterations += 1

    return best
# EVOLVE-BLOCK-END
