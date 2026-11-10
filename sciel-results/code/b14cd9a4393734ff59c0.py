"""Rewritten program: admissible-tuple maximization in [0, 5000]."""
import time
import random
from collections import defaultdict

N = 5000
_DEADLINE = 295.0  # safety margin before 330s kill


def _primes(limit):
    """Sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]


class _AdmissibleTracker:
    """Track residue-class coverage for each prime efficiently."""

    def __init__(self, primes):
        self.primes = primes
        self.residue_counts = {}
        for p in primes:
            self.residue_counts[p] = [0] * p
        self.covered_count = {}
        for p in primes:
            self.covered_count[p] = 0

    def add(self, v):
        """Add integer v; return True if still admissible (no prime fully covered)."""
        for p in self.primes:
            r = v % p
            if self.residue_counts[p][r] == 0:
                self.covered_count[p] += 1
                self.residue_counts[p][r] = 1
                if self.covered_count[p] == p:
                    return False
            else:
                self.residue_counts[p][r] += 1
        return True

    def remove(self, v):
        """Remove integer v."""
        for p in self.primes:
            r = v % p
            self.residue_counts[p][r] -= 1
            if self.residue_counts[p][r] == 0:
                self.covered_count[p] -= 1

    def would_complete(self, v):
        """Check if adding v would make any prime fully covered."""
        for p in self.primes:
            if self.covered_count[p] == p - 1:
                if self.residue_counts[p][v % p] == 0:
                    return True
        return False

    def is_admissible(self):
        """Check if current state is admissible."""
        for p in self.primes:
            if self.covered_count[p] == p:
                return False
        return True

    def rebuild(self, points):
        """Rebuild from scratch given a set of points."""
        for p in self.primes:
            self.residue_counts[p] = [0] * p
            self.covered_count[p] = 0
        for v in points:
            for p in self.primes:
                r = v % p
                if self.residue_counts[p][r] == 0:
                    self.covered_count[p] += 1
                self.residue_counts[p][r] += 1


def _greedy_removal(points, primes, deadline):
    """Greedy removal: repeatedly find a fully-covered prime, remove least-populated class."""
    point_set = set(points)
    tracker = _AdmissibleTracker(primes)
    tracker.rebuild(point_set)

    while time.time() < deadline:
        # Find all fully-covered primes
        covered_primes = [p for p in primes if tracker.covered_count[p] == p]
        if not covered_primes:
            break

        # Pick the smallest covered prime (most constrained)
        p = min(covered_primes)

        # Count elements in each residue class mod p
        class_counts = defaultdict(int)
        for v in point_set:
            class_counts[v % p] += 1

        # Remove the least populated class
        min_class = min(class_counts, key=lambda r: class_counts[r])
        to_remove = [v for v in point_set if v % p == min_class]

        for v in to_remove:
            point_set.discard(v)
            tracker.remove(v)

    return list(point_set)


def _add_back_search(points, primes, deadline):
    """Try to add back elements to increase tuple size while maintaining admissibility."""
    point_set = set(points)
    tracker = _AdmissibleTracker(primes)
    tracker.rebuild(point_set)

    all_candidates = set(range(N + 1)) - point_set

    def _relevant_primes(current_size):
        return [p for p in primes if p <= current_size + 1]

    improved = True
    while improved and time.time() < deadline:
        improved = False
        check_primes = _relevant_primes(len(point_set))
        if not check_primes:
            break

        candidates = list(all_candidates)

        def _safety_score(v):
            score = 0
            for p in check_primes:
                r = v % p
                if tracker.residue_counts[p][r] == 0:
                    score += 1
            return score

        candidates.sort(key=_safety_score)

        for v in candidates:
            if time.time() >= deadline:
                break
            if not tracker.would_complete(v):
                tracker.add(v)
                point_set.add(v)
                all_candidates.discard(v)
                improved = True

    return list(point_set)


def _full_pipeline(points, primes, deadline):
    """Run greedy removal then add-back search."""
    points = _greedy_removal(points, primes, deadline)
    points = _add_back_search(points, primes, deadline)
    return points


def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE

    all_primes = _primes(N + 1)
    best = []

    # Strategy 1: Start from full set [0, N] and sieve down
    if time.time() < deadline - 20:
        init_full = list(range(N + 1))
        result_full = _full_pipeline(init_full, all_primes, deadline)
        if len(result_full) > len(best):
            best = result_full

    # Strategy 2: Random restarts with different densities
    attempts = 0
    while time.time() < deadline - 20 and attempts < 8:
        fraction = 0.2 + 0.6 * (attempts % 6) / 6.0
        seed = attempts * 7919 + 12345
        random.seed(seed)
        init = random.sample(range(N + 1), int(fraction * (N + 1)))
        result = _full_pipeline(init, all_primes, deadline)
        if len(result) > len(best):
            best = result
        attempts += 1

    # Final: one more add-back on the best
    if time.time() < deadline - 5 and best:
        best = _add_back_search(best, all_primes, deadline)

    return best
