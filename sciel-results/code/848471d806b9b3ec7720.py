"""Improved program for the narrow admissible tuple task.

Strategy: efficient greedy construction with maintained residue coverage counts,
randomized restarts, and fast local search (add-back).
Key optimization: maintain per-prime residue coverage for O(1) add/remove checks.
"""

import time
import random
import math

N = 5000
DEADLINE = 280  # seconds; leave safety margin before 330s kill

def _primes(limit):
    """Sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p*p::p] = bytearray(len(sieve[p*p::p]))
    return [p for p in range(2, limit + 1) if sieve[p]]

# Precompute primes up to N
_PRIMES_UP_TO_N = _primes(N)

def _is_admissible(points, primes_needed):
    """Check if points form an admissible tuple with respect to primes_needed."""
    n_points = len(points)
    if n_points == 0:
        return True
    for p in primes_needed:
        if p > n_points:
            break
        seen = [False] * p
        count = 0
        for v in points:
            r = v % p
            if not seen[r]:
                seen[r] = True
                count += 1
                if count == p:
                    return False
    return True

class _ResidueTracker:
    """Maintains residue counts and coverage for each prime."""
    
    def __init__(self, points, primes):
        self.primes = primes
        self.n_points = len(points)
        # For each prime: residue_counts[r] = number of points with that residue
        # coverage[p] = number of distinct residues covered (count > 0)
        self.residue_counts = {}
        self.coverage = {}
        for p in primes:
            counts = [0] * p
            for v in points:
                counts[v % p] += 1
            self.residue_counts[p] = counts
            self.coverage[p] = sum(1 for c in counts if c > 0)
    
    def is_violated(self, p):
        """Check if prime p has all residues covered."""
        return self.coverage[p] == p
    
    def get_violated_primes(self):
        """Return list of violated primes."""
        result = []
        for p in self.primes:
            if p > self.n_points:
                break
            if self.coverage[p] == p:
                result.append(p)
        return result
    
    def can_add(self, v):
        """Check if adding v maintains admissibility. Does NOT modify state."""
        new_n = self.n_points + 1
        for p in self.primes:
            if p > new_n:
                break
            r = v % p
            if self.residue_counts[p][r] == 0:
                # This residue is currently uncovered
                # After adding, coverage would increase by 1
                # Violation if new coverage == p
                if self.coverage[p] + 1 == p:
                    return False
        return True
    
    def add_point(self, v):
        """Add a point. Assumes can_add(v) returned True."""
        for p in self.primes:
            if p > self.n_points + 1:
                break
            r = v % p
            if self.residue_counts[p][r] == 0:
                self.coverage[p] += 1
            self.residue_counts[p][r] += 1
        self.n_points += 1
    
    def remove_point(self, v):
        """Remove a point."""
        for p in self.primes:
            if p > self.n_points:
                break
            r = v % p
            self.residue_counts[p][r] -= 1
            if self.residue_counts[p][r] == 0:
                self.coverage[p] -= 1
        self.n_points -= 1
    
    def get_min_class_residues(self, p):
        """Get residues with minimum count for prime p."""
        counts = self.residue_counts[p]
        min_size = min(counts)
        return [r for r in range(p) if counts[r] == min_size]

def _greedy_remove_efficient(points, primes, rng):
    """
    Greedy removal using ResidueTracker.
    """
    points_set = set(points)
    tracker = _ResidueTracker(list(points), primes)
    
    while True:
        if time.time() > DEADLINE - 5:
            break
        violated = tracker.get_violated_primes()
        if not violated:
            break
        
        p = violated[0]
        min_residues = tracker.get_min_class_residues(p)
        if not min_residues or tracker.residue_counts[p][min_residues[0]] == 0:
            break
        
        chosen_r = rng.choice(min_residues)
        to_remove = [v for v in points_set if v % p == chosen_r]
        for v in to_remove:
            points_set.discard(v)
            tracker.remove_point(v)
    
    return list(points_set)

def _add_back_efficient(points, all_candidates, primes, rng, max_rounds=50, start_time=0, deadline=DEADLINE):
    """
    Try to add back removed elements using efficient residue tracking.
    """
    points_set = set(points)
    tracker = _ResidueTracker(list(points), primes)
    removed = [v for v in all_candidates if v not in points_set]
    
    improved = True
    round_num = 0
    while improved and round_num < max_rounds:
        if time.time() - start_time > deadline - 5:
            break
        improved = False
        round_num += 1
        rng.shuffle(removed)
        
        for v in removed:
            if time.time() - start_time > deadline - 5:
                break
            if v in points_set:
                continue
            
            if tracker.can_add(v):
                tracker.add_point(v)
                points_set.add(v)
                improved = True
    
    return list(points_set)

def _construct_one_strategy(rng, strategy):
    """
    Construct an admissible tuple using a specific strategy.
    """
    all_candidates = set(range(N + 1))
    
    if strategy == 0:
        points = list(all_candidates)
        points = _greedy_remove_efficient(points, _PRIMES_UP_TO_N, rng)
    elif strategy == 1:
        density = 0.7
        points = [v for v in range(N + 1) if rng.random() < density]
        points = _greedy_remove_efficient(points, _PRIMES_UP_TO_N, rng)
    elif strategy == 2:
        points = list(all_candidates)
        points = _greedy_remove_efficient(points, _PRIMES_UP_TO_N, rng)
        points = _add_back_efficient(points, all_candidates, _PRIMES_UP_TO_N, rng, max_rounds=30)
    elif strategy == 3:
        density = rng.uniform(0.5, 0.8)
        points = [v for v in range(N + 1) if rng.random() < density]
        points = _greedy_remove_efficient(points, _PRIMES_UP_TO_N, rng)
        points = _add_back_efficient(points, all_candidates, _PRIMES_UP_TO_N, rng, max_rounds=50)
    else:
        density = 0.9
        points = [v for v in range(N + 1) if rng.random() < density]
        points = _greedy_remove_efficient(points, _PRIMES_UP_TO_N, rng)
        points = _add_back_efficient(points, all_candidates, _PRIMES_UP_TO_N, rng, max_rounds=100)
    
    return points

def construct():
    """
    Main entry point.
    """
    start_time = time.time()
    best_points = []
    best_size = 0
    
    rng0 = random.Random(42)
    baseline = _construct_one_strategy(rng0, 0)
    if _is_admissible(baseline, [p for p in _PRIMES_UP_TO_N if p <= len(baseline)]):
        best_points = baseline
        best_size = len(baseline)
    
    seed = 12345
    rng = random.Random(seed)
    num_restarts = 0
    all_candidates_set = set(range(N + 1))
    
    while time.time() - start_time < DEADLINE - 10:
        strategy = rng.randint(0, 4)
        points = _construct_one_strategy(rng, strategy)
        
        if not _is_admissible(points, [p for p in _PRIMES_UP_TO_N if p <= len(points)]):
            points = _greedy_remove_efficient(points, _PRIMES_UP_TO_N, rng)
        
        if len(points) > best_size:
            best_size = len(points)
            best_points = points
            if time.time() - start_time < DEADLINE - 20:
                improved = _add_back_efficient(best_points, all_candidates_set, _PRIMES_UP_TO_N, rng, max_rounds=10, start_time=start_time, deadline=DEADLINE)
                if _is_admissible(improved, [p for p in _PRIMES_UP_TO_N if p <= len(improved)]):
                    if len(improved) > best_size:
                        best_size = len(improved)
                        best_points = improved
        
        num_restarts += 1
        if num_restarts % 5 == 0 and time.time() - start_time < DEADLINE - 30:
            improved = _add_back_efficient(best_points, all_candidates_set, _PRIMES_UP_TO_N, rng, max_rounds=20, start_time=start_time, deadline=DEADLINE)
            if _is_admissible(improved, [p for p in _PRIMES_UP_TO_N if p <= len(improved)]):
                if len(improved) > best_size:
                    best_size = len(improved)
                    best_points = improved
    
    if not _is_admissible(best_points, [p for p in _PRIMES_UP_TO_N if p <= len(best_points)]):
        best_points = _greedy_remove_efficient(best_points, _PRIMES_UP_TO_N, random.Random(0))
    
    return best_points
