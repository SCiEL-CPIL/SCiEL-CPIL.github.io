"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 5000
import time
import random

_DEADLINE = 320.0  # safety margin before 330s kill

# EVOLVE-BLOCK-START
def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]


class _AdmissibleTracker:
    """Incremental tracker for admissibility using residue class counts per prime."""
    
    def __init__(self, points, primes):
        self.point_set = set(points)
        self.primes = primes
        # For each prime, track count per residue class and number of non-zero classes
        self.residue_counts = {}  # p -> list of length p
        self.nonzero_classes = {}  # p -> number of residue classes with count > 0
        for p in primes:
            counts = [0] * p
            for v in self.point_set:
                counts[v % p] += 1
            self.residue_counts[p] = counts
            self.nonzero_classes[p] = sum(1 for c in counts if c > 0)
    
    def is_covered(self, p):
        """Check if all p residue classes are covered (admissibility violation)."""
        return self.nonzero_classes[p] == p
    
    def would_cover(self, v, p):
        """Check if adding v would make all p residue classes covered."""
        r = v % p
        counts = self.residue_counts[p]
        if counts[r] == 0:
            # This would be a new class
            return self.nonzero_classes[p] == p - 1
        return self.is_covered(p)
    
    def add(self, v):
        """Add a point, updating all prime residue counts."""
        if v in self.point_set:
            return
        self.point_set.add(v)
        for p in self.primes:
            r = v % p
            counts = self.residue_counts[p]
            if counts[r] == 0:
                self.nonzero_classes[p] += 1
            counts[r] += 1
    
    def remove(self, v):
        """Remove a point, updating all prime residue counts."""
        if v not in self.point_set:
            return
        self.point_set.remove(v)
        for p in self.primes:
            r = v % p
            counts = self.residue_counts[p]
            counts[r] -= 1
            if counts[r] == 0:
                self.nonzero_classes[p] -= 1
    
    def covered_primes(self):
        """Return list of primes where all residue classes are covered."""
        result = []
        for p in self.primes:
            if p > len(self.point_set):
                continue
            if self.nonzero_classes[p] == p:
                result.append(p)
        return result
    
    def safety_score(self, v):
        """Score how 'risky' adding v is: count primes where v fills a new class."""
        score = 0
        for p in self.primes:
            if p > len(self.point_set) + 1:
                continue
            r = v % p
            if self.residue_counts[p][r] == 0:
                score += 1
        return score
    
    def can_add(self, v):
        """Check if v can be added without violating admissibility."""
        for p in self.primes:
            if p > len(self.point_set) + 1:
                continue
            if self.would_cover(v, p):
                return False
        return True


def _greedy_removal(points, primes_to_check, deadline):
    """Greedy removal using incremental tracker: find covered prime, remove least-populated class."""
    tracker = _AdmissibleTracker(points, primes_to_check)
    
    while time.time() < deadline:
        covered = tracker.covered_primes()
        if not covered:
            break
        
        # Pick the smallest covered prime (most constrained)
        p = min(covered)
        
        # Find the least populated residue class mod p
        counts = tracker.residue_counts[p]
        min_class = min(range(p), key=lambda r: counts[r])
        
        # Remove all points in that class
        to_remove = [v for v in tracker.point_set if v % p == min_class]
        for v in to_remove:
            tracker.remove(v)
    
    return list(tracker.point_set)


def _add_back_search(points, primes_to_check, deadline):
    """Add-back search using incremental tracker with safety-scored ordering."""
    tracker = _AdmissibleTracker(points, primes_to_check)
    all_candidates = set(range(N + 1)) - tracker.point_set
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Compute safety scores for all candidates
        # Use a heap for efficient lowest-score-first selection
        import heapq
        heap = []
        for v in all_candidates:
            s = tracker.safety_score(v)
            heapq.heappush(heap, (s, v))
        
        while heap and time.time() < deadline:
            s, v = heapq.heappop(heap)
            if v not in all_candidates:
                continue
            if tracker.can_add(v):
                tracker.add(v)
                all_candidates.discard(v)
                improved = True
                # After adding, some previously safe candidates may become unsafe
                # but we'll re-evaluate in the next outer loop iteration
                break  # Restart with fresh safety scores
    
    return list(tracker.point_set)


def _initial_full_set():
    """Create the full set [0, N]."""
    return list(range(N + 1))


def _initial_random_set(fraction=0.5, seed=None):
    """Create a random initial set of given fraction of [0, N]."""
    if seed is not None:
        random.seed(seed)
    return random.sample(range(N + 1), int(fraction * (N + 1)))


def _full_pipeline(points, deadline):
    """Run greedy removal then add-back search using incremental tracker."""
    primes_to_check = _primes(N + 1)
    
    # Phase 1: Greedy removal to make admissible
    points = _greedy_removal(points, primes_to_check, deadline)
    
    # Phase 2: Add-back search to maximize size
    points = _add_back_search(points, primes_to_check, deadline)
    
    return points


def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    # Get primes up to N+1 for checking
    all_primes = _primes(N + 1)
    
    best = []
    
    # Strategy 1: Start from full set [0, N] and sieve down
    if time.time() < deadline - 10:
        init_full = _initial_full_set()
        result_full = _full_pipeline(init_full, deadline)
        if len(result_full) > len(best):
            best = result_full
    
    # Strategy 2: If we have time, try random restarts with different densities
    attempts = 0
    while time.time() < deadline - 15 and attempts < 5:
        fraction = 0.3 + 0.4 * (attempts % 4) / 4.0
        seed = attempts * 7919 + 12345
        init = _initial_random_set(fraction=fraction, seed=seed)
        result = _full_pipeline(init, deadline)
        if len(result) > len(best):
            best = result
        attempts += 1
    
    # Final: if we have time, do one more add-back on the best
    if time.time() < deadline - 5 and best:
        best = _add_back_search(best, all_primes, deadline)
    
    return best
# EVOLVE-BLOCK-END
