"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 5000
import time
import random
import heapq
import math

_DEADLINE = 280.0  # reduced deadline to allow safety margin

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
        self.residue_counts = {}
        self.nonzero_classes = {}
        self.empty_class = {} # For active primes (nonzero == p-1)
        self.active_primes = set()
        
        for p in primes:
            counts = [0] * p
            for v in self.point_set:
                counts[v % p] += 1
            self.residue_counts[p] = counts
            nz = sum(1 for c in counts if c > 0)
            self.nonzero_classes[p] = nz
            if nz == p - 1:
                self.active_primes.add(p)
                # Find the empty class
                for r in range(p):
                    if counts[r] == 0:
                        self.empty_class[p] = r
                        break
    
    def is_admissible(self):
        """Check if the current set is admissible."""
        for p in self.primes:
            if p > len(self.point_set):
                continue
            if self.nonzero_classes[p] == p:
                return False
        return True

    def can_add(self, v):
        """Check if v can be added without violating admissibility."""
        for p in self.active_primes:
            if v % p == self.empty_class[p]:
                return False
        return True

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
                nz = self.nonzero_classes[p]
                
                if nz == p - 1:
                    self.active_primes.add(p)
                    for rr in range(p):
                        if counts[rr] == 0:
                            self.empty_class[p] = rr
                            break
            
            counts[r] += 1

    def remove(self, v):
        """Remove a point, updating all prime residue counts."""
        if v not in self.point_set:
            return
        self.point_set.remove(v)
        
        for p in self.primes:
            r = v % p
            counts = self.residue_counts[p]
            
            if counts[r] == 1:
                self.nonzero_classes[p] -= 1
                if p in self.active_primes:
                    self.active_primes.remove(p)
            
            counts[r] -= 1

    def covered_primes(self):
        """Return list of primes where all residue classes are covered."""
        result = []
        for p in self.primes:
            if p > len(self.point_set):
                continue
            if self.nonzero_classes[p] == p:
                result.append(p)
        return result


def _greedy_removal(points, primes_to_check, deadline, strategy="smallest"):
    """Greedy removal using incremental tracker with time checks in inner loop."""
    tracker = _AdmissibleTracker(points, primes_to_check)
    active_set = set(points)
    
    while time.time() < deadline:
        covered = tracker.covered_primes()
        if not covered:
            break
        
        if strategy == "smallest":
            p = min(covered)
        elif strategy == "largest":
            p = max(covered)
        elif strategy == "min_class":
            p = min(covered, key=lambda q: min(tracker.residue_counts[q]))
        elif strategy == "random":
            p = random.choice(covered)
        else:
            p = min(covered)
        
        counts = tracker.residue_counts[p]
        min_class = min(range(p), key=lambda r: counts[r])
        
        # Remove all points in that class
        to_remove = [v for v in active_set if v % p == min_class]
        for v in to_remove:
            if time.time() > deadline:
                break
            tracker.remove(v)
            active_set.discard(v)
    
    return list(active_set)


def _fast_add_back(points, primes_to_check, deadline):
    """Optimized add-back using precomputed valid candidates and incremental filtering."""
    tracker = _AdmissibleTracker(points, primes_to_check)
    active_set = set(points)
    all_candidates = set(range(N + 1)) - active_set
        
    # Precompute forbidden residues for current active primes
    forbidden = {}
    for p in tracker.active_primes:
        forbidden[p] = tracker.empty_class[p]
            
    # Filter candidates based on current forbidden residues
    valid_candidates = []
    for v in all_candidates:
        is_valid = True
        for p, r in forbidden.items():
            if v % p == r:
                is_valid = False
                break
        if is_valid:
            valid_candidates.append(v)
                
    while time.time() < deadline:
        if not valid_candidates:
            # No valid candidates. Try a simple local search: remove 1, add 2?
            # For now, break to avoid infinite loop.
            break
                
        # Pick the best candidate from valid_candidates
        # Heuristic: minimize new active primes created
        best_v = None
        best_score = float('inf')
            
        # Sample if too many to keep it fast
        sample_size = 100
        if len(valid_candidates) > sample_size:
            import random
            random.seed(int(time.time() * 1000) % 100000)
            sample = random.sample(valid_candidates, sample_size)
        else:
            sample = valid_candidates
                
        for v in sample:
            score = 0
            for p in tracker.primes:
                if p > len(active_set) + 1:
                    continue
                r = v % p
                if tracker.residue_counts[p][r] == 0 and tracker.nonzero_classes[p] == p - 2:
                    score += 1
            if score < best_score:
                best_score = score
                best_v = v
            if score == 0:
                break
            
        if best_v is None:
            best_v = valid_candidates[0]
                
        # Add best_v
        tracker.add(best_v)
        active_set.add(best_v)
        all_candidates.discard(best_v)
        valid_candidates.remove(best_v)
            
        # Update forbidden residues
        # Check if any prime became active
        new_forbidden = {}
        for p in tracker.active_primes:
            new_forbidden[p] = tracker.empty_class[p]
                
        # Filter valid_candidates with new forbidden residues
        if new_forbidden != forbidden:
            filtered_valid = []
            for v in valid_candidates:
                is_valid = True
                for p, r in new_forbidden.items():
                    if v % p == r:
                        is_valid = False
                        break
                if is_valid:
                    filtered_valid.append(v)
            valid_candidates = filtered_valid
            forbidden = new_forbidden
                
    return list(active_set)


def _full_pipeline(points, deadline, strategy="smallest"):
    """Run greedy removal then fast add-back."""
    primes_to_check = _primes(N + 1)
    
    # Phase 1: Greedy removal to make admissible
    points = _greedy_removal(points, primes_to_check, deadline, strategy)
    
    # Phase 2: Fast Add-back search
    points = _fast_add_back(points, primes_to_check, deadline)
    
    return points


def _simulate_annealing(initial_points, primes, deadline, max_iterations=500000):
    """
    Simulated annealing with swap moves to maximize admissible tuple size.
    
    State: a subset S of [0, N]
    Move: remove one element from S, add one element not in S
    Objective: maximize |S| subject to admissibility
    """
    point_set = set(initial_points)
    size = len(point_set)
    
    # Precompute primes to check (only those <= max possible size)
    # For admissibility, we need to check primes p <= |S|
    # Max |S| is N+1 = 5001, so primes up to 5001
    
    # Track residue counts for each prime
    # residue_counts[p][r] = number of elements in S with x % p == r
    residue_counts = {}
    for p in primes:
        counts = [0] * p
        for v in point_set:
            counts[v % p] += 1
        residue_counts[p] = counts
    
    # Track which primes are "active" (all p residue classes covered)
    # A prime p is violated if nonzero_classes[p] == p
    nonzero_classes = {}
    for p in primes:
        nonzero_classes[p] = sum(1 for c in residue_counts[p] if c > 0)
    
    def is_admissible():
        """Check if current set is admissible."""
        s_size = len(point_set)
        for p in primes:
            if p > s_size:
                break
            if nonzero_classes[p] == p:
                return False
        return True
    
    def get_violated_primes():
        """Return set of primes that are violated (all classes covered)."""
        s_size = len(point_set)
        violated = set()
        for p in primes:
            if p > s_size:
                break
            if nonzero_classes[p] == p:
                violated.add(p)
        return violated
    
    # Initial state
    if not is_admissible():
        # Need to fix admissibility first
        violated = get_violated_primes()
        while violated and time.time() < deadline - 10:
            p = min(violated)
            # Find the residue class with minimum count
            counts = residue_counts[p]
            min_r = min(range(p), key=lambda r: counts[r])
            # Remove one element from that class
            candidates = [v for v in point_set if v % p == min_r]
            if not candidates:
                break
            v_to_remove = random.choice(candidates)
            # Remove v_to_remove
            point_set.remove(v_to_remove)
            for q in primes:
                r = v_to_remove % q
                old_count = residue_counts[q][r]
                residue_counts[q][r] = old_count - 1
                if old_count == 1:
                    nonzero_classes[q] -= 1
            violated = get_violated_primes()
    
    best_size = size
    best_set = set(point_set)
    
    # Temperature schedule
    t0 = 1.0
    t_end = 0.01
    cooling_rate = 0.99999
    
    current_temp = t0
    iteration = 0
    
    while time.time() < deadline - 15 and iteration < max_iterations:
        iteration += 1
        
        # Propose a move: remove one element, add one element
        if not point_set:
            # Add a random element
            v_to_add = random.randint(0, N)
            if v_to_add not in point_set:
                point_set.add(v_to_add)
                for p in primes:
                    r = v_to_add % p
                    old_count = residue_counts[p][r]
                    residue_counts[p][r] = old_count + 1
                    if old_count == 0:
                        nonzero_classes[p] += 1
            size += 1
            if size > best_size:
                best_size = size
                best_set = set(point_set)
            continue
        
        v_to_remove = random.choice(tuple(point_set))
        v_to_add = random.randint(0, N)
        
        if v_to_add in point_set:
            continue
        
        # Try the swap
        # First remove v_to_remove
        for p in primes:
            r = v_to_remove % p
            old_count = residue_counts[p][r]
            residue_counts[p][r] = old_count - 1
            if old_count == 1:
                nonzero_classes[p] -= 1
        point_set.discard(v_to_remove)
        
        # Then add v_to_add
        for p in primes:
            r = v_to_add % p
            old_count = residue_counts[p][r]
            residue_counts[p][r] = old_count + 1
            if old_count == 0:
                nonzero_classes[p] += 1
        point_set.add(v_to_add)
        
        new_size = len(point_set)
        new_admissible = is_admissible()
        
        # Energy: we want to maximize size, so energy = -size
        # But we also need admissibility
        # Let's use: if admissible, energy = -new_size; else energy = +infinity
        # Better: use a penalty for violations
        violated_count = 0
        if not new_admissible:
            violated = get_violated_primes()
            violated_count = len(violated)
        
        old_admissible = True  # We assume current state is admissible or we're improving
        # Actually, let's track whether current state was admissible
        # For simplicity, let's use: accept if new_size > size and new_admissible
        # Or if new_size == size and new_admissible and current not admissible
        
        # Simpler approach: 
        # - If new state is admissible and size >= current size, accept
        # - If new state is admissible and size < current size, accept with probability exp((new_size - size) / temp)
        # - If new state is not admissible, accept with probability exp(-violated_count / temp)
        
        accept = False
        if new_admissible:
            if new_size >= size:
                accept = True
            else:
                delta = new_size - size  # negative
                if random.random() < math.exp(delta / current_temp):
                    accept = True
        else:
            # Not admissible, accept with low probability
            if random.random() < math.exp(-violated_count * 2 / current_temp):
                accept = True
        
        if not accept:
            # Undo the move
            # Remove v_to_add
            point_set.discard(v_to_add)
            for p in primes:
                r = v_to_add % p
                old_count = residue_counts[p][r]
                residue_counts[p][r] = old_count - 1
                if old_count == 1:
                    nonzero_classes[p] -= 1
            
            # Add back v_to_remove
            point_set.add(v_to_remove)
            for p in primes:
                r = v_to_remove % p
                old_count = residue_counts[p][r]
                residue_counts[p][r] = old_count + 1
                if old_count == 0:
                    nonzero_classes[p] += 1
        
        size = len(point_set)
        
        if size > best_size:
            best_size = size
            best_set = set(point_set)
        
        # Cool down
        current_temp *= cooling_rate
        if current_temp < t_end:
            current_temp = t_end
    
    # Return the best admissible set found
    # Ensure it's admissible
    if is_admissible():
        return list(point_set)
    else:
        return list(best_set)


def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    # Get primes up to N+1 for checking
    all_primes = _primes(N + 1)
    
    best = []
    
    # Strategy 1: Start from full set [0, N] with different removal heuristics
    strategies = ["smallest", "min_class", "largest", "random"]
    
    for strat in strategies:
        if time.time() > deadline - 30:
            break
            
        init_full = list(range(N + 1))
        result = _full_pipeline(init_full, deadline, strat)
        if len(result) > len(best):
            best = result
    
    # Strategy 2: If we have time, try random restarts with different densities
    attempts = 0
    while time.time() < deadline - 60 and attempts < 3:
        fraction = 0.7 + attempts * 0.05
        if fraction > 1.0:
            fraction = 1.0
        seed = attempts * 7919 + 12345
        random.seed(seed)
        init = random.sample(range(N + 1), int(fraction * (N + 1)))
        result = _full_pipeline(init, deadline, "min_class")
        if len(result) > len(best):
            best = result
        attempts += 1
    
    # Strategy 3: Simulated annealing polish on the best result
    if time.time() < deadline - 20 and best:
        sa_result = _simulate_annealing(best, all_primes, deadline)
        if len(sa_result) > len(best):
            best = sa_result
    
    # Final: if we have time, do one more add-back on the best
    if time.time() < deadline - 5 and best:
        best = _fast_add_back(best, all_primes, deadline)
    
    return best
# EVOLVE-BLOCK-END
