import time
import sys
import random

N = 5000
DEADLINE_MARGIN = 5.0
T_START = time.time()

# Precompute primes up to N+10
_MAX_PRIME_LIMIT = N + 10
_primes_cache = {}

def _get_primes(limit):
    if limit in _primes_cache:
        return _primes_cache[limit]
    if limit > 100 and limit <= _MAX_PRIME_LIMIT:
        if _MAX_PRIME_LIMIT not in _primes_cache:
            sieve = bytearray([1]) * (_MAX_PRIME_LIMIT + 1)
            sieve[0] = sieve[1] = 0
            for p in range(2, int(_MAX_PRIME_LIMIT ** 0.5) + 1):
                if sieve[p]:
                    sieve[p * p : _MAX_PRIME_LIMIT + 1 : p] = bytearray(len(sieve[p * p : _MAX_PRIME_LIMIT + 1 : p]))
            _primes_cache[_MAX_PRIME_LIMIT] = [i for i in range(2, _MAX_PRIME_LIMIT + 1) if sieve[i]]
        res = [p for p in _primes_cache[_MAX_PRIME_LIMIT] if p <= limit]
        _primes_cache[limit] = res
        return res
    else:
        if limit < 2:
            res = []
        else:
            sieve = bytearray([1]) * (limit + 1)
            sieve[0] = sieve[1] = 0
            for p in range(2, int(limit ** 0.5) + 1):
                if sieve[p]:
                    sieve[p * p : limit + 1 : p] = bytearray(len(sieve[p * p : limit + 1 : p]))
            res = [i for i in range(2, limit + 1) if sieve[i]]
        _primes_cache[limit] = res
        return res

# Initialize prime cache
_all_primes = _get_primes(_MAX_PRIME_LIMIT)

def _time_left():
    return (330.0 - DEADLINE_MARGIN) - (time.time() - T_START)

def _check_admissible(points):
    if not points:
        return True
    m = len(points)
    primes = _get_primes(m)
    for p in primes:
        seen = [0] * p
        for v in points:
            seen[v % p] = 1
        if all(seen):
            return False
    return True

def _greedy_repair(points):
    """Seed repair: delete least-populated class for each violating prime."""
    changed = True
    while changed:
        changed = False
        m = len(points)
        primes = _get_primes(m)
        for p in primes:
            classes = {}
            for v in points:
                classes.setdefault(v % p, []).append(v)
            if len(classes) == p:
                smallest = min(classes.values(), key=len)
                to_remove = set(smallest)
                points = [v for v in points if v not in to_remove]
                changed = True
                break
    return points

class SimulatedAnnealingSolver:
    def __init__(self, n):
        self.n = n
        # We consider primes p <= N.
        # For each prime p, we assign a residue r_p.
        # The set S is {x in [0, N] : for all p, x % p != r_p}.
        # We want to maximize |S|.
        # This is equivalent to minimizing |Removed|.
        # Removed = Union of classes C_{p, r_p}.
        
        self.primes = [p for p in _all_primes if p <= n]
        self.num_primes = len(self.primes)
        
        # cover_count[x] = number of primes p such that x % p == r_p
        self.cover_count = [0] * (n + 1)
        self.removed_count = 0
        
        # Initial assignment: random residues
        self.r = {}
        for p in self.primes:
            self.r[p] = random.randint(0, p - 1)
            # Update cover counts
            for x in range(self.r[p], n + 1, p):
                self.cover_count[x] += 1
                if self.cover_count[x] == 1:
                    self.removed_count += 1
                    
        self.best_removed = self.removed_count
        self.best_r = dict(self.r)
        
    def _calculate_size(self):
        return self.n + 1 - self.removed_count

    def solve(self):
        T = 50.0
        cooling = 0.9999
        min_T = 0.01
        
        while _time_left() > 1.0 and T > min_T:
            # Pick a random prime to change
            idx = random.randint(0, self.num_primes - 1)
            p = self.primes[idx]
            old_r = self.r[p]
            
            # Pick a new random residue
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                continue
                
            # Calculate delta in removed_count
            # We remove class old_r and add class new_r.
            # For x in class old_r: cover_count[x] decreases by 1.
            #   If it becomes 0, removed_count decreases by 1.
            # For x in class new_r: cover_count[x] increases by 1.
            #   If it was 0, removed_count increases by 1.
            
            delta = 0
            # Process removal of old_r
            for x in range(old_r, self.n + 1, p):
                self.cover_count[x] -= 1
                if self.cover_count[x] == 0:
                    delta -= 1
                    
            # Process addition of new_r
            for x in range(new_r, self.n + 1, p):
                if self.cover_count[x] == 0:
                    delta += 1
                self.cover_count[x] += 1
                
            new_removed = self.removed_count + delta
            
            # Acceptance criterion
            if new_removed < self.removed_count:
                # Better solution
                self.removed_count = new_removed
                self.r[p] = new_r
                if self.removed_count < self.best_removed:
                    self.best_removed = self.removed_count
                    self.best_r = dict(self.r)
            else:
                # Worse or equal
                # Probability of accepting worse solution
                # delta_energy = new_removed - old_removed = delta
                # We want to minimize removed, so energy = removed_count.
                # delta_E = delta
                # Accept with prob exp(-delta_E / T)
                if delta > 0:
                    prob = 1.0 / (1.0 + exp(delta / T)) # Logistic or Exp? Standard SA uses exp(-dE/T)
                    # Actually standard is exp(-dE/T). If dE > 0, exp(-dE/T) < 1.
                    # Let's use standard exp.
                    # But exp can underflow.
                    # Use math.exp
                    import math
                    try:
                        prob = math.exp(-delta / T)
                    except OverflowError:
                        prob = 0.0
                    
                    if random.random() < prob:
                        self.removed_count = new_removed
                        self.r[p] = new_r
                    else:
                        # Revert changes
                        # Add back old_r
                        for x in range(old_r, self.n + 1, p):
                            self.cover_count[x] += 1
                        # Remove new_r
                        for x in range(new_r, self.n + 1, p):
                            self.cover_count[x] -= 1
                else:
                    # delta == 0, accept
                    self.removed_count = new_removed
                    self.r[p] = new_r
                    if self.removed_count < self.best_removed:
                        self.best_removed = self.removed_count
                        self.best_r = dict(self.r)
                        
            T *= cooling
            
        return self.n + 1 - self.best_removed, self.best_r

def construct():
    # 1. Get baseline from greedy repair
    points = list(range(N + 1))
    points = _greedy_repair(points)
    best = points if _check_admissible(points) else []
    best_size = len(best)
    
    # 2. Try Simulated Annealing
    if _time_left() > 10.0:
        try:
            solver = SimulatedAnnealingSolver(N)
            sa_size, assignment = solver.solve()
            
            if sa_size > best_size:
                # Reconstruct set from assignment
                # S = { x in [0, N] : for all p in assignment, x % p != assignment[p] }
                new_points = []
                for x in range(N + 1):
                    valid = True
                    for p, r in assignment.items():
                        if x % p == r:
                            valid = False
                            break
                    if valid:
                        new_points.append(x)
                
                if _check_admissible(new_points):
                    best = new_points
                    best_size = len(best)
        except Exception as e:
            pass
            
    return list(best)
