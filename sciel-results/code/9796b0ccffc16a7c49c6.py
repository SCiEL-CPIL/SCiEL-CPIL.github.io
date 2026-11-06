import time
import sys
import random
import heapq

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

class ConflictGraphMIS:
    def __init__(self, n):
        self.n = n
        self.nodes = list(range(n + 1))
        self.m = n + 1
        # Precompute residue counts for each prime p <= m
        # counts[p][r] = number of x in [0, n] such that x % p == r
        self.primes = [p for p in _all_primes if p <= self.m]
        self.residue_counts = {}
        for p in self.primes:
            counts = [0] * p
            for x in range(self.n + 1):
                counts[x % p] += 1
            self.residue_counts[p] = counts
            
        # We maintain current counts of available nodes per residue class
        # available_counts[p][r]
        self.available_counts = {}
        for p in self.primes:
            self.available_counts[p] = self.residue_counts[p][:]
            
        self.in_set = [False] * self.m
        self.removed = [False] * self.m # True if removed from consideration (not in set)
        
    def solve(self):
        # Greedy MIS with dynamic ordering
        # Heuristic: At each step, pick the node that "conflicts" the least with future requirements?
        # Or simply: Pick node with smallest "cost" (impact on available counts)?
        # Actually, standard Greedy Coloring for MIS:
        # 1. Pick a vertex v.
        # 2. Add v to independent set.
        # 3. Remove v and all its neighbors from the graph.
        
        # Here, "neighbors" of x are all y such that {x, y} completes a residue class?
        # No, the condition is: for every p <= |S|, S misses a residue.
        # This is not a standard pairwise conflict graph MIS.
        # The conflict is global: if we include x, it "uses up" residue x%p.
        # If we include enough elements to cover all residues mod p, we violate admissibility.
        # So we must stop adding elements that complete a residue class for any p <= current_size.
        
        # Let's simulate the process of adding elements.
        # We want to maximize size.
        # We can try adding elements in a specific order.
        # If we add x, we update counts.
        # If for any p <= current_size, count[p][x%p] becomes 0 (meaning all other residues are filled? No, we track available).
        # Let's track "filled" counts.
        # filled[p][r] = number of elements in S with x % p == r.
        # Condition for admissibility: for all p <= |S|, there exists r such that filled[p][r] == 0.
        
        # Strategy:
        # 1. Order nodes.
        # 2. Iterate through nodes.
        # 3. If adding node x keeps the set admissible, add it.
        
        # How to check admissibility quickly?
        # Maintain for each p <= m, the number of residues with count 0.
        # zero_residues[p] = number of r in 0..p-1 such that filled[p][r] == 0.
        # Admissible iff for all p <= |S|, zero_residues[p] > 0.
        
        # When we add x:
        # For each p <= m:
        #   r = x % p
        #   filled[p][r] += 1
        #   if filled[p][r] == 1: zero_residues[p] -= 1
        #   if zero_residues[p] == 0 and p <= new_size:
        #       # Conflict! We must revert.
        
        # This allows O(P) check per addition, where P is number of primes <= m.
        # P ~ 600.
        # Total operations ~ N * P = 5000 * 600 = 3M. Very fast.
        
        # What order?
        # Try different orderings:
        # 1. Random
        # 2. Increasing x
        # 3. Decreasing x
        # 4. "Sparse" first?
        
        # Let's run multiple random restarts.
        
        best_size = 0
        best_set = []
        
        start_time = time.time()
        max_time = _time_left() - 2.0
        if max_time < 1.0:
            return best_size, best_set
            
        # We will run greedy trials until time runs out.
        # Each trial is O(N * P).
        
        while time.time() - start_time < max_time:
            # Random shuffle
            nodes = list(range(self.n + 1))
            random.shuffle(nodes)
            
            S = []
            filled = {}
            zero_residues = {}
            
            # Initialize filled and zero_residues
            for p in self.primes:
                filled[p] = [0] * p
                zero_residues[p] = p
            
            for x in nodes:
                # Check if adding x is safe
                # We need to check if it causes any p <= len(S)+1 to have zero_residues[p] == 0
                # Note: p must be <= new_size = len(S) + 1
                
                # Optimization: Only check p <= len(S) + 1
                # But we need to update filled for all p?
                # No, we only care about p <= new_size.
                # However, filled[p] for p > new_size might be needed later?
                # Yes, because new_size increases.
                # So we must update filled[p] for all p.
                
                # But we can defer the "check" until necessary?
                # No, we need to know if it's safe before committing.
                
                # Let's simulate the update for all p, and track if any p <= new_size becomes 0.
                
                new_size = len(S) + 1
                safe = True
                updates = [] # (p, r, old_zero)
                
                for p in self.primes:
                    if p > self.m: break
                    r = x % p
                    old_filled = filled[p][r]
                    new_filled = old_filled + 1
                    filled[p][r] = new_filled
                    updates.append((p, r, old_filled))
                    
                    if old_filled == 0:
                        zero_residues[p] -= 1
                        if p <= new_size and zero_residues[p] == 0:
                            safe = False
                            # We can break early?
                            # But we need to revert updates.
                            # Let's collect all updates first, or revert immediately if we break?
                            # Reverting immediately is safer.
                            pass
                            
                if safe:
                    S.append(x)
                else:
                    # Revert
                    for p, r, old_filled in updates:
                        filled[p][r] = old_filled
                        if old_filled == 0:
                            zero_residues[p] += 1
                    # Note: We updated filled for all primes, even those > new_size.
                    # This is necessary for future steps where new_size increases.
                    
            if len(S) > best_size:
                best_size = len(S)
                best_set = S[:]
                
        return best_size, best_set

def construct():
    # 1. Get baseline from greedy repair
    points = list(range(N + 1))
    points = _greedy_repair(points)
    best = points if _check_admissible(points) else []
    best_size = len(best)
    
    # 2. Try Graph MIS approach
    if _time_left() > 10.0:
        try:
            solver = ConflictGraphMIS(N)
            mis_size, mis_set = solver.solve()
            
            if mis_size > best_size:
                # Verify
                if _check_admissible(mis_set):
                    best = mis_set
                    best_size = mis_size
        except Exception:
            pass
            
    return list(best)
