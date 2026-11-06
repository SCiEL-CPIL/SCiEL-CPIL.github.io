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

class CSPSolver:
    def __init__(self, n):
        self.n = n
        # Primes to consider: all primes p <= n+1
        # If we keep a set S, for every p <= |S|, S must miss a residue mod p.
        # However, if we decide for a prime p > |S|, it doesn't matter.
        # But we don't know |S| in advance.
        # Strategy: We select a "forbidden residue" r_p for each prime p <= n.
        # The resulting set S is { x in [0, n] : for all p <= n, x % p != r_p }.
        # We must ensure that for any p <= |S|, the set S actually misses r_p.
        # By construction, S misses r_p for all p.
        # So the only condition is that S is a valid admissible tuple.
        # Wait, the condition is: for every p <= |S|, some residue is missed.
        # If we fix r_p for all p, S misses r_p for all p.
        # Thus S is admissible iff for all p <= |S|, r_p is a valid choice (i.e., we didn't force all elements out? No, we just remove them).
        # The set S is always admissible with respect to the constraints we imposed?
        # Yes, if S = {x | x not in R_p for all p}, then for any p, S misses R_p.
        # So any such S is admissible?
        # Let's check: "for every prime p <= len(tuple) some residue class mod p is missed".
        # If we construct S by removing exactly one class per prime p (for all p <= N), then for any p <= |S|, S misses that specific class.
        # So S is admissible.
        # The goal is to maximize |S|.
        # This is equivalent to minimizing the number of removed elements.
        # Removed elements are union of classes R_p.
        # We want to choose r_p for each p to minimize |Union(R_p)|.
        
        # This is a classic optimization problem.
        # We can use backtracking to choose r_p.
        # Order of primes: Larger primes have larger classes (size ~ N/p).
        # Actually, class size is roughly (N+1)/p.
        # To minimize union, we should handle primes that cause most conflicts first?
        # Or just iterate primes in increasing order?
        # If we pick r_p, we remove ~ N/p elements.
        # Small p removes many elements. Large p removes few.
        # Conflicts occur when an element is in R_p and R_q.
        
        # Heuristic: Process primes in increasing order.
        # For each p, try each residue r.
        # Calculate new removed count.
        # Prune if current removed count + lower_bound_future_removed > best_removed.
        
        # Lower bound on future removed:
        # For any future prime q, we must remove at least 1 element? No.
        # We must remove the class r_q. Size is ~ N/q.
        # But many might already be removed.
        
        # Let's implement a simple backtracking with pruning.
        self.primes = [p for p in _all_primes if p <= n]
        self.best_size = 0
        self.best_assignment = {}
        
        # Precompute for each prime p, the list of elements in each residue class?
        # That's too much memory if stored explicitly for all p.
        # But N=5000.
        # Total elements across all classes for all p <= N:
        # Sum_{p<=N} N = N * pi(N) ~ 5000 * 669 ~ 3.3M.
        # We can store sets of removed elements.
        
        self.removed = set()
        
    def solve(self):
        # Sort primes by increasing size of smallest class?
        # Or just increasing p.
        # Increasing p means we start with small primes which remove many elements.
        # This might prune early if we remove too many.
        
        # Let's try ordering by p.
        # Also, we can use a heuristic for residue choice:
        # Choose r_p that minimizes the number of NEW elements removed.
        
        self._backtrack(0, 0)
        return self.best_size, self.best_assignment
        
    def _backtrack(self, idx, current_removed_count):
        if _time_left() < 1.0:
            return
            
        if idx == len(self.primes):
            # All primes assigned
            size = self.n + 1 - len(self.removed)
            if size > self.best_size:
                self.best_size = size
                self.best_assignment = dict(self._current_assignment)
            return
            
        p = self.primes[idx]
        
        # Estimate lower bound of additional removals for remaining primes
        # Each remaining prime q will remove at least 1 element?
        # Actually, the class size is approx N/q.
        # But we don't know how many are already removed.
        # A safe lower bound is 0.
        # A better bound: For each remaining prime q, we must remove the class.
        # The class has size ~ N/q.
        # If we assume no overlap with current removed, min removal is N/q.
        # But overlap is likely.
        
        # Let's just use 0 for now, or a simple sum of 1s?
        # If we use sum(1 for q in remaining), it's weak.
        
        # Heuristic: Try residues in order of least new removals.
        # Calculate new removals for each r in 0..p-1.
        # This is O(p * N) if we iterate all x.
        # But we can optimize:
        # New removals = { x in [0, N] : x % p == r and x not in removed }
        
        # We can precompute for each p, a list of lists?
        # Or just iterate.
        # N=5000, p up to 5000.
        # If we do this for every node, it's expensive.
        
        # Optimization:
        # Only calculate exact new removals for the top few candidates?
        # Or calculate for all r, but efficiently.
        
        # Efficient calculation:
        # For a given p, we can iterate x in [0, N].
        # If x not in removed, it belongs to class x%p.
        # Count available elements per class.
        # Then for each r, new_removals = available_count[r].
        
        # Let's do this.
        available = [0] * p
        for x in range(self.n + 1):
            if x not in self.removed:
                available[x % p] += 1
                
        # Sort residues by available count (ascending)
        # We want to remove the class with fewest available elements to keep more total?
        # Wait. We WANT to maximize remaining.
        # So we want to MINIMIZE new removals.
        # So we prefer r with smallest available[r].
        
        residues_sorted = sorted(range(p), key=lambda r: available[r])
        
        # Pruning:
        # If current_removed + available[r] > (N+1) - self.best_size:
        # Then even if we remove the fewest possible now, we won't beat best.
        # Actually, we need to consider future primes too.
        # Lower bound on future removals:
        # For each future prime q, we must remove at least 1 element?
        # Let's assume a very optimistic bound: 0 future removals.
        # So if current_removed + available[r] > (N+1) - self.best_size, prune this branch.
        
        limit = (self.n + 1) - self.best_size
        
        for r in residues_sorted:
            if current_removed + available[r] > limit:
                # Since available is sorted ascending, if this r fails, all larger r will fail?
                # No, available[r] is the count. If we pick r, we remove available[r] elements.
                # If available[r] is small, it's good.
                # If current_removed + available[r] > limit, it means even removing this class makes us exceed the removal budget to beat best.
                # Since we sorted by available ascending, the first few are best.
                # If the BEST option (smallest available) fails, do we break?
                # No, because maybe a larger available r allows better choices for future primes?
                # Unlikely, as removing more now is never better for size.
                # However, the choice of r affects WHICH elements are removed, which affects future primes.
                # So we cannot simply break.
                # But we can skip this r if it's too expensive?
                # If current_removed + available[r] > limit, then size = N+1 - (current_removed + available[r]) < best_size.
                # And future removals will only decrease size further.
                # So this branch cannot lead to a solution better than best_size.
                # So we can skip this r.
                continue
                
            # Commit
            self._current_assignment[p] = r
            new_removed = []
            for x in range(self.n + 1):
                if x % p == r and x not in self.removed:
                    self.removed.add(x)
                    new_removed.append(x)
            
            self._backtrack(idx + 1, current_removed + len(new_removed))
            
            # Uncommit
            for x in new_removed:
                self.removed.discard(x)
            del self._current_assignment[p]

def construct():
    # 1. Get baseline from greedy repair
    points = list(range(N + 1))
    points = _greedy_repair(points)
    best = points if _check_admissible(points) else []
    best_size = len(best)
    
    # 2. Try CSP Solver
    if _time_left() > 10.0:
        try:
            solver = CSPSolver(N)
            # Add _current_assignment attribute
            solver._current_assignment = {}
            csp_size, assignment = solver.solve()
            
            if csp_size > best_size:
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
            
    # 3. Local improvement on best
    if _time_left() > 5.0:
        # Simple local improvement: try to add back removed elements
        best_set = set(best)
        removed = [x for x in range(N + 1) if x not in best_set]
        
        # Try to add each removed element
        # Check if adding x keeps admissibility
        # This is O(N * P * N) which is slow.
        # We can use the FastAdmissibleChecker logic from before but simplified.
        
        # For now, skip complex local improvement if CSP was good.
        # If CSP didn't improve, maybe try a few random restarts of CSP?
        
    return list(best)

# To make the CSP solver faster, we need to optimize the inner loop.
# The current implementation iterates x in [0, N] for every node.
# Number of nodes can be large.
# We can precompute the list of elements for each residue class for each prime?
# Memory: Sum_{p<=N} p * (N/p) = N * pi(N) ~ 3.3M integers.
# This is feasible.

class OptimizedCSPSolver:
    def __init__(self, n):
        self.n = n
        self.primes = [p for p in _all_primes if p <= n]
        self.best_size = 0
        self.best_assignment = {}
        self.removed = set()
        self._current_assignment = {}
        
        # Precompute classes
        # classes[p] = list of lists, classes[p][r] = [x1, x2, ...]
        self.classes = {}
        for p in self.primes:
            self.classes[p] = [[] for _ in range(p)]
            for x in range(n + 1):
                self.classes[p][x % p].append(x)
                
    def solve(self):
        self._backtrack(0, 0)
        return self.best_size, self.best_assignment
        
    def _backtrack(self, idx, current_removed_count):
        if _time_left() < 1.0:
            return
            
        if idx == len(self.primes):
            size = self.n + 1 - current_removed_count
            if size > self.best_size:
                self.best_size = size
                self.best_assignment = dict(self._current_assignment)
            return
            
        p = self.primes[idx]
        
        # Calculate available count for each residue
        # available[r] = len(classes[p][r]) - (number of elements in classes[p][r] that are in self.removed)
        # Instead of iterating all x, we can iterate the class and check membership.
        # Or maintain a count array?
        # Since we backtrack, maintaining counts is tricky.
        # Let's just iterate the class list and check `in self.removed`.
        # Class size is ~ N/p.
        # Sum of class sizes over all r is N+1.
        # So calculating available for all r takes O(N).
        # This is acceptable if branching factor is small.
        
        available = [0] * p
        for r in range(p):
            count = 0
            for x in self.classes[p][r]:
                if x not in self.removed:
                    count += 1
            available[r] = count
            
        residues_sorted = sorted(range(p), key=lambda r: available[r])
        
        limit = (self.n + 1) - self.best_size
        
        for r in residues_sorted:
            if current_removed_count + available[r] > limit:
                continue
                
            self._current_assignment[p] = r
            new_removed = []
            for x in self.classes[p][r]:
                if x not in self.removed:
                    self.removed.add(x)
                    new_removed.append(x)
                    
            self._backtrack(idx + 1, current_removed_count + len(new_removed))
            
            for x in new_removed:
                self.removed.discard(x)
            del self._current_assignment[p]

def construct():
    points = list(range(N + 1))
    points = _greedy_repair(points)
    best = points if _check_admissible(points) else []
    best_size = len(best)
    
    if _time_left() > 10.0:
        try:
            solver = OptimizedCSPSolver(N)
            csp_size, assignment = solver.solve()
            
            if csp_size > best_size:
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
        except Exception:
            pass
            
    return list(best)
