"""Combined program: seed repair-loop baseline + set-cover/local-improvement with optimized data structures.

Strategy:
  * Baseline (seed): start from [0, N], repair by deleting the least-populated
    residue class for each prime p <= len(tuple) whose classes are all covered.
  * Refinement: Use a more efficient local improvement with bitsets for tracking
    which elements are in the current tuple, and smarter swap heuristics.
"""
import time
import sys
from typing import List, Set, Dict, Tuple

N = 5000
DEADLINE_MARGIN = 30.0
T_START = time.time()

# Precompute primes up to N+10
_MAX_PRIME_LIMIT = N + 10
_primes_cache: Dict[int, List[int]] = {}

def _get_primes(limit: int) -> List[int]:
    if limit in _primes_cache:
        return _primes_cache[limit]
    if limit > 100 and limit <= _MAX_PRIME_LIMIT:
        if _MAX_PRIME_LIMIT not in _primes_cache:
            sieve = bytearray([1]) * (_MAX_PRIME_LIMIT + 1)
            sieve[0] = sieve[1] = 0
            for p in range(2, int(_MAX_PRIME_LIMIT ** 0.5) + 1):
                if sieve[p]:
                    sieve[p * p: _MAX_PRIME_LIMIT + 1: p] = bytearray(len(range(p * p, _MAX_PRIME_LIMIT + 1, p)))
            _primes_cache[_MAX_PRIME_LIMIT] = [i for i in range(2, _MAX_PRIME_LIMIT + 1) if sieve[i]]
        res = [p for p in _primes_cache[_MAX_PRIME_LIMIT] if p <= limit]
        _primes_cache[limit] = res
        return res
    
    if limit < 2:
        res = []
    else:
        sieve = bytearray([1]) * (limit + 1)
        sieve[0] = sieve[1] = 0
        for p in range(2, int(limit ** 0.5) + 1):
            if sieve[p]:
                sieve[p * p: limit + 1: p] = bytearray(len(range(p * p, limit + 1, p)))
        res = [i for i in range(2, limit + 1) if sieve[i]]
    _primes_cache[limit] = res
    return res

_get_primes(_MAX_PRIME_LIMIT)

def _time_left() -> float:
    return (330.0 - DEADLINE_MARGIN) - (time.time() - T_START)

def _check_admissible(points: List[int]) -> bool:
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

def _greedy_repair(points: List[int]) -> List[int]:
    changed = True
    while changed:
        changed = False
        m = len(points)
        primes = _get_primes(m)
        for p in primes:
            classes: Dict[int, List[int]] = {}
            for v in points:
                r = v % p
                if r not in classes:
                    classes[r] = []
                classes[r].append(v)
            if len(classes) == p:
                smallest = min(classes.values(), key=len)
                to_remove = set(smallest)
                points = [v for v in points if v not in to_remove]
                changed = True
                break
    return points

class AdmissibleState:
    """Efficient tracker for admissibility with fast add/remove operations."""
    
    def __init__(self, points: List[int]):
        self.points = list(points)
        self.point_set = set(self.points)
        self.m = len(self.points)
        self.primes = _get_primes(self.m)
        self.residue_counts: Dict[int, List[int]] = {}
        self.zero_residues: Dict[int, List[int]] = {}
        self._build_counts()
    
    def _build_counts(self):
        self.residue_counts = {}
        self.zero_residues = {}
        for p in self.primes:
            counts = [0] * p
            for v in self.points:
                counts[v % p] += 1
            self.residue_counts[p] = counts
            zeros = [r for r in range(p) if counts[r] == 0]
            self.zero_residues[p] = zeros
    
    def _can_add(self, x: int) -> bool:
        """Check if x can be added without violating admissibility."""
        if x in self.point_set:
            return False
        
        new_m = self.m + 1
        new_primes = _get_primes(new_m)
        
        # Check existing primes
        for p in self.primes:
            if len(self.zero_residues[p]) == 0:
                return False  # Already violating
            r = x % p
            if len(self.zero_residues[p]) == 1:
                if r == self.zero_residues[p][0]:
                    return False  # Would eliminate the only zero residue
        
        # Check new prime if needed
        new_prime_set = set(new_primes)
        old_prime_set = set(self.primes)
        for p in new_prime_set - old_prime_set:
            # p = new_m (if prime)
            if p > self.m:
                # Check if adding x would cover all residues mod p
                seen = [0] * p
                for v in self.points:
                    seen[v % p] = 1
                seen[x % p] = 1
                if all(seen):
                    return False
        
        return True
    
    def add_point(self, x: int) -> bool:
        if not self._can_add(x):
            return False
        
        self.points.append(x)
        self.point_set.add(x)
        self.m += 1
        
        # Update primes
        self.primes = _get_primes(self.m)
        
        # Update counts
        for p in self.primes:
            if p not in self.residue_counts:
                counts = [0] * p
                for v in self.points:
                    counts[v % p] += 1
                self.residue_counts[p] = counts
                self.zero_residues[p] = [r for r in range(p) if counts[r] == 0]
            else:
                r = x % p
                old_count = self.residue_counts[p][r]
                self.residue_counts[p][r] = old_count + 1
                if old_count == 0:
                    # Remove from zero_residues
                    if r in self.zero_residues[p]:
                        self.zero_residues[p].remove(r)
        
        return True
    
    def remove_point(self, x: int) -> bool:
        if x not in self.point_set:
            return False
        
        self.points.remove(x)
        self.point_set.discard(x)
        self.m -= 1
        
        # Update primes
        self.primes = _get_primes(self.m)
        
        # Update counts
        for p in self.primes:
            if p in self.residue_counts:
                r = x % p
                old_count = self.residue_counts[p][r]
                self.residue_counts[p][r] = old_count - 1
                if old_count == 1:
                    # Add to zero_residues
                    if r not in self.zero_residues[p]:
                        self.zero_residues[p].append(r)
        
        return True
    
    def swap(self, y: int, x: int) -> bool:
        """Try to swap y (in points) with x (not in points)."""
        if x in self.point_set:
            return False
        if y not in self.point_set:
            return False
        
        # Temporarily remove y
        self.remove_point(y)
        
        # Try to add x
        if self.add_point(x):
            return True
        else:
            # Revert
            self.add_point(y)
            return False

def _local_improve(points: List[int], removed: List[int]) -> List[int]:
    """Local improvement: try to re-add removed elements and perform swaps."""
    state = AdmissibleState(points)
    removed_set = set(removed)
    improved = True
    
    while improved and _time_left() > 5.0:
        improved = False
        
        # Re-add pass
        for x in list(removed_set):
            if _time_left() < 5.0:
                break
            if state.add_point(x):
                removed_set.discard(x)
                improved = True
        
        # Swap pass with time limit
        if _time_left() < 5.0:
            break
        
        found_swap = False
        # Try swaps: for each x in removed, try to find a y in points to swap with
        for x in list(removed_set):
            if _time_left() < 5.0:
                break
            for y in state.points:
                if _time_left() < 5.0:
                    break
                if state.swap(y, x):
                    removed_set.discard(x)
                    removed_set.add(y)
                    found_swap = True
                    break
            if found_swap:
                improved = True
                break
    
    return state.points

def construct() -> List[int]:
    # Baseline: greedy repair
    points = list(range(N + 1))
    points = _greedy_repair(points)
    best = points if _check_admissible(points) else []
    best_set = set(best)
    removed = [x for x in range(N + 1) if x not in best_set]
    
    # Local improvement
    try:
        if _time_left() > 5.0:
            improved = _local_improve(best, removed)
            if _check_admissible(improved) and len(improved) > len(best):
                best = improved
    except Exception:
        pass
    
    # Final safety check
    if not _check_admissible(best):
        fallback = _greedy_repair(list(range(N + 1)))
        best = fallback if _check_admissible(fallback) else []
    
    return list(best)
