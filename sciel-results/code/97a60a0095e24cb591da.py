"""Rewritten program: backtracking search with forward-checking for divisibility-incomparable subset sums."""

import time
import random
from functools import lru_cache

def construct():
    deadline = time.time() + 80.0  # safety margin before 90s kill
    best_set = [31, 32, 34]  # known valid seed of size 3
    best_size = 3

    # Precompute divisibility: for each value v in 1..256, list of multiples up to 256
    # Actually we need: for a sum s, which other sums t satisfy s|t or t|s
    # We'll maintain the set of all nonempty subset sums and check incrementally.

    def get_subset_sums(elements):
        """Return set of all nonempty subset sums of elements."""
        sums = set()
        for e in elements:
            new_sums = set()
            for s in sums:
                new_sums.add(s + e)
            new_sums.add(e)
            sums |= new_sums
        return sums

    def is_valid_set(elements):
        """Check if all nonempty subset sums are distinct and no two divide each other."""
        sums = get_subset_sums(elements)
        # Check distinctness: the number of nonempty subsets is 2^n - 1
        n = len(elements)
        if len(sums) != (1 << n) - 1:
            return False
        # Check divisibility: no two distinct sums where one divides the other
        sorted_sums = sorted(sums)
        for i, a in enumerate(sorted_sums):
            for j in range(i + 1, len(sorted_sums)):
                b = sorted_sums[j]
                if b % a == 0:
                    return False
        return True

    def try_add(sums_set, candidate):
        """
        Try to add candidate to current set with given subset sums.
        Returns new sums set if valid, else None.
        """
        # New sums = old sums ∪ {candidate} ∪ {s + candidate for s in old sums}
        new_sums = set(sums_set)
        new_sums.add(candidate)
        for s in sums_set:
            new_sums.add(s + candidate)

        # Check distinctness: should have |old| + 2^(n) new sums... 
        # Actually, if any s + candidate == candidate, that means s=0 which isn't in set.
        # If any s1 + candidate == s2 + candidate, that means s1==s2, already distinct.
        # If any s + candidate == t (existing), that's a collision -> invalid.
        # If any s + candidate == candidate + s' where s != s', same thing.
        # So we just need to check no new sum equals an existing sum (other than itself).
        # Actually simpler: check that len(new_sums) == len(sums_set) + 1 + len(sums_set) = 2*len(sums_set) + 1
        # Wait, that's only if all s+candidate are distinct and not in old sums and not equal to candidate.
        
        expected_size = 2 * len(sums_set) + 1
        if len(new_sums) != expected_size:
            return None

        # Check divisibility: for each new sum, check against all existing sums
        # New sums are: candidate, and {s + candidate for s in sums_set}
        # We need to check no pair in new_sums divides each other.
        # Optimized: only check new sums against all, since old sums were already valid.
        new_only = set()
        new_only.add(candidate)
        for s in sums_set:
            new_only.add(s + candidate)
        
        for a in new_only:
            for b in new_sums:
                if a == b:
                    continue
                if a > b:
                    if a % b == 0:
                        return None
                else:
                    if b % a == 0:
                        return None
        return new_sums

    def backtrack(elements, sums_set, remaining_candidates, depth):
        nonlocal best_set, best_size
        if time.time() > deadline:
            return
        if len(elements) > best_size:
            best_set = list(elements)
            best_size = len(elements)
        
        if not remaining_candidates:
            return
        
        # Try each candidate
        for i, cand in enumerate(remaining_candidates):
            if time.time() > deadline:
                return
            new_sums = try_add(sums_set, cand)
            if new_sums is not None:
                backtrack(elements + [cand], new_sums, remaining_candidates[i+1:], depth + 1)
                # If we found a large set, we can stop early in some cases
                if len(elements) + 1 >= best_size and best_size >= 15:
                    return  # don't explore further if already good enough for now

    # Candidate pool: use a range of integers
    # Strategy: try different candidate pools and orderings
    candidate_pool = list(range(2, 257))
    
    # Try multiple random orderings
    rng = random.Random(42)
    attempts = 0
    while time.time() < deadline and attempts < 200:
        attempts += 1
        # Shuffle candidates
        shuffled = candidate_pool[:]
        rng.shuffle(shuffled)
        
        # Start backtracking
        backtrack([], set(), shuffled, 0)
        
        # If we found a large set, try to improve
        if best_size >= 12:
            # Try to extend the best set
            best_elements = best_set[:]
            best_sums = get_subset_sums(best_elements)
            remaining = [c for c in candidate_pool if c not in best_elements]
            rng.shuffle(remaining)
            backtrack(best_elements, best_sums, remaining, 0)
            if best_size >= 15:
                break

    # Final validation
    if not is_valid_set(best_set):
        return [31, 32, 34]
    
    return list(best_set)

if __name__ == "__main__":
    result = construct()
    print(f"Found set of size {len(result)}: {result}")
