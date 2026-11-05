"""Improved: diverse prime-set candidates + vectorized greedy + local refinement with better heuristics."""
import time
from fractions import Fraction
import numpy as np
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    def _time_ok():
        return time.time() < deadline

    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]

    primes = sieve(1000)
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    large_primes = [p for p in primes if p > 23]

    MAX_L = 100000

    # ---- Build candidate prime sets ----
    seen_sets = set()
    candidate_sets = []

    def add_set(P):
        P = sorted(P)
        if not P:
            return
        key = tuple(P)
        if key in seen_sets:
            return
        # budget check
        s = sum(Fraction(1, p) for p in P)
        if s > 1:
            return
        seen_sets.add(key)
        candidate_sets.append(P)

    # Seed: primes >= 13
    seed_set = [p for p in primes if p >= 13]
    add_set(seed_set)

    # Subsets of small primes, fill rest with smallest large primes under budget
    for r in range(len(small_primes) + 1):
        if not _time_ok():
            break
        for subset in itertools.combinations(small_primes, r):
            current_sum = sum(Fraction(1, p) for p in subset)
            if current_sum > 1:
                continue
            current_set = list(subset)
            for p in large_primes:
                if current_sum + Fraction(1, p) <= 1:
                    current_set.append(p)
                    current_sum += Fraction(1, p)
                else:
                    break
            add_set(current_set)

    # Targeted swaps: remove some large primes, add a small prime
    for small in [11, 7, 5, 3, 2]:
        if not _time_ok():
            break
        for remove_list in [[], [13], [17], [19], [13, 17], [13, 19], [17, 19],
                            [13, 17, 19], [13, 17, 19, 23], [23, 29], [13, 17, 23]]:
            s = [p for p in seed_set if p not in remove_list]
            s_sum = sum(Fraction(1, p) for p in s)
            if s_sum + Fraction(1, small) <= 1:
                s.append(small)
                add_set(s)

    # ---- Vectorized greedy residue assignment with better heuristics ----
    def evaluate_set(P, max_check=MAX_L):
        """Assign residues to cover the longest prefix 1..L using vectorized greedy."""
        if not P:
            return 0, {}
        P_sorted = sorted(P)
        
        covered = np.zeros(max_check + 1, dtype=np.uint8)
        residues = {}
        assigned_primes = set()

        # 1. Initial Greedy Pass - pick smallest unused prime for each uncovered n
        for n in range(1, max_check + 1):
            if not _time_ok():
                break
            if covered[n]:
                continue
            
            # Heuristic: Pick the smallest unused prime
            best_p = None
            for p in P_sorted:
                if p not in assigned_primes:
                    best_p = p
                    break
            
            if best_p is None:
                break
                
            r = n % best_p
            residues[best_p] = r
            assigned_primes.add(best_p)
            
            # Update coverage
            idx = np.arange(r, max_check + 1, best_p, dtype=np.int64)
            if idx.size:
                covered[idx] = 1

        # 2. Calculate L
        def get_L():
            if max_check < 1:
                return 0
            seg = covered[1:max_check + 1]
            z = np.nonzero(seg == 0)[0]
            if z.size:
                return int(z[0])
            return max_check

        L = get_L()
        return L, residues

    # Prioritize candidate sets: more small primes first, then more total primes
    def small_prime_count(P):
        return sum(1 for p in P if p <= 23)

    candidate_sets.sort(key=lambda P: (-small_prime_count(P), -len(P)))

    best_L = 0
    best_classes = [[p, (i + 1) % p] for i, p in enumerate(seed_set)]

    for P in candidate_sets:
        if not _time_ok():
            break
        L, residues = evaluate_set(P)
        if L > best_L:
            best_L = L
            best_classes = [[p, residues[p]] for p in sorted(P) if p in residues]

    # ---- Local refinement: try to extend the best prefix by re-optimizing residues ----
    if best_classes and _time_ok():
        best_P = sorted(p for p, r in best_classes)
        # Re-run with a slightly larger check range to see if we can extend
        L2, residues2 = evaluate_set(best_P, max_check=min(MAX_L, best_L + 50000))
        if L2 > best_L:
            best_L = L2
            best_classes = [[p, residues2[p]] for p in sorted(best_P) if p in residues2]

    # ---- Advanced refinement: local search on residues to extend coverage ----
    if best_classes and _time_ok():
        best_P = sorted(p for p, r in best_classes)
        current_L = best_L
        
        # Build coverage array for current assignment
        covered = np.zeros(MAX_L + 1, dtype=np.uint8)
        residues_dict = {p: r for p, r in best_classes}
        for p, r in best_classes:
            idx = np.arange(r, MAX_L + 1, p, dtype=np.int64)
            if idx.size:
                covered[idx] = 1
        
        # Find first gap
        def find_first_gap(covered_arr):
            seg = covered_arr[1:MAX_L + 1]
            z = np.nonzero(seg == 0)[0]
            if z.size:
                return int(z[0])
            return MAX_L
        
        # Try to fix gaps by adjusting residues of primes
        for iteration in range(20):
            if not _time_ok():
                break
            
            gap = find_first_gap(covered)
            if gap >= MAX_L:
                break
                
            # Find primes that could cover this gap
            possible_primes = []
            for p, r in best_classes:
                # Check if changing r to gap % p would cover the gap
                new_r = gap % p
                if new_r != r:
                    possible_primes.append((p, r, new_r))
            
            # Try each possible prime change
            best_improvement = 0
            best_change = None
            for p, old_r, new_r in possible_primes:
                # Simulate the change
                # Remove old coverage
                old_idx = np.arange(old_r, MAX_L + 1, p, dtype=np.int64)
                # Add new coverage
                new_idx = np.arange(new_r, MAX_L + 1, p, dtype=np.int64)
                
                # Count how many new positions are covered
                # Positions covered by new but not by old
                new_set = set(new_idx.tolist())
                old_set = set(old_idx.tolist())
                gain = len(new_set - old_set)
                
                # Also check if this helps cover the specific gap
                if gap in new_set and gap not in old_set:
                    gain += 1000  # Bonus for covering the gap
                
                if gain > best_improvement:
                    best_improvement = gain
                    best_change = (p, old_r, new_r)
            
            if best_change is not None and best_improvement > 0:
                p, old_r, new_r = best_change
                # Apply the change
                for i, (pp, rr) in enumerate(best_classes):
                    if pp == p:
                        best_classes[i] = [pp, new_r]
                        break
                
                # Update coverage
                old_idx = np.arange(old_r, MAX_L + 1, p, dtype=np.int64)
                new_idx = np.arange(new_r, MAX_L + 1, p, dtype=np.int64)
                
                # Recompute coverage from scratch for simplicity
                covered = np.zeros(MAX_L + 1, dtype=np.uint8)
                for pp, rr in best_classes:
                    idx = np.arange(rr, MAX_L + 1, pp, dtype=np.int64)
                    if idx.size:
                        covered[idx] = 1
                
                new_L = find_first_gap(covered)
                if new_L > current_L:
                    current_L = new_L
        
        # Update best_L if improved
        if current_L > best_L:
            best_L = current_L

    # Final verification
    if best_classes:
        final_sum = sum(Fraction(1, p) for p, r in best_classes)
        if final_sum > 1:
            best_classes = [[p, (i + 1) % p] for i, p in enumerate(seed_set)]
        prime_list = [p for p, r in best_classes]
        if len(prime_list) != len(set(prime_list)):
            best_classes = [[p, (i + 1) % p] for i, p in enumerate(seed_set)]

    return best_classes
# EVOLVE-BLOCK-END
