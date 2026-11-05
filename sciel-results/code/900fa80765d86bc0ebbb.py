"""Improved: diverse prime-set candidates + max-impact greedy + local refinement with better heuristics."""
import time
from fractions import Fraction
import numpy as np
import itertools

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
    
    # Sets including 2
    for residue_2 in [0, 1]:
        P = [2]
        current_sum = Fraction(1, 2)
        for p in primes:
            if p == 2: continue
            if current_sum + Fraction(1, p) <= 1:
                P.append(p)
                current_sum += Fraction(1, p)
            else:
                break
        add_set(P)
        
    # Sets including 2 and 3
    for residue_2 in [0, 1]:
        for residue_3 in [0, 1, 2]:
            P = [2, 3]
            current_sum = Fraction(1, 2) + Fraction(1, 3)
            for p in primes:
                if p in [2, 3]: continue
                if current_sum + Fraction(1, p) <= 1:
                    P.append(p)
                    current_sum += Fraction(1, p)
                else:
                    break
            add_set(P)

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

    # ---- Max-impact greedy residue assignment with local refinement ----
    def evaluate_set(P, max_check=MAX_L):
        """Assign residues to cover the longest prefix 1..L using max-impact greedy + local search."""
        if not P:
            return 0, {}
        P_sorted = sorted(P)
        
        covered = np.zeros(max_check + 1, dtype=np.uint8)
        residues = {}
        assigned_primes = set()
        
        # 1. Greedy Pass - pick prime/residue that covers most uncovered numbers
        for n in range(1, max_check + 1):
            if not _time_ok():
                break
            if covered[n]:
                continue
            
            # Find the best unused prime to cover n
            # Use max-impact: count how many uncovered numbers in [n, n+limit] would be covered
            best_p = None
            best_gain = -1
            limit = min(2000, max_check - n)
            
            for p in P_sorted:
                if p in assigned_primes:
                    continue
                r = n % p
                # Count uncovered numbers in [n, n+limit] covered by (p, r)
                idx = np.arange(n, n + limit + 1, p, dtype=np.int64)
                if idx.size:
                    gain = np.sum(covered[idx] == 0)
                else:
                    gain = 0
                # Tie-break: prefer smaller primes (higher density)
                if gain > best_gain or (gain == best_gain and best_p is not None and p < best_p):
                    best_gain = gain
                    best_p = p
            
            if best_p is None:
                break
                
            r = n % best_p
            residues[best_p] = r
            assigned_primes.add(best_p)
            
            # Update coverage
            idx = np.arange(r, max_check + 1, best_p, dtype=np.int64)
            if idx.size:
                covered[idx] = 1

        def get_L():
            if max_check < 1:
                return 0
            seg = covered[1:max_check + 1]
            z = np.nonzero(seg == 0)[0]
            if z.size:
                return int(z[0])
            return max_check

        L = get_L()
        
        # 2. Local Search: Try to extend by shifting residues of large primes
        if L < max_check and _time_ok():
            for _ in range(50):  # up to 50 refinement steps
                if not _time_ok():
                    break
                # Find first gap
                z = np.nonzero(covered[1:L + 1000] == 0)[0]
                if z.size == 0:
                    break
                gap = z[0] + 1
                
                # Try to shift a prime's residue to cover gap
                improved = False
                # Sort primes by size descending (larger primes have less impact)
                for p in sorted(residues.keys(), reverse=True):
                    if p < 50:  # Only try large primes to minimize disruption
                        continue
                    old_r = residues[p]
                    new_r = gap % p
                    if new_r == old_r:
                        continue
                    
                    # Check if shifting uncovers anything in [1, gap-1]
                    # Old coverage: old_r, old_r+p, old_r+2p, ...
                    # New coverage: new_r, new_r+p, new_r+2p, ...
                    # Numbers that lose coverage: those in old but not new, in [1, gap-1]
                    # Numbers that gain coverage: those in new but not old, in [1, gap-1]
                    
                    old_idx = np.arange(old_r, gap, p, dtype=np.int64)
                    new_idx = np.arange(new_r, gap, p, dtype=np.int64)
                    
                    # Check if any number in old_idx is ONLY covered by p
                    safe = True
                    for x in old_idx:
                        if x < 1:
                            continue
                        # Check if x is covered by any other prime
                        covered_by_other = False
                        for q, r2 in residues.items():
                            if q == p:
                                continue
                            if x % q == r2:
                                covered_by_other = True
                                break
                        if not covered_by_other:
                            safe = False
                            break
                    
                    if safe:
                        # Apply the shift
                        residues[p] = new_r
                        # Update coverage
                        # Remove old coverage
                        old_full_idx = np.arange(old_r, max_check + 1, p, dtype=np.int64)
                        # Add new coverage
                        new_full_idx = np.arange(new_r, max_check + 1, p, dtype=np.int64)
                        
                        # For numbers only in old_full_idx, check if still covered
                        # For numbers only in new_full_idx, they're now covered
                        # This is complex; let's just rebuild coverage for this prime
                        # and verify the prefix is still covered
                        
                        # Simple approach: rebuild full coverage and check
                        temp_covered = np.zeros(max_check + 1, dtype=np.uint8)
                        for q, r2 in residues.items():
                            if q == p:
                                idx = np.arange(new_r, max_check + 1, q, dtype=np.int64)
                            else:
                                idx = np.arange(r2, max_check + 1, q, dtype=np.int64)
                            if idx.size:
                                temp_covered[idx] = 1
                        
                        # Check if prefix [1, L] is still covered
                        if np.all(temp_covered[1:L + 1] == 1):
                            covered = temp_covered
                            # Recalculate L
                            z2 = np.nonzero(covered[1:max_check + 1] == 0)[0]
                            new_L = int(z2[0]) if z2.size else max_check
                            if new_L > L:
                                L = new_L
                                improved = True
                                break
                            else:
                                # Revert
                                residues[p] = old_r
                                covered = temp_covered.copy()
                                # Rebuild coverage with old residue
                                for q, r2 in residues.items():
                                    idx = np.arange(r2, max_check + 1, q, dtype=np.int64)
                                    if idx.size:
                                        covered[idx] = 1
                                break
                        else:
                            # Revert
                            residues[p] = old_r
                            covered = temp_covered.copy()
                            for q, r2 in residues.items():
                                idx = np.arange(r2, max_check + 1, q, dtype=np.int64)
                                if idx.size:
                                    covered[idx] = 1
                            break
                
                if not improved:
                    break
        
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
        
        # Additional Refinement: Try to shift residues to cover the first gap
        if _time_ok():
            # Re-evaluate to get current coverage
            L_curr, res_curr = evaluate_set(best_P, max_check=min(MAX_L, best_L + 10000))
            if L_curr > best_L:
                best_L = L_curr
                best_classes = [[p, res_curr[p]] for p in sorted(best_P) if p in res_curr]
            
            # Try to fix the first gap
            # Get coverage array
            covered = np.zeros(min(MAX_L, best_L + 10000) + 1, dtype=np.uint8)
            for p, r in best_classes:
                idx = np.arange(r, min(MAX_L, best_L + 10000) + 1, p, dtype=np.int64)
                if idx.size:
                    covered[idx] = 1
            
            # Find first gap
            z = np.nonzero(covered[1:best_L + 10000] == 0)[0]
            if z.size > 0:
                gap = z[0] + 1
                # Try to change residue of some prime to cover gap
                for p, r in best_classes:
                    new_r = gap % p
                    if new_r == r:
                        continue
                    # Check if changing p to new_r uncovers anything in 1..gap-1
                    # Old coverage by p: r, r+p, ...
                    # New coverage by p: new_r, new_r+p, ...
                    # We need to ensure that any number x < gap that was ONLY covered by p
                    # is still covered by some other prime.
                    # This is expensive to check exactly.
                    # Heuristic: Only try if p is large (small impact on small numbers).
                    if p > 100:
                        # Check if gap is covered by new_r
                        # And check if 1..gap-1 remain covered
                        # This is a simplification.
                        pass

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
