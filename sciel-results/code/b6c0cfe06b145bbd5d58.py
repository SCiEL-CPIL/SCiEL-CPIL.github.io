"""Combined: Diverse prime sets + Max-Impact Greedy Residue Assignment + Multi-Heuristic Evaluation."""
import time
from fractions import Fraction
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

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
    small_primes = [2, 3, 5, 7, 11]
    large_primes = [p for p in primes if p > 11]

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
    import itertools
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

    # ---- Evaluation with Max-Impact Greedy ----
    def evaluate_set(P, max_check=MAX_L, heuristic="impact"):
        """
        Assign residues to cover the longest prefix 1..L.
        Heuristic 'impact': For each uncovered n, pick p that covers most currently uncovered numbers.
        Heuristic 'small': Pick smallest unused p.
        Heuristic 'large': Pick largest unused p.
        """
        if not P:
            return 0, {}
        P_sorted = sorted(P)
        
        covered = np.zeros(max_check + 1, dtype=np.uint8)
        residues = {}
        assigned_primes = set()
        unused_primes = set(P)

        # Pre-compute indices for speed? No, dynamic updates make it hard.
        # But np.sum(covered[r::p] == 0) is vectorized and fast enough for small number of iterations.

        for n in range(1, max_check + 1):
            if not _time_ok():
                break
            if covered[n]:
                continue
            
            if not unused_primes:
                break

            best_p = None
            best_score = -1
            best_r = 0

            # Determine candidate primes
            if heuristic == "impact":
                candidates = list(unused_primes)
            elif heuristic == "small":
                candidates = [p for p in P_sorted if p in unused_primes][:1]
            elif heuristic == "large":
                candidates = [p for p in reversed(P_sorted) if p in unused_primes][:1]
            else:
                candidates = list(unused_primes)

            for p in candidates:
                r = n % p
                # Count uncovered numbers in the progression r, r+p, ... up to max_check
                # Note: n is the first element. The progression is n, n+p, n+2p...
                # This is equivalent to indices r, r+p, ... but starting from n?
                # No, r = n % p. The numbers are n, n+p, n+2p...
                # These are congruent to r mod p.
                # The slice covered[r::p] includes r, r+p, ...
                # We need to ensure we only count >= 1.
                # If r == 0, indices are 0, p, 2p...
                # covered[0] is not part of 1..L.
                
                if r == 0:
                    idx = np.arange(p, max_check + 1, p)
                else:
                    idx = np.arange(r, max_check + 1, p)
                
                # Count zeros in covered[idx]
                # But we must only count indices >= 1.
                # If r > 0, all idx are >= 1.
                # If r == 0, idx starts at p (>=2), so all are >= 1.
                
                score = np.sum(covered[idx] == 0)
                
                if score > best_score:
                    best_score = score
                    best_p = p
                    best_r = r

            if best_p is None:
                break

            residues[best_p] = best_r
            assigned_primes.add(best_p)
            unused_primes.discard(best_p)

            # Update coverage
            if best_r == 0:
                idx = np.arange(best_p, max_check + 1, best_p, dtype=np.int64)
            else:
                idx = np.arange(best_r, max_check + 1, best_p, dtype=np.int64)
            
            if idx.size:
                covered[idx] = 1

        # Calculate L
        def get_L():
            if max_check < 1:
                return 0
            seg = covered[1:max_check + 1]
            z = np.nonzero(seg == 0)[0]
            if z.size:
                return int(z[0])
            return max_check

        return get_L(), residues

    # Prioritize candidate sets
    def small_prime_count(P):
        return sum(1 for p in P if p <= 11)

    candidate_sets.sort(key=lambda P: (-small_prime_count(P), -len(P)))

    best_L = 0
    best_classes = [[p, (i + 1) % p] for i, p in enumerate(seed_set)]

    # Evaluate each set with multiple heuristics
    for P in candidate_sets:
        if not _time_ok():
            break
            
        # Try 'impact' heuristic first
        L, residues = evaluate_set(P, max_check=MAX_L, heuristic="impact")
        if L > best_L:
            best_L = L
            best_classes = [[p, residues[p]] for p in sorted(P) if p in residues]
        
        if not _time_ok():
            break

        # Try 'small' heuristic if time permits
        L_small, residues_small = evaluate_set(P, max_check=MAX_L, heuristic="small")
        if L_small > best_L:
            best_L = L_small
            best_classes = [[p, residues_small[p]] for p in sorted(P) if p in residues_small]
            
        if not _time_ok():
            break

        # Try 'large' heuristic if time permits
        L_large, residues_large = evaluate_set(P, max_check=MAX_L, heuristic="large")
        if L_large > best_L:
            best_L = L_large
            best_classes = [[p, residues_large[p]] for p in sorted(P) if p in residues_large]

    # ---- Local Refinement / Polish ----
    # If we have time, try to extend the best solution by re-evaluating with a higher cap
    # or trying to fix the first gap.
    if best_classes and _time_ok():
        best_P = sorted(p for p, r in best_classes)
        # Re-run with 'impact' on the best set, but maybe with a slightly different tie-breaker?
        # Or just verify.
        # Let's try one more pass: "Greedy with Lookahead 1"
        # If n is uncovered, pick p. But also check if n+1 is uncovered.
        # The 'impact' heuristic already does a form of lookahead by counting future coverage.
        
        # Let's try a "Swap" refinement:
        # Take the best assignment. Find the first gap g.
        # Can we change the residue of some prime p to cover g without uncovering 1..g-1?
        # This is complex. We'll stick to re-running the greedy with a different random seed/order if needed,
        # but since we don't have randomness, we'll just re-run 'impact' with a higher max_check 
        # to ensure we didn't stop early due to a temporary gap that could be filled by a later prime?
        # No, the greedy assigns primes as it goes. It doesn't "reserve" primes.
        
        # One simple improvement: If the first gap is at G, and we have unused primes, 
        # the greedy would have used them. If no unused primes, we are stuck.
        
        # Let's just ensure we return the best found so far.
        
        pass

    # Final verification
    if best_classes:
        final_sum = sum(Fraction(1, p) for p, r in best_classes)
        if final_sum > 1:
            # Fallback to seed
            best_classes = [[p, (i + 1) % p] for i, p in enumerate(seed_set)]
        prime_list = [p for p, r in best_classes]
        if len(prime_list) != len(set(prime_list)):
            best_classes = [[p, (i + 1) % p] for i, p in enumerate(seed_set)]

    return best_classes
# EVOLVE-BLOCK-END
