import time
import random
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    random.seed(42)

    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]

    primes = sieve(1000)

    # Strategy from Inspiration 1: Build high-density sets by filling budget with smallest primes
    def build_set(base_primes):
        s = sum(Fraction(1, p) for p in base_primes)
        if s > 1:
            return []
        final_set = list(base_primes)
        for p in primes:
            if p in base_primes:
                continue
            if s + Fraction(1, p) <= 1:
                final_set.append(p)
                s += Fraction(1, p)
        return final_set

    # Generate candidate prime sets. 
    # Bases chosen to allow many small primes to be added, maximizing density.
    candidates = []
    # {2, 5, 11} is a strong base. Sum = 0.5 + 0.2 + 0.09 = 0.79. Remaining 0.21.
    # Can add 13, 17, 19 (sum ~0.188). Next 23 (0.043) too big. 47 (0.021) fits.
    candidates.append(build_set([2, 5, 11]))
    
    # {2, 3, 11}. Sum = 0.5 + 0.33 + 0.09 = 0.92. Remaining 0.08.
    # Can add 13 (0.077). Next 17 (0.059) too big.
    candidates.append(build_set([2, 3, 11]))
    
    # {2, 7, 13}. Sum = 0.5 + 0.142 + 0.077 = 0.719. Remaining 0.281.
    # Can add 17, 19, 23, 29, 31, 37, 41, 43...
    candidates.append(build_set([2, 7, 13]))
    
    # {3, 5, 7}. Sum = 0.33 + 0.2 + 0.142 = 0.672. Remaining 0.328.
    # Can add 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47...
    candidates.append(build_set([3, 5, 7]))
    
    # {5, 7, 11}. Sum = 0.2 + 0.142 + 0.09 = 0.432. Remaining 0.568.
    # Can add 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97...
    candidates.append(build_set([5, 7, 11]))
    
    # {2, 3, 7}. Sum = 0.5 + 0.33 + 0.142 = 0.972. Remaining 0.028.
    # Can add 47 (0.021).
    candidates.append(build_set([2, 3, 7]))

    # Fast coverage check using NumPy
    def compute_L(primes_list, residues, max_check=100000):
        """Compute the largest L such that all 1..L are covered."""
        if not primes_list:
            return 0
        covered = np.zeros(max_check + 1, dtype=bool)
        for p, r in zip(primes_list, residues):
            if r == 0:
                start = p
            else:
                start = r
            if start <= max_check:
                covered[start::p] = True
        # Find first gap
        gaps = np.where(~covered[1:])[0]
        if len(gaps) == 0:
            return max_check
        return gaps[0]

    def greedy_init(primes_list):
        """Greedy assignment: for each number 1, 2, 3..., assign the first unused prime that covers it."""
        P_sorted = sorted(primes_list)
        residues = [0] * len(P_sorted)
        used = set()
        max_check = 100000
        covered = np.zeros(max_check + 1, dtype=bool)
        
        # Map prime to index in P_sorted for quick access
        p_to_idx = {p: i for i, p in enumerate(P_sorted)}

        for n in range(1, max_check + 1):
            if time.time() > deadline - 10:
                break
            if covered[n]:
                continue
            
            assigned = False
            # Try primes in order of size (smaller primes cover more)
            for p in P_sorted:
                if p in used:
                    continue
                r = n % p
                idx = p_to_idx[p]
                residues[idx] = r
                used.add(p)
                start = r if r > 0 else p
                if start <= max_check:
                    covered[start::p] = True
                assigned = True
                break
            if not assigned:
                break

        return residues

    def local_search(primes_list, init_residues, max_iter=1000):
        """
        Local search with repair and kicks.
        1. Find first gap.
        2. Try to move a residue of a prime that covers a neighbor of the gap to cover the gap.
        3. If no improvement, random kick.
        """
        P_sorted = sorted(primes_list)
        residues = list(init_residues)
        current_L = compute_L(P_sorted, residues)
        max_check = 100000
        
        # Precompute which primes cover which numbers? No, too memory intensive.
        # Instead, for a gap at g, we look for primes p where (g - r_p) % p == 0.
        # i.e., r_p == g % p.
        
        for iter_num in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Check for improvement
            new_L = compute_L(P_sorted, residues)
            if new_L > current_L:
                current_L = new_L
                if current_L >= max_check:
                    break
                continue
            
            # Find the first gap
            # We can reuse the covered array from compute_L if we modify it, but for simplicity we recalculate or store it.
            # To save time, let's store the covered array from the last compute_L? 
            # compute_L doesn't return it. Let's make a version that does.
            
            # Optimization: Inline the gap finding to avoid re-computing covered if possible?
            # Actually, compute_L is fast enough with NumPy for 10k iterations if we are careful.
            
            # Find gap
            covered = np.zeros(max_check + 1, dtype=bool)
            for p, r in zip(P_sorted, residues):
                start = r if r > 0 else p
                if start <= max_check:
                    covered[start::p] = True
            
            gaps = np.where(~covered[1:])[0]
            if len(gaps) == 0:
                break
            gap_pos = gaps[0] + 1 # 1-indexed
            
            # Try to fix the gap
            improved = False
            # Candidate primes: those that are not too large, or just all primes
            # For each prime p, the required residue to cover gap_pos is r_req = gap_pos % p
            # If we set residues[i] = r_req, does it cover gap_pos? Yes.
            # Does it break existing coverage? Potentially.
            
            # Heuristic: Try primes in random order or sorted by size
            indices = list(range(len(P_sorted)))
            random.shuffle(indices)
            
            for i in indices:
                p = P_sorted[i]
                old_r = residues[i]
                new_r = gap_pos % p
                
                if new_r == old_r:
                    continue
                
                residues[i] = new_r
                # Check new L
                # We can do a quick check: does it cover gap_pos? Yes.
                # What is the new first gap?
                # Recompute L
                new_L_candidate = compute_L(P_sorted, residues)
                
                if new_L_candidate > current_L:
                    current_L = new_L_candidate
                    improved = True
                    break
                else:
                    residues[i] = old_r
            
            if not improved:
                # Kick: Randomly change 1-3 residues
                num_kicks = random.randint(1, 3)
                for _ in range(num_kicks):
                    i = random.randint(0, len(P_sorted) - 1)
                    p = P_sorted[i]
                    residues[i] = random.randint(0, p - 1)
                
                # After kick, recompute L to see if we escaped a local minimum
                new_L_after_kick = compute_L(P_sorted, residues)
                if new_L_after_kick > current_L:
                    current_L = new_L_after_kick
                # If not improved, we stay at the kicked state and try to repair again next iter
                # This allows exploring new basins

        return current_L, [[p, r] for p, r in zip(P_sorted, residues)]

    best_L = 0
    best_classes = []

    for P in candidates:
        if time.time() > deadline - 20:
            break
        if not P:
            continue

        # 1. Greedy Init
        init_residues = greedy_init(P)
        L_greedy = compute_L(P, init_residues)
        
        if L_greedy > best_L:
            best_L = L_greedy
            best_classes = [[p, r] for p, r in zip(sorted(P), init_residues)]

        # 2. Local Search from Greedy
        if time.time() > deadline - 15:
            continue
        L_ls, classes_ls = local_search(P, init_residues, max_iter=2000)
        if L_ls > best_L:
            best_L = L_ls
            best_classes = classes_ls

        # 3. Random Init + Local Search (to escape global minima)
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            random_residues = [random.randint(0, p - 1) for p in P]
            L_rand, classes_rand = local_search(P, random_residues, max_iter=500)
            if L_rand > best_L:
                best_L = L_rand
                best_classes = classes_rand

    return best_classes
