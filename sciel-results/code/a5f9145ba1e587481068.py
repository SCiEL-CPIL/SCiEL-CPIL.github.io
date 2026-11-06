"""
Strategy: Global Greedy Removal via Set Cover Approximation.
Instead of adding elements to a set S, we construct the complement set E (removed elements).
Constraint: For every prime p <= |S|, S must miss at least one residue class mod p.
This is equivalent to: For every prime p <= |S|, E must contain at least one complete residue class mod p.
We want to minimize |E| to maximize |S| = N + 1 - |E|.

Algorithm:
1. Start with E = empty set.
2. Iterate:
   a. Determine current k = N + 1 - |E|.
   b. Identify primes p <= k.
   c. For each such prime p, check if E already contains a full residue class mod p.
      If not, we MUST add one.
   d. To minimize growth of E, for each p not yet satisfied, find the residue r that minimizes 
      the number of NEW elements added to E (i.e., elements in AP(r, p) not already in E).
   e. Among all unsatisfied primes, pick the one with the minimum cost to satisfy.
   f. Add that residue class to E.
   g. If no unsatisfied primes, we are done (or check if k changed enough to drop large primes).
   h. Repeat until stable or time limit.
   
   Refinement: Sometimes satisfying a prime p forces us to satisfy larger primes indirectly? No, 
   satisfying p just adds elements. It might help satisfy larger primes q if the APs overlap.
   We should prioritize primes that are "hard" to satisfy (high cost) or small primes which are 
   mandatory for any large k.
   
   Actually, a better heuristic: 
   - Maintain a list of "active" primes p <= k.
   - For each active prime, compute the cost of satisfying it (min new elements).
   - If cost is 0, it's already satisfied.
   - Pick the prime with the smallest positive cost? Or the largest prime?
   - Usually small primes are the bottleneck. 
   - Let's always satisfy the smallest unsatisfied prime first, as it is the most restrictive 
     for the existence of large k. If we can't satisfy small primes cheaply, k must be small.
     But wait, if k is small, we don't need to satisfy large primes.
     So, we should dynamically adjust k.
     
   Refined Loop:
   - Start with k = N+1.
   - While True:
     - Let P_active = {p in Primes | p <= k}
     - For each p in P_active, determine if E satisfies p.
     - If all satisfied, break (we found a valid E for this k).
     - Else, find the p in P_active that is NOT satisfied.
     - Among unsatisfied p, pick the one that minimizes the cost of adding its best AP.
     - Add that AP to E.
     - Update k = N + 1 - |E|.
     - If k has decreased, some large primes might drop out of P_active, which is good.
     
   This greedy approach is deterministic and avoids local minima of random walks.
"""

import time
import random
import numpy as np

N = 5000

# EVOLVE-BLOCK-START
def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def construct():
    deadline = time.time() + 290
    primes = _primes(N + 1)
    
    # We will run multiple restarts with slightly different tie-breaking heuristics
    # to escape potential greedy traps, though the deterministic nature is strong.
    # For now, we implement the core greedy removal.
    
    best_E = None
    best_k = 0
    
    # Precompute APs for all primes? 
    # APs for p=2: 2500 elements each.
    # APs for p=3: ~1667 elements each.
    # Storing these as sets might be memory heavy but N=5000 is small.
    # Total elements in all APs for p<=5000 is sum_{p<=5000} p * (N/p) ~ N * pi(N) ~ 5000 * 669 ~ 3.3M.
    # This is manageable.
    
    # Let's precompute the elements for each residue class of each prime.
    # ap_elements[p][r] = set of x in [0, N] such that x % p == r
    ap_elements = {}
    for p in primes:
        if p > N + 1:
            break
        classes = [set() for _ in range(p)]
        for x in range(N + 1):
            classes[x % p].add(x)
        ap_elements[p] = classes
        
    def calculate_cost(p, E_set):
        """
        Returns the minimum number of new elements added to E if we pick the best residue class for p.
        Also returns the best residue class.
        """
        min_new = float('inf')
        best_r = -1
        classes = ap_elements[p]
        for r in range(p):
            # new_elements = classes[r] - E_set
            # To speed up, we can check intersection size
            # len(classes[r] - E_set) = len(classes[r]) - len(classes[r] & E_set)
            # Since len(classes[r]) is fixed for a given p, we just need max intersection.
            inter_len = len(classes[r] & E_set)
            new_len = len(classes[r]) - inter_len
            if new_len < min_new:
                min_new = new_len
                best_r = r
        return min_new, best_r

    # We will try a few variations of the greedy choice:
    # 1. Always pick the smallest unsatisfied prime.
    # 2. Pick the unsatisfied prime with the lowest cost.
    # 3. Pick the unsatisfied prime with the highest cost (worst-case avoidance? No, we must satisfy it).
    
    strategies = [
        "smallest_prime",
        "lowest_cost",
        "random_tiebreak_smallest"
    ]
    
    for strat in strategies:
        E_set = set()
        k = N + 1
        
        # Optimization: Use a boolean array for E for fast intersection checks?
        # E_arr[x] = 1 if x in E.
        # Intersection length can be computed via sum(E_arr[x] for x in classes[r])
        # This is O(size of class).
        
        E_arr = np.zeros(N + 1, dtype=np.int8)
        
        # To speed up cost calculation, we can maintain counts of how many elements of each AP are in E.
        # ap_count[p][r] = number of x in classes[p][r] that are in E.
        ap_count = {}
        for p in primes:
            if p > N + 1:
                break
            ap_count[p] = np.zeros(p, dtype=np.int32)
            
        # Initial state: E is empty, so ap_count is all 0.
        
        while time.time() < deadline - 10:
            # Determine active primes
            # Primes p <= k
            active_primes = [p for p in primes if p <= k]
            
            unsatisfied = []
            for p in active_primes:
                # Check if any residue class is fully covered by E
                # i.e., ap_count[p][r] == len(ap_elements[p][r])
                # Note: len(ap_elements[p][r]) varies slightly (floor/ceil), but we can precompute sizes.
                # Actually, if ap_count[p][r] == size, it's satisfied.
                
                # To avoid checking all p every time, we can just check.
                # N=5000, primes ~ 669.
                # Checking 669 primes * 5000 steps is fine.
                
                satisfied = False
                for r in range(p):
                    if ap_count[p][r] == len(ap_elements[p][r]):
                        satisfied = True
                        break
                if not satisfied:
                    # Calculate cost
                    min_new, best_r = calculate_cost(p, E_set)
                    # Optimize calculate_cost using ap_count
                    # min_new = min(len(ap_elements[p][r]) - ap_count[p][r])
                    min_new_opt = float('inf')
                    best_r_opt = -1
                    for r in range(p):
                        new = len(ap_elements[p][r]) - ap_count[p][r]
                        if new < min_new_opt:
                            min_new_opt = new
                            best_r_opt = r
                    unsatisfied.append((p, min_new_opt, best_r_opt))
            
            if not unsatisfied:
                # All active primes satisfied. We have a valid set.
                break
            
            # Choose which prime to satisfy
            if strat == "smallest_prime":
                # Pick smallest p
                unsatisfied.sort(key=lambda x: x[0])
                p_to_satisfy, cost, r_to_add = unsatisfied[0]
            elif strat == "lowest_cost":
                # Pick lowest cost
                unsatisfied.sort(key=lambda x: x[1])
                p_to_satisfy, cost, r_to_add = unsatisfied[0]
            else:
                # Random tiebreak among smallest primes?
                # Just pick smallest for now
                unsatisfied.sort(key=lambda x: x[0])
                p_to_satisfy, cost, r_to_add = unsatisfied[0]
            
            # Add the residue class to E
            # Update E_set, E_arr, ap_count
            elements_to_add = ap_elements[p_to_satisfy][r_to_add]
            for x in elements_to_add:
                if E_arr[x] == 0:
                    E_arr[x] = 1
                    E_set.add(x)
                    # Update ap_count for all primes q that have x in their classes
                    # This is expensive if done naively.
                    # We need to update ap_count[q][x%q] for all q <= N+1.
                    # There are ~669 primes.
                    # For each x added, we do 669 updates.
                    # If we add ~1000 elements, that's 669,000 updates.
                    # This is fast in Python? Maybe.
                    # Let's optimize: Only update ap_count for primes q where q <= N+1.
                    for q in primes:
                        if q > N + 1:
                            break
                        # x is in class x%q of q
                        ap_count[q][x % q] += 1
            
            k = N + 1 - len(E_set)
            
            # If k is very small, we might stop early if we can't improve?
            # But we want the LARGEST tuple, so we want MINIMAL E.
            # This greedy process minimizes E.
            
        # End of one strategy run
        current_k = N + 1 - len(E_set)
        if current_k > best_k:
            best_k = current_k
            best_E = E_set.copy()
        
    # Final check: Is best_E valid?
    # The greedy ensures validity for the k it ends up with.
    # But we should verify.
    
    if best_E is None:
        # Fallback to empty set (S = all elements) - likely invalid for N=5000
        # But for small N it might work.
        return list(range(N + 1))
        
    S = [x for x in range(N + 1) if x not in best_E]
    
    # Verify admissibility
    k = len(S)
    S_set = set(S)
    valid = True
    for p in primes:
        if p > k:
            break
        residues_seen = set()
        for x in S_set:
            residues_seen.add(x % p)
        if len(residues_seen) == p:
            valid = False
            break
            
    if valid:
        return S
    else:
        # If invalid, return a known valid small set or empty?
        # Better to return the result from a simpler heuristic if this fails.
        # But the greedy should be valid.
        # If it's invalid, it means my logic "E contains full AP" is sufficient but maybe not necessary?
        # No, "S misses r mod p" <=> "All x in [0,N] with x% p == r are in E".
        # So E MUST contain a full AP.
        # My greedy ensures this.
        # Why would it be invalid?
        # Maybe k changed during the loop?
        # If k decreases, we might have satisfied a prime p > k_new, which is fine.
        # But we might have FAILED to satisfy a prime p <= k_new?
        # No, we only stop when ALL p <= k are satisfied.
        # If k decreases, the set of required primes shrinks.
        # So if we satisfied the larger set, we satisfy the smaller set.
        # So it should be valid.
        
        # Just in case, return a greedy removal from full set (seed-like)
        points = set(range(N + 1))
        for p in primes:
            if p > len(points):
                break
            residues = [set() for _ in range(p)]
            for x in points:
                residues[x % p].add(x)
            if all(len(r) > 0 for r in residues):
                min_idx = min(range(p), key=lambda i: len(residues[i]))
                points -= residues[min_idx]
        return list(points)

# EVOLVE-BLOCK-END
