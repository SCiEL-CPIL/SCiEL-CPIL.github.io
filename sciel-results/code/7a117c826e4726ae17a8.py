import time
from fractions import Fraction
import numpy as np

# EVOLVE-BLOCK-START

def _sieve(n):
    if n < 2:
        return []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    primes = _sieve(1000)
    N = 100000  # Cap for L
    
    # Precompute coverage masks as bitsets (Python integers) for fast OR operations
    # mask[p][r] has bit i set if i % p == r and 1 <= i <= N
    masks = {}
    for p in primes:
        masks[p] = []
        for r in range(p):
            m = 0
            start = r if r > 0 else p
            if start <= N:
                val = 0
                for i in range(start, N + 1, p):
                    val |= (1 << i)
                masks[p].append(val)
            else:
                masks[p].append(0)
        
    # Helper to get L from a combined mask
    def get_L_from_mask(mask):
        if mask == 0:
            return 0
        x = mask >> 1
        mask_limit = (1 << N) - 1
        x = x & mask_limit
        if x == 0:
            return 0
        y = (~x) & mask_limit
        if y == 0:
            return N
        pos = (y & -y).bit_length() - 1
        return pos

    # Strategy: Backtracking Search with Pruning
    # We want to maximize L.
    # State: (current_mask, current_L, current_budget, used_primes_set)
    # We choose the next prime to add.
    
    # Heuristic for choosing the next prime:
    # 1. Small primes are generally better because they cover more numbers per unit budget? 
    #    Actually, 1/p is the cost. Small p -> high cost, high coverage density.
    #    Large p -> low cost, low coverage density.
    #    To maximize L quickly, we likely want small primes.
    # 2. However, we must fit within budget 1.
    
    # Let's pre-select a candidate set of primes.
    # Since sum(1/p) diverges, we can take many primes.
    # But adding a prime costs 1/p.
    # Let's try to include primes up to a certain limit.
    # If we take primes 2..97, sum 1/p is approx 2.5. Too much.
    # Sum 1/p for p<=43 is approx 1.7.
    # Sum 1/p for p<=31 is approx 1.5.
    # Sum 1/p for p<=23 is approx 1.3.
    # Sum 1/p for p<=19 is approx 1.1.
    # Sum 1/p for p<=17 is approx 1.0.
    
    # Wait, the budget is 1.
    # Let's calculate exact sums for small primes.
    # 1/2 + 1/3 + 1/5 + 1/7 + 1/11 + 1/13 + 1/17 + 1/19
    # 0.5 + 0.333 + 0.2 + 0.142 + 0.09 + 0.076 + 0.058 + 0.052 = ~1.35
    # So we can't even take all primes up to 19.
    
    # We need to select a subset of primes such that sum(1/p) <= 1.
    # This is a knapsack-like problem but with the objective of maximizing coverage L.
    
    # Let's define the search space:
    # We iterate through primes in increasing order.
    # At each step, we decide: include this prime (and choose a residue) or skip it.
    # If we include it, we must choose a residue r.
    # There are p choices for r.
    # To prune:
    # - If budget < 1/p, we cannot include p.
    # - If we include p, we update mask and L.
    # - We want to explore residues that maximize L.
    
    # Optimization:
    # Instead of branching on "skip" vs "include" for every prime, we can just branch on residues for a fixed set of primes?
    # No, the set of primes matters.
    
    # Let's use a recursive search:
    # def search(idx, mask, L, budget):
    #   if time up: return
    #   if idx == len(primes): update best, return
    #   p = primes[idx]
    #   
    #   # Option 1: Skip p
    #   search(idx + 1, mask, L, budget)
    #   
    #   # Option 2: Include p
    #   if budget >= 1/p:
    #       for r in range(p):
    #           new_mask = mask | masks[p][r]
    #           new_L = get_L_from_mask(new_mask)
    #           # Pruning: if new_L is not better than current best potential? 
    #           # Hard to know potential.
    #           # But if new_L < L, it's useless (though adding coverage never decreases L, so new_L >= L always).
    #           # So we always explore.
    #           search(idx + 1, new_mask, new_L, budget - 1/p)
    
    # This is exponential. Number of primes is ~168.
    # We need strong pruning.
    
    # Insight:
    # The "frontier" L is the key.
    # If we skip a small prime, we might be forced to use larger primes later to cover the same gap, which is less efficient.
    # So, including small primes is almost always beneficial.
    # Let's assume we *must* include the smallest available primes until budget is tight?
    # Not necessarily. Maybe skipping a prime allows a better residue assignment for others?
    # But residues are independent. Adding a prime only adds coverage.
    # So, if we have budget for a prime, we should ALWAYS include it?
    # Yes! Because adding a class (p, r) adds bits to the mask. It never removes bits.
    # So, for any fixed set of primes, we want to choose residues to maximize L.
    # And for any budget, we want to choose the subset of primes that maximizes L.
    # Since adding a prime is monotonic (coverage only increases), we should include a prime if we can afford it?
    # Wait. If we include prime p, we spend 1/p budget.
    # If we don't include p, we save 1/p budget for a larger prime q.
    # Is it better to have p (small, dense) or q (large, sparse)?
    # p covers 1/p of numbers. q covers 1/q.
    # p is denser.
    # However, the *placement* of the covered numbers (the residue r) matters for contiguous coverage.
    # A small prime p has fewer residue choices (p) than a large prime q (q).
    # But p has more "flexibility" in terms of density? No, less flexibility in count of choices, but each choice covers more ground.
    
    # Hypothesis: The optimal set of primes is the set of smallest primes that fit in the budget.
    # Why? Because small primes provide higher density coverage.
    # Let's test this hypothesis.
    # If we take the smallest primes until budget < 1/p_next, we get a set S.
    # Then we solve the residue assignment problem for S to maximize L.
    
    # How to solve residue assignment for a fixed set S?
    # This is still a CSP.
    # Variables: r_p for p in S.
    # Objective: Maximize L.
    # We can use backtracking on residues.
    # Order of variables: Smallest p first?
    # Heuristic: For the current frontier L, the next gap is at L+1.
    # We want some prime p in S to cover L+1.
    # Ideally, the smallest p that can cover L+1 with a residue that doesn't "waste" coverage?
    
    # Let's implement a backtracking solver for a FIXED set of primes.
    # Then we try different sets of primes (e.g., greedy smallest, or skipping some).
    
    def solve_residues(prime_set):
        # prime_set: list of primes
        # Returns best_L and residues dict
        n_primes = len(prime_set)
        if n_primes == 0:
            return 0, {}
        
        # Order primes by size? 
        # Smallest primes have the most impact on L.
        ordered_primes = sorted(prime_set)
        
        best_L = 0
        best_residues = {}
        
        def backtrack(idx, current_mask, current_L):
            nonlocal best_L, best_residues
            
            if time.time() > deadline - 1:
                return
            
            if idx == n_primes:
                if current_L > best_L:
                    best_L = current_L
                    # We don't store best_residues here to save time, 
                    # but we need to return it. 
                    # We can reconstruct or store it.
                    # Storing dict at every leaf is expensive.
                    # Let's just track best_L here, and do a second pass or store path.
                    # For now, let's just return best_L.
                return
            
            p = ordered_primes[idx]
            
            # Heuristic: Try residues that cover the current frontier gap first.
            # Gap is at current_L + 1.
            # Target residue r* = (current_L + 1) % p
            # Try r* first, then others.
            
            # To avoid trying all p residues (which can be up to 1000),
            # we can argue that only residues that cover "interesting" numbers matter.
            # But for small p, p is small.
            
            # Optimization:
            # If we pick r, new_L = get_L_from_mask(current_mask | masks[p][r])
            # We want to maximize new_L.
            
            # Let's sort residues by "potential" to extend L.
            # The residue that covers current_L + 1 is the best candidate.
            # Other residues might cover current_L + 2, etc., but if current_L + 1 is not covered, L doesn't increase.
            # So, ONLY residues r such that (current_L + 1) % p == r can increase L immediately.
            # Wait. If (current_L + 1) % p != r, then current_L + 1 is not covered by this prime.
            # Is it covered by others?
            # current_L is the first gap in current_mask.
            # So current_L + 1 is NOT in current_mask.
            # So if this prime doesn't cover it, it remains a gap.
            # So L will NOT increase.
            # Thus, to increase L, we MUST pick r = (current_L + 1) % p.
            
            # This means the choice is deterministic for extending L!
            # If we want to extend L, we have only ONE choice for the current prime p:
            # r = (current_L + 1) % p.
            # If we pick any other r, L stays the same.
            # So, should we ever pick a different r?
            # Only if picking the "best" r leads to a dead end later?
            # But adding coverage never hurts.
            # If we pick r_best, L increases to L_new.
            # If we pick r_other, L stays L_old.
            # Since L_new >= L_old + 1, the state with L_new is strictly "better" or equal in terms of covered prefix.
            # However, the *mask* is different.
            # The mask with r_best covers more numbers (at least L_old+1).
            # The mask with r_other covers different numbers.
            # Could r_other lead to a better final L?
            # Suppose L_old = 10.
            # p=7. r_best = 11%7 = 4. Covers 11, 18, 25...
            # L becomes 11 (if 12 is covered by others, etc).
            # r_other = 5. Covers 12, 19, 26...
            # 11 is still a gap. L stays 10.
            # Next prime q.
            # If we used r_best, L is 11. Gap at 12.
            # If we used r_other, L is 10. Gap at 11.
            # It seems always better to extend L as soon as possible.
            # So, for a fixed set of primes, the greedy assignment of residues (always cover the current gap) is likely optimal?
            # Let's verify this logic.
            # If greedy is optimal for a fixed set, then we just need to find the best SET of primes.
            
            # Let's implement the greedy residue assignment for a fixed set.
            # Then we search for the best set of primes.
            
            # Actually, let's just do the greedy assignment inside the prime set selection search.
            
            # If greedy residue assignment is optimal for a fixed set, we don't need backtracking on residues.
            # We just need to search for the best subset of primes.
            
            # How to search for the best subset?
            # We iterate primes in increasing order.
            # At each step, we can include or exclude.
            # If we include, we apply the greedy residue update.
            
            # This reduces the problem to:
            # Maximize L(prime_subset) where L is computed by greedy residue assignment.
            
            # We can use DFS on the prime subset.
            
            return # Disable this recursive residue search, use greedy residue logic in the outer loop
        
        # Since we decided greedy residue assignment is likely optimal for a fixed set,
        # we will implement a solver that takes a list of primes and returns the L from greedy assignment.
        
        current_mask = 0
        current_L = 0
        residues = {}
        
        for p in ordered_primes:
            target = current_L + 1
            if target > N:
                break
            r = target % p
            residues[p] = r
            m = masks[p][r]
            new_mask = current_mask | m
            current_mask = new_mask
            current_L = get_L_from_mask(current_mask)
            
        return current_L, residues

    # Now, search for the best subset of primes.
    # We'll use a recursive search over primes in increasing order.
    # State: (idx, current_mask, current_L, current_budget)
    
    best_L_global = 0
    best_residues_global = {}
    
    # To speed up, we can prune if current_L + (max possible extension) <= best_L_global.
    # Max possible extension is N - current_L. Not helpful.
    
    # Let's just do DFS with time limit.
    # Pruning: If budget is 0, stop.
    
    # Precompute 1/p as Fractions for exact budget tracking.
    inv_primes = {p: Fraction(1, p) for p in primes}
    
    # We will search subsets of the first K primes.
    # How many primes?
    # Sum of reciprocals of first 10 primes (2..29) is ~1.5.
    # So we can only pick a subset of these.
    # Let's limit the search to primes <= 50.
    # There are 15 primes <= 50.
    # 2^15 = 32768 subsets. This is very fast!
    # For each subset, we compute L via greedy assignment.
    # This is a complete search over small prime subsets.
    
    # What if we need primes > 50?
    # Unlikely, as small primes are more efficient.
    # But let's try primes up to 100 (25 primes). 2^25 is too big for exhaustive.
    # We'll use DFS with pruning.
    
    search_primes = [p for p in primes if p <= 50]
    
    def search_subset(idx, current_mask, current_L, current_budget):
        nonlocal best_L_global, best_residues_global
        
        if time.time() > deadline - 1:
            return
            
        if current_L > best_L_global:
            # Update best? We need residues.
            # We don't have residues stored in this state easily without backtracking them.
            # Let's just track best_L here.
            # After search, we can reconstruct or store the path.
            # Or, we can just store the best prime set.
            pass
            
        if idx == len(search_primes):
            if current_L > best_L_global:
                best_L_global = current_L
                # Store the set of primes used?
                # We need to know which primes were included.
                # This DFS doesn't track the "path" of included primes easily in the state.
                # We can pass a list of chosen primes.
            return
        
        p = search_primes[idx]
        
        # Option 1: Skip p
        search_subset(idx + 1, current_mask, current_L, current_budget)
        
        # Option 2: Include p
        cost = inv_primes[p]
        if current_budget >= cost:
            # Apply greedy residue for p
            target = current_L + 1
            if target <= N:
                r = target % p
                m = masks[p][r]
                new_mask = current_mask | m
                new_L = get_L_from_mask(new_mask)
                search_subset(idx + 1, new_mask, new_L, current_budget - cost)
    
    # To get the actual residues for the best set, we need to know the set.
    # Let's modify the search to store the best set of primes.
    
    best_prime_set = []
    
    def search_subset_v2(idx, current_mask, current_L, current_budget, chosen_primes):
        nonlocal best_L_global, best_prime_set
        
        if time.time() > deadline - 1:
            return
            
        if current_L > best_L_global:
            best_L_global = current_L
            best_prime_set = list(chosen_primes)
            
        if idx == len(search_primes):
            return
        
        p = search_primes[idx]
        
        # Skip
        search_subset_v2(idx + 1, current_mask, current_L, current_budget, chosen_primes)
        
        # Include
        cost = inv_primes[p]
        if current_budget >= cost:
            target = current_L + 1
            if target <= N:
                r = target % p
                m = masks[p][r]
                new_mask = current_mask | m
                new_L = get_L_from_mask(new_mask)
                chosen_primes.append(p)
                search_subset_v2(idx + 1, new_mask, new_L, current_budget - cost, chosen_primes)
                chosen_primes.pop()
    
    # Initial call
    search_subset_v2(0, 0, 0, Fraction(1, 1), [])
    
    # If we found a best set, compute the residues for it using the greedy method.
    if best_prime_set:
        _, best_residues = solve_residues(best_prime_set)
        result = [[p, best_residues[p]] for p in best_prime_set]
    else:
        # Fallback: Greedy on smallest primes
        ps = [p for p in primes if p <= 20]
        _, best_residues = solve_residues(ps)
        result = [[p, best_residues[p]] for p in ps]
        
    return result

# EVOLVE-BLOCK-END
