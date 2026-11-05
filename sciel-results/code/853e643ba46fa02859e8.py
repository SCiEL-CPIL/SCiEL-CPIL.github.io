import time
import math
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    MAX_M = 128
    candidates = list(range(2, MAX_M + 1))
    n = len(candidates)
    
    # Precompute gcds
    gcd_tab = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                gcd_tab[i][j] = math.gcd(candidates[i], candidates[j])
    
    # Precompute prime factor counts for ordering
    def count_prime_factors(x):
        count = 0
        d = 2
        while d * d <= x:
            while x % d == 0:
                count += 1
                x //= d
            d += 1
        if x > 1:
            count += 1
        return count
    
    # Sort indices by descending number of prime factors, then by modulus
    # This is the "dynamic variable ordering" hint: process moduli with more 
    # prime factors first to maximize early conflict detection
    prime_counts = [count_prime_factors(m) for m in candidates]
    # We want to process moduli that are "constrained" first.
    # Moduli with many factors have more divisors, hence more potential conflicts?
    # Or moduli that are highly composite?
    # Actually, the hint says "descending order of their number of prime factors".
    # Let's try that.
    order = sorted(range(n), key=lambda i: (-prime_counts[i], -candidates[i]))
    
    # We will maintain a list of selected (idx, residue)
    # And for each unselected modulus, the set of valid residues
    
    def get_valid_residues(selected, idx):
        """
        Given a list of selected (idx, r) pairs, compute valid residues for candidates[idx].
        Two classes (r1, m1) and (r2, m2) are disjoint iff gcd(m1, m2) does NOT divide (r1 - r2).
        So for a new (r, m_new) and existing (r_old, m_old), we require:
        gcd(m_new, m_old) % (r - r_old) != 0  =>  (r - r_old) % gcd(m_new, m_old) != 0
        """
        m = candidates[idx]
        valid = set(range(m))
        for (sel_idx, sel_r) in selected:
            g = gcd_tab[idx][sel_idx]
            # We need (r - sel_r) % g != 0
            # So r % g != sel_r % g
            forbidden_residue_mod_g = sel_r % g
            # Remove all r in [0, m) such that r % g == forbidden_residue_mod_g
            base = forbidden_residue_mod_g
            # r = base, base+g, base+2g, ... < m
            for r in range(base, m, g):
                if r in valid:
                    valid.discard(r)
        return sorted(valid)

    def solve_backtrack(selected, available_indices, depth=0):
        """
        Backtracking solver with forward checking (implicit via get_valid_residues).
        We process available_indices in a specific order? 
        The hint suggests dynamic ordering. We can pick the available index with the 
        fewest valid residues (MRV heuristic) at each step.
        """
        if time.time() > deadline - 2:
            return selected
        
        if not available_indices:
            return selected
        
        # MRV: Choose the index with the fewest valid residues
        best_idx = None
        best_residues = None
        min_count = float('inf')
        
        # We might want to check all available indices, or just a subset if n is large.
        # n=127, so checking all is fine.
        for idx in available_indices:
            residues = get_valid_residues(selected, idx)
            count = len(residues)
            if count == 0:
                # Dead end, but we are choosing best_idx to proceed.
                # If any has 0, we can't pick it.
                continue
            if count < min_count:
                min_count = count
                best_idx = idx
                best_residues = residues
        
        if best_idx is None:
            return selected
        
        # Try residues in best_residues
        # To maximize success, we might try residues that are "least constraining" for others?
        # Or just iterate.
        # Let's iterate in a deterministic order first, maybe randomize if we get stuck?
        # For now, deterministic order of residues.
        for r in best_residues:
            new_selected = selected + [(best_idx, r)]
            remaining = [idx for idx in available_indices if idx != best_idx]
            result = solve_backtrack(new_selected, remaining, depth + 1)
            if len(result) > len(selected):
                return result
        
        return selected

    # Initial greedy pass to get a good starting point
    # Using the specific order derived from prime factors
    selected = []
    available = list(range(n))
    
    # Greedy phase:
    # Pick the next modulus based on MRV or the fixed order?
    # Let's try a greedy approach that picks the modulus with the fewest valid residues
    # among the remaining ones, similar to the backtracking but without backtracking.
    
    # Actually, let's just run the backtracking solver directly on the full set.
    # The backtracking solver uses MRV, which is a form of dynamic ordering.
    
    # We need to be careful with recursion depth. n=127.
    # Python default recursion limit is 1000, so 127 is fine.
    
    # Let's run the backtracking solver.
    # We start with an empty selection.
    # To speed up, we can provide a good initial order for the *first* few steps?
    # No, MRV is dynamic.
    
    # One optimization: if we find a solution of size k, we can stop if we hit the deadline.
    # We want the MAXIMUM size.
    
    # Let's try to improve the backtracking with a heuristic for residue selection.
    # When we pick a residue for a modulus, we should prefer residues that leave the most
    # options for other moduli? That's expensive to compute.
    
    # Let's stick to the simple MRV backtracking.
    
    # To ensure we get a good result quickly, let's do a few random restarts or perturbations
    # if the first run doesn't find a large solution?
    # Or just run it once with a tight deadline.
    
    # Let's try running it once.
    result = solve_backtrack([], list(range(n)), 0)
    
    # If the backtracking didn't find a large solution (e.g. got stuck early),
    # we might want to try a different strategy.
    # But MRV backtracking is generally strong for CSPs.
    
    # Let's verify the result size.
    # If it's small, maybe we should try a greedy approach first to seed it?
    # The backtracking starts from scratch.
    
    # Let's add a check: if the result is very small, try a greedy fill first.
    if len(result) < 50:
        # Greedy fill
        greedy_selected = []
        remaining = list(range(n))
        while remaining and time.time() < deadline - 5:
            # Find MRV
            best_idx = None
            best_residues = None
            min_count = float('inf')
            for idx in remaining:
                residues = get_valid_residues(greedy_selected, idx)
                if len(residues) < min_count:
                    min_count = len(residues)
                    best_idx = idx
                    best_residues = residues
            
            if best_idx is None or not best_residues:
                break
            
            # Pick the first valid residue (or random?)
            # Let's pick the one that seems most "central" or just the first.
            r = best_residues[0]
            greedy_selected.append((best_idx, r))
            remaining.remove(best_idx)
        
        # Now try to improve greedy_selected with backtracking?
        # Or just use greedy_selected if it's better.
        if len(greedy_selected) > len(result):
            result = greedy_selected
            
            # Try to extend greedy_selected with backtracking
            selected_idx_set = set(idx for (idx, _) in result)
            available = [idx for idx in range(n) if idx not in selected_idx_set]
            extended = solve_backtrack(result, available, 0)
            if len(extended) > len(result):
                result = extended

    # Convert to required format
    final_result = []
    for (idx, r) in result:
        final_result.append([r, candidates[idx]])
        
    return final_result

# Test locally?
# if __name__ == "__main__":
#     res = construct()
#     print(len(res))
#     # Verify disjointness
#     for i in range(len(res)):
#         for j in range(i+1, len(res)):
#             r1, m1 = res[i]
#             r2, m2 = res[j]
#             g = math.gcd(m1, m2)
#             if (r1 - r2) % g == 0:
#                 print("Conflict!", res[i], res[j])
