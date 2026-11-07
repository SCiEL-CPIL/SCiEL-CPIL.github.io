"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    divisors = DIVISORS
    n = len(divisors)
    L = reduce(_lcm, divisors)
    
    # Precompute: for each divisor d, the set of residues r in [0, d) such that
    # a window of width W starting at X (where X % d == r) contains a multiple of d.
    # Actually, we need: for window [X, X+W], multiples of d are at positions
    # k*d where X <= k*d <= X+W.
    # This is equivalent to: there exists k such that X <= k*d <= X+W.
    # i.e., ceil(X/d) * d <= X + W.
    # i.e., (X + d - 1) // d * d <= X + W.
    
    # Let's use a different, faster approach.
    # For a fixed W, iterate over possible X values.
    # Optimization: X only needs to be checked at positions where the "feasibility"
    # of at least one divisor changes. These are positions near multiples of divisors.
    
    # However, a simpler and very effective optimization:
    # The problem is periodic with period L.
    # But L is huge (~3M).
    # Let's look at the structure of the divisors.
    # Divisors: 6, 10, 14, 15, 21, 22, 26, 33, 35, 39
    
    # Let's try a "local search" or "smart sliding" approach.
    # Instead of checking every X in [0, L-1], we can note that if a solution exists
    # for X, it might also exist for X+1 if the set of candidates doesn't change "too much".
    # But the matching changes.
    
    # Let's stick to the sliding window but optimize the inner loop.
    # The bottleneck is the bipartite matching.
    # Can we make the matching faster?
    # The graph is sparse.
    
    # Alternative:
    # For small W, the number of candidates per divisor is small (1 or 2).
    # If W < min(divisors) = 6, each divisor has at most 1 candidate in any window of length W.
    # Wait, if W < 6, and d >= 6, then a window of length W can contain at most 1 multiple of d.
    # So if W < 6, the problem reduces to: is there an X such that the single multiple
    # of each d in [X, X+W] are all distinct?
    
    # Let's check W from 9 upwards.
    # For W >= 6, divisors can have 2 or more candidates.
    
    # Let's optimize the candidate generation and matching.
    
    def solve_for_W(W):
        """
        Find a valid assignment for span W.
        Returns list of values or None.
        """
        # Iterate X.
        # Optimization: We can skip X values where the "pattern" of candidates is the same
        # as a previously checked X.
        # The pattern changes when X crosses a multiple of some divisor.
        # Specifically, the set of candidates for divisor d changes when X or X+W
        # crosses a multiple of d.
        # So we only need to check X in a specific set of "critical" points.
        
        # Critical points for divisor d:
        # X such that X is a multiple of d (new multiple enters from left)
        # X such that X+W is a multiple of d (multiple leaves from right)
        # i.e., X = k*d or X = k*d - W.
        
        # Let's collect all critical X values in [0, L-1].
        # But this set is large.
        
        # Simpler: Just iterate X from 0 to L-1, but break early if found.
        # With the previous code, it was slow. Why?
        # Probably because for small W, most X are infeasible (some divisor has no candidate).
        # We can filter X quickly.
        
        # Precompute for each divisor d, the "good" X ranges where d has at least one candidate.
        # For divisor d, X is good if [X, X+W] contains a multiple of d.
        # This is true if and only if:
        # There exists k such that X <= k*d <= X+W.
        # This is equivalent to:
        # ceil(X/d) * d <= X + W
        # Let r = X % d. Then X = q*d + r.
        # ceil(X/d) = q if r==0 else q+1.
        # If r==0: q*d <= q*d + r + W => 0 <= r + W (always true for W>=0, r>=0)
        # If r>0: (q+1)*d <= q*d + r + W => d <= r + W => r >= d - W.
        
        # So for divisor d, X is good iff (X % d == 0) OR (X % d >= d - W).
        # If W >= d, then d - W <= 0, so all residues are good.
        # If W < d, then only residues in [0, 0] (i.e., 0) and [d-W, d-1] are good.
        # i.e., X % d is in {0} U [d-W, d-1].
        
        # So we can quickly check if X is feasible for ALL divisors.
        
        # Let's precompute for each divisor d, the set of "bad" residues mod d.
        # Bad residues are those r in [1, d-W-1] if W < d, else none.
        
        bad_residues = []
        for d in divisors:
            if W >= d:
                bad_residues.append(set()) # All residues are good
            else:
                # Good: 0, d-W, d-W+1, ..., d-1
                # Bad: 1, 2, ..., d-W-1
                bad = set(range(1, d - W))
                bad_residues.append(bad)
        
        # Now iterate X.
        # We can jump over "bad" X values.
        # But the badness is defined modulo different d's.
        # The combined badness is periodic with period L.
        
        # Let's just iterate X from 0 to L-1, but with a fast check.
        # The fast check is: for all i, X % divisors[i] not in bad_residues[i].
        
        # To speed this up, we can use the Chinese Remainder Theorem idea or just bitmasks?
        # No, just simple modulo checks are fast enough in C, but in Python loops they are slow.
        
        # Let's try to generate only the "good" X values.
        # This is a system of congruences / intervals.
        
        # Alternative: Iterate over X, but in chunks?
        
        # Let's try a different optimization:
        # Sort divisors by "difficulty" (largest first) to fail fast in matching?
        # No, the feasibility check (has any candidate) is the first filter.
        
        # Let's optimize the feasibility check using precomputed bitsets or similar?
        # Maybe just unrolling the check?
        
        # Let's try iterating X and checking feasibility.
        # If feasible, run matching.
        
        # To avoid iterating all L values, let's note that the pattern of feasibility
        # repeats every L. But L is 3M. 3M iterations * 10 mod checks = 30M ops.
        # In Python, 30M ops might take ~3-5 seconds. That's acceptable.
        
        # The matching part is the bottleneck if many X are feasible.
        # How many X are feasible?
        # For large W, most X are feasible.
        # For small W, fewer.
        
        # Let's assume the first feasible X with a valid matching is found quickly.
        
        # One more optimization:
        # If we find a matching for X, we can return immediately.
        
        # Let's implement the fast feasibility check.
        
        def is_feasible(X):
            for i in range(n):
                if X % divisors[i] in bad_residues[i]:
                    return False
            return True
        
        # Iterate X
        # We can step by 1.
        # But we can also step by the GCD of something? No.
        
        # Let's just loop.
        for X in range(L):
            if time.time() > deadline:
                return None
                
            if not is_feasible(X):
                continue
                
            # Build candidates
            candidates = []
            valid = True
            for i in range(n):
                d = divisors[i]
                # First multiple >= X
                if X % d == 0:
                    first = X
                else:
                    first = X + (d - X % d)
                    
                if first > X + W:
                    # Should not happen if is_feasible returned True
                    valid = False
                    break
                    
                cands = []
                v = first
                while v <= X + W:
                    cands.append(v)
                    v += d
                    
                if not cands:
                    valid = False
                    break
                candidates.append(cands)
                
            if not valid:
                continue
                
            # Bipartite matching
            # Use a simple DFS augmenting path.
            # To speed up, we can try to match divisors with fewer candidates first.
            # Let's reorder the divisors based on candidate count.
            
            # Create a list of (count, index)
            order = sorted(range(n), key=lambda i: len(candidates[i]))
            
            # Remap candidates to match the order
            # Actually, the matching algorithm needs to handle the original indices for the result.
            # Let's just run the matching on the original indices but iterate in 'order'.
            
            all_vals = set()
            for c in candidates:
                all_vals.update(c)
            if len(all_vals) < n:
                continue
                
            val_to_idx = {v: i for i, v in enumerate(sorted(all_vals))}
            idx_to_val = {i: v for v, i in val_to_idx.items()}
            m = len(all_vals)
            
            match_to = [-1] * m
            match_from = [-1] * n
            
            def try_kun(u, seen):
                for v_val in candidates[u]:
                    v_idx = val_to_idx[v_val]
                    if seen[v_idx]:
                        continue
                    seen[v_idx] = True
                    if match_to[v_idx] == -1 or try_kun(match_to[v_idx], seen):
                        match_to[v_idx] = u
                        match_from[u] = v_idx
                        return True
                return False
                
            success = True
            for i in order:
                seen = [False] * m
                if not try_kun(i, seen):
                    success = False
                    break
                    
            if success:
                result = [0] * n
                for i in range(n):
                    result[i] = idx_to_val[match_from[i]]
                # Verify distinctness and divisibility (sanity check)
                # The matching guarantees distinctness.
                # Divisibility is guaranteed by construction.
                return result
                
        return None

    # Search for minimal W
    # Minimum span for 10 distinct integers is 9
    for W in range(9, 200):
        if time.time() > deadline:
            break
            
        result = solve_for_W(W)
        if result is not None:
            return [int(v) for v in result]
            
    # Fallback
    return [int(d) for d in divisors]
# EVOLVE-BLOCK-END
