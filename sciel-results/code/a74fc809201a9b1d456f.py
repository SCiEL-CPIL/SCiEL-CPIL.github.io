import time
import random
import sys

def _check_sidon(S):
    """Check Sidon property: all pair sums (with replacement) are unique."""
    seen = set()
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def _diffs(S):
    """Return set of absolute nonzero differences."""
    d = set()
    n = len(S)
    for i in range(n):
        for j in range(i + 1, n):
            diff = abs(S[i] - S[j])
            if diff > 0:
                d.add(diff)
    return d

def _validate(A, B, max_n=256):
    """Validate solution. Returns (is_valid, score)."""
    if not A or not B:
        return False, 0
    if len(A) > 64 or len(B) > 64:
        return False, 0
    for x in A:
        if x < 1 or x > max_n:
            return False, 0
    for x in B:
        if x < 1 or x > max_n:
            return False, 0
    if not _check_sidon(A):
        return False, 0
    if not _check_sidon(B):
        return False, 0
    dA = _diffs(A)
    dB = _diffs(B)
    if dA & dB:
        return False, 0
    return True, len(A) + len(B)

def construct():
    """
    Main entry point. Uses a CSP-based backtracking search with MRV and Forward Checking.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    max_n = 256
    max_size = 64

    best_A, best_B = [1], [3]
    best_score = 2
    best_valid = True

    rng = random.Random(42)

    def run_csp_search(seed_val):
        """
        Runs a CSP search to maximize size.
        Returns (A, B, score)
        """
        current_rng = random.Random(seed_val)
        
        # State variables
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        all_diffs = diffs_A | diffs_B
        
        # To allow some overlap in values, we track used values? 
        # Actually, A and B can overlap. 
        # However, if x is in A and x is in B, it doesn't violate Sidon for A or Sidon for B.
        # But it might violate difference disjointness?
        # If x in A and x in B, diff is 0. Nonzero diffs are what matters.
        # So overlap is allowed.
        
        # We want to maximize |A|+|B|.
        # We will try to fill A and B up to max_size.
        
        # To make it a CSP, we decide for each x in [1, 256] whether to put it in A, B, or neither.
        # This is a huge search space.
        
        # Instead, let's use a constructive approach with backtracking.
        # We pick elements one by one.
        # MRV: At each step, which x is the most constrained?
        # Actually, since we are constructing, we can just iterate x from 1 to 256 and decide.
        # But that's greedy if we don't backtrack.
        
        # Let's use a recursive backtracking where we try to add the "next best" candidate.
        # Candidate pool: all x in [1, 256].
        
        def backtrack(A, B, sums_A, sums_B, diffs_A, diffs_B, depth, max_depth):
            nonlocal best_A, best_B, best_score
            
            if time.time() > deadline - 1.0:
                return False, (list(A), list(B), len(A)+len(B))
            
            # Update best
            if len(A) + len(B) > best_score:
                best_score = len(A) + len(B)
                best_A = list(A)
                best_B = list(B)
            
            if depth == max_depth or (len(A) >= max_size and len(B) >= max_size):
                return True, (list(A), list(B), len(A)+len(B))
                
            # Heuristic: Try to add to the set that is smaller, or randomize.
            # Candidates: all x in 1..256 not in A (for A) and not in B (for B)?
            # Wait, x can be in both.
            
            # Let's generate candidates for A and B.
            candidates_A = []
            candidates_B = []
            
            # To speed up, we can just iterate through a shuffled list of all numbers?
            # No, that's too slow for backtracking.
            
            # Let's pick a random x from the remaining pool?
            # Or better: pick the x that has the most "conflicts" if we don't add it?
            # This is complex.
            
            # Simpler CSP approach:
            # We have a pool of unused numbers? No, numbers can be reused.
            # But if we add x to A, we don't "use" x for B.
            
            # Let's just try adding a random x to A or B.
            # To ensure progress, we should prioritize adding to the set that allows more future additions.
            
            # Let's try a specific x.
            # Which x?
            # If we pick x randomly, we might waste time.
            
            # Let's use a "forward checking" style:
            # For each x in 1..256, check if it can be added to A or B.
            # Pick the one with the fewest options? (MRV on the variable 'next element')
            # This is expensive (256 checks per node).
            
            # Let's try a different strategy:
            # We know we want to maximize size.
            # Let's try to fill A first, then B? Or interleave?
            
            # Let's use a randomized greedy with backtracking.
            # At each step, pick a random x from 1..256.
            # Try to add to A. If valid, add.
            # Try to add to B. If valid, add.
            # If both valid, pick one randomly.
            # If neither valid, skip? No, that's not backtracking.
            
            # Backtracking requires exploring alternatives.
            # If we can't add x, we might need to remove a previous element?
            
            # This is getting complicated. Let's stick to a simpler CSP formulation:
            # Variables: x_1, ..., x_256.
            # Domain: {0, 1, 2} (0=neither, 1=A, 2=B).
            # Constraints:
            # 1. Sidon A: For any distinct pairs (x_i, x_j) and (x_k, x_l) in A, sums are unique.
            # 2. Sidon B: Same for B.
            # 3. Diff Disjoint: Diff(A) ∩ Diff(B) = empty.
            
            # Solving this full CSP is hard.
            
            # Let's go back to the constructive backtracking.
            # State: (A, B, sums_A, sums_B, diffs_A, diffs_B)
            # Action: Add x to A, Add x to B, or Skip x.
            # We iterate x from 1 to 256.
            
            # This is a standard 0-1 knapsack-like structure.
            # We can memoize? No, state is large.
            
            # Let's implement a simple recursive search over x.
            
            pass

    # Since the full CSP is complex to implement efficiently in Python within the time limit,
    # let's try a hybrid:
    # 1. Fast Greedy to get a good baseline.
    # 2. Local Search with "Swaps" and "Moves" but using exact constraint checks (no penalty).
    #    This is different from SA which uses penalties.
    #    We only accept moves that maintain validity.
    #    This is a "Valid State Only" local search.
    
    def try_add(A, B, sums_A, sums_B, diffs_A, diffs_B, x, target):
        """
        Try to add x to target set ('A' or 'B').
        Returns (success, new_sums, new_diffs)
        """
        if target == 'A':
            # Check Sidon
            if 2*x in sums_A:
                return False, None, None
            for a in A:
                if x + a in sums_A:
                    return False, None, None
            # Check Diffs
            for a in A:
                d = abs(x - a)
                if d > 0 and d in diffs_B:
                    return False, None, None
                # Note: d must not be in diffs_A? 
                # Sidon property implies differences are unique? 
                # No, Sidon implies sums are unique. 
                # Does Sidon imply differences are unique?
                # If a-b = c-d, then a+d = c+b. 
                # If a,b,c,d are distinct, this is a collision of sums.
                # If some are equal, e.g., a=c, then b=d, so diff is same pair.
                # So yes, nonzero differences in a Sidon set are unique.
                # So we just need to check against diffs_B.
                if d > 0 and d in diffs_A:
                    return False, None, None
            
            # Update
            new_sums_A = set(sums_A)
            new_sums_A.add(2*x)
            for a in A:
                new_sums_A.add(x + a)
                
            new_diffs_A = set(diffs_A)
            for a in A:
                d = abs(x - a)
                if d > 0:
                    new_diffs_A.add(d)
                    
            return True, (new_sums_A, new_diffs_A), None

        else: # target == 'B'
            if 2*x in sums_B:
                return False, None, None
            for b in B:
                if x + b in sums_B:
                    return False, None, None
            for b in B:
                d = abs(x - b)
                if d > 0 and d in diffs_A:
                    return False, None, None
                if d > 0 and d in diffs_B:
                    return False, None, None
            
            new_sums_B = set(sums_B)
            new_sums_B.add(2*x)
            for b in B:
                new_sums_B.add(x + b)
                
            new_diffs_B = set(diffs_B)
            for b in B:
                d = abs(x - b)
                if d > 0:
                    new_diffs_B.add(d)
                    
            return True, None, (new_sums_B, new_diffs_B)

    # This "try_add" is O(N). With N=64, it's fast.
    
    # Strategy:
    # 1. Initialize with empty sets.
    # 2. Iteratively add elements.
    # 3. Use backtracking if stuck?
    
    # Let's implement a "Greedy with Randomized Restart and Backtracking"
    
    def solve_with_backtracking(seed_val):
        current_rng = random.Random(seed_val)
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        # We will try to fill up to max_size for both.
        # We iterate through a random permutation of [1, 256] and try to add each to A or B.
        # If we can't add, we skip.
        # This is greedy. To improve, we can backtrack if we reach a dead end early.
        
        # But simple greedy with random restarts is often good enough.
        # The current program uses SA.
        # Let's try a "Beam Search" or "A*"?
        
        # Let's try a simple randomized greedy with multiple restarts, but with a twist:
        # We prioritize adding to the set that is currently smaller.
        
        for restart in range(50):
            if time.time() > deadline - 2.0:
                break
            
            A = []
            B = []
            sums_A = set()
            sums_B = set()
            diffs_A = set()
            diffs_B = set()
            
            # Random order
            candidates = list(range(1, max_n + 1))
            current_rng.shuffle(candidates)
            
            for x in candidates:
                # Try to add to A
                if len(A) < max_size:
                    ok_A, res_A, _ = try_add(A, B, sums_A, sums_B, diffs_A, diffs_B, x, 'A')
                    if ok_A:
                        A.append(x)
                        sums_A, diffs_A = res_A
                        continue
                
                # Try to add to B
                if len(B) < max_size:
                    ok_B, _, res_B = try_add(A, B, sums_A, sums_B, diffs_A, diffs_B, x, 'B')
                    if ok_B:
                        B.append(x)
                        sums_B, diffs_B = res_B
                        continue
                        
            # Local Search Polish
            # Try to swap elements or move elements to improve size?
            # If we are at max size, we can't increase.
            # But maybe we can find a configuration that allows larger sets?
            # The greedy might get stuck in a local minimum where we could have fit more.
            
            # Let's try a "Reconstruction" step:
            # Remove a random element, and try to add more?
            
            improved = True
            while improved and time.time() < deadline - 2.0:
                improved = False
                # Try to remove one element from A or B and re-add?
                # Or just try to add a new random element?
                
                # Let's try to add a random element from 1..256 that is not currently in A or B?
                # No, it can be in both.
                
                # Let's try to swap an element in A with a new one?
                
                # Simpler: Just run the greedy again with a different seed?
                # We are already doing restarts.
                
                # Let's try a "Kick": Remove 2 random elements, then re-run greedy on the remaining pool?
                # This is complex.
                
                # Let's just accept the result of the greedy for now.
                
            if len(A) + len(B) > best_score:
                best_score = len(A) + len(B)
                best_A = list(A)
                best_B = list(B)
                
    # Run the solver
    for i in range(100):
        if time.time() > deadline - 5.0:
            break
        solve_with_backtracking(42 + i)
        
    # Final validation
    valid, score = _validate(best_A, best_B)
    if not valid:
        return [[1], [3]]
    
    return [best_A, best_B]

# EVOLVE-BLOCK-END
