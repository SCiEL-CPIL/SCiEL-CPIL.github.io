import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    def valid_sidon(sets):
        """Check if a set is a Sidon set (unique pair sums including a+a)."""
        s = sorted(sets)
        seen = set()
        for i in range(len(s)):
            for j in range(i, len(s)):
                val = s[i] + s[j]
                if val in seen:
                    return False
                seen.add(val)
        return True

    def get_diffs(sets):
        """Get the set of nonzero differences for a set."""
        s = sorted(sets)
        diffs = set()
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                diffs.add(s[j] - s[i])
        return diffs

    def greedy_build(order, side, A, B, diff_A, diff_B):
        """
        Greedily add elements from 'order' to side 'side' (0 for A, 1 for B).
        Returns the updated sets and diff sets.
        """
        for x in order:
            if time.time() > deadline:
                break
            
            # Determine which set we are adding to
            if side == 0:
                target = A
                other = B
                target_diffs = diff_A
                other_diffs = diff_B
            else:
                target = B
                other = A
                target_diffs = diff_B
                other_diffs = diff_A
            
            # Candidate set
            candidate = target + [x]
            
            # Check Sidon property for candidate
            if not valid_sidon(candidate):
                continue
            
            # Check difference disjointness
            new_diffs = get_diffs(candidate)
            # The new differences introduced by x are x - a and a - x for a in target
            # But get_diffs(candidate) includes all diffs. We just need to check intersection with other_diffs.
            # Note: new_diffs includes diffs within 'target' which were already checked against other_diffs in previous steps.
            # So we only need to check if (new_diffs & other_diffs) is empty.
            if new_diffs & other_diffs:
                continue
            
            # Also, we must ensure that the differences between the NEW element x and existing elements
            # do not conflict with the *other* set's differences.
            # This is covered by new_diffs & other_diffs.
            
            # Accept x
            if side == 0:
                A.append(x)
                diff_A = new_diffs
            else:
                B.append(x)
                diff_B = new_diffs
                
        return A, B, diff_A, diff_B

    best_size = 0
    best_A = []
    best_B = []
    
    # Strategy: Random restarts with shuffled orders
    # We can also try different heuristics for ordering, e.g., random, ascending, descending
    # Let's stick to random permutations for simplicity and diversity.
    
    while time.time() < deadline:
        A = []
        B = []
        diff_A = set()
        diff_B = set()
        
        # Generate a random order of elements from 1 to 256
        order = list(range(1, 257))
        random.shuffle(order)
        
        # We can interleave additions to A and B or do them sequentially.
        # The original code did x=1..128, then for each x tried A then B.
        # Here, we have a single global order. We can decide on the fly which set to add to.
        # A simple strategy: for each x in order, try adding to A, then B? 
        # Or randomly assign? 
        # Let's try: for each x, try adding to the set that currently has fewer elements, 
        # or randomly. Or just try both and pick the one that succeeds?
        
        # Let's try a specific strategy: 
        # For each x in the shuffled order:
        # 1. Try adding to A. If valid, add to A.
        # 2. Else, try adding to B. If valid, add to B.
        # This might be biased. 
        # Alternative: Randomly choose which set to try first, or try both.
        
        # Let's try: For each x, try adding to A. If it fails, try adding to B.
        # This is similar to the seed but with random order.
        
        for x in order:
            if time.time() > deadline:
                break
            
            # Try A
            candidate_A = A + [x]
            if valid_sidon(candidate_A):
                new_diffs_A = get_diffs(candidate_A)
                if not (new_diffs_A & diff_B):
                    A = candidate_A
                    diff_A = new_diffs_A
                    continue
            
            # Try B
            candidate_B = B + [x]
            if valid_sidon(candidate_B):
                new_diffs_B = get_diffs(candidate_B)
                if not (new_diffs_B & diff_A):
                    B = candidate_B
                    diff_B = new_diffs_B
                    continue
        
        current_size = len(A) + len(B)
        if current_size > best_size:
            best_size = current_size
            best_A = A[:]
            best_B = B[:]
            
            # If we hit the theoretical max or a very high number, we might stop early?
            # Max Sidon size in [1, 256] is roughly sqrt(256) = 16? No, Sidon sets can be larger.
            # Upper bound for Sidon set in [1, N] is approx sqrt(N). 
            # Wait, for N=256, sqrt(256)=16. But we can have two sets.
            # Actually, the maximum size of a Sidon set in [1, N] is asymptotically sqrt(N).
            # So 16 is a rough estimate. But we are looking for disjoint differences too.
            # If we get > 30, that's good.
            
    # Ensure non-empty
    if not best_A:
        best_A = [1]
    if not best_B:
        # Find a valid element for B
        for x in range(1, 257):
            if x not in best_A:
                candidate_B = [x]
                if valid_sidon(candidate_B):
                    # Check diff disjointness (empty diff set for single element is fine, 
                    # but wait, diff of [x] is empty. So it's always disjoint from anything).
                    best_B = candidate_B
                    break
        if not best_B:
            best_B = [2] # Fallback, though validity might be an issue if 2 in A and diffs conflict? 
                         # Single element has no diffs. So [2] is always valid for Sidon.
                         # And diff set is empty, so disjoint from anything.
    
    return [best_A, best_B]
