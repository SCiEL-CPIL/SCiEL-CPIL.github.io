import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 55.0  # Leave 5s margin for 60s limit
    
    n = 256
    best_size = 0
    best_A = []
    best_B = []
    
    # Helper to check Sidon property for adding x to S
    def can_add_sidon(x, S, sums_S):
        for a in S:
            s = x + a
            if s in sums_S:
                return False
        if 2 * x in sums_S:
            return False
        return True

    def add_sidon(x, S, sums_S, diffs_S):
        for a in S:
            sums_S.add(x + a)
            diffs_S.add(abs(x - a))
        sums_S.add(2 * x)
        S.append(x)

    # Function to find a Sidon set of target_size with disjoint diffs from forbidden_diffs
    def find_sidon_set(target_size, forbidden_diffs, max_time=1.0):
        # Use randomized backtracking
        # We will try to build the set from random candidates
        # To make it faster, we can sort candidates or just pick randomly
        
        def backtrack(S, sums_S, diffs_S, remaining, start_idx, candidates):
            if time.time() - start_time > time_limit:
                return None
            if len(S) == target_size:
                return S[:]
            
            # Heuristic: pick a random candidate from remaining
            # To improve success rate, we can try a few random ones
            
            # We need to ensure we don't pick too small numbers if we want to leave room?
            # Actually, for Sidon sets, smaller numbers are often better for packing?
            # Or larger? 
            # Let's just randomize.
            
            # Optimization: Pre-filter candidates that don't conflict with existing diffs
            valid_candidates = []
            for i in range(start_idx, len(candidates)):
                x = candidates[i]
                # Check diff disjointness
                ok = True
                for a in S:
                    d = abs(x - a)
                    if d in forbidden_diffs:
                        ok = False
                        break
                if not ok:
                    continue
                # Check Sidon
                if not can_add_sidon(x, S, sums_S):
                    continue
                valid_candidates.append(i)
            
            if not valid_candidates:
                return None
            
            # Randomly select a candidate
            # Bias towards earlier indices? Or fully random?
            # Fully random might explore more.
            idx = random.choice(valid_candidates)
            x = candidates[idx]
            
            # Add x
            new_S = S + [x]
            new_sums_S = set(sums_S)
            new_diffs_S = set(diffs_S)
            for a in S:
                new_sums_S.add(x + a)
                new_diffs_S.add(abs(x - a))
            new_sums_S.add(2 * x)
            
            # Recurse
            # We can pass the rest of the candidates
            res = backtrack(new_S, new_sums_S, new_diffs_S, remaining - 1, idx + 1, candidates)
            if res is not None:
                return res
            
            # If failed, try another random candidate
            # To avoid infinite loop, limit retries or just return None
            # For efficiency, we'll just try one random path per call in this simple version.
            # But we can add a loop to try a few random choices.
            
            # Let's try up to 5 random choices
            attempts = 0
            while attempts < 5:
                idx = random.choice(valid_candidates)
                x = candidates[idx]
                new_S = S + [x]
                new_sums_S = set(sums_S)
                new_diffs_S = set(diffs_S)
                for a in S:
                    new_sums_S.add(x + a)
                    new_diffs_S.add(abs(x - a))
                new_sums_S.add(2 * x)
                
                res = backtrack(new_S, new_sums_S, new_diffs_S, remaining - 1, idx + 1, candidates)
                if res is not None:
                    return res
                attempts += 1
                
            return None

        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        # Initial call
        return backtrack([], set(), set(), target_size, 0, candidates)

    # Main loop: Try to find A and B
    # We aim for |A|=16, |B|=16
    # If that fails, try 15/17, 14/18, etc.
    
    target_sizes = [
        (16, 16),
        (15, 17),
        (17, 15),
        (14, 18),
        (18, 14),
        (13, 19),
        (19, 13),
        (12, 20),
        (20, 12)
    ]
    
    # We can also just try to maximize greedily if backtracking is too slow
    # But backtracking for 16 should be fast enough with good pruning.
    
    for size_a, size_b in target_sizes:
        if time.time() - start_time > time_limit - 5:
            break
            
        found = False
        for _ in range(100): # 100 attempts for each target pair
            if time.time() - start_time > time_limit - 5:
                break
                
            # Find A
            A = find_sidon_set(size_a, set(), max_time=0.5)
            if A is None:
                continue
            
            # Calculate D(A)
            D_A = set()
            for i in range(len(A)):
                for j in range(i + 1, len(A)):
                    D_A.add(abs(A[i] - A[j]))
            
            # Find B
            B = find_sidon_set(size_b, D_A, max_time=0.5)
            if B is None:
                continue
            
            # Verify
            if len(A) + len(B) > best_size:
                best_size = len(A) + len(B)
                best_A = A
                best_B = B
                found = True
                if best_size == 32: # Max possible
                    return [best_A, best_B]
        
        if found:
            # If we found a valid pair for this size, we can stop if it's large enough
            # Or continue to see if we can find larger? 
            # Since we iterate from largest, we can stop.
            break
        
    # Fallback: Greedy if backtracking didn't find anything or timed out
    if best_size == 0:
        # Simple greedy
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        for x in candidates:
            # Try A
            if can_add_sidon(x, A, sums_A):
                # Check diffs
                ok = True
                for a in A:
                    if abs(x - a) in diffs_B:
                        ok = False
                        break
                if ok:
                    add_sidon(x, A, sums_A, diffs_A)
                    continue
            
            # Try B
            if can_add_sidon(x, B, sums_B):
                ok = True
                for b in B:
                    if abs(x - b) in diffs_A:
                        ok = False
                        break
                if ok:
                    add_sidon(x, B, sums_B, diffs_B)
                    continue
        
        if len(A) + len(B) > best_size:
            best_A = A
            best_B = B
    
    if best_size == 0:
        return [[1], [2]]
        
    return [best_A, best_B]
