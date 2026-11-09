import random
import time

N = 256

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 25.0  # Safety margin
    
    best_set = None
    best_size = N + 1
    
    # Helper to check if x can be added to S given current sums
    def can_add(x, S, sums):
        # Check 2x
        if 2 * x in sums:
            return False
        # Check x + a for all a in S
        for a in S:
            s = x + a
            if s in sums:
                return False
        return True

    # Helper to check if S is maximal
    def is_maximal(S):
        S_set = set(S)
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                sums.add(S[i] + S[j])
        
        for x in range(1, N + 1):
            if x in S_set:
                continue
            # Check if x can be added
            if 2 * x in sums:
                continue
            valid = True
            for a in S:
                if (x + a) in sums:
                    valid = False
                    break
            if valid:
                return False
        return True

    # Initial greedy to get a baseline quickly
    # Ascending greedy
    S_greedy = []
    sums_greedy = set()
    for x in range(1, N + 1):
        if can_add(x, S_greedy, sums_greedy):
            S_greedy.append(x)
            sums_greedy.update([2 * x] + [x + a for a in S_greedy])
    
    if is_maximal(S_greedy):
        best_set = S_greedy[:]
        best_size = len(S_greedy)
    else:
        # If not maximal (unlikely for greedy, but possible), we still use it as a fallback
        # Actually, greedy usually produces a maximal set because we process all x.
        # If we process all x in 1..N and add if possible, the result is maximal.
        # Proof: If x was skipped, it couldn't be added given the set at that time.
        # Since we only add elements, the set only grows, so sums only grow.
        # If x couldn't be added earlier, it can't be added later.
        # So the final set is maximal.
        pass

    # Randomized greedy attempts
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        random.shuffle(range(1, N + 1))
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        S = []
        sums = set()
        
        for x in candidates:
            if can_add(x, S, sums):
                S.append(x)
                # Update sums
                sums.add(2 * x)
                for a in S[:-1]: # S[:-1] is S before adding x
                    sums.add(x + a)
        
        # The resulting S is maximal by the same logic as above.
        # We just need to verify it's a valid Sidon set (it is by construction)
        # and check its size.
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = S[:]
            if best_size <= 8: # Theoretical lower bound is ~sqrt(N) ~ 16, but maximal sets can be smaller?
                # Actually, minimal maximal Sidon sets are known to be smaller than maximum.
                # For N=256, sqrt(256)=16.
                # A maximal set must cover all numbers.
                # Let's see if we can break 16.
                pass

        # If we find a very small set, we might want to stop early?
        # No, keep searching until deadline.
        
    return best_set if best_set else []
# EVOLVE-BLOCK-END
