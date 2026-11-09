import time
import random
import numpy as np

def construct():
    N = 1000
    max_val = 1000
    max_sum = 2 * max_val
    deadline = time.time() + 115  # Safety margin for 150s limit

    # --- Core data structures and helpers ---
    
    def make_state():
        """Create a fresh state container."""
        return {
            'S': [],
            'in_S': [False] * (max_val + 1),
            'counts': [0] * (max_sum + 1)
        }

    def try_add(state, x):
        S = state['S']
        in_S = state['in_S']
        counts = state['counts']
        
        if in_S[x]:
            return False
            
        # Check 2*x
        if counts[2 * x] >= 2:
            return False
        # Check x + y for y in S
        for y in S:
            s = x + y
            if counts[s] >= 2:
                return False
        
        # Add
        S.append(x)
        in_S[x] = True
        counts[2 * x] += 1
        for y in S:
            if y != x:
                counts[x + y] += 1
        return True

    def try_remove(state, x):
        S = state['S']
        in_S = state['in_S']
        counts = state['counts']
        
        if not in_S[x]:
            return False
            
        S.remove(x)
        in_S[x] = False
        counts[2 * x] -= 1
        for y in S:
            if y != x:
                counts[x + y] -= 1
        return True

    def validate_set(S):
        counts = [0] * (max_sum + 1)
        for i, a in enumerate(S):
            counts[2 * a] += 1
            for b in S[i:]:
                if b != a:
                    counts[a + b] += 1
        return all(c <= 2 for c in counts)

    # --- Heuristic 1: Deterministic Greedy (Ascending) ---
    # This provides a strong, consistent baseline.
    def greedy_ascending():
        state = make_state()
        S = state['S']
        in_S = state['in_S']
        counts = state['counts']
        
        for x in range(1, max_val + 1):
            # Check if x can be added
            if counts[2 * x] >= 2:
                continue
            valid = True
            for y in S:
                if counts[x + y] >= 2:
                    valid = False
                    break
            if valid:
                try_add(state, x)
        return S[:]

    # --- Heuristic 2: Randomized Greedy ---
    def greedy_random(seed):
        rng = random.Random(seed)
        state = make_state()
        candidates = list(range(1, max_val + 1))
        rng.shuffle(candidates)
        
        for x in candidates:
            try_add(state, x)
        return state['S'][:]

    # --- Heuristic 3: Local Search with Blocker Analysis ---
    # This is the core refinement. It attempts to add blocked elements by swapping out blockers.
    
    def local_search(S_init, time_limit, seed=42):
        rng = random.Random(seed)
        state = make_state()
        for x in S_init:
            try_add(state, x)
        
        S = state['S']
        in_S = state['in_S']
        counts = state['counts']
        
        # Precompute sum pairs for blocker identification
        # This is expensive (O(n^2)), so we do it once at the start of this phase
        # and update incrementally? No, updating sum pairs incrementally is complex.
        # Instead, we will recompute "conflicts" on the fly for specific candidates.
        
        # To speed up blocker detection, we maintain a dictionary of sum -> list of pairs
        # But rebuilding this every swap is O(n^2).
        # Let's stick to the O(n) check for validity, and O(n) for finding blockers.
        
        # Blocker finding:
        # If x is blocked, it means there exists some y in S such that counts[x+y] >= 2.
        # The pairs (a,b) in S with a+b = x+y are the blockers.
        
        def find_blockers(x):
            blockers = set()
            if counts[2 * x] >= 2:
                # Find pairs (a,b) in S with a+b = 2x
                for i, a in enumerate(S):
                    for b in S[i:]:
                        if a + b == 2 * x:
                            blockers.add(a)
                            blockers.add(b)
            for y in S:
                s = x + y
                if counts[s] >= 2:
                    # Find pairs (a,b) in S with a+b = s
                    for i, a in enumerate(S):
                        for b in S[i:]:
                            if a + b == s:
                                blockers.add(a)
                                blockers.add(b)
            return [b for b in blockers if in_S[b]]

        # We will iterate through candidates not in S, and try to swap in.
        # To avoid O(N^2) per step in the inner loop, we limit the search.
        
        # Strategy:
        # 1. Maintain a list of "blocked" candidates.
        # 2. Pick a candidate.
        # 3. Find blockers.
        # 4. Try swapping a blocker with the candidate.
        
        # To make this efficient, we'll just loop through random candidates.
        
        candidates_not_in_S = [x for x in range(1, max_val + 1) if not in_S[x]]
        
        # Shuffle to avoid bias
        rng.shuffle(candidates_not_in_S)
        
        # We'll use a pointer to cycle through them, or just pick random ones.
        # Let's pick random ones from the list.
        
        while time.time() < time_limit:
            if not candidates_not_in_S:
                break
            
            # Pick a random candidate
            idx = rng.randint(0, len(candidates_not_in_S) - 1)
            c = candidates_not_in_S[idx]
            
            if in_S[c]:
                # Should not happen if we maintain the list correctly, but just in case
                continue
            
            # Try direct add
            if try_add(state, c):
                # Success! Remove from candidates list
                candidates_not_in_S.pop(idx)
                continue
            
            # Blocked. Find blockers.
            # This O(n^2) part is the bottleneck.
            # Optimization: Only check a subset of S for blockers?
            # Or, precompute sum_counts and use that?
            
            # Let's try to find blockers more efficiently.
            # We know c is blocked if:
            # 1. counts[2c] >= 2
            # 2. Exists y in S s.t. counts[c+y] >= 2
            
            # Let's identify *which* sums are critical.
            critical_sums = []
            if counts[2 * c] >= 2:
                critical_sums.append(2 * c)
            for y in S:
                if counts[c + y] >= 2:
                    critical_sums.append(c + y)
            
            if not critical_sums:
                # Shouldn't happen if try_add failed
                continue
            
            # Now find elements in S that participate in these critical sums.
            # We can do this by iterating through S and checking if they form a pair summing to a critical sum.
            # This is still O(n * |critical_sums|).
            
            # Better: Precompute a map sum -> list of elements in S that can form this sum with another element in S?
            # That's basically the sum_pairs dict.
            
            # Let's build a lightweight sum_pairs map for the current S.
            # Since S size is ~300-400, n^2 is ~160k operations.
            # Doing this every iteration (1000s of iterations) is too slow.
            
            # Alternative:
            # Instead of finding ALL blockers, just pick a random element from S that is "likely" a blocker.
            # An element y in S is a blocker for c if there exists z in S such that y+z is a critical sum.
            
            # Let's just pick a random element from S to swap out, and see if c fits.
            # This is a stochastic hill climb.
            
            if len(S) == 0:
                break
                
            # Pick a random element to remove
            out_idx = rng.randint(0, len(S) - 1)
            x_out = S[out_idx]
            
            # Try swap
            try_remove(state, x_out)
            if try_add(state, c):
                # Success
                candidates_not_in_S.pop(idx)
                # Update S and in_S are already handled by try_add/remove
                # We don't need to update candidates_not_in_S for x_out as it's not in S anymore
                # But we should ensure c is not in candidates_not_in_S (it was popped)
                pass
            else:
                # Swap failed, restore
                try_add(state, x_out)
                # Try a few more random swaps?
                for _ in range(5):
                    out_idx2 = rng.randint(0, len(S) - 1)
                    x_out2 = S[out_idx2]
                    try_remove(state, x_out2)
                    if try_add(state, c):
                        candidates_not_in_S.pop(idx)
                        break
                    else:
                        try_add(state, x_out2)
                
                # If still failed, maybe try removing two? (Expensive, skip for now)

        # Final polish: try to add any remaining candidates
        for c in candidates_not_in_S:
            if try_add(state, c):
                pass
        # Note: candidates_not_in_S might still contain elements that became addable,
        # or elements that were already added (if we didn't pop them correctly in all paths).
        # Let's do a final sweep of all 1..N
        final_S = []
        for x in range(1, max_val + 1):
            if in_S[x]:
                final_S.append(x)
        
        return final_S

    # --- Main Execution Loop ---

    best_S = []
    best_size = 0

    # 1. Deterministic Greedy
    S_det = greedy_ascending()
    best_S = S_det[:]
    best_size = len(S_det)

    # 2. Try a few randomized greedy starts
    # Time budget for each start + local search
    # Total time ~115s.
    # Let's do 5 trials.
    
    t_start = time.time()
    time_per_trial = 20  # seconds
    
    for trial in range(5):
        if time.time() > deadline - 5:
            break
            
        seed = trial * 1000 + 7
        S_rand = greedy_random(seed)
        
        # If the random start is already better, update best
        if len(S_rand) > best_size:
            best_S = S_rand[:]
            best_size = len(S_rand)
        
        # Run local search on the better of (S_det, S_rand) or just S_rand?
        # Let's run on S_rand, but if S_det is much better, maybe start from S_det?
        # Actually, local search from a good start is better.
        # Let's run local search on S_rand.
        
        remaining_time = deadline - time.time() - 10 # 10s safety buffer
        if remaining_time < 5:
            continue
            
        # Allocate time for local search
        ls_time = min(time_per_trial, remaining_time)
        
        S_ls = local_search(S_rand, time.time() + ls_time, seed=seed)
        
        if len(S_ls) > best_size:
            best_size = len(S_ls)
            best_S = S_ls[:]

    # 3. If time remains, try local search on the deterministic start
    if time.time() < deadline - 15:
        remaining_time = deadline - time.time() - 5
        if remaining_time > 10:
            S_ls_det = local_search(S_det, time.time() + remaining_time, seed=999)
            if len(S_ls_det) > best_size:
                best_size = len(S_ls_det)
                best_S = S_ls_det[:]

    # Final Validation
    if not validate_set(best_S):
        # Fallback to deterministic
        best_S = S_det[:]
        best_size = len(S_det)
        if not validate_set(best_S):
            # This should never happen
            return list(range(1, 50))

    return best_S
