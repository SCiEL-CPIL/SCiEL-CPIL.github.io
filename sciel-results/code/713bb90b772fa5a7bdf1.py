import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    N = 32
    MAX_VAL = 512
    
    # Precompute APs for each difference d.
    # ap_sets[d] is a list of sets (or bitsets) of {a, a+d, a+2d}
    # Actually, for speed, let's precompute for each d, the list of possible starting points a.
    # We will use a bitset representation for the current set S.
    # S_bitset: numpy array of bools or int8 of length MAX_VAL+1.
    
    # To count new diffs when adding x to S:
    # A difference d is realized if there exists a such that a, a+d, a+2d are in S.
    # If we add x, new APs are formed if x is part of an AP with two other elements already in S.
    # Specifically:
    # 1. x = a: need a+d, a+2d in S. (d = x - (a+d) ... wait, d is the diff).
    #    If x is the first term: x, x+d, x+2d. Need x+d, x+2d in S.
    # 2. x = a+d: need a, a+2d in S. Here a = x-d, a+2d = x+d. Need x-d, x+d in S.
    # 3. x = a+2d: need a, a+d in S. Here a = x-2d, a+d = x-d. Need x-2d, x-d in S.
    
    # So, for a given x and current S, the set of NEW diffs d is:
    # { d > 0 | (x+d in S and x+2d in S) OR (x-d in S and x+d in S) OR (x-2d in S and x-d in S) }
    # Note: We must ensure the AP is fully within 1..512.
    
    # We can precompute for each x, the list of (d, required_elements) tuples.
    # But that's 512 * ~250 * 3 ~ 384k tuples. Might be heavy to store but fast to check.
    # Alternatively, compute on the fly using bitsets?
    
    # Let's use a bitset for S.
    # S_bits = np.zeros(MAX_VAL + 1, dtype=bool)
    
    # For a candidate x, to find which d are realized:
    # We can iterate d from 1 to MAX_VAL//2.
    # Check conditions.
    # This is O(MAX_VAL) per candidate.
    # Number of candidates: MAX_VAL.
    # Total per step: MAX_VAL^2 = 512^2 = 262k operations.
    # 32 steps * 262k = 8.4M operations.
    # In Python, this is too slow (seconds).
    # In NumPy, we can vectorize the check for a fixed x?
    # Or vectorize over x?
    
    # Let's try to vectorize over x for a fixed step.
    # We want to find the x that maximizes new_diffs.
    # For each d, we can identify which x would complete an AP for that d.
    # If we have S, for a fixed d, the x's that complete an AP are:
    # - x such that x+d, x+2d in S. (x is start)
    # - x such that x-d, x+d in S. (x is middle)
    # - x such that x-2d, x-d in S. (x is end)
    
    # Let's precompute for each d, the masks:
    # mask_start[d]: bitset where bit i is 1 if i+d and i+2d are in S? No, S changes.
    
    # Better approach:
    # Maintain S as a set.
    # For each step, iterate through all candidates x not in S.
    # For each x, count new diffs.
    # To speed up, we can precompute the "potential" diffs for each x based on S.
    
    # Let's stick to the simple O(N * MAX_VAL * MAX_VAL) logic but optimize with NumPy for the inner loop if possible.
    # Actually, pure Python with sets might be fast enough if we are clever.
    # 32 steps.
    # For each step, 512 candidates.
    # For each candidate, check ~250 diffs.
    # 512 * 250 = 128k checks per step.
    # 32 * 128k = 4M checks.
    # Each check is a few set lookups.
    # 4M * 5 ops = 20M ops.
    # Python: ~2-5 seconds.
    # This is feasible!
    
    # Strategy: Greedy Construction
    # 1. Start with empty set or random small set.
    # 2. Iteratively add the element x that maximizes the number of NEW distinct differences.
    # 3. If tie, break randomly or by heuristic (e.g., smaller x to keep density low? or larger?).
    # 4. After reaching size 32, do local search (swap) to improve.
    
    def get_new_diffs(S_set, x):
        """Return set of new differences realized by adding x to S_set."""
        new_diffs = set()
        # x as start: x, x+d, x+2d
        # x+d <= 512 => d <= 512-x
        # x+2d <= 512 => d <= (512-x)//2
        max_d_start = (512 - x) // 2
        for d in range(1, max_d_start + 1):
            if (x + d) in S_set and (x + 2 * d) in S_set:
                new_diffs.add(d)
                
        # x as middle: x-d, x, x+d
        # x-d >= 1 => d <= x-1
        # x+d <= 512 => d <= 512-x
        max_d_mid = min(x - 1, 512 - x)
        for d in range(1, max_d_mid + 1):
            if (x - d) in S_set and (x + d) in S_set:
                new_diffs.add(d)
                
        # x as end: x-2d, x-d, x
        # x-2d >= 1 => 2d <= x-1 => d <= (x-1)//2
        # x-d >= 1 => d <= x-1
        max_d_end = (x - 1) // 2
        for d in range(1, max_d_end + 1):
            if (x - 2 * d) in S_set and (x - d) in S_set:
                new_diffs.add(d)
                
        return new_diffs

    def count_all_diffs(S_set):
        """Count distinct d > 0 such that AP exists in S_set."""
        diffs = set()
        S_list = list(S_set)
        for i in range(len(S_list)):
            for j in range(len(S_list)):
                if i == j: continue
                d = S_list[j] - S_list[i]
                if d > 0:
                    a = S_list[i]
                    c = a + 2 * d
                    if c in S_set:
                        diffs.add(d)
        return len(diffs)

    best_score = 0
    best_set = None
    
    # Run multiple restarts with different initial seeds or random tie-breaking
    num_restarts = 10
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        S_set = set()
        current_diffs = set()
        
        # Initialize with a random element or empty
        # Let's start empty and let greedy pick.
        
        for step in range(N):
            if time.time() > deadline - 10:
                break
                
            best_x = -1
            best_new_count = -1
            best_new_diffs = None
            candidates = [x for x in range(1, 513) if x not in S_set]
            
            # To speed up, we can sample candidates if the set is large?
            # 512 candidates is small.
            
            for x in candidates:
                new_diffs = get_new_diffs(S_set, x)
                # Filter out diffs already in current_diffs
                truly_new = new_diffs - current_diffs
                cnt = len(truly_new)
                
                if cnt > best_new_count:
                    best_new_count = cnt
                    best_x = x
                    best_new_diffs = truly_new
                elif cnt == best_new_count and cnt > 0:
                    # Tie breaking: random
                    if random.random() < 0.5:
                        best_x = x
                        best_new_diffs = truly_new
            
            if best_x == -1:
                # No improvement possible? Pick random
                best_x = random.choice(candidates)
                best_new_diffs = set()
                
            S_set.add(best_x)
            if best_new_diffs:
                current_diffs.update(best_new_diffs)
            
        # After construction, we have a set of size 32.
        # Calculate actual score
        score = count_all_diffs(S_set)
        
        if score > best_score:
            best_score = score
            best_set = list(S_set)
            
        # Local Search Improvement
        # Try swapping elements to increase score
        # Budget: 2 seconds per restart?
        ls_start = time.time()
        ls_deadline = ls_start + 2.0
        
        S_list = list(S_set)
        S_set_local = set(S_list)
        current_score = score
        
        while time.time() < ls_deadline:
            # Pick random i, j
            i = random.randint(0, N - 1)
            # Pick random x not in S
            x = random.randint(1, 512)
            while x in S_set_local:
                x = random.randint(1, 512)
            
            old_x = S_list[i]
            
            # Try swap
            S_set_local.remove(old_x)
            S_set_local.add(x)
            
            new_score = count_all_diffs(S_set_local)
            
            if new_score > current_score:
                current_score = new_score
                S_list[i] = x
                # Accept
            else:
                # Revert
                S_set_local.remove(x)
                S_set_local.add(old_x)
            
            if current_score > best_score:
                best_score = current_score
                best_set = list(S_set_local)
                
    if best_set is None:
        return list(range(1, 33))
        
    return best_set

# EVOLVE-BLOCK-END
