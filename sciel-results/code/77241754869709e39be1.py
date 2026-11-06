import time
import random
import math

def construct():
    start_time = time.time()
    # Explicit deadline guard: 150s limit, keep 10s safety margin
    deadline = start_time + 140.0
    
    # Helper to calculate score and the set of valid differences
    # Returns (score, diff_set)
    def analyze_set(S):
        S_sorted = sorted(S)
        diff_set = set()
        n = len(S_sorted)
        contains = S.__contains__
        for i in range(n):
            a = S_sorted[i]
            for j in range(i + 1, n):
                b = S_sorted[j]
                d = b - a
                # Check if a + 2d is in S
                # a + 2d = b + d
                if contains(b + d):
                    diff_set.add(d)
        return len(diff_set), diff_set

    # Greedy Construction: Start with a random seed, add elements that maximize new differences
    def greedy_construct():
        # Start with a small random set
        current_set = set(random.sample(range(1, 513), 8))
        current_score, current_diffs = analyze_set(current_set)
        
        # Pre-compute all possible differences? No, just check candidates.
        # To speed up, we can maintain a list of "promising" candidates or just iterate.
        # Since N=32 is small, we can iterate through all 512 values for each addition step.
        
        for _ in range(24):  # Add 24 more elements to reach 32
            best_val = -1
            best_new_score = -1
            best_new_diffs = set()
            
            # To save time, we don't need to check all 512 if we are time-constrained,
            # but 512 * 24 * (cost of analyze) is manageable if analyze is fast.
            # However, analyze_set is O(N^2). 
            # Let's optimize the inner loop: 
            # When adding a new element x, which new differences are created?
            # A difference d is new if it wasn't in current_diffs and there exist a, a+d, a+2d 
            # such that one of them is x.
            # This is complex to update incrementally without full re-evaluation for correctness.
            # Given N=32, full re-evaluation is ~32*31/2 = 496 checks. 
            # 512 candidates * 496 checks = 250k ops per step. 
            # 24 steps = 6M ops. This is fast in Python if optimized.
            
            # We'll just pick a random subset of candidates to speed up if needed, 
            # but let's try all 512 first as it's likely fast enough.
            
            candidates = [x for x in range(1, 513) if x not in current_set]
            # Shuffle to break ties randomly
            random.shuffle(candidates)
            
            # To further speed up, we can limit candidates if time is tight, 
            # but let's assume full search is fast.
            
            for x in candidates:
                new_set = current_set.union([x])
                # Quick check: does x form any new progression?
                # We can just call analyze_set.
                new_score, new_diffs = analyze_set(new_set)
                if new_score > best_new_score:
                    best_new_score = new_score
                    best_new_diffs = new_diffs
                    best_val = x
            
            if best_val != -1:
                current_set.add(best_val)
                current_score = best_new_score
                current_diffs = best_new_diffs
            else:
                # Fallback: pick random
                current_set.add(random.choice(candidates))
                current_score, current_diffs = analyze_set(current_set)
                
        return current_set

    # Local Search: Swap elements to improve score
    def local_search(initial_set, time_limit):
        current_set = set(initial_set)
        current_score, _ = analyze_set(current_set)
        best_set = set(current_set)
        best_score = current_score
        
        # Pre-compute list of elements for random choice
        while time.time() < time_limit:
            # Generate a random swap
            if len(current_set) < 2:
                break
            
            # Pick two distinct elements
            elems = list(current_set)
            i, j = random.sample(range(len(elems)), 2)
            a = elems[i]
            b = elems[j]
            
            # Try swapping a with a new random value
            new_val = random.randint(1, 512)
            if new_val in current_set and new_val != a and new_val != b:
                continue
            if new_val == a or new_val == b:
                continue
                
            # Construct new set
            new_set = set(current_set)
            new_set.remove(a)
            new_set.add(new_val)
            
            new_score, _ = analyze_set(new_set)
            
            if new_score > best_score:
                best_score = new_score
                best_set = set(new_set)
                current_set = new_set
                current_score = new_score
            elif new_score >= current_score:
                # Accept neutral or positive moves to explore
                current_set = new_set
                current_score = new_score
            
            # Occasionally try a "kick" (random restart of local search) if stuck?
            # For now, simple hill climbing with random restarts outside.

        return best_set

    best_global_set = None
    best_global_score = -1
    
    # Strategy: 
    # 1. Run several greedy constructions.
    # 2. For each, run local search for a fixed time chunk.
    # 3. Keep the best.
    
    num_attempts = 10
    # Allocate time: 50ms per attempt setup, rest for local search?
    # Let's do 5 attempts with 20s local search each.
    attempts_per_phase = 5
    time_per_attempt = 20.0
    
    for phase in range(3):  # 3 phases of 5 attempts = 15 attempts total
        if time.time() > deadline - 5.0:
            break
            
        for i in range(attempts_per_phase):
            if time.time() > deadline - 5.0:
                break
                
            # 1. Greedy Construct
            g_set = greedy_construct()
            g_score, _ = analyze_set(g_set)
            
            if g_score > best_global_score:
                best_global_score = g_score
                best_global_set = set(g_set)
                
            # 2. Local Search
            # Time limit for this specific local search
            ls_deadline = time.time() + time_per_attempt
            # Ensure we don't exceed global deadline
            if ls_deadline > deadline - 5.0:
                ls_deadline = deadline - 5.0
                
            if time.time() < ls_deadline:
                ls_set = local_search(g_set, ls_deadline)
                ls_score, _ = analyze_set(ls_set)
                
                if ls_score > best_global_score:
                    best_global_score = ls_score
                    best_global_set = set(ls_set)
                    
    # Final check: if we have time, do one more high-intensity local search on the best set found
    if time.time() < deadline - 10.0 and best_global_set is not None:
        final_deadline = deadline - 5.0
        final_set = local_search(best_global_set, final_deadline)
        final_score, _ = analyze_set(final_set)
        if final_score > best_global_score:
            best_global_score = final_score
            best_global_set = set(final_set)

    if best_global_set is None:
        # Fallback: simple random
        return sorted(random.sample(range(1, 513), 32))
        
    return sorted(best_global_set)
