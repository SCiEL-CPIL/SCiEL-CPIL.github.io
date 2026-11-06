import time
import random
import itertools

def is_sidon(subset):
    # Check if subset has all unordered pair sums unique INCLUDING diagonals
    # i.e., a[i] + a[j] == a[k] + a[l] implies {i,j} == {k,l}
    # Since we iterate i <= j, we just need to check if any sum repeats.
    s = sorted(subset)
    n = len(s)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            val = s[i] + s[j]
            if val in sums:
                return False
            sums.add(val)
    return True

def max_sidon_size(S):
    # Find largest subset of S that is Sidon
    # |S| = 12. C(12, k) is small.
    # k=12: 1
    # k=11: 12
    # k=10: 66
    # k=9: 220
    # k=8: 495
    # k=7: 792
    # k=6: 924
    # Total ~ 2500 checks. Each check is O(k^2) ~ 144 ops. Total ~ 360k ops. Very fast.
    for k in range(len(S), 0, -1):
        for comb in itertools.combinations(S, k):
            if is_sidon(list(comb)):
                return k
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin for 90s limit
    
    # Start with a random set in a smaller range to force collisions
    # The knowledge suggests [0, 25] or similar is better than [0, 128]
    range_max = 30
    
    # Initial best: try a few random starts
    best_set = None
    best_score = 13 # Max possible is 12, so 13 is infinity for comparison
    
    # Try a few deterministic starts in small range
    starts = [
        list(range(12)), # 0..11
        [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14],
        [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16]
    ]
    
    for s in starts:
        if len(set(s)) == 12 and all(0 <= x <= 128 for x in s):
            score = max_sidon_size(s)
            if score < best_score:
                best_score = score
                best_set = s.copy()
    
    if best_set is None:
        best_set = list(range(12))
        best_score = max_sidon_size(best_set)
        
    current_best = best_set
    current_best_val = best_score
    
    # Local search / Hill Climbing
    # We want to minimize the value.
    # Swap elements with values in the small range [0, range_max]
    
    if time.time() < deadline - 5:
        # Random restarts / hill climbing
        # We can afford many iterations because max_sidon_size is fast
        for _ in range(5000):
            if time.time() > deadline: break
            
            # Pick a random element to change
            idx = random.randint(0, 11)
            old_val = current_best[idx]
            
            # Try a few random replacements in the small range
            for _ in range(5):
                if time.time() > deadline: break
                new_val = random.randint(0, range_max)
                if new_val == old_val:
                    continue
                
                # Check if new_val is already in the set (excluding the one we're replacing)
                # Since we replace current_best[idx], we check if new_val is in current_best
                # but not at idx? No, just check if new_val is in the set of other elements.
                # Simpler: create candidate, check distinctness.
                
                candidate = current_best.copy()
                candidate[idx] = new_val
                
                # Quick distinctness check
                if len(set(candidate)) != 12:
                    continue
                    
                # Sort for consistent representation (though max_sidon_size sorts anyway)
                # No need to sort before passing, max_sidon_size handles it.
                
                val = max_sidon_size(candidate)
                if val < current_best_val:
                    current_best_val = val
                    current_best = candidate
                    break # Move to next iteration of outer loop to try another swap?
                    
    return current_best
