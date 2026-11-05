import time
import itertools
import sys
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    if n == 1:
        return "1"
    
    if n == 2:
        return "121"

    # 1. Generate standard superpermutation
    # The recursive construction S_m = concat( p + [m] + p for p in perms(S_{m-1}) ) with overlaps.
    # This produces a valid superpermutation.
    # Length for n=7 is 5905.
    
    word = [1]
    for m in range(2, n + 1):
        if time.time() > deadline:
            break
            
        # Extract unique (m-1)-permutations from word
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Build new word
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find max overlap with current new_word
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
        
        word = new_word
    
    S = list(word)
    
    all_perms = set(itertools.permutations(range(1, n + 1)))
    
    # Initialize counts
    # We track counts of all n-permutations that appear in S.
    # For removal validity, we only care if a count drops to 0.
    # We can store counts in a dict.
    
    perm_counts = {}
    for i in range(len(S) - n + 1):
        p = tuple(S[i:i + n])
        perm_counts[p] = perm_counts.get(p, 0) + 1
        
    best_S = S[:]
    best_len = len(S)

    # Helper to check if removal at index i is valid
    # Returns (valid, new_S, deltas) where deltas is dict of perm -> count_change
    # Note: This is O(n) per check.
    
    def is_valid_incremental(S, i, perm_counts):
        L = len(S)
        if L <= n:
            return False, None, None
        
        # Old windows affected: indices j such that j <= i < j+n
        # j in [i-n+1, i] intersect [0, L-n]
        start_min = max(0, i - n + 1)
        start_max = min(i, L - n)
        
        # New windows formed by straddling i
        # New length is L-1.
        # Window k in new string corresponds to S[k:k+n] if k < i, or S[k+1:k+1+n] if k >= i (shifted).
        # Straddling windows are those that include position i in the original string.
        # In the new string, the position i is removed.
        # The windows that change are those that covered i.
        # In the new string, these are windows starting at k where the window includes the junction.
        # The junction is between S[i-1] and S[i+1].
        # A window of length n straddles the junction if it starts at k such that:
        # k < i <= k+n-1 in the new string coordinates?
        # Let's map indices.
        # New index x corresponds to old index x if x < i, and old index x+1 if x >= i.
        # A window [k, k+n-1] in new string straddles if it contains both an index < i and an index >= i (in new coords).
        # This means k < i <= k+n-1.
        # So k in [i-n+1, i-1].
        # Also k must be valid: 0 <= k <= (L-1)-n = L-n-1.
        
        start_new_min = max(0, i - n + 1)
        start_new_max = min(i - 1, L - n - 1)
        
        deltas = {}
        
        # Decrement old windows
        for j in range(start_min, start_max + 1):
            w = tuple(S[j:j+n])
            deltas[w] = deltas.get(w, 0) - 1
            # Check if count would drop to 0
            current_count = perm_counts.get(w, 0)
            if current_count + deltas[w] < 1:
                return False, None, None
        
        # Increment new windows
        for k in range(start_new_min, start_new_max + 1):
            # Construct new window
            # Indices in S:
            # new idx x -> old idx y
            # if x < i: y = x
            # if x >= i: y = x + 1
            new_win = []
            for x in range(k, k+n):
                y = x if x < i else x + 1
                new_win.append(S[y])
            w = tuple(new_win)
            deltas[w] = deltas.get(w, 0) + 1
            
        return True, S[:i] + S[i+1:], deltas

    # Pruning Loop
    # We will iterate through the string and try to remove characters.
    # To avoid getting stuck in a local minimum of "no single removal is valid",
    # we can try multiple passes.
    # Also, we can try removing from the end first, or in random order?
    # The structure of the recursive superpermutation suggests that the ends are "looser".
    # Let's try scanning from the end to the beginning.
    
    removed_any = True
    passes = 0
    max_passes = 1000  # Safety cap
    
    while removed_any and passes < max_passes and time.time() < deadline - 5:
        removed_any = False
        passes += 1
        
        # Iterate from end to start
        # This is safe because removing at i shifts indices > i, but we are going down.
        # However, removing at i changes the validity of indices < i.
        # So a single pass from end to start might miss opportunities that become valid after a removal.
        # But it's a good heuristic.
        
        for i in range(len(S) - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            if len(S) <= n:
                break
            
            valid, new_S, deltas = is_valid_incremental(S, i, perm_counts)
            if valid:
                S = new_S
                for p, delta in deltas.items():
                    perm_counts[p] = perm_counts.get(p, 0) + delta
                    if perm_counts[p] == 0:
                        del perm_counts[p]
                removed_any = True
                # Update best
                if len(S) < best_len:
                    best_S = S[:]
                    best_len = len(S)
                # Continue scanning downwards (i-1)
                # Note: The string length changed, but we are iterating i downwards.
                # The next i is i-1, which is still valid in the new string?
                # If we removed index i, the elements at indices < i are unchanged.
                # So index i-1 in the new string is the same as index i-1 in the old string.
                # So we can safely continue with i-1.
                
    # If we still have time, try a random restart or shuffle?
    # Given the complexity, a single greedy pass is likely the best we can do in the time limit
    # for n=7, as the search space is huge.
    # However, the "greedy set-cover" strategy mentioned in the prompt suggests a different approach.
    # But the current program structure is a local search.
    # Let's stick to the local search but maybe try a few random shuffles of the removal order?
    # No, that's risky.
    
    # One improvement: If we get stuck, try to swap two adjacent elements?
    # Swapping is much more complex to validate.
    
    # Let's just return the best found.
    return "".join(map(str, best_S))

# Test locally if needed
# if __name__ == "__main__":
#     print(construct(3))
#     print(construct(4))
#     print(construct(5))
#     print(len(construct(6)))
