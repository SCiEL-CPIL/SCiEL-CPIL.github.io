import time
import random

def construct(n):
    # Standard construction
    def standard_superpermutation(n):
        if n == 1:
            return [1]
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                # Find max overlap
                max_overlap = 0
                for o in range(min(len(new), len(block)), -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        max_overlap = o
                        break
                new += block[max_overlap:]
            word = new
        return word

    # Validity check
    def is_valid(word, n):
        if len(word) < n:
            return False
        windows = set()
        for i in range(len(word) - n + 1):
            windows.add(tuple(word[i:i+n]))
        return len(windows) == len(windows) # Wait, need to check if it contains ALL permutations
        # Actually, for n=7, 5040 permutations.
        # If len(windows) == 5040, it's valid.
        # But we need to ensure they are permutations of 1..n.
        # Since we start with a valid one and only delete, 
        # if a window is not a permutation, it wouldn't have been in the original set?
        # No, deleting can create new windows that are NOT permutations.
        # Example: 121312321. Delete '1' at start -> 21312321.
        # Window 2131 is not a permutation of 1,2,3? It is.
        # But if we delete such that a window has duplicates, e.g. 112, it's not a perm.
        # So we must check that all windows are permutations of 1..n.
        
    def is_valid_full(word, n):
        if len(word) < n:
            return False
        target_len = 1
        for i in range(2, n+1):
            target_len *= i
        if len(word) - n + 1 < target_len:
            return False
            
        windows = set()
        for i in range(len(word) - n + 1):
            w = word[i:i+n]
            # Check if w is a permutation of 1..n
            if len(set(w)) != n:
                return False
            windows.add(tuple(w))
        return len(windows) == target_len

    # Get standard word
    word = standard_superpermutation(n)
    
    if n < 7:
        return "".join(map(str, word))
        
    if n == 7:
        start_time = time.time()
        deadline = start_time + 110 # 10s safety margin
        best_word = word
        best_len = len(word)
        
        # Local search: try to delete characters
        # We use a random index
        while time.time() < deadline:
            if len(word) <= best_len:
                # If we found a shorter one, update best
                if len(word) < best_len:
                    best_word = word[:]
                    best_len = len(word)
            
            # Try to delete a random character
            # Avoid first and last to keep it simple? No, can delete anywhere.
            # But deleting first/last might break the "superpermutation" property more easily?
            # Actually, any deletion is a candidate.
            idx = random.randint(0, len(word) - 1)
            
            # Optimization: if idx is 0 or len-1, it's less likely to be redundant?
            # Just pick random.
            
            # Create candidate
            candidate = word[:idx] + word[idx+1:]
            
            # Check validity
            if is_valid_full(candidate, n):
                word = candidate
                # Loop continues, will update best_word at top of next iter or here
                if len(word) < best_len:
                    best_word = word[:]
                    best_len = len(word)
            
        return "".join(map(str, best_word))
        
    else:
        # For n > 7, standard is likely best we can do
        return "".join(map(str, word))
