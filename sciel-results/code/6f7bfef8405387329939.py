import time
import random
from collections import defaultdict
import itertools

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # 1. Recursive Construction
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
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    
    # Initialize counts
    perm_count = defaultdict(int)
    for i in range(len(word) - n + 1):
        p = tuple(word[i:i+n])
        perm_count[p] += 1
        
    # Helper to update counts for a window change
    def try_delete(i):
        """
        Attempts to delete word[i].
        Returns True if successful, False otherwise.
        Modifies word, perm_count in-place if successful.
        """
        L = len(word)
        if L <= n:
            return False
            
        # Windows affected in OLD word: j such that j <= i <= j+n-1
        start_j = max(0, i - n + 1)
        end_j = min(i, L - n)
        
        # Fast validity check:
        # If any old window is critical (count == 1), deletion is invalid.
        # Because that permutation would be lost (it cannot be reconstructed in new windows 
        # because new windows are different permutations).
        for j in range(start_j, end_j + 1):
            p = tuple(word[j:j+n])
            if perm_count[p] == 1:
                return False
                
        # Validity confirmed. Now perform deletion.
        
        # 1. Decrement old windows
        for j in range(start_j, end_j + 1):
            p = tuple(word[j:j+n])
            perm_count[p] -= 1
            
        # 2. Compute and increment new windows
        # New word length L-1.
        # New windows spanning the gap:
        # In new word, the gap is at index i.
        # Windows j' such that j' <= i <= j'+n-1 in the NEW word.
        # j' ranges from max(0, i-n+1) to min(i, (L-1)-n)
        start_j_new = max(0, i - n + 1)
        end_j_new = min(i, L - 1 - n)
        
        for j in range(start_j_new, end_j_new + 1):
            # Construct new window content
            new_w = []
            for k in range(n):
                idx = j + k
                if idx < i:
                    new_w.append(word[idx])
                else:
                    new_w.append(word[idx+1])
            p_new = tuple(new_w)
            perm_count[p_new] += 1
            
        # 3. Commit
        word.pop(i)
        return True

    def deletion_score(i):
        """
        Score a deletion position: higher is better.
        Prefer deletions where the affected windows have higher redundancy.
        """
        L = len(word)
        start_j = max(0, i - n + 1)
        end_j = min(i, L - n)
        
        score = 0.0
        for j in range(start_j, end_j + 1):
            p = tuple(word[j:j+n])
            cnt = perm_count[p]
            if cnt > 1:
                # Higher redundancy = safer to delete
                score += cnt - 1
        return score

    # Multi-pass deletion with different strategies
    best_word = list(word)
    best_len = len(word)
    
    # Strategy 1: Greedy deletion with random shuffle (original approach)
    for seed in [42, 123, 777]:
        if time.time() > deadline - 5:
            break
        
        # Restore best word
        word[:] = best_word
        # Rebuild perm_count
        perm_count.clear()
        for i in range(len(word) - n + 1):
            p = tuple(word[i:i+n])
            perm_count[p] += 1
        
        random.seed(seed)
        
        while time.time() < deadline - 5:
            progress = False
            L = len(word)
            indices = list(range(L))
            random.shuffle(indices)
            
            deletions_before = 0
            for idx in indices:
                if time.time() > deadline - 5:
                    break
                current_idx = idx - deletions_before
                if current_idx < 0 or current_idx >= len(word):
                    continue
                if try_delete(current_idx):
                    progress = True
                    deletions_before += 1
                    
            if not progress:
                break
            
            # Update best if improved
            if len(word) < best_len:
                best_len = len(word)
                best_word = list(word)
    
    # Strategy 2: Score-based deletion (prefer high-redundancy deletions)
    if time.time() < deadline - 5:
        # Restore best word
        word[:] = best_word
        # Rebuild perm_count
        perm_count.clear()
        for i in range(len(word) - n + 1):
            p = tuple(word[i:i+n])
            perm_count[p] += 1
        
        while time.time() < deadline - 5:
            progress = False
            L = len(word)
            
            # Compute scores for all valid deletion positions
            candidates = []
            for i in range(L):
                # Check if deletion is valid
                start_j = max(0, i - n + 1)
                end_j = min(i, L - n)
                valid = True
                for j in range(start_j, end_j + 1):
                    p = tuple(word[j:j+n])
                    if perm_count[p] == 1:
                        valid = False
                        break
                if valid:
                    s = deletion_score(i)
                    candidates.append((s, i))
            
            if not candidates:
                break
            
            # Sort by score descending, add small random noise to break ties
            candidates.sort(key=lambda x: -x[0])
            
            deletions_before = 0
            for s, idx in candidates:
                if time.time() > deadline - 5:
                    break
                current_idx = idx - deletions_before
                if current_idx < 0 or current_idx >= len(word):
                    continue
                if try_delete(current_idx):
                    progress = True
                    deletions_before += 1
            
            if not progress:
                break
            
            # Update best if improved
            if len(word) < best_len:
                best_len = len(word)
                best_word = list(word)
    
    # Final: use the best word found
    return "".join(map(str, best_word))
