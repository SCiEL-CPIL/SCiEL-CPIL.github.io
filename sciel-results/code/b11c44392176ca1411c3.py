import time
import random
import heapq
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
        
    # For each position, compute how many of its n affected windows have count > 1
    # Position i is affected by windows j where j <= i <= j+n-1, i.e., j in [max(0,i-n+1), min(i, L-n)]
    # Safety score = number of such windows with perm_count > 1
    
    def compute_safety(i, L):
        """Compute safety score for position i: number of affected windows with count > 1."""
        start_j = max(0, i - n + 1)
        end_j = min(i, L - n)
        score = 0
        for j in range(start_j, end_j + 1):
            if perm_count[tuple(word[j:j+n])] > 1:
                score += 1
        return score

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

    # Priority-based deletion: use a max-heap of (safety_score, position)
    # We want to try positions with highest safety score first
    
    def build_heap(L):
        """Build a max-heap of (safety, position) for all positions."""
        heap = []
        for i in range(L):
            safety = compute_safety(i, L)
            if safety > 0:  # Only consider positions that might be deletable
                heapq.heappush(heap, (-safety, i))
        return heap

    # Main deletion loop with priority-based strategy
    while time.time() < deadline:
        L = len(word)
        if L <= n:
            break
            
        # Build priority heap
        heap = build_heap(L)
        
        if not heap:
            break
            
        progress = False
        deletions_before = 0
        
        while heap:
            if time.time() > deadline:
                break
                
            neg_safety, idx = heapq.heappop(heap)
            current_idx = idx - deletions_before
            
            if current_idx < 0 or current_idx >= len(word):
                continue
                
            # Re-verify safety since deletions may have changed counts
            L_now = len(word)
            if current_idx < 0 or current_idx >= L_now:
                continue
                
            if try_delete(current_idx):
                progress = True
                deletions_before += 1
                
                # After deletion, positions after current_idx shift left by 1
                # We need to update the heap entries for positions > current_idx
                # For simplicity, we'll rebuild the heap periodically or just continue
                # with remaining heap entries (some may be stale)
                
        if not progress:
            # Try a random pass as fallback
            indices = list(range(len(word)))
            random.shuffle(indices)
            deletions_before = 0
            for idx in indices:
                if time.time() > deadline:
                    break
                current_idx = idx - deletions_before
                if current_idx < 0 or current_idx >= len(word):
                    continue
                if try_delete(current_idx):
                    progress = True
                    deletions_before += 1
                    
            if not progress:
                break
                
    return "".join(map(str, word))
