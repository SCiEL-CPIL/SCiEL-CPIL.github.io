import time
import random
import sys

def construct(n):
    start_time = time.time()
    deadline = start_time + 135  # Safety margin
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "123412314231243121342132413214321"
    
    # Phase 1: Build standard recursive superpermutation
    word = "1"
    for m in range(2, n + 1):
        if time.time() > deadline - 5:
            break
        perms = []
        seen = set()
        for i in range(len(word) - (m - 1) + 1):
            window = word[i:i + m - 1]
            if len(set(window)) == m - 1:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        new_word = ""
        for p in perms:
            block = p + str(m) + p
            max_overlap = 0
            limit = min(len(new_word), len(block))
            for o in range(limit, 0, -1):
                if new_word[-o:] == block[:o]:
                    max_overlap = o
                    break
            new_word += block[max_overlap:]
        word = new_word

    # Convert to list of integers
    arr = [int(c) for c in word]
    L = len(arr)
    
    if time.time() > deadline - 10:
        return word

    # Phase 2: Local Search to shorten
    # Precompute all valid permutations
    from itertools import permutations
    all_perms = set(permutations(range(1, n + 1)))
    
    # Initialize perm_count
    perm_count = {}
    for i in range(L - n + 1):
        perm = tuple(arr[i:i + n])
        if perm in all_perms:
            perm_count[perm] = perm_count.get(perm, 0) + 1
            
    # Helper to check if deletion at i is safe
    def is_deletion_safe(i):
        # Windows in old array affected by removing i
        # j such that j <= i < j+n
        start_j = max(0, i - n + 1)
        end_j = min(i, L - n)
        
        for j in range(start_j, end_j + 1):
            perm = tuple(arr[j:j + n])
            if perm in all_perms:
                if perm_count[perm] <= 1:
                    return False
        return True

    # Helper to execute deletion and update counts
    def execute_deletion(i):
        nonlocal L
        # 1. Decrement counts for old windows
        start_j = max(0, i - n + 1)
        end_j = min(i, L - n)
        for j in range(start_j, end_j + 1):
            perm = tuple(arr[j:j + n])
            if perm in all_perms:
                perm_count[perm] -= 1
                if perm_count[perm] == 0:
                    del perm_count[perm]
        
        # 2. Pop the element
        arr.pop(i)
        L -= 1
        
        # 3. Increment counts for new windows
        # New windows are those that now cross the "gap"
        # In the new array, the gap is at i.
        # Windows starting at j in new array are affected if they include the new position i
        # which corresponds to old i+1.
        # The windows that change are those starting at j in [i-n+1, i-1] in NEW array.
        # Note: New array length is L.
        # Valid j range: [max(0, i-n+1), min(i-1, L-n)]
        
        new_start_j = max(0, i - n + 1)
        new_end_j = min(i - 1, L - n)
        
        for j in range(new_start_j, new_end_j + 1):
            perm = tuple(arr[j:j + n])
            if perm in all_perms:
                perm_count[perm] = perm_count.get(perm, 0) + 1

    # Search loop
    steps = 0
    max_steps = 200000
    stuck_count = 0
    
    while time.time() < deadline - 5 and steps < max_steps:
        if L <= n:
            break
            
        # Try a random deletion
        i = random.randint(0, L - 1)
        
        if is_deletion_safe(i):
            execute_deletion(i)
            stuck_count = 0
            steps += 1
        else:
            stuck_count += 1
            # If stuck for too long, we might try a swap to unstick
            # But for now, just keep trying random deletions.
            # If we are truly stuck, no single deletion is safe.
            # This means we are at a local minimum for deletion.
            # We could break and return.
            # However, random order might find a different path? No, if NO deletion is safe, we are stuck.
            # How to check if ANY deletion is safe?
            # That would be O(L * n) per step, which is too slow.
            # We just rely on random sampling. If we hit many failures, we might be stuck.
            
            if stuck_count > 1000:
                # Likely stuck at a local minimum.
                # We could try to "shake" by swapping two random chars if it doesn't break coverage.
                # But implementing safe swap is complex.
                # For n=7, the gap is small. We might just be at the optimum or near it.
                # Let's try a few swaps.
                # Swap i, j
                # Check if swap is safe.
                # Affected windows: union of windows covering i and j.
                
                # Simple heuristic: if stuck, break.
                break

    return ''.join(str(x) for x in arr)
