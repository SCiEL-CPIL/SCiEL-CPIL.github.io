import time
import itertools

def construct(n):
    start_time = time.time()
    deadline = 145.0  # 5s margin
    
    # 1. Generate classic superpermutation
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
            overlap = 0
            # Find max overlap
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # 2. Greedy Deletion Passes
    # Maintain counts of permutations
    # For n=7, n! is 5040.
    # We can use a dict or an array if we encode perms.
    # Encoding: map perm to integer 0..5039.
    
    def encode_perm(p):
        # p is tuple of n ints
        # Rank in lexicographic order
        # This is slow. Better to just use the tuple as key in a dict.
        return p

    # Initialize counts
    perm_counts = {}
    for i in range(len(word) - n + 1):
        p = tuple(word[i:i+n])
        perm_counts[p] = perm_counts.get(p, 0) + 1
        
    # Check if all perms are present
    # Total perms n!
    # We need to ensure len(perm_counts) == n! and all counts > 0
    
    def is_valid(counts, n):
        if len(counts) != int(math.factorial(n)):
            return False
        for v in counts.values():
            if v == 0:
                return False
        return True

    import math
    target_perms = math.factorial(n)
    
    # Initial check
    if len(perm_counts) != target_perms:
        # This shouldn't happen for classic construction
        pass

    # Greedy deletion
    # We try to remove word[i]
    # This affects windows starting at i-n+1 to i
    # We need to decrement counts for these windows
    # If any count becomes 0, revert
    
    # We can just do a simple loop
    # To speed up, we can pre-calculate the affected windows
    
    # Let's do a few passes
    for pass_num in range(10):
        if time.time() > deadline:
            break
            
        changed = False
        # Left to right
        for i in range(len(word)):
            if time.time() > deadline:
                break
                
            # Get affected windows
            # Window starts at j, where j <= i <= j+n-1
            # j in [i-n+1, i]
            # Valid j: 0 <= j <= len(word)-n
            
            affected = []
            min_j = max(0, i - n + 1)
            max_j = min(i, len(word) - n)
            
            for j in range(min_j, max_j + 1):
                p = tuple(word[j:j+n])
                affected.append((j, p))
            
            # Decrement counts
            valid = True
            for j, p in affected:
                perm_counts[p] -= 1
                if perm_counts[p] == 0:
                    valid = False
                    break
            
            if valid:
                # Remove word[i]
                word.pop(i)
                changed = True
                # Shift indices for subsequent windows?
                # The list `word` is modified.
                # The `perm_counts` is updated.
                # We don't need to shift indices in `affected` because we only use them to revert if invalid.
                # But we must be careful: if we remove word[i], the indices of elements after i shift left.
                # The loop continues with i+1.
                # The next element is now at index i.
                # So we should not increment i?
                # Or we can just restart the pass?
                # Restarting is safer and simpler.
                # But restarting is O(N^2). N=6000. 36M ops. Might be ok.
                # Let's just break and restart the pass.
                break
            else:
                # Revert
                for j, p in affected:
                    perm_counts[p] += 1
        
        if not changed:
            # Try right to left
            for i in range(len(word) - 1, -1, -1):
                if time.time() > deadline:
                    break
                    
                affected = []
                min_j = max(0, i - n + 1)
                max_j = min(i, len(word) - n)
                
                for j in range(min_j, max_j + 1):
                    p = tuple(word[j:j+n])
                    affected.append((j, p))
                
                valid = True
                for j, p in affected:
                    perm_counts[p] -= 1
                    if perm_counts[p] == 0:
                        valid = False
                        break
                
                if valid:
                    word.pop(i)
                    changed = True
                    break
                else:
                    for j, p in affected:
                        perm_counts[p] += 1
            
            if not changed:
                break

    return "".join(map(str, word))
