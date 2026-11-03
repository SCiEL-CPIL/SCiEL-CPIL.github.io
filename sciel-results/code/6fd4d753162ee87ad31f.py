import time
import itertools

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    # 1. Generate initial superpermutation (recursive construction)
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        # Extract all (m-1)-length windows that are permutations of 1..m-1
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find max overlap
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # 2. Prepare for reduction
    # Map permutations to indices
    perms_list = list(itertools.permutations(range(1, n + 1)))
    perm_to_idx = {p: i for i, p in enumerate(perms_list)}
    num_perms = len(perms_list)
    
    # Initial counts
    counts = [0] * num_perms
    L = len(word)
    for i in range(L - n + 1):
        window = tuple(word[i:i + n])
        if window in perm_to_idx:
            counts[perm_to_idx[window]] += 1
    
    # Check if valid initially (should be)
    if any(c == 0 for c in counts):
        # Fallback: return the initial word if something went wrong
        return "".join(map(str, word))
    
    # 3. Reduction Loop
    while True:
        if time.time() > deadline:
            break
        
        L = len(word)
        removed_any = False
        
        for i in range(L):
            if time.time() > deadline:
                break
            
            # Try to remove word[i]
            # Identify affected windows in the original word
            # Old windows containing i: start j in [i-n+1, i]
            old_starts = []
            for j in range(max(0, i - n + 1), min(i + 1, L - n + 1)):
                old_starts.append(j)
            
            # Decrement counts for old windows
            for j in old_starts:
                window = tuple(word[j:j + n])
                idx = perm_to_idx.get(window, -1)
                if idx != -1:
                    counts[idx] -= 1
            
            # Perform removal
            removed_symbol = word.pop(i)
            new_L = L - 1
            
            # Identify new windows in the new word that span the gap
            # The gap is at index i in the new word (where removed_symbol was)
            # New windows containing index i: start j' in [i-n+1, i]
            new_starts = []
            for j_prime in range(max(0, i - n + 1), min(i + 1, new_L - n + 1)):
                new_starts.append(j_prime)
            
            # Increment counts for new windows
            for j_prime in new_starts:
                window = tuple(word[j_prime:j_prime + n])
                idx = perm_to_idx.get(window, -1)
                if idx != -1:
                    counts[idx] += 1
            
            # Check validity
            valid = True
            for c in counts:
                if c == 0:
                    valid = False
                    break
            
            if valid:
                removed_any = True
                L = new_L
                # Continue scanning from i (since the next element shifted to i)
                # We do NOT restart the outer loop immediately, but we do restart the 
                # "no removal" check. Actually, just continuing the for loop is fine 
                # because the index i now points to the new element at that position.
                # However, the range of the for loop is fixed at the start of the pass.
                # So we might miss some. But we will do another pass.
            else:
                # Revert removal
                word.insert(i, removed_symbol)
                L = L # L was already new_L, restore
                # Restore counts
                for j_prime in new_starts:
                    window = tuple(word[j_prime:j_prime + n])
                    idx = perm_to_idx.get(window, -1)
                    if idx != -1:
                        counts[idx] -= 1
                for j in old_starts:
                    window = tuple(word[j:j + n])
                    idx = perm_to_idx.get(window, -1)
                    if idx != -1:
                        counts[idx] += 1
        
        if not removed_any:
            break
    
    return "".join(map(str, word))
