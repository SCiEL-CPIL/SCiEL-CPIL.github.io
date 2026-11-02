import itertools

def construct(n):
    # Step 1: Generate the standard recursive superpermutation
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
        
    # Step 2: Compress using greedy deletion with incremental validity check
    if n <= 1:
        return "".join(map(str, word))
        
    # Precompute permutation IDs
    all_perms = list(itertools.permutations(range(1, n + 1)))
    perm_to_id = {p: i for i, p in enumerate(all_perms)}
    num_perms = len(all_perms)
    
    counts = [0] * num_perms
    unique_count = 0
    
    # Initialize counts for the initial word
    for i in range(len(word) - n + 1):
        w = tuple(word[i:i + n])
        if w in perm_to_id:
            pid = perm_to_id[w]
            if counts[pid] == 0:
                unique_count += 1
            counts[pid] += 1
            
    # Greedy Deletion
    # We iterate until no more deletions are possible
    # To avoid O(L^2) in the worst case if we don't break, we break on first success
    # and restart the scan.
    
    # Helper function to perform one pass of greedy deletion
    def greedy_deletion_pass(reverse=False):
        nonlocal word, counts, unique_count
        L = len(word)
        range_func = range(L - 1, -1, -1) if reverse else range(L)
        for i in range_func:
            # Identify old windows covering i
            start = max(0, i - n + 1)
            end = min(i, L - n)
            
            old_ids = []
            # Remove old windows
            for j in range(start, end + 1):
                w = tuple(word[j:j + n])
                if w in perm_to_id:
                    pid = perm_to_id[w]
                    old_ids.append(pid)
                    counts[pid] -= 1
                    if counts[pid] == 0:
                        unique_count -= 1
                        
            # Create new word
            new_word = word[:i] + word[i + 1:]
            new_L = L - 1
            
            # Add new windows
            new_ids = []
            # The affected range in the new word is [start, end]
            # Note: if i was the last char, end might be out of bounds for new word windows
            # max start index for a window in new word is new_L - n
            max_new_start = new_L - n
            if max_new_start < start:
                # No new windows in the affected range?
                # This happens if we delete from the very end and n is large?
                # If new_L < n, no windows.
                pass
            else:
                for j in range(start, min(end, max_new_start) + 1):
                    if j < 0: continue
                    w = tuple(new_word[j:j + n])
                    if w in perm_to_id:
                        pid = perm_to_id[w]
                        new_ids.append(pid)
                        if counts[pid] == 0:
                            unique_count += 1
                        counts[pid] += 1
                        
            # Check validity
            if unique_count == num_perms:
                # Accept deletion
                word = new_word
                return True
            else:
                # Revert
                for pid in new_ids:
                    counts[pid] -= 1
                    if counts[pid] == 0:
                        unique_count -= 1
                for pid in old_ids:
                    counts[pid] += 1
                    if counts[pid] == 1:
                        unique_count += 1
        return False

    # Perform multiple passes, alternating direction
    improved = True
    while improved:
        improved = False
        if greedy_deletion_pass(False):
            improved = True
        if greedy_deletion_pass(True):
            improved = True

    return "".join(map(str, word))
