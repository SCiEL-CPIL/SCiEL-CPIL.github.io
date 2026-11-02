import time
import itertools

# EVOLVE-BLOCK-START
def construct(n):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin

    # 1. Generate initial recursive superpermutation
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
    
    if n <= 4:
        return "".join(map(str, word))

    # 2. Optimization
    word = optimize(word, n, deadline)
    
    return "".join(map(str, word))

def optimize(word, n, deadline):
    perms = list(itertools.permutations(range(1, n+1)))
    perm_to_rank = {p: i for i, p in enumerate(perms)}
    num_perms = len(perms)
    counts = [0] * num_perms
    
    # Initialize counts
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i+n])
        if len(set(window)) == n:
            try:
                rank = perm_to_rank[window]
                counts[rank] += 1
            except KeyError:
                pass

    # Helper to get rank of a window, assuming it's a valid perm
    def get_rank(w):
        return perm_to_rank[w]

    # Greedy Deletion
    # We try to delete characters from left to right.
    # If valid, delete and restart.
    
    # To handle "stuck" states, we can try swaps if no deletions are found in a full pass.
    # But let's first see if pure deletion works.
    
    # We will loop until no deletion is possible or time runs out.
    while time.time() < deadline:
        deleted = False
        L = len(word)
        for i in range(L):
            if time.time() > deadline:
                break
            
            # Identify affected windows
            # Old windows: start indices i-n+1 to i
            # New windows: start indices i-n+1 to i-1 (shifted)
            
            old_starts = list(range(max(0, i-n+1), min(i+1, L-n+1)))
            new_starts = list(range(max(0, i-n+1), min(i, L-n))) # Note: L decreases by 1, so max start is L-1-n
            
            # Decrement old
            valid = True
            touched = []
            for k in old_starts:
                w = tuple(word[k:k+n])
                if len(set(w)) == n:
                    rank = perm_to_rank[w]
                    counts[rank] -= 1
                    touched.append(rank)
                    if counts[rank] == 0:
                        valid = False
                        break
            
            if not valid:
                # Revert
                for rank in touched:
                    counts[rank] += 1
                continue
            
            # Increment new
            # The new windows are formed by shifting.
            # For each new start k, the window is word[k..k+n-1] AFTER deletion.
            # We can simulate this by taking word, removing i, and slicing.
            # But we want to avoid creating a new list for every check.
            # We can construct the tuple manually.
            
            # Actually, if we didn't revert, we can just build the new window tuples.
            for k in new_starts:
                # Construct window from word, skipping index i
                # If k < i: window is word[k..i-1] + word[i+1..k+n]
                # If k >= i: window is word[k..k+n-1] (but indices are shifted)
                # Wait, the indices in `word` are still the old ones.
                # We are simulating deletion at i.
                
                # Let's just build the tuple.
                # Since n is small (7), this is fast.
                w_list = []
                count = 0
                for j in range(k, k+n):
                    if j == i:
                        continue
                    if j < 0 or j >= L:
                        break
                    w_list.append(word[j])
                    count += 1
                    if count == n:
                        break
                
                if len(w_list) == n and len(set(w_list)) == n:
                    w = tuple(w_list)
                    rank = perm_to_rank[w]
                    counts[rank] += 1
            
            # If we reach here, the deletion is valid
            # Perform deletion
            del word[i]
            deleted = True
            break # Restart loop
            
        if not deleted:
            # No deletions found in this pass
            # Try swaps?
            # For now, break to avoid infinite loop if no swaps implemented
            break
            
    return word

# EVOLVE-BLOCK-END
