import itertools
import random
import time

def construct(n):
    if n <= 1:
        return "1"
    
    # Generate all permutations
    all_perms = list(itertools.permutations(range(1, n + 1)))
    N = len(all_perms)
    
    # Map permutation to index
    perm_to_idx = {p: i for i, p in enumerate(all_perms)}
    
    # Precompute prefix sets
    # prefix_tuple -> set of indices
    # We need prefixes of length 1 to n-1
    prefix_sets = {}
    for i, p in enumerate(all_perms):
        for l in range(1, n):
            prefix = p[:l]
            if prefix not in prefix_sets:
                prefix_sets[prefix] = set()
            prefix_sets[prefix].add(i)
            
    best_word = None
    best_len = float('inf')
    
    # Run multiple trials to find a shorter path
    # Time limit is 120s. Let's run for a few seconds or a fixed number of trials.
    start_time = time.time()
    max_time = 60.0 # Leave some buffer
    
    trial = 0
    while time.time() - start_time < max_time:
        trial += 1
        # Random start
        start_idx = random.randint(0, N - 1)
        visited = {start_idx}
        word = list(all_perms[start_idx])
        last_idx = start_idx
        
        for _ in range(N - 1):
            curr_perm = all_perms[last_idx]
            found_next = False
            
            # Try longest overlap first
            # Suffix of curr_perm of length L is curr_perm[n-L:]
            # We need a perm starting with this suffix.
            for L in range(n - 1, 0, -1):
                suffix = tuple(curr_perm[n - L:])
                candidates = prefix_sets.get(suffix, set()) - visited
                
                if candidates:
                    # Pick a candidate
                    # Random pick to explore different paths
                    next_idx = random.choice(list(candidates))
                    
                    # Append the non-overlapping part
                    # The next perm is all_perms[next_idx]
                    # It starts with suffix (length L)
                    # So we append all_perms[next_idx][L:]
                    word.extend(all_perms[next_idx][L:])
                    visited.add(next_idx)
                    last_idx = next_idx
                    found_next = True
                    break
            
            if not found_next:
                # This is a dead end. In a superpermutation, this shouldn't happen
                # unless we've visited all. But we check len(visited) < N.
                # If it happens, we break and this trial is invalid.
                break
        
        if len(visited) == N:
            # Valid superpermutation
            if len(word) < best_len:
                best_len = len(word)
                best_word = word.copy()
                
        # Optional: Add a small local search improvement step on the best_word found so far?
        # The directive says "keep this program's approach and improve it" OR "design a new program".
        # I am designing a new program. I can add a simple local search on the resulting word
        # to delete redundant characters, similar to the original, but applied to the greedy result.
        # However, the greedy result is already quite compact.
        # Let's stick to the greedy construction for now to ensure speed.
        
    if best_word is None:
        # Fallback to a simple valid construction if greedy failed (unlikely)
        # Use the classic recursive one as fallback
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
                for o in range(min(len(new), len(block)), -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))
        
    return "".join(map(str, best_word))
