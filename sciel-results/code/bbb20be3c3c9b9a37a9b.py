import time
import random
import itertools
from collections import defaultdict

def construct(n):
    """
    Constructs a superpermutation of length n using a randomized greedy algorithm
    with prefix bucketing for efficiency.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second margin before 120s limit
    
    if n == 1:
        return "1"
    
    # Generate all permutations
    perms = list(itertools.permutations(range(1, n + 1)))
    num_perms = len(perms)
    
    # Pre-compute prefixes for all lengths 1 to n-1 for each permutation
    # This is expensive but done once per restart? No, perms are static.
    # We can pre-compute the prefixes for each perm once.
    perm_prefixes = []
    for p in perms:
        prefixes = [p[:k] for k in range(1, n)]
        perm_prefixes.append(prefixes)
        
    best_word = None
    best_len = float('inf')
    
    # Run multiple random restarts
    while time.time() < deadline:
        # Initialize buckets
        # buckets[k] is a dict mapping prefix (tuple of length k) to a set of indices of perms
        # We store indices to avoid storing tuples in sets repeatedly?
        # Actually, storing the perm tuple itself is fine.
        
        # We need to be able to remove a perm from all its prefix buckets.
        # Let's use a list of sets for each prefix length.
        # buckets[k] = { prefix: set_of_perm_indices }
        
        buckets = [defaultdict(set) for _ in range(n)] # index 0..n-1 for lengths 1..n
        
        # Populate buckets
        for idx, p in enumerate(perms):
            for k in range(1, n):
                prefix = p[:k]
                buckets[k-1][prefix].add(idx) # k-1 is index for length k
        
        # Start with a random permutation
        start_idx = random.randint(0, num_perms - 1)
        current_perm = perms[start_idx]
        word = list(current_perm)
        
        # Remove start_idx from buckets
        for k in range(1, n):
            prefix = current_perm[:k]
            buckets[k-1][prefix].discard(start_idx)
        
        used = {start_idx}
        
        # Greedy construction
        while len(used) < num_perms:
            # Check time
            if time.time() > deadline:
                break
                
            # Current tail is the last n-1 elements of the word
            # But we only need to check overlaps up to n-1
            tail = tuple(word[-(n-1):])
            
            chosen_idx = None
            chosen_overlap = 0
            
            # Try to find the longest prefix match
            # k goes from n-1 down to 1
            # The prefix to match is tail[-k:]
            for k in range(n-1, 0, -1):
                prefix_to_match = tail[-k:]
                # Look in buckets[k-1]
                candidates = buckets[k-1].get(prefix_to_match, None)
                if candidates:
                    # Pick a random candidate
                    # Convert set to list? Or just pop one?
                    # If we pop, we remove it from the set immediately, which is good.
                    # But we need to remove it from other buckets too.
                    # Let's pick one randomly.
                    # Using next(iter(...)) is arbitrary. random.choice needs a list.
                    # For small sets, list() is fine.
                    # To be safe and fast, we can just pick one.
                    # Since we want randomization, let's convert to list if small, or just pick arbitrary for speed?
                    # Arbitrary might lead to deterministic behavior if not careful.
                    # Let's use random.choice on a list.
                    cand_list = list(candidates)
                    chosen_idx = random.choice(cand_list)
                    chosen_overlap = k
                    break
            
            if chosen_idx is None:
                # No overlap found, pick any remaining perm
                # Find any unused index
                # This is slow if we scan. 
                # We can maintain a set of all unused indices.
                # Let's add an 'all_unused' set.
                # For now, fallback: scan buckets[0] (length 1 prefixes)
                # There must be something there.
                # Pick from any non-empty set in buckets[0]
                for prefix, cands in buckets[0].items():
                    if cands:
                        chosen_idx = random.choice(list(cands))
                        chosen_overlap = 1
                        break
            
            if chosen_idx is None:
                # Should not happen unless all used
                break
                
            chosen_perm = perms[chosen_idx]
            
            # Append the non-overlapping part
            # Overlap is chosen_overlap
            # Append chosen_perm[chosen_overlap:]
            word.extend(chosen_perm[chosen_overlap:])
            
            # Remove chosen_idx from all buckets
            for k in range(1, n):
                prefix = chosen_perm[:k]
                buckets[k-1][prefix].discard(chosen_idx)
                
            used.add(chosen_idx)
            
        # End of one run
        if len(used) == num_perms:
            current_len = len(word)
            if current_len < best_len:
                best_len = current_len
                best_word = word[:]
                # Optional: break if we hit a known good bound?
                # 5905 is the best known. If we get close, keep going.
        
    if best_word is None:
        # Fallback to classic recursive if greedy failed (shouldn't happen)
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
        return "".join(map(str, word))
        
    return "".join(map(str, best_word))
