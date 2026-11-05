import time
import sys
import itertools
import random

def construct(n):
    if n < 7:
        # Classic recursive construction is optimal for n <= 6
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
                # Find overlap
                max_overlap = min(len(new), len(block))
                overlap = 0
                for o in range(max_overlap, -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))
    
    if n != 7:
        # Fallback for other n
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
                max_overlap = min(len(new), len(block))
                overlap = 0
                for o in range(max_overlap, -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    # n == 7 specific search
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    # Precompute permutations and transitions
    perms = []
    perm_to_id = {}
    for p in itertools.permutations(range(1, n+1)):
        perm_to_id[p] = len(perms)
        perms.append(p)
    
    num_perms = len(perms)
    full_mask = (1 << num_perms) - 1
    
    # Precompute next states for suffixes
    # Suffix is a tuple of n-1 distinct symbols
    # Next symbol c -> new suffix, new window
    # We can cache this
    transition_cache = {}
    def get_transitions(suffix):
        if suffix in transition_cache:
            return transition_cache[suffix]
        
        trans = []
        s_list = list(suffix)
        for c in range(1, n+1):
            new_suffix = tuple(s_list[1:] + [c])
            window = tuple(s_list + [c])
            
            # Check if window is a valid permutation
            # Since suffix has distinct symbols, window is a perm iff c is not in suffix
            if c not in s_list:
                pid = perm_to_id.get(window, -1)
                if pid != -1:
                    trans.append((new_suffix, 1 << pid))
                else:
                    trans.append((new_suffix, 0))
            else:
                # Window is not a permutation (repeated symbol)
                trans.append((new_suffix, 0))
                
        transition_cache[suffix] = trans
        return trans

    # Beam Search
    beam_width = 20
    best_solution = None
    best_length = 5913
    
    # Initialize beam
    # Start with a few random suffixes
    # Suffix must be a permutation of n-1 symbols
    start_suffixes = []
    # Use the first n-1 symbols of the standard construction
    # Or random
    all_suffixes = [p[:n-1] for p in perms]
    random.shuffle(all_suffixes)
    
    # Initial states: (suffix, mask, path)
    # Path is the list of symbols so far
    # To save memory, we can store path as a string or list
    # Since n=7, path length ~6000, list of ints is fine
    
    initial_beam = []
    for s in all_suffixes[:beam_width]:
        # Start with the permutation that has this suffix
        # Actually, we just start with the suffix. The first window is not formed yet?
        # No, the word starts with the first n symbols.
        # If we start with suffix s (len n-1), we need to add 1 symbol to form the first window.
        # So path = list(s) + [c]
        # But we don't know c yet.
        # Let's start with a full permutation.
        # Pick a perm p that starts with s? No, s is the suffix.
        # Let's just pick a random permutation to start.
        pass
        
    # Better initialization:
    # Start with a random permutation p0.
    # Path = list(p0)
    # Suffix = p0[1:]
    # Mask = 1 << id(p0)
    
    for _ in range(10): # 10 restarts
        if time.time() > deadline:
            break
            
        start_perm = random.choice(perms)
        path = list(start_perm)
        suffix = start_perm[1:]
        mask = 1 << perm_to_id[start_perm]
        
        beam = [(suffix, mask, path)]
        
        for step in range(1, 10000):
            if time.time() > deadline:
                break
                
            new_beam = []
            for suffix, mask, path in beam:
                trans = get_transitions(suffix)
                for new_suffix, new_mask_bit in trans:
                    new_mask = mask | new_mask_bit
                    new_path = path + [new_suffix[-1]] # The added symbol
                    
                    # Check if we found a solution
                    if new_mask == full_mask:
                        curr_len = len(new_path)
                        if curr_len < best_length:
                            best_length = curr_len
                            best_solution = new_path[:]
                            if best_length <= 5905:
                                return "".join(map(str, best_solution))
                        continue
                        
                    # Score: prioritize covering new perms
                    # new_mask_bit != 0 means we covered a new perm
                    # We want to maximize new perms covered
                    # If tie, we want to keep options open
                    # Simple score: 1 if new, 0 if not
                    # To break ties, we can use a random factor or a heuristic
                    # For now, just add to new_beam
                    
                    new_beam.append((new_suffix, new_mask, new_path, (1 if new_mask_bit else 0)))
                    
            if not new_beam:
                break
                
            # Sort by score (descending) and keep top beam_width
            # We need to handle the score properly
            # The score is just "did we cover a new perm?".
            # This is too greedy. It might get stuck.
            # Better score: Number of new perms covered in this step (0 or 1).
            # If 0, it's a "wasted" step.
            # We want to minimize wasted steps.
            
            # To avoid getting stuck in a loop of 0s, we can prioritize states
            # that have "more options" for the next step.
            # But calculating that is expensive.
            
            # Let's just sort by:
            # 1. New perm covered (1 > 0)
            # 2. Random tie-breaker to explore
            
            random.shuffle(new_beam) # Add randomness to tie-breaks
            new_beam.sort(key=lambda x: x[3], reverse=True)
            
            beam = [ (s, m, p) for s, m, p, _ in new_beam[:beam_width] ]
            
    if best_solution:
        return "".join(map(str, best_solution))
    else:
        # Fallback to classic
        word = [1]
        for m in range(2, n + 1):
            perms_list, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms_list.append(window)
            new = []
            for p in perms_list:
                block = list(p) + [m] + list(p)
                max_overlap = min(len(new), len(block))
                overlap = 0
                for o in range(max_overlap, -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))
