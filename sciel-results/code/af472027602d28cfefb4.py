import math
import random

def construct(n):
    # Generate all permutations
    symbols = list(range(1, n + 1))
    perms = []
    def permute(s):
        if len(s) == 1:
            perms.append(tuple(s))
        else:
            for i in range(len(s)):
                permute(s[:i] + s[i+1:])
    permute(symbols)
    
    N = len(perms)
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    bit_for_perm = [1 << i for i in range(N)]
    
    # Initialize beam
    # States are (suffix, mask, parent_idx, char)
    # parent_idx is index in all_states, -1 for start
    all_states = []
    
    # Start with all unique suffixes (n-1)!
    # For each suffix S (perm of 1..n-1), find the unique perm P ending in S
    # P = [missing_elem] + list(S)
    seen_suffixes = set()
    initial_beam = []
    for p in perms:
        s = p[1:]
        if s not in seen_suffixes:
            seen_suffixes.add(s)
            idx = perm_to_idx[p]
            mask = bit_for_perm[idx]
            # State: suffix, mask, parent_idx (-1), char (last char of p? No, we store the whole path implicitly)
            # We need to reconstruct the path.
            # For start states, path is just p.
            # We can store the full path for start states in a separate dict?
            # Or just store the parent chain.
            # Let's store the full path as a string for the start states?
            # No, let's just store the parent chain.
            # For start states, parent is -1. We need to know the chars.
            # We can store a 'start_path' for parent -1?
            # Or simply, if parent is -1, the path is determined by the state?
            # No, the state only has suffix.
            # We need to store the full initial path.
            
            # Let's modify state to: (suffix, mask, parent_idx, char, start_path_str)
            # If parent_idx == -1, start_path_str is the full path.
            # Else, start_path_str is empty/None.
            
            start_path = "".join(map(str, p))
            state = (s, mask, -1, None, start_path)
            all_states.append(state)
            initial_beam.append(len(all_states) - 1)
            
    # Prune initial beam? It's (n-1)! = 720.
    # We can keep all 720.
    
    K = 500 # Beam width
    
    current_beam = initial_beam
    
    for step in range(n, n * N + 1): # Max length
        candidates = []
        
        for idx in current_beam:
            suffix, mask, parent, char, start_path = all_states[idx]
            
            for c in range(1, n + 1):
                new_suffix = suffix[1:] + (c,)
                new_perm = suffix + (c,)
                
                if new_perm in perm_to_idx:
                    p_idx = perm_to_idx[new_perm]
                    new_mask = mask | bit_for_perm[p_idx]
                else:
                    new_mask = mask
                    
                # New state
                # parent is idx
                # char is c
                # start_path is None (inherited)
                new_state = (new_suffix, new_mask, idx, c, None)
                all_states.append(new_state)
                candidates.append((len(all_states) - 1, new_mask.bit_count()))
                
        # Prune candidates
        # Sort by bit_count descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Keep top K
        # Also, we can deduplicate by (suffix, mask)
        # If we have multiple states with same suffix and mask, keep one (the one with shorter path? No, same length)
        # Just keep the first one encountered in the sorted list?
        
        seen_keys = set()
        new_beam = []
        for idx, count in candidates:
            suffix, mask, parent, char, start_path = all_states[idx]
            key = (suffix, mask)
            if key not in seen_keys:
                seen_keys.add(key)
                new_beam.append(idx)
                if len(new_beam) == K:
                    break
        
        current_beam = new_beam
        
        # Check if any state is complete
        for idx in current_beam:
            suffix, mask, parent, char, start_path = all_states[idx]
            if mask.bit_count() == N:
                # Reconstruct path
                path_chars = []
                curr = idx
                while curr != -1:
                    s = all_states[curr]
                    path_chars.append(s[3])
                    curr = s[2]
                # The last char added is at the end of the list
                # But the start_path contains the first n chars.
                # The chain contains the chars added *after* the start.
                # Wait, my state definition:
                # Start state: parent=-1, char=None, start_path="1234567"
                # Next state: parent=start_idx, char='8' (if n=7, c is 1..7)
                
                # Reconstruct:
                # Reverse path_chars
                path_chars.reverse()
                
                # Get start path of the root
                root_idx = curr # -1
                # Trace to root to get start_path
                curr = idx
                while all_states[curr][2] != -1:
                    curr = all_states[curr][2]
                start_str = all_states[curr][4]
                
                final_path = start_str + "".join(map(str, path_chars))
                return final_path

    # If loop finishes without finding, return the best one found
    # This shouldn't happen
    return "".join(map(str, symbols)) # Fallback

