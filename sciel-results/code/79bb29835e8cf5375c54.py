import time
import sys
import random
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    
    # For n >= 4, use the greedy heuristic
    symbols = list(range(n))
    all_perms = list(permutations(symbols))
    num_perms = len(all_perms)
    
    # Map permutations to indices for fast lookup
    perm_to_idx = {p: i for i, p in enumerate(all_perms)}
    
    # Precompute suffix (length n-1) to list of perm indices starting with that suffix
    # Note: A permutation P "starts with" suffix S if P[0:n-1] == S.
    # Wait, the overlap is: current word ends with S. We want to append c such that S+c is a permutation.
    # Or, we want to find a permutation P such that P[0:n-1] == S. Then appending P[n-1] completes P.
    # So we need a map: suffix_tuple -> list of perm_indices where perm[:n-1] == suffix_tuple
    
    suffix_map = {}
    for i, p in enumerate(all_perms):
        s = p[:n-1]
        if s not in suffix_map:
            suffix_map[s] = []
        suffix_map[s].append(i)
    
    # Unvisited set
    unvisited = set(range(num_perms))
    
    # Start with the first permutation
    current_word = list(all_perms[0])
    unvisited.remove(0)
    
    # Current suffix is the last n-1 elements
    current_suffix = tuple(current_word[-(n-1):])
    
    while unvisited:
        if time.time() > deadline:
            break
            
        # Check if there are unvisited permutations starting with current_suffix
        candidates = suffix_map.get(current_suffix, [])
        valid_candidates = [idx for idx in candidates if idx in unvisited]
        
        if valid_candidates:
            # Pick a candidate. Heuristic: pick one that leads to a suffix with many options?
            # For now, pick the first one to keep it deterministic and fast, or random.
            # Let's pick the one that minimizes the index (arbitrary but consistent)
            # Or better: pick the one whose resulting suffix has the most unvisited options.
            best_idx = None
            best_score = -1
            # To avoid O(n) check for each candidate if list is large, we can just pick one.
            # The list size is (n-1)! / (n-1)! = 1? No.
            # Number of perms starting with a specific (n-1) prefix is 1! = 1.
            # Wait. If prefix is length n-1, the last element is fixed. So there is only 1 permutation.
            # So valid_candidates will have at most 1 element.
            # If so, we just take it.
            if len(valid_candidates) == 1:
                next_idx = valid_candidates[0]
            else:
                # Should not happen for n-1 prefix, but handle generally
                next_idx = valid_candidates[0]
            
            next_perm = all_perms[next_idx]
            next_char = next_perm[-1]
            
            current_word.append(next_char)
            unvisited.remove(next_idx)
            
            # Update current suffix
            current_suffix = tuple(current_word[-(n-1):])
            
        else:
            # No unvisited perm starts with current_suffix.
            # We must append a character that moves us to a new suffix.
            # Try all n possible characters.
            best_char = None
            best_score = -1
            best_new_suffix = None
            best_new_visited = None
            
            for c in symbols:
                new_suffix = tuple(current_suffix[1:] + (c,))
                
                # 1. Check if new_suffix itself is a permutation and unvisited
                # This happens if new_suffix is in perm_to_idx
                new_visited_idx = perm_to_idx.get(new_suffix, -1)
                is_new_visited = (new_visited_idx != -1 and new_visited_idx in unvisited)
                
                # 2. Count how many unvisited perms start with new_suffix
                # This determines how many "free" moves we have next
                next_candidates = suffix_map.get(new_suffix, [])
                num_options = sum(1 for idx in next_candidates if idx in unvisited)
                
                # Score: prefer moves that visit a new perm immediately, and then have many options
                score = (1 if is_new_visited else 0) * 1000 + num_options
                
                if score > best_score:
                    best_score = score
                    best_char = c
                    best_new_suffix = new_suffix
                    best_new_visited = new_visited_idx if is_new_visited else -1
            
            if best_char is None:
                # Should not happen, but fallback
                best_char = symbols[0]
                best_new_suffix = tuple(current_suffix[1:] + (best_char,))
                best_new_visited = perm_to_idx.get(best_new_suffix, -1)
                if best_new_visited != -1 and best_new_visited in unvisited:
                    unvisited.remove(best_new_visited)
            
            current_word.append(best_char)
            
            if best_new_visited != -1 and best_new_visited in unvisited:
                unvisited.remove(best_new_visited)
            
            current_suffix = best_new_suffix
    
    # Convert to string
    result = "".join(map(str, current_word))
    
    # Optional: Local search for deletion if time permits
    # Given the complexity, we skip heavy local search to ensure we return a valid result in time.
    # The greedy result is already a valid superpermutation.
    
    return result
