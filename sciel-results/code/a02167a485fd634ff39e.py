"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s limit to be safe
    
    N = 192
    MAX_K = 48
    
    # Initial greedy construction: Descending order
    # This is a strong baseline because large numbers are "safer"
    selected = []
    for x in range(N, 0, -1):
        if _is_valid_addition(x, selected):
            selected.append(x)
            if len(selected) >= MAX_K:
                break
        if time.time() > deadline:
            break
    
    best_set = set(selected)
    
    # Local search with swap operations to improve
    # Try to swap out smaller elements for larger unpicked ones
    # Or just try adding unpicked elements if possible (though greedy should have done this)
    # The main gain comes from trying different orders or repairing.
    
    # Let's try a simple local search:
    # 1. Identify candidates to remove (smaller ones) and candidates to add (larger ones not in set)
    # 2. Try swapping one out and one in.
    
    # Sort selected ascending to find smallest first for removal
    selected_list = sorted(selected)
    
    # Unpicked numbers
    picked_set = set(selected)
    unpicked = [x for x in range(N, 0, -1) if x not in picked_set]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing each element from the smallest up
        # and see if we can add a larger element
        # To be efficient, we only try to remove elements that are "blocking" many additions?
        # Simple heuristic: try removing the smallest element, then try adding the largest possible unpicked.
        
        if not selected_list:
            break
            
        # Try removing the smallest element
        # Actually, removing the smallest might not help if it's not causing conflicts.
        # Better: try swapping.
        
        # For each element 'out' in selected (start with smallest):
        #   For each element 'in' in unpicked (start with largest):
        #     If we can swap (remove 'out', add 'in'), do it.
        
        # This is O(N^2 * Check). Check is O(N^3). Might be too slow.
        # Let's limit the number of swaps we attempt per iteration.
        
        # Strategy:
        # 1. Try to add any unpicked element directly? (Greedy should have failed to add them)
        #    If greedy failed, it's because of conflict.
        # 2. So we must remove something.
        
        # Let's try removing the smallest element and see if we can add a larger one.
        # If yes, do it.
        
        out_idx = 0
        out_val = selected_list[0]
        
        # Try adding the largest unpicked element
        in_val = None
        for candidate in unpicked:
            # Check if we can add 'candidate' if we remove 'out_val'
            # New set = (selected - {out_val}) + {candidate}
            # This is equivalent to checking if 'candidate' is valid in (selected - {out_val})
            # AND if all other elements in (selected - {out_val}) are still valid with 'candidate'
            
            # We can just construct the new list and check validity of the whole set?
            # No, that's O(N^4).
            # We can check:
            # 1. 'candidate' is valid in S' = S - {out_val}
            # 2. For every a in S', a is valid with 'candidate' and other elements in S'
            #    Since S was valid, S' is valid.
            #    We only need to check conflicts involving 'candidate'.
            
            S_prime = [x for x in selected if x != out_val]
            
            # Check 1: candidate doesn't divide sum of 3 in S_prime
            # Check 2: No a in S_prime divides sum of 3 in S_prime + [candidate] involving candidate
            
            if _is_valid_addition(candidate, S_prime):
                in_val = candidate
                break
                
        if in_val is not None:
            # Perform swap
            selected.remove(out_val)
            selected.append(in_val)
            picked_set.remove(out_val)
            picked_set.add(in_val)
            unpicked.remove(in_val)
            unpicked.append(out_val)
            unpicked.sort(reverse=True)
            selected_list = sorted(selected)
            improved = True
            best_set = set(selected)
            if len(selected) >= MAX_K:
                break
        else:
            # If we couldn't swap with the smallest, maybe we should try removing a different element?
            # For now, if the smallest swap fails, we might be stuck or need to try other 'out' values.
            # Let's try the next smallest?
            # To avoid infinite loop or excessive time, let's just try a few.
            
            # If we can't improve by swapping the smallest, we might try random swaps or give up.
            # Given time constraints, let's just break if the smallest swap fails, assuming local optimum.
            # Or try a few more.
            
            # Let's try up to 5 smallest elements to remove
            found_swap = False
            for idx in range(1, min(5, len(selected_list))):
                out_val = selected_list[idx]
                for candidate in unpicked:
                    S_prime = [x for x in selected if x != out_val]
                    if _is_valid_addition(candidate, S_prime):
                        selected.remove(out_val)
                        selected.append(candidate)
                        picked_set.remove(out_val)
                        picked_set.add(candidate)
                        unpicked.remove(candidate)
                        unpicked.append(out_val)
                        unpicked.sort(reverse=True)
                        selected_list = sorted(selected)
                        improved = True
                        found_swap = True
                        best_set = set(selected)
                        break
                if found_swap:
                    break
            
            if not found_swap:
                break

    return list(best_set)

def _is_valid_addition(x, S):
    """
    Check if x can be added to S such that the new set is valid.
    S is a list of selected numbers.
    """
    # 1. x must not divide sum of any 3 distinct in S
    n = len(S)
    if n >= 3:
        for i in range(n):
            for j in range(i+1, n):
                s2 = S[i] + S[j]
                target_res = (-s2) % x
                for k in range(j+1, n):
                    if (S[i] + S[j] + S[k]) % x == 0:
                        return False
                # Check k < i and i < k < j?
                # The loop i<j<k covers all distinct triples.
                # We need to check ALL distinct triples.
                # My loop:
                # for i
                #   for j > i
                #     for k > j
                # This covers all sets {i,j,k}.
                # So yes, it covers all.
                
    # 2. No a in S divides sum of any 3 distinct in S + [x] involving x
    # Sums involving x are x + S[i] + S[j] for distinct i, j in S.
    # We need to check for each a in S:
    # Does a divide (x + S[i] + S[j]) for any distinct i, j?
    
    if n >= 2:
        for a in S:
            target_res = (-x) % a
            found = False
            for i in range(n):
                for j in range(i+1, n):
                    if (S[i] + S[j]) % a == target_res:
                        found = True
                        break
                if found:
                    break
            if found:
                return False
                
    return True
# EVOLVE-BLOCK-END
