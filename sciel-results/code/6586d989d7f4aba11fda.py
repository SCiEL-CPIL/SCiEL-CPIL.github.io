import time
import math
import itertools
import random

def construct(n):
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s kill
    
    # Generate classic superpermutation
    word = generate_classic(n)
    
    target_count = math.factorial(n)
    
    # Local search with optimized tracking
    word = local_search_optimize(word, n, target_count, start_time, deadline)
    
    return "".join(map(str, word))

def generate_classic(n):
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
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1)
                          if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    return word

def local_search_optimize(word, n, target_count, start_time, deadline):
    """
    Local search using a lookup table for permutation ranks.
    Uses conservative checks for speed, and exact checks for stuck states.
    """
    L = len(word)
    if L < target_count + n - 1:
        return word
    
    # Precompute lookup table for permutation ranks
    # Map permutation tuple to rank (0 to n!-1)
    perms = list(itertools.permutations(range(1, n + 1)))
    rank_lookup = {p: i for i, p in enumerate(perms)}
    
    def get_rank(window_tuple):
        return rank_lookup[window_tuple]

    def is_valid_exact(candidate, n, target_count):
        L_c = len(candidate)
        if L_c < target_count + n - 1:
            return False
        seen = bytearray(target_count)
        for i in range(L_c - n + 1):
            r = get_rank(tuple(candidate[i:i+n]))
            seen[r] = 1
        # Check if all permutations are seen
        for v in seen:
            if v == 0:
                return False
        return True

    best_word = word[:]
    
    while time.time() < deadline:
        L = len(best_word)
        
        # Compute ranks and counts for current best_word
        window_ranks = [0] * (L - n + 1)
        rank_counts = [0] * target_count
        for i in range(L - n + 1):
            r = get_rank(tuple(best_word[i:i+n]))
            window_ranks[i] = r
            rank_counts[r] += 1
        
        improved = False
        
        # Phase 1: Conservative Deletion Search
        # Try to delete a character where the affected windows are "redundant" (count > 1)
        # This is O(L * n) per pass, but we break on first success.
        for i in range(1, L - 1):
            if time.time() > deadline:
                break
            
            # Check if all affected windows have count > 1
            # Affected windows: j in [i-n+1, i]
            j_min = max(0, i - n + 1)
            j_max = min(i, L - n)
            
            valid = True
            for j in range(j_min, j_max + 1):
                if rank_counts[window_ranks[j]] == 1:
                    valid = False
                    break
            
            if valid:
                # Perform deletion
                best_word = best_word[:i] + best_word[i+1:]
                improved = True
                break
        
        if not improved:
            # Phase 2: Randomized Exact Deletion Search
            # If conservative search failed, try random deletions with exact check
            # This helps escape local minima where unique perms might be reformed
            random_attempts = 200
            for _ in range(random_attempts):
                if time.time() > deadline:
                    break
                
                i = random.randint(1, L - 2)
                candidate = best_word[:i] + best_word[i+1:]
                
                if is_valid_exact(candidate, n, target_count):
                    best_word = candidate
                    improved = True
                    break
            
            if not improved:
                # Phase 3: Randomized Swaps (optional, if time permits)
                # Swaps can enable future deletions
                swap_attempts = 50
                for _ in range(swap_attempts):
                    if time.time() > deadline:
                        break
                    
                    i = random.randint(1, L - 2)
                    # Swap i and i+1
                    best_word[i], best_word[i+1] = best_word[i+1], best_word[i]
                    
                    # Check if this swap allows a deletion? 
                    # Or just check if the swap itself is valid (it is, since it's a permutation of the same set of windows? No, windows change).
                    # Actually, a swap changes the set of windows. It might break validity.
                    # So we must check validity of the swapped word.
                    if is_valid_exact(best_word, n, target_count):
                        # If valid, we don't gain length, but we might be in a better spot.
                        # However, we want to reduce length. 
                        # If the swap is valid, we can try to delete again.
                        # But for now, if no deletion was found, a valid swap doesn't help immediately.
                        # Let's revert the swap if it doesn't lead to immediate improvement?
                        # No, let's keep it if valid, and loop back to Phase 1.
                        break
                    else:
                        # Revert swap
                        best_word[i], best_word[i+1] = best_word[i+1], best_word[i]
                
                # If a valid swap was found, we break out of the random deletion loop
                # and go back to the main while loop to try Phase 1 again.
                # If no valid swap, we are stuck.
                if not improved:
                    # Check if we did a valid swap
                    # The 'improved' flag is not set by swaps.
                    # We need a separate flag.
                    pass # Simplification: If no deletion found, we are likely stuck.
                    # Let's just break to save time if we can't find a deletion.
                    break

        if not improved:
            break

    return best_word
