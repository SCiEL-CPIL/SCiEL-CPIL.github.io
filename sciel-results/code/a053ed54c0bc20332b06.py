"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Classic recursive construction
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
            # Find maximum overlap
            max_overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if new[len(new) - o:] == block[:o]:
                    max_overlap = o
                    break
            new += block[max_overlap:]
        word = new
    
    # Try to shorten the word
    if n >= 3 and time.time() < deadline:
        word = try_shorten_fast(word, n, deadline)
    
    return "".join(map(str, word))

def try_shorten_fast(word, n, deadline):
    """
    Optimized local search to remove redundant symbols.
    Uses the fact that deleting a symbol only affects O(n) windows.
    Improved: multiple passes, left-to-right and right-to-left, with early termination.
    """
    word = list(word)
    
    def get_perm_indices(w):
        perm_indices = defaultdict(list)
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i + n])
            if len(set(window)) == n:
                perm_indices[window].append(i)
        return perm_indices
    
    perm_indices = get_perm_indices(word)
    total_perms = len(perm_indices)
    
    # Try multiple passes in different directions
    for direction in ['right_to_left', 'left_to_right', 'right_to_left', 'left_to_right']:
        if time.time() >= deadline:
            break
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            if direction == 'right_to_left':
                indices_range = range(len(word) - 1, -1, -1)
            else:
                indices_range = range(len(word))
            
            for i in indices_range:
                if time.time() >= deadline:
                    return word
                
                # Identify windows affected by deletion at i
                start_k = max(0, i - n + 1)
                end_k = min(i, len(word) - n)
                
                if start_k > end_k:
                    continue
                
                # Collect permutations that ONLY appear in the affected range
                at_risk = False
                for k in range(start_k, end_k + 1):
                    p = tuple(word[k:k + n])
                    if len(set(p)) == n:
                        indices = perm_indices[p]
                        # If all indices are in [start_k, end_k], it is at risk
                        has_safe_index = False
                        for idx in indices:
                            if idx < start_k or idx > end_k:
                                has_safe_index = True
                                break
                        if not has_safe_index:
                            at_risk = True
                            break
                
                # If no at-risk permutations, deletion is safe
                if not at_risk:
                    word = word[:i] + word[i+1:]
                    perm_indices = get_perm_indices(word)
                    improved = True
                    break
            
    return word
# EVOLVE-BLOCK-END
