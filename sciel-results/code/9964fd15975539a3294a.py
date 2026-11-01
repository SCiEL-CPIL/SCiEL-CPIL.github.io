import time
import random
import itertools
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = start_time + 115

    # Precompute all permutations
    all_perms = list(itertools.permutations(range(1, n + 1)))
    total_perms = len(all_perms)
    
    # Build a mapping from permutation to index for fast counting
    perm_to_idx = {p: i for i, p in enumerate(all_perms)}
    
    # Helper to compute permutation counts using integer array
    def compute_counts(w):
        counts = [0] * total_perms
        for i in range(len(w) - n + 1):
            t = tuple(w[i:i+n])
            idx = perm_to_idx.get(t)
            if idx is not None:
                counts[idx] += 1
        return counts
    
    def is_valid_from_counts(counts):
        for c in counts:
            if c < 1:
                return False
        return True
    
    # Fast validity check: only check affected windows
    def check_move_affected(w, old_w, changed_indices, counts):
        """Check if a move is valid by only examining affected windows.
        changed_indices: set of positions that changed in w vs old_w
        Returns new_counts if valid, None otherwise."""
        L = len(w)
        if L < n:
            return None
        
        # Determine which window positions are affected
        affected_windows = set()
        for idx in changed_indices:
            # Windows that include position idx
            start_lo = max(0, idx - n + 1)
            start_hi = min(idx, L - n)
            for s in range(start_lo, start_hi + 1):
                affected_windows.add(s)
        
        # Decrement counts for old windows
        for s in affected_windows:
            if s < len(old_w) - n + 1:
                t = tuple(old_w[s:s+n])
                idx = perm_to_idx.get(t)
                if idx is not None:
                    counts[idx] -= 1
        
        # Increment counts for new windows
        for s in affected_windows:
            if s < L - n + 1:
                t = tuple(w[s:s+n])
                idx = perm_to_idx.get(t)
                if idx is not None:
                    counts[idx] += 1
        
        # Check validity
        for c in counts:
            if c < 1:
                # Rollback
                for s in affected_windows:
                    if s < len(old_w) - n + 1:
                        t = tuple(old_w[s:s+n])
                        idx = perm_to_idx.get(t)
                        if idx is not None:
                            counts[idx] += 1
                    if s < L - n + 1:
                        t = tuple(w[s:s+n])
                        idx = perm_to_idx.get(t)
                        if idx is not None:
                            counts[idx] -= 1
                return None
        
        return counts
    
    # Generate the naive superpermutation (recursive construction)
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
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    best_word = word[:]
    current_len = len(best_word)
    current_counts = compute_counts(best_word)
    
    if not is_valid_from_counts(current_counts):
        return "".join(map(str, best_word))
    
    # Main search loop with multiple phases
    while time.time() < deadline:
        found_improvement = False
        
        # Phase 1: Try to find redundant positions to remove
        # More thorough: try all positions, use fast check
        L = len(best_word)
        if L > n:
            # Precompute which windows are unique
            unique_windows = set()
            for i in range(L - n + 1):
                t = tuple(best_word[i:i+n])
                pi = perm_to_idx.get(t)
                if pi is not None and current_counts[pi] == 1:
                    unique_windows.add(i)
            
            # For each position, check if it's in any unique window
            for idx in range(L):
                start_lo = max(0, idx - n + 1)
                start_hi = min(idx, L - n)
                
                in_unique = False
                for s in range(start_lo, start_hi + 1):
                    if s in unique_windows:
                        in_unique = True
                        break
                
                if not in_unique:
                    candidate = best_word[:]
                    del candidate[idx]
                    new_counts = compute_counts(candidate)
                    if is_valid_from_counts(new_counts):
                        best_word = candidate
                        current_len = len(best_word)
                        current_counts = new_counts
                        found_improvement = True
                        break
        
        # Phase 2: Try swap moves with fast check
        if not found_improvement and time.time() < deadline:
            for _ in range(2000):
                if time.time() > deadline:
                    break
                if len(best_word) <= n:
                    break
                i = random.randint(0, len(best_word) - 1)
                j = random.randint(0, len(best_word) - 1)
                if i == j:
                    continue
                candidate = best_word[:]
                candidate[i], candidate[j] = candidate[j], candidate[i]
                
                changed = {i, j}
                new_counts = check_move_affected(candidate, best_word, changed, current_counts[:])
                if new_counts is not None:
                    best_word = candidate
                    current_len = len(best_word)
                    current_counts = new_counts
                    found_improvement = True
                    break
        
        # Phase 3: Try insert-move with fast check
        if not found_improvement and time.time() < deadline:
            for _ in range(3000):
                if time.time() > deadline:
                    break
                if len(best_word) <= n:
                    break
                idx = random.randint(0, len(best_word) - 1)
                new_idx = random.randint(0, len(best_word) - 1)
                if idx == new_idx:
                    continue
                candidate = best_word[:]
                val = candidate.pop(idx)
                if new_idx > idx:
                    new_idx -= 1
                candidate.insert(new_idx, val)
                
                # Changed positions: idx and new_idx (and everything between if moved)
                changed = set()
                if idx < new_idx:
                    changed = set(range(idx, new_idx + 1))
                else:
                    changed = set(range(new_idx, idx + 1))
                
                new_counts = check_move_affected(candidate, best_word, changed, current_counts[:])
                if new_counts is not None:
                    best_word = candidate
                    current_len = len(best_word)
                    current_counts = new_counts
                    found_improvement = True
                    break
        
        # Phase 4: If no improvement, try a larger perturbation
        if not found_improvement and time.time() < deadline:
            if len(best_word) > n:
                seg_len = random.randint(1, min(5, len(best_word) - n))
                start_pos = random.randint(0, len(best_word) - seg_len)
                candidate = best_word[:]
                segment = candidate[start_pos:start_pos+seg_len]
                del candidate[start_pos:start_pos+seg_len]
                insert_pos = random.randint(0, len(candidate))
                candidate[insert_pos:insert_pos] = segment
                
                # Determine changed positions
                changed = set()
                if start_pos < insert_pos:
                    changed = set(range(start_pos, insert_pos + seg_len))
                else:
                    changed = set(range(insert_pos, start_pos + seg_len))
                
                new_counts = check_move_affected(candidate, best_word, changed, current_counts[:])
                if new_counts is not None:
                    best_word = candidate
                    current_len = len(best_word)
                    current_counts = new_counts
                    found_improvement = True
    
    return "".join(map(str, best_word))
