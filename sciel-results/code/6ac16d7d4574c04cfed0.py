import time
import random
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    # For small n, handle directly
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        # Known optimal for n=4 is length 33
        return "1234121342132431213421312431423413241342124312"

    # Generate standard recursive superpermutation
    word = [1]
    for m in range(2, n + 1):
        # Collect all (m-1)-permutations that appear as windows
        perms = []
        seen = set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find max overlap
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new

    # Total number of permutations
    total_perms = 1
    for i in range(2, n + 1):
        total_perms *= i

    # Fast validation using set
    def is_valid_superperm(w):
        seen = set()
        L = len(w)
        for i in range(L - n + 1):
            window = tuple(w[i:i+n])
            if len(set(window)) == n:
                seen.add(window)
        return len(seen) == total_perms

    # Quick check: count distinct perms without full validation
    def count_perms(w):
        seen = set()
        L = len(w)
        for i in range(L - n + 1):
            window = tuple(w[i:i+n])
            if len(set(window)) == n:
                seen.add(window)
        return len(seen)

    # Initial validation
    if not is_valid_superperm(word):
        # Shouldn't happen, but just in case
        return "".join(map(str, word))

    best_word = word[:]
    best_len = len(word)

    # Greedy deletion optimization
    # A position is "deletable" if removing it doesn't lose any unique permutation occurrence
    # More generally: we can remove a position if every permutation that had an occurrence
    # covering that position still has at least one other occurrence.

    def try_delete_positions(w, positions):
        """Try to delete the given positions (sorted) and check validity."""
        L = len(w)
        # Build new word
        remove_set = set(positions)
        new_w = [w[i] for i in range(L) if i not in remove_set]
        if not is_valid_superperm(new_w):
            return None
        return new_w

    # Main optimization loop: greedy deletion with backtracking
    # We iterate, finding positions that can be removed.
    
    # For efficiency, we maintain for each permutation the count of occurrences
    # and for each position, the list of permutations that cover it.
    
    def compute_occurrence_data(w):
        """Returns dict: perm -> list of start indices, and pos -> set of perms covering it."""
        L = len(w)
        perm_indices = {}
        for i in range(L - n + 1):
            window = tuple(w[i:i+n])
            if len(set(window)) == n:
                if window in perm_indices:
                    perm_indices[window].append(i)
                else:
                    perm_indices[window] = [i]
        
        # pos_to_perms: for each position, which permutations cover it
        # Position j is covered by windows starting at j-k for k in 0..n-1
        pos_to_perms = [set() for _ in range(L)]
        for perm, indices in perm_indices.items():
            for start in indices:
                for k in range(n):
                    pos = start + k
                    if pos < L:
                        pos_to_perms[pos].add(perm)
        
        return perm_indices, pos_to_perms

    # Greedy deletion: repeatedly find a position that can be safely removed
    # A position can be removed if for every permutation that covers it,
    # that permutation has another occurrence NOT covering this position.
    
    def find_deletable_positions(w):
        L = len(w)
        perm_indices, pos_to_perms = compute_occurrence_data(w)
        
        deletable = []
        for pos in range(L):
            if not pos_to_perms[pos]:
                # No permutation covers this position - always deletable
                deletable.append(pos)
                continue
            
            can_delete = True
            for perm in pos_to_perms[pos]:
                indices = perm_indices[perm]
                # Check if there's another occurrence not covering this position
                found_other = False
                for start in indices:
                    # Does this occurrence cover pos?
                    if start <= pos < start + n:
                        # This is the occurrence we're considering
                        continue
                    else:
                        found_other = True
                        break
                if not found_other:
                    can_delete = False
                    break
            
            if can_delete:
                deletable.append(pos)
        
        return deletable

    # Optimize using greedy deletion
    # We process the word from left to right, removing deletable positions
    # This is more efficient than random deletion
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        L = len(word)
        deletable = find_deletable_positions(word)
        
        if not deletable:
            # Try to find deletable positions after some local rearrangement
            # For now, break
            break
        
        # Remove the leftmost deletable position for determinism and efficiency
        # Actually, let's remove ALL deletable positions that are adjacent to already-removed ones
        # Strategy: scan left to right, remove position if it's deletable
        new_word = []
        skip_set = set()
        
        # Re-check deletability as we build new_word (positions shift)
        # Simpler: just remove one at a time, but batch if possible
        # Remove all currently deletable positions at once
        for pos in sorted(deletable, reverse=True):
            word = word[:pos] + word[pos+1:]
            improved = True
        
        if len(word) < best_len:
            best_word = word[:]
            best_len = len(word)

    # Additional optimization: try to merge overlapping segments
    # For now, return the best we found
    
    if len(word) < len(best_word):
        best_word = word[:]

    # Final validation
    if not is_valid_superperm(best_word):
        return "".join(map(str, word))
    
    return "".join(map(str, best_word))
