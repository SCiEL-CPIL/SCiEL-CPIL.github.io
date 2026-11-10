# EVOLVE-BLOCK-START
import numpy as np
from collections import deque
import itertools


def construct(q, n, radius):
    """Construct a covering code of length n over alphabet q with covering radius
    radius.  For the football-pool task this is (q, n, radius) = (3, 6, 1).
    
    Strategy:
      1. Greedy set cover to find an initial valid code.
      2. Local search with Delete and Directed Swap moves to minimize size.
    """
    total = q ** n
    
    def encode(word):
        v = 0
        for sym in word:
            v = v * q + sym
        return v

    def decode(v):
        out = []
        for _ in range(n):
            v, r = divmod(v, q)
            out.append(r)
        return tuple(reversed(out))

    # Precompute neighbors
    # neighbors[w] = list of words within Hamming distance <= radius
    neighbors = [None] * total
    for w in range(total):
        word = decode(w)
        neigh = set([w])
        if radius >= 1:
            for i in range(n):
                orig = word[i]
                for sym in range(q):
                    if sym != orig:
                        new_word = list(word)
                        new_word[i] = sym
                        neigh.add(encode(new_word))
        if radius >= 2:
            for i in range(n):
                for j in range(i + 1, n):
                    orig_i = word[i]
                    orig_j = word[j]
                    for sym_i in range(q):
                        for sym_j in range(q):
                            if sym_i == orig_i and sym_j == orig_j:
                                continue
                            new_word = list(word)
                            new_word[i] = sym_i
                            new_word[j] = sym_j
                            neigh.add(encode(new_word))
        neighbors[w] = list(neigh)

    # 1. Greedy Initialization
    uncovered = set(range(total))
    code = []
    while uncovered:
        best_w = -1
        best_gain = 0
        for w in range(total):
            gain = 0
            for nb in neighbors[w]:
                if nb in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_w = w
        code.append(best_w)
        for nb in neighbors[best_w]:
            uncovered.discard(nb)
            
    # Convert code to list of ints for processing
    code_ints = code[:]
    
    # 2. Local Search
    # covered_by[w] = number of codewords covering w
    covered_by = [0] * total
    
    def update_coverage(cw_int, delta):
        for nb in neighbors[cw_int]:
            covered_by[nb] += delta

    # Initialize coverage
    for cw in code_ints:
        update_coverage(cw, 1)
        
    def is_valid():
        return all(c > 0 for c in covered_by)
        
    # Optimization loop
    max_iter = 1000
    for _ in range(max_iter):
        improved = False
        
        # Phase 1: Try to delete
        i = 0
        while i < len(code_ints):
            cw = code_ints[i]
            update_coverage(cw, -1)
            if is_valid():
                code_ints.pop(i)
                improved = True
                # Restart deletion phase with new list
                break 
            else:
                update_coverage(cw, 1)
                i += 1
                
        if improved:
            continue
            
        # Phase 2: Try to swap
        # For each codeword, try to replace it with another to enable a deletion
        swapped = False
        for i in range(len(code_ints)):
            cw_old = code_ints[i]
            update_coverage(cw_old, -1)
            
            # Find uncovered words
            uncovered_now = [w for w in range(total) if covered_by[w] == 0]
            if not uncovered_now:
                # This would have been caught in delete phase
                continue
                
            # Find candidates that cover all uncovered_now
            # Since radius is small, uncovered_now is small
            # We can just iterate through all possible codewords? 
            # Or restrict to those close to uncovered words?
            # For now, iterate all, it's 729 * 73 * 729 operations worst case?
            # 729 * 73 = 53k swaps to check. Each check involves finding candidates.
            # Finding candidates: for each w in uncovered_now, intersect neighbors[w].
            
            # Intersect neighbors of all uncovered words
            if len(uncovered_now) == 1:
                candidates = set(neighbors[uncovered_now[0]])
            else:
                candidates = set(neighbors[uncovered_now[0]])
                for w in uncovered_now[1:]:
                    candidates.intersection_update(neighbors[w])
                    if not candidates:
                        break
            
            if not candidates:
                update_coverage(cw_old, 1)
                continue
                
            # Try each candidate
            found_swap = False
            for cw_new in candidates:
                if cw_new == cw_old:
                    continue
                    
                # Tentatively add cw_new
                update_coverage(cw_new, 1)
                
                # Check if we can delete any codeword now
                # Note: cw_old is currently NOT in code_ints (we removed it)
                # We need to check if any remaining codeword can be removed
                # Or if cw_new itself is redundant (unlikely immediately)
                
                can_delete = False
                delete_idx = -1
                for j in range(len(code_ints)):
                    cw_test = code_ints[j]
                    update_coverage(cw_test, -1)
                    if is_valid():
                        can_delete = True
                        delete_idx = j
                        break
                    else:
                        update_coverage(cw_test, 1)
                        
                if can_delete:
                    # Accept swap and deletion
                    # 1. Remove cw_old (already done)
                    # 2. Add cw_new (already done)
                    # 3. Remove code_ints[delete_idx]
                    cw_del = code_ints[delete_idx]
                    update_coverage(cw_del, -1)
                    code_ints.pop(delete_idx)
                    
                    # We also need to add cw_new to code_ints
                    # Wait, we removed cw_old at index i.
                    # code_ints currently does NOT contain cw_old.
                    # We added cw_new to coverage, but not to list.
                    # We removed code_ints[delete_idx] from list and coverage.
                    # So we just need to add cw_new to the list.
                    code_ints.append(cw_new)
                    
                    swapped = True
                    found_swap = True
                    improved = True
                    break
                else:
                    # Revert cw_new
                    update_coverage(cw_new, -1)
                    
            if found_swap:
                # Revert cw_old removal? No, we replaced it.
                # But we need to fix the list state.
                # We popped cw_old? No, we didn't pop it from the list yet in this loop iteration?
                # Let's trace carefully.
                # Start: code_ints has cw_old at i.
                # update_coverage(cw_old, -1).
                # ...
                # If swap accepted:
                #   We added cw_new to coverage.
                #   We deleted code_ints[delete_idx] from list and coverage.
                #   We appended cw_new to list.
                #   But cw_old is still in code_ints at index i (or shifted).
                #   We need to remove cw_old from the list.
                
                # Find cw_old in code_ints and remove it.
                # It might have shifted if delete_idx < i.
                try:
                    idx_old = code_ints.index(cw_old)
                    code_ints.pop(idx_old)
                except ValueError:
                    pass # Should not happen
                
                break # Break out of for i loop, restart outer loop
            else:
                # Revert cw_old
                update_coverage(cw_old, 1)
                
        if swapped:
            continue
            
        # If no improvement, break
        break
        
    # Return as list of lists of ints
    result = [list(decode(cw)) for cw in code_ints]
    return result
# EVOLVE-BLOCK-END
