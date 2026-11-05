"""Construct a small ternary covering code of length 8, radius 1.

Strategy:
- Use bitset representation (Python integers) for fast coverage operations.
- Greedy set cover to find an initial solution.
- Local search (removal and swap) to minimize the number of codewords.
"""

import itertools
import random
import time

# Global cache
_all_words = None
_word_to_idx = None
_cover_masks = None

def _precompute():
    """Precompute all words and their coverage masks as integers."""
    n = 8
    q = 3
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    total_words = len(all_words)
    
    cover_masks = []
    for w in all_words:
        mask = 0
        # Self
        mask |= (1 << word_to_idx[w])
        # Neighbors
        for pos in range(n):
            for alt in range(q):
                if alt == w[pos]:
                    continue
                w2 = list(w)
                w2[pos] = alt
                w2 = tuple(w2)
                mask |= (1 << word_to_idx[w2])
        cover_masks.append(mask)
        
    return all_words, word_to_idx, cover_masks

def _ensure_precomputed():
    global _all_words, _word_to_idx, _cover_masks
    if _all_words is None:
        _all_words, _word_to_idx, _cover_masks = _precompute()

def construct(q, n, radius):
    """Return a list of codewords (each a list of n ints in 0..q-1) covering
    all of {0..q-1}^n within Hamming distance radius."""
    _ensure_precomputed()
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback: trivial construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    all_words = _all_words
    cover_masks = _cover_masks
    total_words = len(all_words)
    full_mask = (1 << total_words) - 1
    
    # 1. Greedy Set Cover
    uncovered = full_mask
    selected = set()
    
    while uncovered:
        best_idx = -1
        best_count = 0
        # Find the codeword covering the most uncovered points
        for i in range(total_words):
            if i in selected:
                continue
            # Count bits in (uncovered & cover_masks[i])
            count = (uncovered & cover_masks[i]).bit_count()
            if count > best_count:
                best_count = count
                best_idx = i
                if best_count == 0:
                    break
        if best_idx < 0:
            break
        selected.add(best_idx)
        uncovered &= ~cover_masks[best_idx]
        
    # Helper to check redundancy
    def is_redundant(idx_to_remove, current_selected):
        # Compute coverage of all selected except idx_to_remove
        union_mask = 0
        for idx in current_selected:
            if idx != idx_to_remove:
                union_mask |= cover_masks[idx]
        return union_mask == full_mask

    # 2. Local Search: Removal
    changed = True
    while changed:
        changed = False
        for idx in list(selected):
            if is_redundant(idx, selected):
                selected.remove(idx)
                changed = True

    # 3. Local Search: Swap / Add-Remove
    # We try to improve the solution by swapping codewords.
    # Strategy: 
    # 1. Pick a random codeword A from selected.
    # 2. Pick a random codeword B from unselected.
    # 3. Replace A with B.
    # 4. If the new set is a valid cover, try to remove redundant codewords.
    # 5. If the size decreased, keep the change.
    
    # To make it more effective, we can try multiple swaps per iteration.
    # Also, we can bias B to be "good" (e.g., covers many points currently covered only by A?).
    # For simplicity, random is a good start.
    
    max_swap_iterations = 100
    no_improvement_count = 0
    last_size = len(selected)
    
    for _ in range(max_swap_iterations):
        if not selected:
            break
            
        # Pick random A
        A = random.choice(list(selected))
        # Pick random B
        B = random.choice(range(total_words))
        
        if A == B:
            continue
            
        # Tentatively swap
        temp_selected = set(selected)
        temp_selected.remove(A)
        temp_selected.add(B)
        
        # Check if valid cover
        union = 0
        for idx in temp_selected:
            union |= cover_masks[idx]
            
        if union != full_mask:
            continue
            
        # Valid cover. Try to remove redundant.
        improved = False
        for idx in list(temp_selected):
            if is_redundant(idx, temp_selected):
                temp_selected.remove(idx)
                improved = True
        
        # If we removed at least one, size decreased
        if improved:
            selected = temp_selected
            last_size = len(selected)
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count > 20:
                break

    # Final cleanup removal
    changed = True
    while changed:
        changed = False
        for idx in list(selected):
            if is_redundant(idx, selected):
                selected.remove(idx)
                changed = True

    result = [list(all_words[idx]) for idx in sorted(selected)]
    return result
