"""Shortest superpermutation construction with greedy block reordering."""
import sys
import time
import random
from itertools import permutations

# Deadline guard
_DEADLINE = 115.0  # seconds, safety margin before 120s limit
_start_time = time.time()


def _elapsed():
    return time.time() - _start_time


def construct(n):
    """Construct a superpermutation on n symbols.
    
    Uses the classic recursive construction with greedy nearest-neighbor
    reordering of permutation blocks to maximize overlaps, plus 2-opt
    local search refinement for the final step.
    """
    global _start_time
    _start_time = time.time()
    
    # For small n, just use the classic construction
    if n <= 1:
        return "1"
    
    # Classic recursive construction
    word = [1]
    
    for m in range(2, n + 1):
        # Extract all distinct (m-1)-permutations from the current word
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Try to reorder perms to maximize overlap
        if len(perms) <= 200:
            perms = _greedy_reorder(perms, m)
        elif m == n and len(perms) <= 1000:
            # For the final step with manageable size, use enhanced reordering
            perms = _enhanced_reorder(perms, m)
        
        # Build new word
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find maximum overlap with end of new
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        
        word = new
        
        # Check deadline
        if _elapsed() > _DEADLINE - 5:
            break
    
    return "".join(map(str, word))


def _compute_overlap_matrix(blocks):
    """Compute overlap matrix: overlap[i][j] = max overlap of end of blocks[i] with start of blocks[j]"""
    n_perms = len(blocks)
    overlap = [[0] * n_perms for _ in range(n_perms)]
    
    for i in range(n_perms):
        for j in range(n_perms):
            if i == j:
                continue
            bi = blocks[i]
            bj = blocks[j]
            max_ov = min(len(bi), len(bj))
            for o in range(max_ov, -1, -1):
                if o == 0 or bi[len(bi) - o:] == bj[:o]:
                    overlap[i][j] = o
                    break
    
    return overlap


def _greedy_order(overlap, n_perms, start=0):
    """Greedy nearest-neighbor ordering from a given start."""
    order = [start]
    remaining = set(range(n_perms)) - {start}
    
    while remaining:
        best_next = None
        best_ov = -1
        for j in remaining:
            last = order[-1]
            ov = overlap[last][j]
            if ov > best_ov:
                best_ov = ov
                best_next = j
        if best_next is None:
            break
        order.append(best_next)
        remaining.remove(best_next)
    
    return order


def _order_length(order, blocks, overlap):
    """Compute total length of the concatenated blocks in the given order."""
    if not order:
        return 0
    total = len(blocks[order[0]])
    for i in range(1, len(order)):
        total += len(blocks[order[i]]) - overlap[order[i-1]][order[i]]
    return total


def _two_opt_refine(order, blocks, overlap, max_iterations=10000):
    """Apply 2-opt local search to improve the ordering."""
    n_perms = len(order)
    if n_perms < 2:
        return order
    
    best_order = list(order)
    best_length = _order_length(best_order, blocks, overlap)
    improved = True
    iterations = 0
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        if _elapsed() > _DEADLINE - 3:
            break
        
        for i in range(n_perms - 1):
            for j in range(i + 1, n_perms):
                # Try reversing segment [i+1..j]
                new_order = best_order[:i+1] + best_order[i+1:j+1][::-1] + best_order[j+1:]
                new_length = _order_length(new_order, blocks, overlap)
                if new_length < best_length:
                    best_order = new_order
                    best_length = new_length
                    improved = True
    
    return best_order


def _greedy_reorder(perms, m):
    """Reorder permutation blocks using greedy nearest-neighbor to maximize overlaps."""
    if len(perms) <= 1:
        return perms
    
    # Precompute the full block for each perm
    blocks = [list(p) + [m] + list(p) for p in perms]
    n_perms = len(perms)
    
    # Compute overlap matrix
    overlap = _compute_overlap_matrix(blocks)
    
    # Try all starting points and pick the best
    if n_perms <= 50:
        best_order = None
        best_length = float('inf')
        
        for start in range(n_perms):
            if _elapsed() > _DEADLINE - 3:
                break
            order = _greedy_order(overlap, n_perms, start)
            if len(order) == n_perms:
                length = _order_length(order, blocks, overlap)
                if length < best_length:
                    best_length = length
                    best_order = order
        
        if best_order is not None:
            return [perms[i] for i in best_order]
    
    else:
        # For larger n, do a single greedy pass from the first element
        order = _greedy_order(overlap, n_perms, 0)
        if len(order) == n_perms:
            return [perms[i] for i in order]
    
    # Fallback: original order
    return perms


def _enhanced_reorder(perms, m):
    """Enhanced reordering with multiple random restarts and 2-opt refinement.
    
    Used for the final step where we have a moderate number of permutations
    (e.g., 720 for n=7).
    """
    if len(perms) <= 1:
        return perms
    
    # Precompute the full block for each perm
    blocks = [list(p) + [m] + list(p) for p in perms]
    n_perms = len(perms)
    
    # Compute overlap matrix
    overlap = _compute_overlap_matrix(blocks)
    
    best_order = None
    best_length = float('inf')
    
    # Try multiple random restarts
    n_restarts = min(50, max(10, (int)(_DEADLINE - _elapsed()) * 5))
    
    for r in range(n_restarts):
        if _elapsed() > _DEADLINE - 3:
            break
        
        # Choose a random start
        start = random.randint(0, n_perms - 1)
        order = _greedy_order(overlap, n_perms, start)
        
        if len(order) != n_perms:
            continue
        
        # Apply 2-opt refinement
        order = _two_opt_refine(order, blocks, overlap, max_iterations=5000)
        
        length = _order_length(order, blocks, overlap)
        if length < best_length:
            best_length = length
            best_order = list(order)
    
    if best_order is not None:
        return [perms[i] for i in best_order]
    
    # Fallback: original order
    return perms
