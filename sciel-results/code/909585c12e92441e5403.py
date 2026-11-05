"""Shortest superpermutation construction with greedy block reordering."""
import sys
import time
from itertools import permutations

# Deadline guard
_DEADLINE = 115.0  # seconds, safety margin before 120s limit
_start_time = time.time()


def _elapsed():
    return time.time() - _start_time


def construct(n):
    """Construct a superpermutation on n symbols.
    
    Uses the classic recursive construction with greedy nearest-neighbor
    reordering of permutation blocks to maximize overlaps.
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
        if len(perms) <= 200:  # Only attempt reordering for manageable sizes
            perms = _greedy_reorder(perms, m)
        
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


def _greedy_reorder(perms, m):
    """Reorder permutation blocks using greedy nearest-neighbor to maximize overlaps.
    
    For each block p, the 'cost' of placing it after block q is determined by
    how well the end of (q + [m] + q) overlaps with the start of (p + [m] + p).
    We want to minimize total added length.
    """
    if len(perms) <= 1:
        return perms
    
    # Precompute the full block for each perm
    blocks = [list(p) + [m] + list(p) for p in perms]
    
    # Compute overlap matrix: overlap[i][j] = max overlap of end of blocks[i] with start of blocks[j]
    n_perms = len(perms)
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
    
    # Greedy: start with the block that has the smallest "penalty"
    # (i.e., the one that, when placed first, allows the best next overlaps)
    # Actually, start with any, then greedily pick the next that maximizes overlap
    
    # Start with the first perm (or we could try all starts for small n)
    if n_perms <= 50:
        # Try all starting points and pick the best
        best_order = None
        best_length = float('inf')
        
        for start in range(n_perms):
            if _elapsed() > _DEADLINE - 3:
                break
            order = [start]
            remaining = set(range(n_perms)) - {start}
            current_length = len(blocks[start])
            
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
                current_length += len(blocks[best_next]) - best_ov
            
            if len(order) == n_perms and current_length < best_length:
                best_length = current_length
                best_order = order
        
        if best_order is not None:
            return [perms[i] for i in best_order]
    
    else:
        # For larger n, just do a single greedy pass from the first element
        order = [0]
        remaining = set(range(1, n_perms))
        
        while remaining:
            if _elapsed() > _DEADLINE - 3:
                break
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
        
        if len(order) == n_perms:
            return [perms[i] for i in order]
    
    # Fallback: original order
    return perms
