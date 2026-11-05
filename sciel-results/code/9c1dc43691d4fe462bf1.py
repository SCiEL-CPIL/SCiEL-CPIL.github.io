"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction
import time

def _subset_sums(elements):
    """Generate all subset sums of reciprocals of elements."""
    n = len(elements)
    sums = [Fraction(0, 1)]
    for i in range(n):
        r = Fraction(1, elements[i])
        new_sums = [s + r for s in sums]
        sums.extend(new_sums)
    return sums

def _has_violation(S_list):
    """Check if S_list has a reciprocal-subsum violation."""
    n = len(S_list)
    if n <= 1:
        return False
    recip = [Fraction(1, x) for x in S_list]
    for i in range(n):
        target = recip[i]
        others = [j for j in range(n) if j != i]
        m = len(others)
        half = m // 2
        left_idx = others[:half]
        right_idx = others[half:]
        
        # Meet in the middle
        left_sums = set()
        for mask in range(1 << len(left_idx)):
            s = Fraction(0, 1)
            for j in range(len(left_idx)):
                if mask & (1 << j):
                    s += recip[left_idx[j]]
            left_sums.add(s)
        
        for mask in range(1 << len(right_idx)):
            s = Fraction(0, 1)
            for j in range(len(right_idx)):
                if mask & (1 << j):
                    s += recip[right_idx[j]]
            if target - s in left_sums:
                # Check nonempty
                if mask != 0 or (target - s) != Fraction(0, 1):
                    return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 75
    
    full_set = set(range(1, 25))
    current = set(full_set)
    
    # Greedy removal
    for _ in range(30):
        if time.time() > deadline:
            break
        if not _has_violation(list(current)):
            break
        # Find a violation and remove an element
        S_list = list(current)
        n = len(S_list)
        recip = [Fraction(1, x) for x in S_list]
        removed = False
        for i in range(n):
            target = recip[i]
            others = [j for j in range(n) if j != i]
            m = len(others)
            half = m // 2
            left_idx = others[:half]
            right_idx = others[half:]
            
            left_sums = set()
            for mask in range(1 << len(left_idx)):
                s = Fraction(0, 1)
                for j in range(len(left_idx)):
                    if mask & (1 << j):
                        s += recip[left_idx[j]]
                left_sums.add(s)
            
            found = False
            for mask in range(1 << len(right_idx)):
                s = Fraction(0, 1)
                for j in range(len(right_idx)):
                    if mask & (1 << j):
                        s += recip[right_idx[j]]
                if target - s in left_sums:
                    if mask != 0 or (target - s) != Fraction(0, 1):
                        # Remove S_list[i]
                        current.discard(S_list[i])
                        found = True
                        removed = True
                        break
            if found:
                break
        if not removed:
            break
    
    # Add back
    removed_elems = full_set - current
    for x in sorted(removed_elems, reverse=True):
        if time.time() > deadline:
            break
        if not _has_violation(list(current | {x})):
            current = current | {x}
    
    # Local search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        removed_elems = full_set - current
        for x in list(current):
            if time.time() > deadline:
                break
            for y in sorted(removed_elems, reverse=True):
                if time.time() > deadline:
                    break
                if not _has_violation(list((current - {x}) | {y})):
                    current = (current - {x}) | {y}
                    improved = True
                    break
            if improved:
                break
    
    return list(current)

# EVOLVE-BLOCK-END
