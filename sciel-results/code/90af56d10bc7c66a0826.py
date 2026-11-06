"""Two Sidon sets with disjoint difference sets – ILS with Bitsets."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations, combinations_with_replacement

N = 256
MAX_EACH = 64
DEADLINE = 55.0  # Safety margin

def _rebuild_bits(elems):
    """Rebuild sum_bits and diff_bits for a list of elements."""
    sum_bits = 0
    for a, b in combinations_with_replacement(elems, 2):
        sum_bits |= (1 << (a + b))
    diff_bits = 0
    for a, b in combinations(elems, 2):
        diff_bits |= (1 << abs(a - b))
    return sum_bits, diff_bits

def _can_add(elem, elems, sum_bits, diff_bits, other_diff_bits):
    """Check if elem can be added to the set defined by elems, sum_bits, diff_bits."""
    # Check sum collisions
    # elem + elem
    if (sum_bits >> (elem + elem)) & 1:
        return False
    # elem + a for each a in elems
    for a in elems:
        if (sum_bits >> (elem + a)) & 1:
            return False
    # Check difference collisions with own set
    for a in elems:
        d = abs(elem - a)
        if (diff_bits >> d) & 1:
            return False
    # Check difference disjointness with other set
    if (other_diff_bits >> abs(elem - 0)) & 1:
        pass  # We need to check differences between elem and all elements in the OTHER set
    # Actually we need to check differences between elem and each element in the other set
    # But other_diff_bits only contains differences within the other set.
    # We need to check if any (elem - x) for x in other_set is already in other_diff_bits? No.
    # We need to check if any (elem - x) for x in other_set creates a difference that is
    # already in the CURRENT set's diff_bits? No, the constraint is that the two sets'
    # difference sets are disjoint. So (elem - x) must not be in other_diff_bits AND
    # (elem - x) must not be in current diff_bits? No.
    
    # Let's re-read the constraint: "Their nonzero difference sets must be disjoint."
    # Diff(A) ∩ Diff(B) = ∅.
    # When adding elem to A, the new differences are {elem - a : a in A}.
    # These new differences must not be in Diff(B).
    # So for each a in A, we need (elem - a) not in Diff(B).
    # But we also need the new differences to be consistent with A's own Sidon property,
    # which is handled by the diff_bits check above (d not in diff_bits).
    
    # So we need to check: for all a in elems, (abs(elem - a)) not in other_diff_bits.
    # Wait, diff_bits stores abs differences. other_diff_bits stores abs differences.
    # So we need abs(elem - a) not in other_diff_bits for all a in elems.
    
    for a in elems:
        d = abs(elem - a)
        if (other_diff_bits >> d) & 1:
            return False
            
    return True

def _greedy_build(start_elem=1, other_diff_bits=0, max_size=64):
    """Greedily build a Sidon set with disjoint differences."""
    elems = []
    sum_bits = 0
    diff_bits = 0
    
    # Try to add start_elem first if valid
    if start_elem >= 1 and start_elem <= N:
        if (other_diff_bits >> 0) & 1:  # diff 0 is never in diff_bits since we use combinations
            pass
        # Check if start_elem can be added to empty set
        # Sum: start_elem + start_elem
        # Diff: none
        # So it's always valid for empty set unless other_diff_bits has issues?
        # Other_diff_bits is from the OTHER set. Since this set is empty, no new diffs.
        # So it's valid.
        elems.append(start_elem)
        sum_bits |= (1 << (start_elem + start_elem))
        diff_bits = 0  # No pairs yet
    
    # Try to add other elements greedily
    for x in range(1, N + 1):
        if x in elems:
            continue
        if len(elems) >= max_size:
            break
        if _can_add(x, elems, sum_bits, diff_bits, other_diff_bits):
            elems.append(x)
            # Update sum_bits
            for a in elems[:-1]:
                sum_bits |= (1 << (x + a))
            sum_bits |= (1 << (x + x))
            # Update diff_bits
            for a in elems[:-1]:
                d = abs(x - a)
                diff_bits |= (1 << d)
    
    return elems, sum_bits, diff_bits

def _local_search(A, B, sum_bits_A, diff_bits_A, sum_bits_B, diff_bits_B, time_limit):
    """Perform local search to improve |A| + |B|."""
    start_time = time.time()
    best_size = len(A) + len(B)
    best_A = list(A)
    best_B = list(B)
    
    # We'll try to add elements to A or B, or do swaps.
    # For simplicity, we'll try to add elements first.
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Try to add an element to A
        for x in range(1, N + 1):
            if x in A:
                continue
            if len(A) >= MAX_EACH:
                break
            if _can_add(x, A, sum_bits_A, diff_bits_A, diff_bits_B):
                A.append(x)
                for a in A[:-1]:
                    sum_bits_A |= (1 << (x + a))
                sum_bits_A |= (1 << (x + x))
                for a in A[:-1]:
                    d = abs(x - a)
                    diff_bits_A |= (1 << d)
                improved = True
                break
        
        # Try to add an element to B
        if not improved:
            for x in range(1, N + 1):
                if x in B:
                    continue
                if len(B) >= MAX_EACH:
                    break
                if _can_add(x, B, sum_bits_B, diff_bits_B, diff_bits_A):
                    B.append(x)
                    for b in B[:-1]:
                        sum_bits_B |= (1 << (x + b))
                    sum_bits_B |= (1 << (x + x))
                    for b in B[:-1]:
                        d = abs(x - b)
                        diff_bits_B |= (1 << d)
                    improved = True
                    break
        
        if not improved:
            break
        
        if len(A) + len(B) > best_size:
            best_size = len(A) + len(B)
            best_A = list(A)
            best_B = list(B)
    
    return best_A, best_B

def construct():
    start_time = time.time()
    
    # Initial greedy build
    # Build A first
    A, sum_bits_A, diff_bits_A = _greedy_build(1, 0, MAX_EACH)
    
    # Build B with A's differences
    B, sum_bits_B, diff_bits_B = _greedy_build(1, diff_bits_A, MAX_EACH)
    
    # Local search
    time_left = DEADLINE - (time.time() - start_time) - 5.0  # Reserve 5s for safety
    if time_left > 1.0:
        A, B = _local_search(A, B, sum_bits_A, diff_bits_A, sum_bits_B, diff_bits_B, time_left)
    
    # Final check and return
    # Verify constraints
    # A and B must be nonempty
    if not A:
        A = [1]
    if not B:
        B = [2] if 2 not in A else [3]
    
    return [A, B]
# EVOLVE-BLOCK-END
