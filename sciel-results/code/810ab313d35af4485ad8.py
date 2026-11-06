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
    if (sum_bits >> (elem + elem)) & 1:
        return False
    for a in elems:
        if (sum_bits >> (elem + a)) & 1:
            return False
    # Check diff collisions within same set
    for a in elems:
        d = abs(elem - a)
        if (diff_bits >> d) & 1:
            return False
    # Check diff collisions with other set
    if (other_diff_bits >> 0) & 1:  # This won't happen since we use abs
        pass
    for a in elems:
        d = abs(elem - a)
        if (other_diff_bits >> d) & 1:
            return False
    return True

def _add_elem(state, elem, side):
    """Add elem to side's set, updating bits. Returns True if successful."""
    elems = state['elems'][side]
    sum_bits = state['sum_bits'][side]
    diff_bits = state['diff_bits'][side]
    other_diff_bits = state['diff_bits'][1 - side]
    
    if not _can_add(elem, elems, sum_bits, diff_bits, other_diff_bits):
        return False
    
    # Update sum_bits
    for a in elems:
        sum_bits |= (1 << (elem + a))
    sum_bits |= (1 << (elem + elem))
    
    # Update diff_bits
    for a in elems:
        d = abs(elem - a)
        diff_bits |= (1 << d)
    
    state['elems'][side].append(elem)
    state['sum_bits'][side] = sum_bits
    state['diff_bits'][side] = diff_bits
    return True

def _remove_elem(state, elem, side):
    """Remove elem from side's set, updating bits. Returns True if successful."""
    elems = state['elems'][side]
    if elem not in elems:
        return False
    
    idx = elems.index(elem)
    elems.pop(idx)
    sum_bits, diff_bits = _rebuild_bits(elems)
    state['sum_bits'][side] = sum_bits
    state['diff_bits'][side] = diff_bits
    return True

def _greedy_build(order_a, order_b):
    """Greedily build two sets given element orders."""
    state = {
        'elems': [[], []],
        'sum_bits': [0, 0],
        'diff_bits': [0, 0]
    }
    
    # Interleave or build separately? Let's build A first then B, or interleave.
    # Better to interleave to balance sizes, or just build both greedily in parallel.
    # Actually, since difference sets must be disjoint, the order matters.
    # Let's try adding elements in a round-robin fashion or just build A then B.
    # Building A then B might be suboptimal if B blocks A's differences.
    # Let's try a combined greedy: at each step, pick the best element from remaining
    # for either set. But that's complex. Let's just do A then B for simplicity in the
    # initial greedy, then use ILS to improve.
    
    # Actually, let's just iterate through all elements in a shuffled order and try to add
    # to either set.
    remaining = list(range(1, N + 1))
    random.shuffle(remaining)
    
    for elem in remaining:
        # Try to add to A
        if len(state['elems'][0]) < MAX_EACH and _can_add(elem, state['elems'][0], state['sum_bits'][0], state['diff_bits'][0], state['diff_bits'][1]):
            _add_elem(state, elem, 0)
        # Try to add to B
        elif len(state['elems'][1]) < MAX_EACH and _can_add(elem, state['elems'][1], state['sum_bits'][1], state['diff_bits'][1], state['diff_bits'][0]):
            _add_elem(state, elem, 1)
    
    return state['elems'][0], state['elems'][1]

def _score(A, B):
    """Return the score (sum of sizes) and validity."""
    # Check Sidon property for A
    sums_a = set()
    for a, b in combinations_with_replacement(A, 2):
        s = a + b
        if s in sums_a:
            return 0, False
        sums_a.add(s)
    
    # Check Sidon property for B
    sums_b = set()
    for a, b in combinations_with_replacement(B, 2):
        s = a + b
        if s in sums_b:
            return 0, False
        sums_b.add(s)
    
    # Check difference sets are disjoint
    diffs_a = set()
    for a, b in combinations(A, 2):
        d = abs(a - b)
        diffs_a.add(d)
    
    diffs_b = set()
    for a, b in combinations(B, 2):
        d = abs(a - b)
        diffs_b.add(d)
    
    if diffs_a & diffs_b:
        return 0, False
    
    return len(A) + len(B), True

def construct():
    start_time = time.time()
    
    # Initial greedy construction
    A, B = _greedy_build(None, None)
    best_A, best_B = A[:], B[:]
    best_score = len(A) + len(B)
    
    # Iterated Local Search (ILS)
    while time.time() - start_time < DEADLINE:
        # Perturbation: remove a random element from A or B
        if best_A:
            elem_to_remove = random.choice(best_A)
            A = [x for x in best_A if x != elem_to_remove]
            B = best_B[:]
        else:
            A = best_A[:]
            B = [x for x in best_B if x != random.choice(best_B)] if best_B else best_B[:]
        
        # Local improvement: try to add back elements
        remaining = [x for x in range(1, N + 1) if x not in A and x not in B]
        random.shuffle(remaining)
        
        for elem in remaining:
            if len(A) < MAX_EACH:
                state_a = {'elems': A, 'sum_bits': _rebuild_bits(A)[0], 'diff_bits': _rebuild_bits(A)[1]}
                state_b = {'elems': B, 'sum_bits': _rebuild_bits(B)[0], 'diff_bits': _rebuild_bits(B)[1]}
                if _can_add(elem, A, state_a['sum_bits'], state_a['diff_bits'], state_b['diff_bits']):
                    A.append(elem)
                    continue
            if len(B) < MAX_EACH:
                state_a = {'elems': A, 'sum_bits': _rebuild_bits(A)[0], 'diff_bits': _rebuild_bits(A)[1]}
                state_b = {'elems': B, 'sum_bits': _rebuild_bits(B)[0], 'diff_bits': _rebuild_bits(B)[1]}
                if _can_add(elem, B, state_b['sum_bits'], state_b['diff_bits'], state_a['diff_bits']):
                    B.append(elem)
        
        score, valid = _score(A, B)
        if valid and score > best_score:
            best_score = score
            best_A, best_B = A[:], B[:]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
