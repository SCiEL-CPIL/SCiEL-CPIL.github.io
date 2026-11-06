"""Two Sidon sets with disjoint difference sets – Bitset + multi-start local search."""
# EVOLVE-BLOCK-START
import time
import random

N = 256
MAX_EACH = 64
DEADLINE = 55.0  # safety margin before 60s kill


def _can_add(elem, elems, sum_bits, diff_bits, other_diff_bits):
    """Check if elem can be added to elems maintaining Sidon + disjoint diffs."""
    # Check elem+elem sum
    s2 = elem + elem
    if s2 <= 512 and (sum_bits >> s2) & 1:
        return False
    # Check elem+a sums and |elem-a| diffs
    for a in elems:
        s = elem + a
        if (sum_bits >> s) & 1:
            return False
        d = abs(elem - a)
        if (diff_bits >> d) & 1 or (other_diff_bits >> d) & 1:
            return False
    return True


def _add_elem(state, side, elem):
    """Add elem to side's set, updating state. Returns True if successful."""
    elems = state['sets'][side]
    other_side = 1 - side
    other_diff = state['diff_bits'][other_side]
    
    if not _can_add(elem, elems, state['sum_bits'][side], state['diff_bits'][side], other_diff):
        return False
    
    # Update sum bits
    for a in elems:
        state['sum_bits'][side] |= 1 << (elem + a)
    state['sum_bits'][side] |= 1 << (elem + elem)
    
    # Update diff bits
    for a in elems:
        d = abs(elem - a)
        state['diff_bits'][side] |= 1 << d
    
    state['sets'][side].append(elem)
    return True


def _remove_elem(state, side, elem):
    """Remove elem from side's set, updating state. Returns True if successful."""
    elems = state['sets'][side]
    if elem not in elems:
        return False
    
    idx = elems.index(elem)
    elems.pop(idx)
    
    # Clear sum bits
    for a in elems:
        state['sum_bits'][side] &= ~(1 << (elem + a))
    state['sum_bits'][side] &= ~(1 << (elem + elem))
    
    # Clear diff bits
    for a in elems:
        d = abs(elem - a)
        state['diff_bits'][side] &= ~(1 << d)
    
    return True


def _greedy_build():
    """Greedy construction of two Sidon sets with disjoint differences."""
    state = {
        'sets': [[], []],
        'sum_bits': [0, 0],
        'diff_bits': [0, 0]
    }
    
    # Try elements in a randomized order
    order = list(range(1, N + 1))
    random.shuffle(order)
    
    for elem in order:
        # Try adding to side 0 first, then side 1
        for side in (0, 1):
            if len(state['sets'][side]) < MAX_EACH:
                if _add_elem(state, side, elem):
                    break
    
    return state


def _swap_improve(state):
    """Try 1-for-1 and 1-for-2 swaps to improve the solution."""
    improved = True
    while improved:
        improved = False
        for side in (0, 1):
            elems = state['sets'][side]
            if not elems:
                continue
            
            # Try removing each element and adding a better one
            for i in range(len(elems)):
                old_elem = elems[i]
                # Try swapping with elements not in either set
                for new_elem in range(1, N + 1):
                    if new_elem in state['sets'][0] or new_elem in state['sets'][1]:
                        continue
                    
                    # Try the swap
                    _remove_elem(state, side, old_elem)
                    if _add_elem(state, side, new_elem):
                        improved = True
                        break
                    else:
                        _add_elem(state, side, old_elem)
    
    return state


def _score(state):
    """Compute the score: |A| + |B|."""
    return len(state['sets'][0]) + len(state['sets'][1])


def construct():
    start_time = time.time()
    best_state = None
    best_score = 0
    
    # Multiple random restarts
    while time.time() - start_time < DEADLINE:
        state = _greedy_build()
        
        # Try to improve with swaps
        state = _swap_improve(state)
        
        score = _score(state)
        if score > best_score:
            best_score = score
            best_state = state
        
        # If we've hit the theoretical max, stop
        if best_score >= 128:
            break
    
    if best_state is None:
        # Fallback: simple greedy
        best_state = _greedy_build()
    
    return best_state['sets']
# EVOLVE-BLOCK-END
