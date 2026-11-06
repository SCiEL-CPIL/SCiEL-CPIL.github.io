"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _valid_insert(selected, x):
    """Check if x can be added to selected without creating a 3-AP."""
    s = set(selected)
    for a in selected:
        for b in selected:
            if a == b:
                continue
            # Check if a, b, x form a 3-AP
            # x is middle: 2x = a + b
            if 2*x == a + b:
                return False
            # a is middle: 2a = b + x
            if 2*a == b + x:
                return False
            # b is middle: 2b = a + x
            if 2*b == a + x:
                return False
    return True

def _reciprocal_sum(selected):
    """Compute sum of reciprocals as float."""
    return sum(1.0/x for x in selected)

def _greedy_ascending():
    selected = []
    for x in range(1, 257):
        if _valid_insert(selected, x):
            selected.append(x)
    return selected

def _local_search(initial, deadline):
    """Try to improve via adding missing elements and swaps."""
    selected = set(initial)
    best_sum = _reciprocal_sum(list(selected))
    best_set = set(selected)
    
    # Try adding each missing element
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for x in range(1, 257):
            if x not in selected:
                if _valid_insert(list(selected), x):
                    selected.add(x)
                    changed = True
                    if _reciprocal_sum(list(selected)) > best_sum:
                        best_sum = _reciprocal_sum(list(selected))
                        best_set = set(selected)
    
    # Try swaps: remove one, add one
    for x in range(1, 257):
        if time.time() > deadline:
            break
        if x not in selected:
            for y in list(selected):
                if time.time() > deadline:
                    break
                new_set = selected - {y}
                if _valid_insert(list(new_set), x):
                    new_set.add(x)
                    new_sum = _reciprocal_sum(list(new_set))
                    if new_sum > best_sum:
                        best_sum = new_sum
                        best_set = new_set
                        selected = set(new_set)
    
    return list(best_set)

def construct():
    deadline = time.time() + 80.0  # Safety margin before 90s limit
    
    # Start with greedy ascending
    initial = _greedy_ascending()
    
    # Try local search
    result = _local_search(initial, deadline)
    
    return result
# EVOLVE-BLOCK-END
