"""
Two Sidon sets with disjoint difference sets.
Uses Iterated Local Search with Bitmasks for fast constraint checking.
"""
# EVOLVE-BLOCK-START
import time
import random

N = 256
MAX_EACH = 64
DEADLINE = 55.0

def _init_state():
    """Initialize empty state."""
    return {
        'A': [],
        'B': [],
        'sum_A': 0,
        'diff_A': 0,
        'sum_B': 0,
        'diff_B': 0
    }

def _check_add(state, side, elem):
    """
    Check if elem can be added to side ('A' or 'B').
    Returns True if valid.
    """
    if side == 'A':
        sums = state['sum_A']
        diffs = state['diff_A']
        other_diffs = state['diff_B']
        elems = state['A']
    else:
        sums = state['sum_B']
        diffs = state['diff_B']
        other_diffs = state['diff_A']
        elems = state['B']
        
    # Check self-sum
    if (sums >> (elem + elem)) & 1:
        return False
        
    # Check sums with existing elements
    for e in elems:
        s = elem + e
        if (sums >> s) & 1:
            return False
        d = abs(elem - e)
        # Check diff against own diffs
        if (diffs >> d) & 1:
            return False
        # Check diff against other set's diffs
        if (other_diffs >> d) & 1:
            return False
            
    return True

def _add_elem(state, side, elem):
    """Add elem to side. Assumes validity has been checked."""
    if side == 'A':
        elems = state['A']
        # Update sums
        state['sum_A'] |= (1 << (elem + elem))
        for e in elems:
            state['sum_A'] |= (1 << (elem + e))
        # Update diffs
        for e in elems:
            d = abs(elem - e)
            state['diff_A'] |= (1 << d)
        elems.append(elem)
    else:
        elems = state['B']
        state['sum_B'] |= (1 << (elem + elem))
        for e in elems:
            state['sum_B'] |= (1 << (elem + e))
        for e in elems:
            d = abs(elem - e)
            state['diff_B'] |= (1 << d)
        elems.append(elem)

def _remove_elem(state, side, elem):
    """Remove elem from side. Rebuild masks for simplicity/safety in local search steps."""
    if side == 'A':
        state['A'].remove(elem)
        # Rebuild masks
        elems = state['A']
        s_mask = 0
        d_mask = 0
        for i in range(len(elems)):
            for j in range(i, len(elems)):
                s_mask |= (1 << (elems[i] + elems[j]))
        for i in range(len(elems)):
            for j in range(len(elems)):
                if i != j:
                    d_mask |= (1 << abs(elems[i] - elems[j]))
        state['sum_A'] = s_mask
        state['diff_A'] = d_mask
    else:
        state['B'].remove(elem)
        elems = state['B']
        s_mask = 0
        d_mask = 0
        for i in range(len(elems)):
            for j in range(i, len(elems)):
                s_mask |= (1 << (elems[i] + elems[j]))
        for i in range(len(elems)):
            for j in range(len(elems)):
                if i != j:
                    d_mask |= (1 << abs(elems[i] - elems[j]))
        state['sum_B'] = s_mask
        state['diff_B'] = d_mask

def _greedy_build(state, side, candidates):
    """Greedily add elements from candidates to side."""
    added = 0
    for c in candidates:
        if len(state[side]) >= MAX_EACH:
            break
        if _check_add(state, side, c):
            _add_elem(state, side, c)
            added += 1
    return added

def _local_search(state):
    """Perform local search until no improving move is found."""
    improved = True
    while improved:
        improved = False
        # 1. Try to add new elements
        for side in ['A', 'B']:
            if len(state[side]) >= MAX_EACH:
                continue
            # Try random candidates
            for _ in range(20):
                c = random.randint(1, N)
                if c in state['A'] or c in state['B']:
                    continue
                if _check_add(state, side, c):
                    _add_elem(state, side, c)
                    improved = True
                    break # Restart search after one addition
                    
        if improved:
            continue
            
        # 2. Try swaps within A
        if len(state['A']) > 0:
            for _ in range(50):
                x_idx = random.randint(0, len(state['A'])-1)
                x = state['A'][x_idx]
                # Try to find y to replace x
                y = random.randint(1, N)
                if y == x or y in state['A']:
                    continue
                # Temporarily remove x
                state['A'].remove(x)
                # Rebuild masks for A to check y
                old_sum_A = state['sum_A']
                old_diff_A = state['diff_A']
                
                # Rebuild quickly
                elems = state['A']
                s_mask = 0
                d_mask = 0
                for i in range(len(elems)):
                    for j in range(i, len(elems)):
                        s_mask |= (1 << (elems[i] + elems[j]))
                for i in range(len(elems)):
                    for j in range(len(elems)):
                        if i != j:
                            d_mask |= (1 << abs(elems[i] - elems[j]))
                
                state['sum_A'] = s_mask
                state['diff_A'] = d_mask
                
                if _check_add(state, 'A', y):
                    _add_elem(state, 'A', y)
                    improved = True
                    break
                else:
                    # Restore
                    state['A'].append(x)
                    state['sum_A'] = old_sum_A
                    state['diff_A'] = old_diff_A
                    # Note: _remove_elem would rebuild, but we manually restored. 
                    # However, manual restore is risky if lists are modified elsewhere. 
                    # Let's use a safer swap mechanism: 
                    # Actually, the manual restore above is valid because we only removed x and didn't modify the list structure otherwise.
                    
        # 3. Try swaps within B
        if len(state['B']) > 0:
            for _ in range(50):
                x_idx = random.randint(0, len(state['B'])-1)
                x = state['B'][x_idx]
                y = random.randint(1, N)
                if y == x or y in state['B']:
                    continue
                state['B'].remove(x)
                old_sum_B = state['sum_B']
                old_diff_B = state['diff_B']
                
                elems = state['B']
                s_mask = 0
                d_mask = 0
                for i in range(len(elems)):
                    for j in range(i, len(elems)):
                        s_mask |= (1 << (elems[i] + elems[j]))
                for i in range(len(elems)):
                    for j in range(len(elems)):
                        if i != j:
                            d_mask |= (1 << abs(elems[i] - elems[j]))
                
                state['sum_B'] = s_mask
                state['diff_B'] = d_mask
                
                if _check_add(state, 'B', y):
                    _add_elem(state, 'B', y)
                    improved = True
                    break
                else:
                    state['B'].append(x)
                    state['sum_B'] = old_sum_B
                    state['diff_B'] = old_diff_B
                    
        # 4. Try moving element from A to B
        if len(state['A']) > 0 and len(state['B']) < MAX_EACH:
             for _ in range(20):
                 x_idx = random.randint(0, len(state['A'])-1)
                 x = state['A'][x_idx]
                 # Remove x from A
                 state['A'].remove(x)
                 old_sum_A = state['sum_A']
                 old_diff_A = state['diff_A']
                 
                 elems = state['A']
                 s_mask = 0
                 d_mask = 0
                 for i in range(len(elems)):
                     for j in range(i, len(elems)):
                         s_mask |= (1 << (elems[i] + elems[j]))
                 for i in range(len(elems)):
                     for j in range(len(elems)):
                         if i != j:
                             d_mask |= (1 << abs(elems[i] - elems[j]))
                 
                 state['sum_A'] = s_mask
                 state['diff_A'] = d_mask
                 
                 # Try add x to B
                 if _check_add(state, 'B', x):
                     _add_elem(state, 'B', x)
                     improved = True
                     break
                 else:
                     # Restore A
                     state['A'].append(x)
                     state['sum_A'] = old_sum_A
                     state['diff_A'] = old_diff_A

        # 5. Try moving element from B to A
        if len(state['B']) > 0 and len(state['A']) < MAX_EACH:
             for _ in range(20):
                 x_idx = random.randint(0, len(state['B'])-1)
                 x = state['B'][x_idx]
                 state['B'].remove(x)
                 old_sum_B = state['sum_B']
                 old_diff_B = state['diff_B']
                 
                 elems = state['B']
                 s_mask = 0
                 d_mask = 0
                 for i in range(len(elems)):
                     for j in range(i, len(elems)):
                         s_mask |= (1 << (elems[i] + elems[j]))
                 for i in range(len(elems)):
                     for j in range(len(elems)):
                         if i != j:
                             d_mask |= (1 << abs(elems[i] - elems[j]))
                 
                 state['sum_B'] = s_mask
                 state['diff_B'] = d_mask
                 
                 if _check_add(state, 'A', x):
                     _add_elem(state, 'A', x)
                     improved = True
                     break
                 else:
                     state['B'].append(x)
                     state['sum_B'] = old_sum_B
                     state['diff_B'] = old_diff_B

def construct():
    start_time = time.time()
    best_size = 0
    best_A = []
    best_B = []
    
    # Random restarts
    while time.time() - start_time < DEADLINE:
        state = _init_state()
        
        # Greedy Initialization
        # Shuffle candidates to avoid bias
        candidates = list(range(1, N+1))
        random.shuffle(candidates)
        
        # Interleave adding to A and B to balance load
        for c in candidates:
            if time.time() - start_time > DEADLINE - 5:
                break
            # Try A first, then B
            if len(state['A']) < MAX_EACH and _check_add(state, 'A', c):
                _add_elem(state, 'A', c)
            elif len(state['B']) < MAX_EACH and _check_add(state, 'B', c):
                _add_elem(state, 'B', c)
                
        # Local Search
        _local_search(state)
        
        current_size = len(state['A']) + len(state['B'])
        if current_size > best_size:
            best_size = current_size
            best_A = state['A'][:]
            best_B = state['B'][:]
            
        # Check time
        if time.time() - start_time > DEADLINE - 2:
            break
            
    return [best_A, best_B]
# EVOLVE-BLOCK-END
