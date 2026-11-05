import time
import random

N = 16
NUM_INPUTS = 1 << N  # 65536
ALL_ONES = (1 << NUM_INPUTS) - 1

# Pre-compute initial states for bit-parallel verification
_initial_states = None

def _get_initial_states():
    global _initial_states
    if _initial_states is None:
        _initial_states = []
        for i in range(N):
            val = 0
            period = 1 << (i + 1)
            half = 1 << i
            period_val = ((1 << half) - 1) << half
            num_periods = NUM_INPUTS // period
            shift = 0
            for _ in range(num_periods):
                val |= period_val << shift
                shift += period
            _initial_states.append(val)
    return _initial_states

def fast_verify(network):
    """Verify a sorting network using bit-parallel zero-one principle."""
    state = _get_initial_states()[:]
    
    for (i, j) in network:
        min_val = state[i] & state[j]
        max_val = state[i] | state[j]
        state[i] = min_val
        state[j] = max_val
    
    for i in range(N - 1):
        if state[i] & ~state[i + 1]:
            return False
    return True

def bitonic_sorter(n):
    """
    Generate Batcher's bitonic sorting network for n wires.
    """
    if n <= 1:
        return []
    
    comparators = []
    
    def bitonic_sort(size, start, ascending):
        if size <= 1:
            return
        half = size // 2
        bitonic_sort(half, start, ascending)
        bitonic_sort(half, start + half, not ascending)
        bitonic_merge(half, start, ascending)
    
    def bitonic_merge(size, start, ascending):
        if size <= 1:
            return
        for i in range(size):
            a = start + i
            b = start + i + size
            if ascending:
                comparators.append((a, b))
            else:
                comparators.append((b, a))
        bitonic_merge(size // 2, start, ascending)
        bitonic_merge(size // 2, start + size, ascending)
    
    bitonic_sort(n, 0, True)
    return comparators

def clean(net):
    """
    Removes all comparators that are no-ops in the current network.
    Returns the reduced network.
    """
    if not net:
        return []
        
    state = _get_initial_states()[:]
    # We need to check redundancy at each step.
    # However, removing a comparator changes the state for subsequent ones.
    # So we must iterate and remove one by one, or iterate and mark, but be careful.
    # The previous `remove_redundant` removed all at once based on the *original* states.
    # Is that safe?
    # If comparator A is a no-op, removing it doesn't change the state entering B.
    # So if B is a no-op given the state entering it (which is the same as if A was there, because A did nothing),
    # then B is also a no-op.
    # So yes, we can identify all no-ops in one pass based on the states *before* any removals,
    # provided we compute states correctly.
    
    # But wait. If A is NOT a no-op, but removing A makes B a no-op?
    # No, if A is not a no-op, it changes the state.
    # So we must remove A, then re-evaluate B.
    # The "remove all at once" logic is:
    # If A is no-op, state after A = state before A.
    # So state before B = state after A = state before A.
    # If B is no-op given state before B, it is no-op.
    # So if we check all comparators against the states *as they flow through the original network*,
    # and find a set S of no-ops, removing all of S is safe?
    # Let A, B in S.
    # If A is before B.
    # State before A = S_A. A is no-op => State after A = S_A.
    # State before B = S_B (from original flow).
    # In original flow, State before B is derived from State after A.
    # Since State after A = State before A, the flow is identical.
    # So S_B is the same in the modified network.
    # If B is no-op given S_B, it is no-op in the modified network.
    # So yes, we can remove all identified no-ops in one pass.
    
    states = [None] * (len(net) + 1)
    current_state = _get_initial_states()[:]
    states[0] = current_state
    
    to_remove = set()
    
    for k, (i, j) in enumerate(net):
        # Check if wire i <= wire j for all inputs before this comparator
        if states[k][i] & ~states[k][j] == 0:
            to_remove.add(k)
            
        # Compute next state
        next_state = current_state[:]
        min_val = current_state[i] & current_state[j]
        max_val = current_state[i] | current_state[j]
        next_state[i] = min_val
        next_state[j] = max_val
        states[k+1] = next_state
        current_state = next_state
        
    if not to_remove:
        return net
        
    new_net = [net[k] for k in range(len(net)) if k not in to_remove]
    return new_net

def construct():
    deadline = time.time() + 85  # 85s to leave 5s margin
    
    # Start from bitonic sorter (80 comparators for n=16)
    net = bitonic_sorter(N)
    net = clean(net)
    
    best_net = list(net)
    best_count = len(net)
    
    random.seed(42)
    
    # Simulated Annealing / Hill Climbing
    # We try to minimize the count.
    # Moves:
    # 1. Swap two comparators
    # 2. Move one comparator
    # 3. Insert one comparator
    
    current_net = list(net)
    current_count = len(net)
    
    # Temperature for accepting non-improving moves (same size)
    # We accept same-size moves with prob 0.5 to explore.
    
    for iteration in range(20000):
        if time.time() > deadline:
            break
            
        move_type = random.random()
        
        if move_type < 0.4:
            # Swap
            if len(current_net) < 2:
                continue
            i1, i2 = random.sample(range(len(current_net)), 2)
            candidate = current_net[:]
            candidate[i1], candidate[i2] = candidate[i2], candidate[i1]
            
            # Check validity
            if fast_verify(candidate):
                # Try to clean
                cleaned = clean(candidate)
                if len(cleaned) < current_count:
                    current_net = cleaned
                    current_count = len(cleaned)
                    if current_count < best_count:
                        best_net = list(current_net)
                        best_count = current_count
                elif len(cleaned) == current_count:
                    # Accept with probability 0.5 to explore
                    if random.random() < 0.5:
                        current_net = cleaned
                        # current_count unchanged
                        # Try to clean again? No, clean is idempotent on clean nets?
                        # Actually, clean removes all no-ops. So cleaned is "clean".
                        
        elif move_type < 0.7:
            # Move
            if len(current_net) < 2:
                continue
            i1 = random.randint(0, len(current_net) - 1)
            i2 = random.randint(0, len(current_net) - 1)
            if i1 == i2:
                continue
            comp = current_net[i1]
            candidate = current_net[:i1] + current_net[i1+1:]
            # Insert at i2 (adjust index if i2 > i1)
            insert_pos = i2 if i2 < i1 else i2 - 1
            candidate.insert(insert_pos, comp)
            
            if fast_verify(candidate):
                cleaned = clean(candidate)
                if len(cleaned) < current_count:
                    current_net = cleaned
                    current_count = len(cleaned)
                    if current_count < best_count:
                        best_net = list(current_net)
                        best_count = current_count
                elif len(cleaned) == current_count:
                    if random.random() < 0.5:
                        current_net = cleaned
                        
        else:
            # Insert
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            pos = random.randint(0, len(current_net))
            candidate = current_net[:pos] + [(i, j)] + current_net[pos:]
            
            # Clean immediately
            cleaned = clean(candidate)
            
            if len(cleaned) < current_count:
                current_net = cleaned
                current_count = len(cleaned)
                if current_count < best_count:
                    best_net = list(current_net)
                    best_count = current_count
            elif len(cleaned) == current_count:
                # The insertion was redundant or didn't help
                # We might still accept it to explore?
                # If it's the same size, it means the inserted comp was removed by clean.
                # So the net is likely the same as before (or equivalent).
                # Accepting it doesn't change the state much.
                # So we don't accept same-size insertions.
                pass
                
    # Final clean
    best_net = clean(best_net)
    
    return best_net
