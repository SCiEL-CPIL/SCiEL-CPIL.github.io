import time
import random

N = 16
NUM_INPUTS = 1 << N

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

def prune_forward(net):
    """
    Removes redundant comparators using forward pass.
    Returns (new_net, final_state)
    """
    if not net:
        return [], _get_initial_states()[:]
    
    state = _get_initial_states()[:]
    new_net = []
    for (i, j) in net:
        # Check if i <= j for all inputs
        if state[i] & ~state[j] == 0:
            # Redundant, skip
            continue
        else:
            # Apply comparator
            min_val = state[i] & state[j]
            max_val = state[i] | state[j]
            state[i] = min_val
            state[j] = max_val
            new_net.append((i, j))
            
    # The state is now the final state of the pruned network
    return new_net, state

def is_sorted_state(final_state):
    for i in range(N - 1):
        if final_state[i] & ~final_state[i + 1]:
            return False
    return True

def bitonic_sorter(n):
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

def odd_even_merge(n):
    if n <= 1:
        return []
    comparators = []
    def oe_merge(size, start, ascending):
        if size <= 1:
            return
        half = size // 2
        for i in range(half):
            a = start + i
            b = start + i + half
            if ascending:
                comparators.append((a, b))
            else:
                comparators.append((b, a))
        oe_merge(half, start, ascending)
        oe_merge(half, start + half, ascending)
    def oe_sort(size, start, ascending):
        if size <= 1:
            return
        half = size // 2
        oe_sort(half, start, ascending)
        oe_sort(half, start + half, ascending)
        oe_merge(half, start, ascending)
    oe_sort(n, 0, True)
    return comparators

def odd_even_transposition(n):
    return [(i, i + 1) for phase in range(n) for i in range(phase % 2, n - 1, 2)]

def local_search(net, deadline, seed):
    rng = random.Random(seed)
    best_net, best_state = prune_forward(net)
    best_count = len(best_net)
    
    current_net = best_net
    current_state = best_state
    
    # If the pruned network is not sorted, it's invalid.
    # But prune_forward only removes no-ops, so if the original was sorted, this should be sorted.
    # We assume inputs are valid sorting networks.
    
    while time.time() < deadline:
        improved = False
        L = len(current_net)
        
        # 1. Try swapping adjacent comparators
        # Randomly pick a few swaps to avoid full O(L^2) scan every time?
        # Or scan all if L is small? L is ~100. L^2 = 10000. 
        # Prune is O(L). 10000 * 100 = 1M ops. Might be too slow per iteration.
        # Let's try a subset of swaps per iteration.
        
        for _ in range(20): # Try 20 random swaps
            if L < 2:
                break
            idx = rng.randint(0, L - 2)
            
            # Create candidate
            candidate = current_net[:]
            candidate[idx], candidate[idx + 1] = candidate[idx + 1], candidate[idx]
            
            # Prune candidate
            cand_net, cand_state = prune_forward(candidate)
            
            if is_sorted_state(cand_state):
                if len(cand_net) < best_count:
                    best_net = cand_net
                    best_count = len(cand_net)
                    current_net = cand_net
                    current_state = cand_state
                    L = len(current_net)
                    improved = True
                elif len(cand_net) == len(current_net):
                    # Accept equal size to explore? Maybe.
                    # Let's only accept if it leads to a different structure?
                    # For now, stick to strict improvement to ensure progress.
                    pass
            else:
                # If not sorted, we cannot accept.
                # But wait, prune_forward might have removed a comparator that was necessary?
                # No, prune_forward only removes NO-OPS. If the network was sorted before,
                # removing no-ops keeps it sorted.
                # So if is_sorted_state is False, it means the *original* candidate was not sorted?
                # Or did we make a mistake?
                # If current_net was sorted, and we swapped two adjacent comparators,
                # the new network might NOT be sorted.
                # So we must reject it.
                pass
        
        # 2. Try removing a random comparator (if it's redundant after some changes?)
        # Actually, just try removing and checking if it's still sorted.
        # This is expensive (full verify).
        # But prune_forward is fast.
        # If we remove a comparator, we can check if it was a no-op in the current state.
        # We already did that in prune_forward. So if current_net is fully pruned,
        # no single removal is valid.
        
        # 3. Try inserting a comparator?
        # This is rarely useful for minimization unless we are stuck.
        
        if not improved:
            # Stagnation? Try a more aggressive move: 
            # Randomly remove a comparator and see if we can fix it? No.
            # Let's just restart from best if we are stuck for a while?
            # Or try a "double swap"?
            
            # For now, just continue. The random swaps will eventually find a path.
            pass
    return best_net

def construct():
    deadline = time.time() + 85
    
    seeds = [
        bitonic_sorter(N),
        odd_even_merge(N),
        odd_even_transposition(N),
    ]
    
    best_net = None
    best_count = float('inf')
    
    for seed_idx, seed_net in enumerate(seeds):
        if time.time() > deadline:
            break
            
        # Initial prune
        pruned, state = prune_forward(seed_net)
        if not is_sorted_state(state):
            continue
            
        if len(pruned) < best_count:
            best_net = pruned
            best_count = len(pruned)
            
        # Local search
        search_deadline = min(deadline, time.time() + 20)
        found = local_search(pruned, search_deadline, seed=seed_idx * 1000)
        
        if len(found) < best_count:
            best_net = found
            best_count = len(found)
            
    if best_net is None:
        best_net = odd_even_transposition(N)
        best_net, _ = prune_forward(best_net)
        
    return best_net
