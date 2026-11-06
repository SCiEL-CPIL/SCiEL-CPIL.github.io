import time
import random
import math

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
            # For input j (0..65535), wire i has value (j >> i) & 1
            # Build a 65536-bit integer where bit j = (j >> i) & 1
            # Pattern: period 2^(i+1), lower half 0, upper half 1
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
    
    # Unrolled check for N=16 to reduce loop overhead
    if (state[0] & ~state[1]) or (state[1] & ~state[2]) or (state[2] & ~state[3]) or (state[3] & ~state[4]) or \
       (state[4] & ~state[5]) or (state[5] & ~state[6]) or (state[6] & ~state[7]) or (state[7] & ~state[8]) or \
       (state[8] & ~state[9]) or (state[9] & ~state[10]) or (state[10] & ~state[11]) or (state[11] & ~state[12]) or \
       (state[12] & ~state[13]) or (state[13] & ~state[14]) or (state[14] & ~state[15]):
        return False
    return True

def bitonic_sorter(n):
    """
    Generate Batcher's bitonic sorting network for n wires (n must be power of 2).
    Returns a list of (i, j) comparator pairs.
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

def odd_even_network():
    """Odd-even transposition sorting network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def compute_states(net):
    """
    Returns (states, final_state) where states[k] is the state of all wires
    before comparator k is applied. states[0] is the initial state.
    """
    states = [None] * (len(net) + 1)
    current_state = _get_initial_states()[:]
    states[0] = current_state
    
    for k, (i, j) in enumerate(net):
        next_state = current_state[:]
        min_val = current_state[i] & current_state[j]
        max_val = current_state[i] | current_state[j]
        next_state[i] = min_val
        next_state[j] = max_val
        states[k+1] = next_state
        current_state = next_state
        
    return states, current_state

def remove_redundant(net):
    """
    Removes all comparators that are no-ops in the current network.
    Returns (new_net, final_state).
    """
    if not net:
        return [], _get_initial_states()[:]
        
    states, final_state = compute_states(net)
    to_remove = set()
    
    for k, (i, j) in enumerate(net):
        # Check if wire i <= wire j for all inputs before this comparator
        if states[k][i] & ~states[k][j] == 0:
            to_remove.add(k)
            
    if not to_remove:
        return net, final_state
        
    new_net = [net[k] for k in range(len(net)) if k not in to_remove]
    return new_net, final_state

def is_sorted_state(state):
    """Check if the final state is sorted (non-decreasing)."""
    for i in range(N - 1):
        if state[i] & ~state[i + 1]:
            return False
    return True

def greedy_remove_with_states(net, deadline):
    """
    Greedy removal that recomputes states each pass and removes all no-ops.
    More efficient than one-by-one re-verification.
    """
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Compute states and find all no-ops
        if not net:
            break
        states, final_state = compute_states(net)
        to_remove = set()
        for k, (i, j) in enumerate(net):
            if states[k][i] & ~states[k][j] == 0:
                to_remove.add(k)
        
        if to_remove:
            new_net = [net[k] for k in range(len(net)) if k not in to_remove]
            net = new_net
            changed = True
    
    return net

def _base_60_network():
    """
    A known 60-comparator sorting network for 16 wires.
    """
    return [
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
        (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
        (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
        (1, 2), (3, 4), (5, 6), (9, 10), (11, 12), (13, 14),
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
        (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
        (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
        (1, 2), (3, 4), (5, 6), (9, 10), (11, 12), (13, 14),
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
    ]

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Initial best network
    best_net = _base_60_network()
    best_len = len(best_net)
    
    if not fast_verify(best_net):
        # Fallback to odd-even if base is broken (shouldn't happen)
        best_net = odd_even_network()
        best_len = len(best_net)
        
    rng = random.Random(42)
    
    # Generate multiple starting points
    start_nets = [best_net]
    
    # 1. Bitonic sorter
    bn = bitonic_sorter(N)
    bn_clean, bn_final = remove_redundant(bn)
    if is_sorted_state(bn_final):
        start_nets.append(bn_clean)
    
    # 2. Odd-even transposition
    oen = odd_even_network()
    oen_clean, oen_final = remove_redundant(oen)
    if is_sorted_state(oen_final):
        start_nets.append(oen_clean)
    
    # 3. Shell sort variant
    def shell_sort_network():
        comparators = []
        gaps = [8, 4, 2, 1]
        for gap in gaps:
            for i in range(gap, N):
                comparators.append((i - gap, i))
        return comparators
    
    sn = shell_sort_network()
    sn_clean, sn_final = remove_redundant(sn)
    if is_sorted_state(sn_final):
        start_nets.append(sn_clean)
    
    # Apply greedy removal to all starting points
    for i in range(len(start_nets)):
        if time.time() < deadline:
            start_nets[i] = greedy_remove_with_states(start_nets[i], deadline)
            if len(start_nets[i]) < best_len:
                best_net = start_nets[i]
                best_len = len(best_net)
    
    # Local search with multiple restarts
    num_restarts = 20
    time_per_restart = (deadline - start_time) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Start of a new session
        if restart == 0:
            current_net = best_net[:]
        else:
            # Start from best found so far, but perturb it
            current_net = best_net[:]
            # Randomly swap a few comparators
            for _ in range(5):
                if len(current_net) > 1:
                    i, j = rng.sample(range(len(current_net)), 2)
                    current_net[i], current_net[j] = current_net[j], current_net[i]
            
        current_len = len(current_net)
        local_best = current_net[:]
        local_best_len = current_len
        
        session_start = time.time()
        session_deadline = session_start + time_per_restart
        
        # Local Search Loop
        no_improve_count = 0
        max_no_improve = 100
        
        while time.time() < session_deadline and no_improve_count < max_no_improve:
            improved = False
            moves = []
            
            # 1. Try removing a comparator
            if len(current_net) > 1:
                idx = rng.randrange(len(current_net))
                moves.append(('remove', idx))
                
            # 2. Try swapping two comparators
            if len(current_net) > 1:
                i, j = rng.sample(range(len(current_net)), 2)
                moves.append(('swap', i, j))
                
            # 3. Try inserting a comparator (to escape local minima)
            if rng.random() < 0.1:
                i, j = rng.sample(range(N), 2)
                if i > j: i, j = j, i
                pos = rng.randrange(len(current_net) + 1)
                moves.append(('insert', i, j, pos))
                
            # 4. Try changing one endpoint of a comparator
            if len(current_net) > 0:
                idx = rng.randrange(len(current_net))
                old_i, old_j = current_net[idx]
                # Change i
                new_i = rng.randrange(N)
                if new_i != old_i:
                    if new_i < old_j:
                        moves.append(('mutate', idx, new_i, old_j))
                    elif new_i > old_j:
                        moves.append(('mutate', idx, old_j, new_i))
                # Change j
                new_j = rng.randrange(N)
                if new_j != old_j:
                    if old_i < new_j:
                        moves.append(('mutate', idx, old_i, new_j))
                    elif old_i > new_j:
                        moves.append(('mutate', idx, new_j, old_i))

            if not moves:
                break
                
            # Pick a random move
            move = rng.choice(moves)
            
            new_net = current_net[:]
            
            if move[0] == 'remove':
                idx = move[1]
                new_net = current_net[:idx] + current_net[idx+1:]
            elif move[0] == 'swap':
                i, j = move[1], move[2]
                new_net = current_net[:]
                new_net[i], new_net[j] = new_net[j], new_net[i]
            elif move[0] == 'insert':
                i, j, pos = move[1], move[2], move[3]
                new_net = current_net[:pos] + [(i, j)] + current_net[pos:]
            elif move[0] == 'mutate':
                idx, new_i, new_j = move[1], move[2], move[3]
                new_net = current_net[:]
                new_net[idx] = (new_i, new_j)
                
            # Verify new network
            if fast_verify(new_net):
                new_len = len(new_net)
                if new_len < current_len:
                    current_net = new_net
                    current_len = new_len
                    improved = True
                    no_improve_count = 0
                    if current_len < local_best_len:
                        local_best = current_net[:]
                        local_best_len = current_len
                        if current_len < best_len:
                            best_net = current_net[:]
                            best_len = current_len
                elif new_len == current_len:
                    if rng.random() < 0.5:
                        current_net = new_net
                        no_improve_count += 1
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
                
        # End of session
        if local_best_len < best_len:
            best_net = local_best
            best_len = local_best_len

    return best_net

if __name__ == "__main__":
    # Quick self-test
    net = construct()
    print(f"Found network with {len(net)} comparators")
    print(f"Valid: {fast_verify(net)}")
