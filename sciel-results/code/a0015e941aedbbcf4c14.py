import time
import random
import math

N = 16
NUM_INPUTS = 1 << N
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
    if not network:
        return True
    state = _get_initial_states()[:]
    
    for (i, j) in network:
        min_val = state[i] & state[j]
        max_val = state[i] | state[j]
        state[i] = min_val
        state[j] = max_val
    
    # Check if sorted
    # Unrolled for speed
    if (state[0] & ~state[1]) or (state[1] & ~state[2]) or (state[2] & ~state[3]) or (state[3] & ~state[4]) or \
       (state[4] & ~state[5]) or (state[5] & ~state[6]) or (state[6] & ~state[7]) or (state[7] & ~state[8]) or \
       (state[8] & ~state[9]) or (state[9] & ~state[10]) or (state[10] & ~state[11]) or (state[11] & ~state[12]) or \
       (state[12] & ~state[13]) or (state[13] & ~state[14]) or (state[14] & ~state[15]):
        return False
    return True

def compute_states(net):
    """
    Returns list of states. states[k] is the state of all wires before comparator k.
    states[0] is initial.
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
        
    return states

def remove_all_redundant(net):
    """
    Removes all comparators that are no-ops.
    """
    if not net:
        return []
    
    changed = True
    while changed:
        changed = False
        states = compute_states(net)
        to_remove = set()
        
        for k, (i, j) in enumerate(net):
            # If wire i is always <= wire j before this comparator, it's redundant
            if states[k][i] & ~states[k][j] == 0:
                to_remove.add(k)
                
        if to_remove:
            net = [net[k] for k in range(len(net)) if k not in to_remove]
            changed = True
            
    return net

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

def odd_even_network():
    """Odd-even transposition sorting network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def shell_sort_network():
    """Shell sort variant."""
    comparators = []
    gaps = [8, 4, 2, 1]
    for gap in gaps:
        for i in range(gap, N):
            comparators.append((i - gap, i))
    return comparators

def merge_sort_network():
    """Merge sort network."""
    comparators = []
    
    def merge_sort(size, start):
        if size <= 1:
            return
        half = size // 2
        merge_sort(half, start)
        merge_sort(half, start + half)
        merge(half, start)
        
    def merge(size, start):
        if size <= 1:
            return
        for i in range(size):
            a = start + i
            b = start + i + size
            comparators.append((a, b))
        merge(size // 2, start)
        merge(size // 2, start + size)
        
    merge_sort(N, 0)
    return comparators

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    rng = random.Random(42)
    
    best_net = None
    best_len = float('inf')
    
    # Generate starting points
    start_nets = []
    
    nets = [
        bitonic_sorter(N),
        odd_even_network(),
        shell_sort_network(),
        merge_sort_network()
    ]
    
    for net in nets:
        # Prune redundant comparators
        pruned = remove_all_redundant(net)
        if fast_verify(pruned):
            start_nets.append(pruned)
            
    # If no valid start (shouldn't happen), use odd-even
    if not start_nets:
        start_nets.append(odd_even_network())
        
    # Simulated Annealing on each start point (or just the best one? 
    # Better to run SA on the best few to allow diversity)
    # Sort start nets by length
    start_nets.sort(key=len)
    
    # We will run SA on the top 3 distinct networks
    unique_nets = []
    for net in start_nets:
        if net not in unique_nets:
            unique_nets.append(net)
            if len(unique_nets) >= 3:
                break
                
    for initial_net in unique_nets:
        if time.time() > deadline:
            break
            
        current_net = initial_net[:]
        current_len = len(current_net)
        
        if current_len < best_len:
            best_net = current_net[:]
            best_len = current_len
            
        # SA Parameters
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        
        session_deadline = time.time() + 20.0 # 20s per start point
        
        while time.time() < session_deadline and T > min_T:
            # Generate neighbor
            move_type = rng.random()
            new_net = current_net[:]
            
            if move_type < 0.6:
                # Remove a comparator
                if len(new_net) > 0:
                    idx = rng.randrange(len(new_net))
                    new_net.pop(idx)
            elif move_type < 0.8:
                # Swap two comparators
                if len(new_net) > 1:
                    i, j = rng.sample(range(len(new_net)), 2)
                    new_net[i], new_net[j] = new_net[j], new_net[i]
            elif move_type < 0.9:
                # Mutate a comparator
                if len(new_net) > 0:
                    idx = rng.randrange(len(new_net))
                    i, j = new_net[idx]
                    # Change one endpoint
                    if rng.random() < 0.5:
                        new_i = rng.randrange(N)
                        if new_i != i:
                            if new_i < j:
                                new_net[idx] = (new_i, j)
                            elif new_i > j:
                                new_net[idx] = (j, new_i)
                    else:
                        new_j = rng.randrange(N)
                        if new_j != j:
                            if i < new_j:
                                new_net[idx] = (i, new_j)
                            elif i > new_j:
                                new_net[idx] = (new_j, i)
            else:
                # Insert a comparator (to escape local minima)
                i, j = rng.sample(range(N), 2)
                if i > j: i, j = j, i
                pos = rng.randrange(len(new_net) + 1)
                new_net.insert(pos, (i, j))
                
            # Verify
            if fast_verify(new_net):
                new_len = len(new_net)
                delta = new_len - current_len
                
                # Accept if better
                if delta < 0:
                    current_net = new_net
                    current_len = new_len
                    if current_len < best_len:
                        best_net = current_net[:]
                        best_len = current_len
                else:
                    # Accept worse with probability exp(-delta/T)
                    if rng.random() < math.exp(-delta / T):
                        current_net = new_net
                        current_len = new_len
                        
                # Periodically prune redundant comparators
                # This helps "snap" to a reduced form
                if current_len < best_len or rng.random() < 0.1:
                    pruned = remove_all_redundant(current_net)
                    if fast_verify(pruned):
                        if len(pruned) < current_len:
                            current_net = pruned
                            current_len = len(current_net)
                            if current_len < best_len:
                                best_net = current_net[:]
                                best_len = current_len
                            
            T *= cooling
            
    return best_net

if __name__ == "__main__":
    net = construct()
    print(f"Found network with {len(net)} comparators")
    print(f"Valid: {fast_verify(net)}")
