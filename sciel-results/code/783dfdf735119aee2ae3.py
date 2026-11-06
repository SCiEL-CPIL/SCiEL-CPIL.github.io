import time
import random
import math

N = 16
NUM_INPUTS = 1 << N

# Precompute initial bitsets for zero-one principle verification
_INITIAL_BITSETS = []
for i in range(N):
    bitset = 0
    for k in range(NUM_INPUTS):
        if k & (1 << i):
            bitset |= (1 << k)
    _INITIAL_BITSETS.append(bitset)

def _verify_zero_one_fast(net):
    """
    Fast verification using bit-slicing.
    Returns True if the network sorts all 2^N boolean inputs.
    """
    if not net:
        return True
    
    state = _INITIAL_BITSETS[:]
    
    for i, j in net:
        new_i = state[i] & state[j]
        new_j = state[i] | state[j]
        state[i] = new_i
        state[j] = new_j
        
    # Check if sorted
    for i in range(N - 1):
        if state[i] & ~state[i + 1]:
            return False
            
    return True

def _compute_states(net):
    """
    Computes the state of the wires before each comparator.
    """
    if not net:
        return [_INITIAL_BITSETS[:]], _INITIAL_BITSETS[:]
    
    states = [_INITIAL_BITSETS[:]]
    current_state = _INITIAL_BITSETS[:]
    
    for i, j in net:
        next_state = current_state[:]
        next_state[i] = current_state[i] & current_state[j]
        next_state[j] = current_state[i] | current_state[j]
        states.append(next_state)
        current_state = next_state
        
    return states, current_state

def _remove_redundant(net):
    """
    Removes all comparators that are no-ops in the current network.
    """
    if not net:
        return []
        
    states, _ = _compute_states(net)
    to_remove = set()
    
    for k, (i, j) in enumerate(net):
        if states[k][i] & ~states[k][j] == 0:
            to_remove.add(k)
            
    if not to_remove:
        return net
        
    new_net = [net[k] for k in range(len(net)) if k not in to_remove]
    return new_net

def _prune_network(net, deadline):
    """
    Repeatedly remove redundant comparators until none are left.
    """
    current_net = net[:]
    changed = True
    while changed and time.time() < deadline:
        new_net = _remove_redundant(current_net)
        if len(new_net) < len(current_net):
            current_net = new_net
            changed = True
        else:
            changed = False
            
    return current_net

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

def _odd_even_network():
    """
    Odd-even transposition network: 120 comparators.
    """
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

def _shell_sort_network():
    """
    Shell sort network with specific gaps.
    """
    comparators = []
    gaps = [8, 4, 2, 1]
    for gap in gaps:
        for i in range(gap, N):
            comparators.append((i - gap, i))
    return comparators

def _merge_sort_network(n):
    """
    Recursive merge sort network.
    """
    if n <= 1:
        return []
    
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
        
    merge_sort(n, 0)
    return comparators

def _sa_search(net, deadline, rng):
    """
    Simulated Annealing search with bias towards validity-preserving moves.
    """
    current_net = net[:]
    best_net = net[:]
    best_len = len(best_net)
    
    # First, prune
    current_net = _prune_network(current_net, deadline)
    if len(current_net) < best_len:
        best_net = current_net[:]
        best_len = len(current_net)
    
    # SA parameters
    T_start = 1.0
    T_end = 0.01
    alpha = 0.995  # Cooling rate
    T = T_start
    
    no_improve = 0
    max_no_improve = 50
    restart_count = 0
    max_restarts = 10
    
    while time.time() < deadline and (no_improve < max_no_improve or restart_count < max_restarts):
        # Generate a neighbor
        move_type = rng.random()
        
        if move_type < 0.35:  # 35% swap
            if len(current_net) < 2:
                no_improve += 1
                continue
            i, j = rng.sample(range(len(current_net)), 2)
            new_net = current_net[:]
            new_net[i], new_net[j] = new_net[j], new_net[i]
        elif move_type < 0.65:  # 30% move
            if len(current_net) < 2:
                no_improve += 1
                continue
            i = rng.randrange(len(current_net))
            j = rng.randrange(len(current_net))
            new_net = current_net[:]
            item = new_net.pop(i)
            new_net.insert(j, item)
        elif move_type < 0.85:  # 20% remove
            if len(current_net) < 2:
                no_improve += 1
                continue
            idx = rng.randrange(len(current_net))
            new_net = current_net[:idx] + current_net[idx+1:]
        else:  # 15% insert
            i, j = rng.sample(range(N), 2)
            if i > j:
                i, j = j, i
            pos = rng.randrange(len(new_net) + 1) if 'new_net' in dir() else rng.randrange(len(current_net) + 1)
            new_net = current_net[:pos] + [(i, j)] + current_net[pos:]
        
        # Verify
        if _verify_zero_one_fast(new_net):
            new_len = len(new_net)
            delta = new_len - len(current_net)
            
            # Accept if better, or with probability exp(-delta/T) if worse
            if delta <= 0 or (delta > 0 and rng.random() < math.exp(-delta / T)):
                current_net = new_net
                if new_len < best_len:
                    best_net = new_net[:]
                    best_len = new_len
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
        else:
            no_improve += 1
        
        # Cool down
        T *= alpha
        
        # Restart if stuck
        if no_improve >= max_no_improve:
            restart_count += 1
            if restart_count >= max_restarts:
                break
            # Perturb best network
            current_net = best_net[:]
            for _ in range(rng.randint(3, 8)):
                if len(current_net) > 1:
                    i, j = rng.sample(range(len(current_net)), 2)
                    current_net[i], current_net[j] = current_net[j], current_net[i]
                if len(current_net) > 1 and rng.random() < 0.5:
                    idx = rng.randrange(len(current_net))
                    current_net.pop(idx)
            # Re-prune
            current_net = _prune_network(current_net, deadline)
            no_improve = 0
            T = T_start * 0.5  # Restart with lower temperature
    
    return best_net

def _polish_pass(net, deadline):
    """
    Final high-precision polish: try removing each comparator one at a time.
    """
    current_net = net[:]
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for idx in range(len(current_net) - 1, -1, -1):
            if time.time() > deadline:
                break
            # Try removing comparator at idx
            candidate = current_net[:idx] + current_net[idx+1:]
            if _verify_zero_one_fast(candidate):
                current_net = candidate
                changed = True
    
    return current_net

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    rng = random.Random(42)
    
    # Generate diverse seeds
    seeds = []
    
    base_net = _base_60_network()
    if _verify_zero_one_fast(base_net):
        seeds.append(base_net)
        
    oe_net = _odd_even_network()
    if _verify_zero_one_fast(oe_net):
        seeds.append(oe_net)
        
    shell_net = _shell_sort_network()
    if _verify_zero_one_fast(shell_net):
        seeds.append(shell_net)
        
    merge_net = _merge_sort_network(N)
    if _verify_zero_one_fast(merge_net):
        seeds.append(merge_net)
        
    if not seeds:
        seeds.append(oe_net)
        
    best_net = None
    best_len = float('inf')
    
    # Phase 1: Prune each seed and run SA on the best
    for seed in seeds:
        if time.time() > deadline - 10:
            break
        pruned = _prune_network(seed, deadline)
        if len(pruned) < best_len:
            best_net = pruned
            best_len = len(pruned)
            
    if best_net is None:
        best_net = seeds[0]
        best_len = len(best_net)
    
    # Phase 2: SA search
    if time.time() < deadline - 15:
        best_net = _sa_search(best_net, deadline - 15, rng)
    
    # Phase 3: Final polish
    if time.time() < deadline - 2:
        best_net = _polish_pass(best_net, deadline - 2)
    
    # Final verification
    if not _verify_zero_one_fast(best_net):
        for seed in seeds:
            if _verify_zero_one_fast(seed):
                return seed
        return _odd_even_network()
        
    return best_net

if __name__ == "__main__":
    net = construct()
    print(f"Found network with {len(net)} comparators")
    print(f"Valid: {_verify_zero_one_fast(net)}")
