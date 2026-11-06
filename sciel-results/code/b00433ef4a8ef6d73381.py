import time
import random

N = 16

def _get_initial_states():
    states = []
    for i in range(N):
        val = 0
        period = 1 << (i + 1)
        half = 1 << i
        period_val = ((1 << half) - 1) << half
        num_periods = (1 << N) // period
        shift = 0
        for _ in range(num_periods):
            val |= period_val << shift
            shift += period
        states.append(val)
    return states

_INITIAL_STATES = _get_initial_states()

def fast_verify(network):
    state = _INITIAL_STATES[:]
    for (i, j) in network:
        min_val = state[i] & state[j]
        max_val = state[i] | state[j]
        state[i] = min_val
        state[j] = max_val
    for i in range(N - 1):
        if state[i] & ~state[i + 1]:
            return False
    return True

def batcher_bitonic(N):
    comps = []
    
    def bitonic_sort(n, offset, direction):
        if n == 2:
            if direction == 1:
                comps.append((offset, offset + 1))
            else:
                comps.append((offset + 1, offset))
            return
        
        half = n // 2
        bitonic_sort(half, offset, 1)
        bitonic_sort(half, offset + half, -1)
        bitonic_merge(n, offset, direction)
        
    def bitonic_merge(n, offset, direction):
        if n == 2:
            if direction == 1:
                comps.append((offset, offset + 1))
            else:
                comps.append((offset + 1, offset))
            return
            
        k = n // 2
        while k >= 1:
            for i in range(0, n, 2 * k):
                for j in range(i, i + k):
                    if direction == 1:
                        comps.append((offset + j, offset + j + k))
                    else:
                        comps.append((offset + j + k, offset + j))
            k //= 2
            
    bitonic_sort(N, 0, 1)
    return comps

def construct():
    deadline = time.time() + 85
    
    net = batcher_bitonic(N)
    
    if not fast_verify(net):
        # Fallback to odd-even if Batcher fails (should not happen)
        net = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
        
    best_net = list(net)
    best_count = len(net)
    
    random.seed(42)
    
    def try_remove(net):
        """Try to remove each comparator; return new net if any removal succeeds."""
        indices = list(range(len(net)))
        random.shuffle(indices)
        for idx in indices:
            candidate = net[:idx] + net[idx+1:]
            if fast_verify(candidate):
                return candidate
        return net

    while time.time() < deadline:
        # Try to remove
        new_net = try_remove(net)
        if len(new_net) < len(net):
            net = new_net
            best_net = list(net)
            best_count = len(net)
            if best_count <= 60:
                return best_net
            continue
        
        # If stuck, perturb
        move_type = random.choice(['insert', 'swap', 'move'])
        
        if move_type == 'insert':
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            pos = random.randint(0, len(net))
            candidate = net[:pos] + [(i, j)] + net[pos:]
            
        elif move_type == 'swap':
            if len(net) < 2:
                continue
            i1 = random.randint(0, len(net) - 1)
            i2 = random.randint(0, len(net) - 1)
            if i1 == i2:
                continue
            candidate = net[:]
            candidate[i1], candidate[i2] = candidate[i2], candidate[i1]
            
        elif move_type == 'move':
            if len(net) < 2:
                continue
            i1 = random.randint(0, len(net) - 1)
            i2 = random.randint(0, len(net) - 1)
            if i1 == i2:
                continue
            candidate = net[:]
            comp = candidate.pop(i1)
            candidate.insert(i2, comp)
            
        # Try to remove from the perturbed network
        improved_net = try_remove(candidate)
        
        # Accept if it's better or same size (to allow wandering)
        if len(improved_net) < len(net):
            net = improved_net
            best_net = list(net)
            best_count = len(net)
            if best_count <= 60:
                return best_net
        else:
            # If not better, we might still want to move to the perturbed state if it's the same size?
            # No, greedy hill climbing is usually better for minimization.
            # But to escape local minima, we might need to accept worse or equal.
            # Let's accept equal size to allow exploration.
            if len(improved_net) == len(net):
                net = improved_net

    return best_net
