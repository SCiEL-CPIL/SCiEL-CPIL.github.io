import time
import random
import math

N = 16
ALL_ONES = (1 << (1 << N)) - 1

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
            num_periods = (1 << N) // period
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

def odd_even_network():
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def batcher_merge(n):
    if n == 1:
        return []
    half = n // 2
    stage1 = [(i, i + half) for i in range(half)]
    left_merge = batcher_merge(half)
    right_merge = batcher_merge(half)
    right_merge = [(i + half, j + half) for i, j in right_merge]
    return stage1 + left_merge + right_merge

def batcher_bitonic(n):
    if n == 1:
        return []
    half = n // 2
    left = batcher_bitonic(half)
    right = batcher_bitonic(half)
    right = [(i + half, j + half) for i, j in right]
    merge = batcher_merge(n)
    return left + right + merge

def construct():
    deadline = time.time() + 85
    start_time = time.time()
    
    # Start from Batcher bitonic (80 comparators)
    net = batcher_bitonic(N)
    best_net = list(net)
    best_count = len(net)
    
    def greedy_remove(net, time_limit=None):
        changed = True
        while changed:
            changed = False
            for idx in range(len(net) - 1, -1, -1):
                if time_limit and time.time() > time_limit:
                    return net
                candidate = net[:idx] + net[idx+1:]
                if fast_verify(candidate):
                    net = candidate
                    changed = True
                    break
        return net
    
    # Phase 1: Greedy removal from Batcher
    net = greedy_remove(net, deadline)
    best_net = list(net)
    best_count = len(net)
    
    random.seed(42)
    
    # Simulated Annealing
    T = 10.0
    cooling_rate = 0.99
    
    for iteration in range(10000):
        if time.time() > deadline:
            break
        
        # Propose a random change
        op_type = random.random()
        candidate = list(net)
        
        if op_type < 0.3:
            # Swap two comparators
            if len(net) < 2:
                continue
            idx1 = random.randint(0, len(net) - 1)
            idx2 = random.randint(0, len(net) - 1)
            candidate[idx1], candidate[idx2] = candidate[idx2], candidate[idx1]
        elif op_type < 0.6:
            # Move a comparator
            if len(net) < 2:
                continue
            from_idx = random.randint(0, len(net) - 1)
            to_idx = random.randint(0, len(net) - 1)
            comp = candidate[from_idx]
            candidate = candidate[:from_idx] + candidate[from_idx+1:]
            if to_idx > from_idx:
                to_idx -= 1
            candidate = candidate[:to_idx] + [comp] + candidate[to_idx:]
        elif op_type < 0.8:
            # Insert a random comparator
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            pos = random.randint(0, len(candidate))
            candidate = candidate[:pos] + [(i, j)] + candidate[pos:]
        else:
            # Remove a random comparator
            if len(net) < 1:
                continue
            idx = random.randint(0, len(net) - 1)
            candidate = candidate[:idx] + candidate[idx+1:]
            
        # Verify candidate
        if fast_verify(candidate):
            new_count = len(candidate)
            old_count = len(net)
            delta = new_count - old_count
            
            if delta < 0:
                # Improvement
                net = candidate
                if new_count < best_count:
                    best_net = list(net)
                    best_count = new_count
            else:
                # Worse, accept with probability exp(-delta/T)
                if random.random() < math.exp(-delta / T):
                    net = candidate
        else:
            # Invalid, reject
            pass
            
        T *= cooling_rate
        if T < 0.01:
            T = 0.01
            
    return best_net
