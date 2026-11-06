import time
import random
import math

N = 18

# EVOLVE-BLOCK-START
_INITIAL_BITSETS = None

def _init_bitsets():
    total_bits = 1 << N
    bitsets = [0] * N
    for k in range(N):
        period = 1 << (k + 1)
        half_period = 1 << k
        pattern = (1 << half_period) - 1
        pattern <<= half_period
        current_val = pattern
        current_len = period
        while current_len < total_bits:
            current_val |= (current_val << current_len)
            current_len *= 2
        bitsets[k] = current_val
    return bitsets

def _get_initial_bitsets():
    global _INITIAL_BITSETS
    if _INITIAL_BITSETS is None:
        _INITIAL_BITSETS = _init_bitsets()
    return _INITIAL_BITSETS

def _is_sorting_network(comparators):
    if not comparators:
        return N <= 1
    bitsets = list(_get_initial_bitsets())
    for i, j in comparators:
        bi = bitsets[i]
        bj = bitsets[j]
        bitsets[i] = bi & bj
        bitsets[j] = bi | bj
    for k in range(N - 1):
        if bitsets[k] & ~bitsets[k + 1]:
            return False
    return True

def _odd_even_transposition():
    net = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            net.append((i, i + 1))
    return net

def _odd_even_mergesort(n):
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    half = n // 2
    left = _odd_even_mergesort(half)
    right = _odd_even_mergesort(n - half)
    right = [(i + half, j + half) for i, j in right]
    
    # Merge phase
    # OES Merge for a=half, b=n-half
    # We implement a simple OES merge
    a = half
    b = n - half
    merge_comps = []
    
    # OES Merge algorithm:
    # min(a, b) phases
    for k in range(min(a, b)):
        if k % 2 == 0:
            # Even phase
            # Compare (i, a+i) for i in [0, a-1]
            # But we must respect boundaries
            for i in range(a):
                if i + k < a: # This logic is tricky
                    pass
            # Correct OES Merge:
            # Phase k:
            # If k even:
            #   (k, a+k), (k+1, a+k+1), ... (a-1, 2a-1) ??
            #   And (a+k, a+k+1), ...
            
            # Let's use a known good implementation for OES Merge
            # From "Sorting Networks" by Knuth or similar
            # For simplicity and correctness, I will use a slightly different seed:
            # Batcher's Batcher for 16 is complex to generate.
            # I will use the OES seed as is, even if it's 147.
            
            # Actually, let's just return OES for the whole N.
            # The merge step in OES is:
            # For k in range(min(a,b)):
            #   if k % 2 == 0:
            #      for i in range(k, a):
            #         merge_comps.append((i, a+i))
            #   else:
            #      for i in range(k, a-1):
            #         merge_comps.append((a+i, a+i+1))
            
            if k % 2 == 0:
                for i in range(k, a):
                    merge_comps.append((i, a + i))
            else:
                for i in range(k, a - 1):
                    merge_comps.append((a + i, a + i + 1))
                    
    return left + right + merge_comps

def _greedy_reduce(comparators, deadline):
    current = list(comparators)
    i = 0
    while i < len(current):
        if time.time() > deadline:
            break
        saved = current[i]
        current.pop(i)
        if _is_sorting_network(current):
            pass
        else:
            current.insert(i, saved)
            i += 1
    return current

def _simulated_annealing(start_network, deadline):
    current = list(start_network)
    best = list(current)
    best_size = len(current)
    
    T = 2.0
    cooling_rate = 0.99
    
    while time.time() < deadline and T > 0.05:
        r = random.random()
        neighbor = list(current)
        
        if r < 0.4:
            if len(neighbor) > 0:
                idx = random.randint(0, len(neighbor) - 1)
                neighbor.pop(idx)
                delta = -1
            else:
                continue
        elif r < 0.7:
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            neighbor.append((i, j))
            delta = 1
        elif r < 0.85:
            if len(neighbor) > 1:
                idx1 = random.randint(0, len(neighbor) - 1)
                idx2 = random.randint(0, len(neighbor) - 1)
                if idx1 != idx2:
                    neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
                    delta = 0
                else:
                    continue
            else:
                continue
        else:
            if len(neighbor) > 1:
                idx1 = random.randint(0, len(neighbor) - 1)
                idx2 = random.randint(0, len(neighbor) - 1)
                if idx1 != idx2:
                    comp = neighbor.pop(idx1)
                    neighbor.insert(idx2, comp)
                    delta = 0
                else:
                    continue
            else:
                continue
        
        if _is_sorting_network(neighbor):
            new_size = len(neighbor)
            if new_size < best_size:
                best = neighbor
                best_size = new_size
            
            if delta < 0:
                current = neighbor
            elif delta == 0:
                current = neighbor
            else:
                if random.random() < math.exp(-1.0 / T):
                    current = neighbor
        
        T *= cooling_rate
        
    return best

def construct():
    # Seed 1: Odd-Even Transposition
    seed_oet = _odd_even_transposition()
    
    # Seed 2: Odd-Even Mergesort
    seed_oem = _odd_even_mergesort(N)
    
    best = seed_oet
    start_time = time.time()
    deadline = start_time + 85
    
    # Initial reduction of both seeds
    best_oet = _greedy_reduce(seed_oet, min(deadline, start_time + 10))
    best_oem = _greedy_reduce(seed_oem, min(deadline, start_time + 10))
    
    if len(best_oem) < len(best_oet):
        best = best_oem
    else:
        best = best_oet
        
    # SA Loop
    while time.time() < deadline:
        # Pick a start point
        if random.random() < 0.5:
            start_net = list(best)
        else:
            # Perturb best
            start_net = list(best)
            for _ in range(random.randint(5, 15)):
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                start_net.append((i, j))
            start_net = _greedy_reduce(start_net, min(deadline, time.time() + 2))
            
        candidate = _simulated_annealing(start_net, deadline)
        
        if len(candidate) < len(best):
            best = candidate
            # Quick greedy
            if time.time() < deadline:
                best = _greedy_reduce(best, min(deadline, time.time() + 2))
                
    return best
# EVOLVE-BLOCK-END
