import time
import random
import numpy as np

N = 16
NUM_INPUTS = 1 << N

# Precompute all 65536 boolean inputs as a (65536, 16) int array (columns = wires).
def _build_inputs():
    arr = np.arange(NUM_INPUTS, dtype=np.uint8).reshape(-1, 1)
    # each row r -> its 16 bits across the 16 wires
    bits = 1 << np.arange(N, dtype=np.uint8)
    x = (arr & bits)
    return x.astype(np.int8)

_INPUTS = _build_inputs()
_SORTED = np.zeros((NUM_INPUTS, N), dtype=np.int8)
_SORTED[:, 0] = 0
for _c in range(1, N):
    _SORTED[:, _c] = _SORTED[:, _c - 1] + 1  # wait, this is wrong for boolean; fix below
# Actually the sorted boolean sequence for a k-hot input is [0]*k + [1]*(16-k).
# Build it properly:
def _build_sorted():
    out = np.zeros((NUM_INPUTS, N), dtype=np.int8)
    counts = np.arange(NUM_INPUTS, dtype=np.int16)
    for c in range(N):
        # wire c is 1 iff c >= popcount(input)
        # popcount of the 16-bit index
        out[:, c] = 0
    # popcount via lookup
    pc = np.zeros(NUM_INPUTS, dtype=np.int8)
    for b in range(N):
        pc += (np.arange(NUM_INPUTS, dtype=np.uint16) >> b) & 1
    for c in range(N):
        out[:, c] = (c >= pc).astype(np.int8)
    return out

_SORTED = _build_sorted()

# Fast network application: list of (i,j) -> returns output array (NUM_INPUTS, N)
def _apply(net, x):
    x = x.copy()
    for i, j in net:
        a = x[:, i]
        b = x[:, j]
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        x[:, i] = lo
        x[:, j] = hi
    return x

def _is_valid(net):
    if not net:
        return False
    out = _apply(net, _INPUTS)
    return bool(np.array_equal(out, _SORTED))

# Depth of a network (parallel layers)
def _depth(net):
    d = np.zeros(N, dtype=np.int32)
    for i, j in net:
        nd = d[i] + 1
        if nd > d[j]:
            d[j] = nd
        if nd > d[i]:
            d[i] = nd
    return int(d.max())

def _bitonic_sort_network(n):
    def bitonic_merge(size, start, direction):
        if size == 2:
            if direction == 1:
                return [(start, start + 1)]
            else:
                return [(start + 1, start)]
        
        half = size // 2
        result = []
        for i in range(half):
            a = start + i
            b = start + i + half
            if direction == 1:
                result.append((a, b))
            else:
                result.append((b, a))
        
        result.extend(bitonic_merge(half, start, direction))
        result.extend(bitonic_merge(half, start + half, direction))
        
        return result

    def bitonic_sort(size, start, direction):
        if size == 2:
            if direction == 1:
                return [(start, start + 1)]
            else:
                return [(start + 1, start)]
        
        half = size // 2
        result = []
        result.extend(bitonic_sort(half, start, direction))
        result.extend(bitonic_sort(half, start + half, 1 - direction))
        result.extend(bitonic_merge(size, start, direction))
        
        return result

    return bitonic_sort(n, 0, 1)

def _batcher_odd_even(n):
    # Batcher's odd-even mergesort network for n (power of 2)
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    half = n // 2
    left = _batcher_odd_even(half)
    right = _batcher_odd_even(half)
    # even-odd stage: compare (0,2),(4,6),... then (1,3),(5,7),...
    stage = []
    for i in range(0, n, 2):
        if i + 1 < n:
            stage.append((i, i + 1))
    # Actually Batcher: sort halves, then even-odd merge
    # Even-odd merge network for n
    def eo_merge(n):
        comps = []
        # even-odd transposition sort on n
        for phase in range(n):
            start = phase % 2
            for i in range(start, n - 1, 2):
                comps.append((i, i + 1))
        return comps
    return left + right + eo_merge(n)

def _initial_networks():
    nets = []
    # Odd-even transposition (120 comparators)
    nets.append([(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)])
    # Batcher
    nets.append(_batcher_odd_even(N))
    return nets

def _crossover(a, b):
    # Layered crossover: split at a random point in terms of "steps"
    # Simpler: take a random prefix of a and random suffix of b, then dedupe/fix
    la, lb = len(a), len(b)
    cut_a = random.randint(0, la)
    cut_b = random.randint(0, lb)
    child = list(a[:cut_a]) + list(b[cut_b:])
    return child

def _mutate(net, rate=0.15):
    net = list(net)
    if not net:
        return net
    # Random insert
    if random.random() < 0.5:
        i = random.randrange(N)
        j = random.randrange(N)
        if i == j:
            j = (j + 1) % N
        if i > j:
            i, j = j, i
        pos = random.randrange(len(net) + 1)
        net.insert(pos, (i, j))
    # Random delete
    if random.random() < 0.5 and net:
        pos = random.randrange(len(net))
        net.pop(pos)
    # Random replace
    if random.random() < 0.5 and net:
        pos = random.randrange(len(net))
        i = random.randrange(N)
        j = random.randrange(N)
        if i == j:
            j = (j + 1) % N
        if i > j:
            i, j = j, i
        net[pos] = (i, j)
    return net

def construct():
    start = time.time()
    deadline = start + 50.0  # Safety margin

    # Start with Bitonic Sort
    net = _bitonic_sort_network(N)
    
    if not _is_valid(net):
        # Fallback to Odd-Even Transposition if Bitonic fails
        net = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

    best_net = list(net)
    best_len = len(net)

    # Local Search: Greedy Removal
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(len(best_net) - 1, -1, -1):
            if time.time() > deadline:
                break
            
            candidate = best_net[:idx] + best_net[idx+1:]
            
            if _is_valid(candidate):
                best_net = candidate
                best_len = len(best_net)
                improved = True
                break
    
    # If time remains, try Batcher's Odd-Even Mergesort as a second baseline
    if time.time() < deadline - 5.0:
        def _batcher_odd_even(n):
            if n <= 1:
                return []
            if n == 2:
                return [(0, 1)]
            half = n // 2
            left = _batcher_odd_even(half)
            right = [(i + n // 2, j + n // 2) for i, j in _batcher_odd_even(half)]
            
            def eo_merge(n):
                comps = []
                for phase in range(n):
                    start_idx = phase % 2
                    for i in range(start_idx, n - 1, 2):
                        comps.append((i, i + 1))
                return comps
            return left + right + eo_merge(n)

        net2 = _batcher_odd_even(N)
        if _is_valid(net2) and len(net2) < best_len:
            best_net = list(net2)
            best_len = len(best_net)
            
            improved = True
            while improved and time.time() < deadline - 5.0:
                improved = False
                for idx in range(len(best_net) - 1, -1, -1):
                    if time.time() > deadline - 5.0:
                        break
                    candidate = best_net[:idx] + best_net[idx+1:]
                    if _is_valid(candidate):
                        best_net = candidate
                        best_len = len(best_net)
                        improved = True
                        break

    return best_net
