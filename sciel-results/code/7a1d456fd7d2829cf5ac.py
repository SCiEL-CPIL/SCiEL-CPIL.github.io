import time
import numpy as np

N = 16
TOTAL_ZERO_ONE = 1 << N

T_START = time.time()
DEADLINE = T_START + 82.0

# Precompute target masks for sorted output
# For input m with popcount p, sorted output has wire k = 0 iff k < 16-p
# i.e., wire k = 0 iff popcount(m) <= 15-k
def _precompute_targets():
    targets = []
    for k in range(N):
        threshold = 15 - k
        mask = 0
        for m in range(TOTAL_ZERO_ONE):
            if bin(m).count('1') <= threshold:
                mask |= (1 << m)
        targets.append(mask)
    return targets

TARGETS = _precompute_targets()

# Precompute initial wires: wire k holds bit k of each input
# _INIT_WIRES[k] is a 65536-bit int where bit m = (m >> k) & 1
_INIT_WIRES = [0] * N
for _k in range(N):
    _w = 0
    for _m in range(TOTAL_ZERO_ONE):
        if (_m >> _k) & 1:
            _w |= (1 << _m)
    _INIT_WIRES[_k] = _w

def _verify(comp_list):
    """Fast verification using big-int bit operations."""
    wires = _INIT_WIRES[:]
    for i, j in comp_list:
        a = wires[i]
        b = wires[j]
        lo = a & b
        hi = a | b
        wires[i] = lo
        wires[j] = hi
    for k in range(N):
        if wires[k] != TARGETS[k]:
            return False
    return True

def _batcher_oe_merge_sort(n):
    def oe_merge(left, right):
        comps = []
        h = len(left)
        if h == 1:
            comps.append((left[0], right[0]))
            return comps
        le = [left[k] for k in range(0, h, 2)]
        lo = [left[k] for k in range(1, h, 2)]
        re = [right[k] for k in range(0, h, 2)]
        ro = [right[k] for k in range(1, h, 2)]
        comps += oe_merge(le, re)
        comps += oe_merge(lo, ro)
        # Fixed: Compare left[k] and right[k-1] for k=1, 3, ...
        for k in range(1, h, 2):
            comps.append((left[k], right[k - 1]))
        return comps

    def sort_block(wires):
        m = len(wires)
        if m == 1:
            return []
        h = m // 2
        left = wires[:h]
        right = wires[h:]
        comps = []
        comps += sort_block(left)
        comps += sort_block(right)
        comps += oe_merge(left, right)
        return comps

    return sort_block(list(range(n)))

def _remove_redundant(comp_list):
    cur = list(comp_list)
    changed = True
    while changed and time.time() < DEADLINE:
        changed = False
        i = 0
        while i < len(cur) and time.time() < DEADLINE:
            trial = cur[:i] + cur[i + 1:]
            if _verify(trial):
                cur = trial
                changed = True
            else:
                i += 1
    return cur

def construct():
    network = _batcher_oe_merge_sort(N)
    
    if not _verify(network):
        # Fallback
        network = [(i, i + 1) for phase in range(N)
                   for i in range(phase % 2, N - 1, 2)]
        
    best = network
    best = _remove_redundant(best)
    
    # Multiple passes of deletion to find more redundancies
    while time.time() < DEADLINE - 5.0:
        prev_len = len(best)
        best = _remove_redundant(best)
        if len(best) == prev_len:
            break
    
    if time.time() < DEADLINE - 2.0:
        if not _verify(best):
            best = network
            
    return [(int(i), int(j)) for (i, j) in best]
