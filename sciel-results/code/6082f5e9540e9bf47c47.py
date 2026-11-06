# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    N = 16
    DEADLINE = 85.0
    start_time = time.time()
    
    def verify_numpy(comps):
        if not comps:
            return False
        masks = np.arange(1 << N, dtype=np.uint16)
        wires = np.zeros((1 << N, N), dtype=np.uint8)
        for i in range(N):
            wires[:, i] = (masks >> i) & 1
        for i, j in comps:
            a = wires[:, i].copy()
            b = wires[:, j].copy()
            wires[:, i] = np.minimum(a, b)
            wires[:, j] = np.maximum(a, b)
        for k in range(N - 1):
            if np.any(wires[:, k] > wires[:, k + 1]):
                return False
        return True
    
    def oe_merge(n1, n2):
        if n1 == 0:
            return []
        if n2 == 0:
            return []
        if n1 == 1 and n2 == 1:
            return [(0, 1)]
        if n1 == 1:
            return [(0, i) for i in range(n2)]
        if n2 == 1:
            return [(i, n1) for i in range(n1)]
        n = n1
        if n == 2:
            return [(0, 2), (1, 3), (1, 2)]
        comps = []
        for i in range(1, n, 2):
            comps.append((i, n + i))
        for i in range(0, n, 2):
            comps.append((i, n + i))
        for (a, b) in oe_merge(n // 2, n // 2):
            comps.append((2 * a, 2 * b))
        for (a, b) in oe_merge(n // 2, n // 2):
            comps.append((2 * a + 1, 2 * b + 1))
        return comps
    
    def oe_merge_sort(n):
        if n == 1:
            return []
        if n == 2:
            return [(0, 1)]
        half = n // 2
        left = oe_merge_sort(half)
        right = [(i + half, j + half) for i, j in oe_merge_sort(half)]
        merge = oe_merge(half, half)
        return left + right + merge
    
    network = oe_merge_sort(16)
    
    if time.time() > DEADLINE:
        return network
    
    if not verify_numpy(network):
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    best = list(network)
    best_count = len(best)
    
    # Multiple passes of greedy removal
    for _ in range(10):
        if time.time() > DEADLINE:
            break
        comps = list(best)
        i = 0
        while i < len(comps):
            if time.time() > DEADLINE:
                break
            new_comps = comps[:i] + comps[i+1:]
            if verify_numpy(new_comps):
                comps = new_comps
            else:
                i += 1
        if len(comps) < best_count:
            best = list(comps)
            best_count = len(comps)
    
    # Random removal
    random.seed(42)
    comps = list(best)
    for _ in range(20000):
        if time.time() > DEADLINE:
            break
        if len(comps) <= best_count:
            break
        idx = random.randint(0, len(comps) - 1)
        new_comps = comps[:idx] + comps[idx+1:]
        if verify_numpy(new_comps):
            comps = new_comps
            if len(comps) < best_count:
                best = list(comps)
                best_count = len(comps)
    
    return best
# EVOLVE-BLOCK-END
