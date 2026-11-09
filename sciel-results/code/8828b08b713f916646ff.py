import numpy as np
import time

N = 16
_IDX = np.arange(65536)
_SHIFTS = np.arange(N)[::-1]

def _odd_even_merge(n):
    if n == 1:
        return [(0, 1)]
    result = []
    for c in _odd_even_merge(n // 2):
        result.append((2 * c[0], 2 * c[1]))
    for c in _odd_even_merge(n // 2):
        result.append((2 * c[0] + 1, 2 * c[1] + 1))
    for i in range(0, 2 * n, 2):
        result.append((i, i + 1))
    return result

def _odd_even_sort(n):
    if n <= 1:
        return []
    h = n // 2
    first = _odd_even_sort(h)
    second = [(i + h, j + h) for i, j in _odd_even_sort(h)]
    merge = _odd_even_merge(h)
    return first + second + merge

def _verify(net):
    inputs = ((_IDX[:, None] >> _SHIFTS) & 1).astype(np.uint8)
    for i, j in net:
        mask = (inputs[:, i] == 1) & (inputs[:, j] == 0)
        if np.any(mask):
            inputs[mask, i] = 0
            inputs[mask, j] = 1
    for k in range(N - 1):
        if np.any((inputs[:, k] == 1) & (inputs[:, k + 1] == 0)):
            return False
    return True

def construct():
    deadline = time.time() + 80
    net = _odd_even_sort(N)
    
    for _ in range(20):
        if time.time() > deadline - 10:
            break
        changed = False
        for idx in range(len(net) - 1, -1, -1):
            if time.time() > deadline - 10:
                break
            candidate = net[:idx] + net[idx + 1:]
            if _verify(candidate):
                net = candidate
                changed = True
        if not changed:
            break
    
    return net
