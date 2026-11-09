"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 17
_DEADLINE = 80  # seconds, safety margin before 90s kill

# EVOLVE-BLOCK-START
def _bitonic_sort(n):
    """Generate a Batcher bitonic sort network for n wires (power of 2)."""
    if n <= 1:
        return []
    import math
    k = int(math.log2(n))
    comps = []
    for stage in range(1, k + 1):
        for j in range(stage - 1, -1, -1):
            step = 1 << j
            for i in range(0, n - step, step * 2):
                idx = i // step
                if (idx >> j) & 1 == 0:
                    comps.append((i, i + step))
                else:
                    comps.append((i + step, i))
    return comps

def _build_seed_network():
    """Build a seed sorting network for 17 wires using 16-wire bitonic sort
    plus linear insertion of the 17th wire."""
    net16 = _bitonic_sort(16)
    insert_comps = [(i, 16) for i in range(15, -1, -1)]
    return net16 + insert_comps

def _verify(network):
    """Verify sorting network using zero-one principle."""
    for mask in range(1 << N):
        for i, j in network:
            vi = (mask >> i) & 1
            vj = (mask >> j) & 1
            if vi > vj:
                mask |= (1 << i)
                mask &= ~(1 << j)
            elif vi < vj:
                mask &= ~(1 << i)
                mask |= (1 << j)
        if mask == 0:
            continue
        tz = 0
        m = mask
        while (m & 1) == 0:
            tz += 1
            m >>= 1
        expected = ((1 << (N - tz)) - 1) << tz
        if mask != expected:
            return False
    return True

def _local_search(network, deadline):
    """Steepest-descent local search to remove redundant comparators."""
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for idx in range(len(network) - 1, -1, -1):
            if time.time() > deadline:
                return network
            candidate = network[:idx] + network[idx + 1:]
            if _verify(candidate):
                network = candidate
                changed = True
                break
        if not changed and time.time() < deadline:
            i = 0
            while i < len(network) and time.time() < deadline:
                candidate = network[:i] + network[i + 1:]
                if _verify(candidate):
                    network = candidate
                    changed = True
                    break
                i += 1
    return network

def construct():
    start_time = time.time()
    network = _build_seed_network()
    if not _verify(network):
        network = [(i, i + 1) for phase in range(N)
                   for i in range(phase % 2, N - 1, 2)]
    network = _local_search(network, _DEADLINE - 2)
    return network
# EVOLVE-BLOCK-END
