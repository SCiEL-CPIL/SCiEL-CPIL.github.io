"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 16
DEADLINE = 85.0  # seconds, safety margin before 90s kill

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # Start with Batcher odd-even mergesort network for N=16
    network = _batcher_oe_mergesort(N)
    best_network = network[:]
    best_count = len(network)
    
    # Verify initial network
    if not _is_valid_network(network):
        # Fallback: use a simple known-good network (odd-even transposition)
        best_network = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
        return best_network
    
    # Greedy removal: try removing each comparator, verify correctness
    changed = True
    while changed and (time.time() - start_time) < DEADLINE:
        changed = False
        # Try removing from the end first
        for idx in range(len(network) - 1, -1, -1):
            if (time.time() - start_time) > DEADLINE - 10:
                break
            # Try removing comparator at idx
            candidate = network[:idx] + network[idx+1:]
            if _is_valid_network(candidate):
                network = candidate
                best_network = network[:]
                best_count = len(network)
                changed = True
    
    return best_network

def _batcher_oe_mergesort(n):
    """Generate Batcher odd-even mergesort network for n wires (n must be power of 2)."""
    comparators = []
    
    def _merge_k(k, offset):
        """Merge two sorted sequences of length 2^k starting at offset."""
        if k == 0:
            comparators.append((offset, offset + 1))
            return
        _merge_k(k - 1, offset)
        _merge_k(k - 1, offset + (1 << (k - 1)))
        # Odd-even merge
        half = 1 << (k - 1)
        total = 1 << k
        for i in range(total - 1):
            if i % 2 == 0:
                a = offset + i
                b = offset + i + half
            else:
                a = offset + i + 1
                b = offset + i + half
            if a > b:
                a, b = b, a
            comparators.append((a, b))
    
    def _sort_k(k, offset):
        """Sort 2^k wires starting at offset."""
        if k == 0:
            return
        _sort_k(k - 1, offset)
        _sort_k(k - 1, offset + (1 << (k - 1)))
        _merge_k(k - 1, offset)
    
    log2n = int(np.log2(n))
    _sort_k(log2n, 0)
    
    # Remove duplicates, ensure i < j
    seen = set()
    unique = []
    for i, j in comparators:
        if i > j:
            i, j = j, i
        if (i, j) not in seen:
            seen.add((i, j))
            unique.append((i, j))
    return unique

def _is_valid_network(network):
    """Check if network sorts all 2^N binary inputs using zero-one principle."""
    if not network:
        return N <= 1
    
    num_inputs = 1 << N
    indices = np.arange(num_inputs, dtype=np.uint16)
    bits = np.zeros((N, num_inputs), dtype=np.uint8)
    for k in range(N):
        bits[k] = (indices >> k) & 1
    
    for i, j in network:
        a = bits[i].copy()
        b = bits[j].copy()
        bits[i] = np.minimum(a, b)
        bits[j] = np.maximum(a, b)
    
    for k in range(N - 1):
        if not np.all(bits[k] >= bits[k+1]):
            return False
    return True
# EVOLVE-BLOCK-END
