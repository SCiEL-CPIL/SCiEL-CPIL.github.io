"""
Search for large graphs with no regular induced subgraph on 6+ vertices.
Strategy: random graph generation with fast bit-based regularity checking.
"""
import random
import time
from itertools import combinations

DEADLINE = 140  # safety margin before 150s kill

def _is_regular_induced(mask, adj_bits, n):
    """Check if the induced subgraph on vertices in mask is regular."""
    k = mask.bit_count()
    if k < 6:
        return False
    # Compute degrees within the induced subgraph
    # For each vertex in mask, count neighbors also in mask
    first = -1
    m = mask
    while m:
        low = m & (-m)
        v = low.bit_length() - 1
        if first == -1:
            first = v
            deg = (adj_bits[v] & mask).bit_count()
        else:
            deg = (adj_bits[v] & mask).bit_count()
            if deg != (adj_bits[first] & mask).bit_count():
                return False
        m ^= low
    return True

def _check_all_subsets(adj_bits, n):
    """Check all subsets of size >= 6 for regularity. Returns True if valid (no regular induced subgraph of size >= 6)."""
    # Check all subsets of size 6
    for combo in combinations(range(n), 6):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        if _is_regular_induced(mask, adj_bits, n):
            return False
    # Check all subsets of size 7
    for combo in combinations(range(n), 7):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        if _is_regular_induced(mask, adj_bits, n):
            return False
    # Check all subsets of size 8
    for combo in combinations(range(n), 8):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        if _is_regular_induced(mask, adj_bits, n):
            return False
    # Check all subsets of size 9
    for combo in combinations(range(n), 9):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        if _is_regular_induced(mask, adj_bits, n):
            return False
    # Check all subsets of size 10
    for combo in combinations(range(n), 10):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        if _is_regular_induced(mask, adj_bits, n):
            return False
    # Check all subsets of size 11
    for combo in combinations(range(n), 11):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        if _is_regular_induced(mask, adj_bits, n):
            return False
    # Check all subsets of size 12
    for combo in combinations(range(n), 12):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        if _is_regular_induced(mask, adj_bits, n):
            return False
    # Check all subsets of size 13
    for combo in combinations(range(n), 13):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        if _is_regular_induced(mask, adj_bits, n):
            return False
    # Check all subsets of size 14
    for combo in combinations(range(n), 14):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        if _is_regular_induced(mask, adj_bits, n):
            return False
    # Check all subsets of size 15
    for combo in combinations(range(n), 15):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        if _is_regular_induced(mask, adj_bits, n):
            return False
    # Check all subsets of size 16
    for combo in combinations(range(n), 16):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        if _is_regular_induced(mask, adj_bits, n):
            return False
    # Check all subsets of size 17
    for combo in combinations(range(n), 17):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        if _is_regular_induced(mask, adj_bits, n):
            return False
    # Check all subsets of size 18
    for combo in combinations(range(n), 18):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        if _is_regular_induced(mask, adj_bits, n):
            return False
    # Check all subsets of size 19
    for combo in combinations(range(n), 19):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        if _is_regular_induced(mask, adj_bits, n):
            return False
    # Check all subsets of size 20
    for combo in combinations(range(n), 20):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        if _is_regular_induced(mask, adj_bits, n):
            return False
    # Check all subsets of size 21
    for combo in combinations(range(n), 21):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        if _is_regular_induced(mask, adj_bits, n):
            return False
    # Check all subsets of size 22
    for combo in combinations(range(n), 22):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        if _is_regular_induced(mask, adj_bits, n):
            return False
    return True


def _fast_check(adj_bits, n):
    """Fast check: only check subsets of size 6 (most common violation)."""
    for combo in combinations(range(n), 6):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        if _is_regular_induced(mask, adj_bits, n):
            return False
    return True


def _full_check(adj_bits, n):
    """Full check all subsets of size >= 6."""
    for combo in combinations(range(n), 6):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        if _is_regular_induced(mask, adj_bits, n):
            return False
    for combo in combinations(range(n), 7):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        if _is_regular_induced(mask, adj_bits, n):
            return False
    for combo in combinations(range(n), 8):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        if _is_regular_induced(mask, adj_bits, n):
            return False
    for combo in combinations(range(n), 9):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        if _is_regular_induced(mask, adj_bits, n):
            return False
    for combo in combinations(range(n), 10):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        if _is_regular_induced(mask, adj_bits, n):
            return False
    for combo in combinations(range(n), 11):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        if _is_regular_induced(mask, adj_bits, n):
            return False
    for combo in combinations(range(n), 12):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        if _is_regular_induced(mask, adj_bits, n):
            return False
    for combo in combinations(range(n), 13):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        if _is_regular_induced(mask, adj_bits, n):
            return False
    for combo in combinations(range(n), 14):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        if _is_regular_induced(mask, adj_bits, n):
            return False
    for combo in combinations(range(n), 15):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        if _is_regular_induced(mask, adj_bits, n):
            return False
    for combo in combinations(range(n), 16):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        if _is_regular_induced(mask, adj_bits, n):
            return False
    for combo in combinations(range(n), 17):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        if _is_regular_induced(mask, adj_bits, n):
            return False
    for combo in combinations(range(n), 18):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        if _is_regular_induced(mask, adj_bits, n):
            return False
    for combo in combinations(range(n), 19):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        if _is_regular_induced(mask, adj_bits, n):
            return False
    for combo in combinations(range(n), 20):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        if _is_regular_induced(mask, adj_bits, n):
            return False
    for combo in combinations(range(n), 21):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        if _is_regular_induced(mask, adj_bits, n):
            return False
    for combo in combinations(range(n), 22):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        if _is_regular_induced(mask, adj_bits, n):
            return False
    return True


def construct():
    start_time = time.time()
    best_n = 5
    best_edges = []
    
    random.seed(42)
    
    # Try to find a 20-vertex graph
    # Random graph with varying edge probabilities
    for n in [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]:
        if time.time() - start_time > DEADLINE:
            break
        
        # Try different edge probabilities
        for p in [0.3, 0.4, 0.5, 0.6, 0.7, 0.35, 0.45, 0.55, 0.65]:
            if time.time() - start_time > DEADLINE:
                break
            
            # Generate random graph
            adj_bits = [0] * n
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < p:
                        adj_bits[i] |= (1 << j)
                        adj_bits[j] |= (1 << i)
                        edges.append([i, j])
            
            # Fast check first (size 6 subsets only)
            if not _fast_check(adj_bits, n):
                continue
            
            # Full check
            if _full_check(adj_bits, n):
                best_n = n
                best_edges = edges
                # If we found n=20, we're done (known maximum)
                if n == 20:
                    break
        if n == 20:
            break
    
    # Try more random graphs if we haven't found n=20
    if best_n < 20 and time.time() - start_time < DEADLINE:
        for attempt in range(500):
            if time.time() - start_time > DEADLINE:
                break
            n = 20
            p = 0.3 + 0.4 * random.random()
            adj_bits = [0] * n
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < p:
                        adj_bits[i] |= (1 << j)
                        adj_bits[j] |= (1 << i)
                        edges.append([i, j])
            
            if not _fast_check(adj_bits, n):
                continue
            
            if _full_check(adj_bits, n):
                best_n = n
                best_edges = edges
                break
    
    return {'n': best_n, 'edges': best_edges}
