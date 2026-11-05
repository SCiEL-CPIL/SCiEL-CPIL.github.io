# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations
from math import comb

def _check_subset_fast(idx, adj):
    """Check if subset idx induces a regular subgraph using NumPy."""
    sub = adj[np.ix_(idx, idx)]
    degs = sub.sum(axis=1)
    return len(set(degs.tolist())) == 1

def _find_violation_bitmask(n, adj_bits, max_check_6=38760, max_check_7=77520):
    """
    Find a violation (regular 6 or 7 subset) using bitmask representation.
    Returns (subset, size) or None.
    """
    # Check all 6-subsets
    for combo in combinations(range(n), 6):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        degs = []
        for i in combo:
            d = bin(adj_bits[i] & mask).count('1')
            degs.append(d)
        if len(set(degs)) == 1:
            return list(combo), 6

    # Check all 7-subsets
    for combo in combinations(range(n), 7):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        degs = []
        for i in combo:
            d = bin(adj_bits[i] & mask).count('1')
            degs.append(d)
        if len(set(degs)) == 1:
            return list(combo), 7

    return None

def _sample_check_8plus(n, adj, samples_per_size=50):
    """Sample check for subsets of size 8 and above."""
    for sz in range(8, n + 1):
        total = comb(n, sz)
        if total == 0:
            continue
        num_samples = min(total, samples_per_size)
        for _ in range(num_samples):
            idx = random.sample(range(n), sz)
            if _check_subset_fast(idx, adj):
                return idx
    return None

def _full_verify(n, adj):
    """
    Full verification: check all 6- and 7-subsets, sample 8+.
    Returns True if no violation found.
    """
    # All 6-subsets
    for combo in combinations(range(n), 6):
        idx = list(combo)
        sub = adj[np.ix_(idx, idx)]
        degs = sub.sum(axis=1)
        if len(set(degs.tolist())) == 1:
            return False

    # All 7-subsets
    for combo in combinations(range(n), 7):
        idx = list(combo)
        sub = adj[np.ix_(idx, idx)]
        degs = sub.sum(axis=1)
        if len(set(degs.tolist())) == 1:
            return False

    # Sample 8+
    for sz in range(8, n + 1):
        total = comb(n, sz)
        if total == 0:
            continue
        num_samples = min(total, 100)
        for _ in range(num_samples):
            idx = random.sample(range(n), sz)
            sub = adj[np.ix_(idx, idx)]
            degs = sub.sum(axis=1)
            if len(set(degs.tolist())) == 1:
                return False

    return True

def _random_adj(n, density):
    """Generate random adjacency matrix."""
    adj = np.zeros((n, n), dtype=np.uint8)
    for u in range(n):
        for v in range(u + 1, n):
            if random.random() < density:
                adj[u, v] = 1
                adj[v, u] = 1
    return adj

def _adj_to_bits(adj):
    """Convert adjacency matrix to bitmask list."""
    n = adj.shape[0]
    bits = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v]:
                bits[u] |= (1 << v)
                bits[v] |= (1 << u)
    return bits

def _bits_to_adj(bits, n):
    """Convert bitmask list to adjacency matrix."""
    adj = np.zeros((n, n), dtype=np.uint8)
    for u in range(n):
        for v in range(u + 1, n):
            if (bits[u] >> v) & 1:
                adj[u, v] = 1
                adj[v, u] = 1
    return adj

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    best_n = 5
    best_edges = []

    # Try n from 20 down to 6
    for n in range(20, 5, -1):
        if time.time() > deadline - 10:
            break

        found = False
        max_restarts = 30 if n <= 15 else 15

        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break

            # Random initialization with varied density
            density = 0.2 + 0.6 * random.random()
            adj = _random_adj(n, density)
            adj_bits = _adj_to_bits(adj)

            # Quick sample check first
            quick_ok = True
            for _ in range(30):
                idx = random.sample(range(n), 6)
                sub = adj[np.ix_(idx, idx)]
                degs = sub.sum(axis=1)
                if len(set(degs.tolist())) == 1:
                    quick_ok = False
                    break

            if not quick_ok:
                continue

            # Local search targeting violations
            best_bad = None
            for step in range(500):
                if time.time() > deadline - 10:
                    break

                # Find a violation using bitmask (fast)
                violation = _find_violation_bitmask(n, adj_bits)

                if violation is None:
                    # No 6 or 7 violation found; do full verification
                    if _full_verify(n, adj):
                        edges = []
                        for u in range(n):
                            for v in range(u + 1, n):
                                if adj[u, v]:
                                    edges.append([u, v])
                        best_n = n
                        best_edges = edges
                        found = True
                        break
                    else:
                        # Full verify found a violation (likely 8+); 
                        # try to fix by random flip
                        u = random.randint(0, n - 1)
                        v = random.randint(0, n - 1)
                        if u != v:
                            adj[u, v] = 1 - adj[u, v]
                            adj[v, u] = adj[u, v]
                            adj_bits = _adj_to_bits(adj)
                        continue

                # Flip an edge within the violation
                subset, size = violation
                u, v = random.sample(subset, 2)
                adj[u, v] = 1 - adj[u, v]
                adj[v, u] = adj[u, v]
                adj_bits = _adj_to_bits(adj)

            if found:
                break

        if found:
            continue

    # If we haven't found n>=6, return trivial
    if best_n < 6:
        best_n = 5
        best_edges = []

    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
