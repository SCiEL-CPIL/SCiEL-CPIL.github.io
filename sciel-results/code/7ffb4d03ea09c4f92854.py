import time
import random
import numpy as np

N = 16
DEADLINE = 85.0  # seconds, safety margin before 90s kill

def _bitonic_network(n):
    """Generate Batcher's bitonic sort network for n=2^k wires.
    Returns list of (i, j) comparators, 0 <= i < j < n.
    """
    if n == 1:
        return []
    comps = []
    def rec(size, offset, ascending):
        if size == 1:
            return
        half = size // 2
        # Sort first half ascending, second half descending (or vice versa)
        rec(half, offset, True)
        rec(half, offset + half, not ascending)
        # Merge the two halves
        def merge(sz, off, asc):
            if sz == 1:
                return
            h = sz // 2
            # Compare elements that are h apart
            for i in range(h):
                a = off + i
                b = off + i + h
                comps.append((a, b))
            # Recursively merge the two resulting halves
            merge(h, off, asc)
            merge(h, off + h, asc)
        merge(size, offset, ascending)
    rec(n, 0, True)
    return comps

def _is_valid(net):
    """Check if network sorts all 2^16 boolean inputs using zero-one principle."""
    for mask in range(1 << N):
        m = mask
        for i, j in net:
            bi = (m >> i) & 1
            bj = (m >> j) & 1
            if bi > bj:
                m &= ~(1 << i)
                m |= (1 << j)
        if (m & (m + 1)) != 0:
            return False
    return True

def _is_valid_fast(net):
    """Vectorized-ish check. Still loops over all masks but uses local vars for speed."""
    net_list = list(net)
    for mask in range(1 << N):
        m = mask
        for i, j in net_list:
            bi = (m >> i) & 1
            bj = (m >> j) & 1
            if bi > bj:
                m &= ~(1 << i)
                m |= (1 << j)
        if (m & (m + 1)) != 0:
            return False
    return True

def construct():
    start_time = time.time()
    
    # Start with bitonic sort (63 comparators for N=16)
    current_net = _bitonic_network(N)
    
    # Verify it's valid (should be)
    if not _is_valid_fast(current_net):
        # Fallback to OET
        current_net = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    best_net = list(current_net)
    best_len = len(current_net)
    
    # Simulated annealing with remove/swap/insert moves
    # Objective: minimize len(net) subject to validity
    # We'll do: try to remove a comparator; if still valid, accept.
    # Also try swap/insert to escape local minima.
    
    rng = random.Random(42)
    
    def elapsed():
        return time.time() - start_time
    
    # Phase 1: Greedy removal (fast, gets us to a local minimum)
    improved = True
    while improved and elapsed() < DEADLINE - 20:
        improved = False
        for i in range(len(current_net)):
            if elapsed() > DEADLINE - 20:
                break
            candidate = current_net[:i] + current_net[i+1:]
            if _is_valid_fast(candidate):
                current_net = candidate
                improved = True
                if len(current_net) < best_len:
                    best_len = len(current_net)
                    best_net = list(current_net)
                break  # restart scan
    
    # Phase 2: Simulated annealing with insert/replace moves to escape local minima
    # Temperature schedule: start high, cool down
    T = 10.0
    T_min = 0.1
    cooling = 0.995
    
    # We'll allow "bad" moves (adding a comparator) with some probability to escape
    # But our objective is just to minimize count, so "cost" = len(net)
    # A move that removes a comp: cost delta = -1 (good)
    # A move that adds a comp: cost delta = +1 (bad, but might enable later removals)
    # A swap: cost delta = 0 (neutral, for exploration)
    
    while elapsed() < DEADLINE - 5 and T > T_min:
        move_type = rng.random()
        if move_type < 0.6:
            # Try to remove a random comparator
            if len(current_net) <= 1:
                T *= cooling
                continue
            idx = rng.randrange(len(current_net))
            candidate = current_net[:idx] + current_net[idx+1:]
            if _is_valid_fast(candidate):
                current_net = candidate
                if len(current_net) < best_len:
                    best_len = len(current_net)
                    best_net = list(current_net)
            T *= cooling
        elif move_type < 0.85:
            # Swap two comparators
            if len(current_net) < 2:
                T *= cooling
                continue
            i, j = rng.sample(range(len(current_net)), 2)
            candidate = list(current_net)
            candidate[i], candidate[j] = candidate[j], candidate[i]
            if _is_valid_fast(candidate):
                # Neutral move, accept with probability based on T
                if rng.random() < 0.5:  # always accept neutral valid moves
                    current_net = candidate
                if len(current_net) < best_len:
                    best_len = len(current_net)
                    best_net = list(current_net)
            T *= cooling
        else:
            # Insert a random comparator (to escape local minima)
            # Pick a random pair (i,j) not already in the network
            existing = set(tuple(sorted(c)) for c in current_net)
            attempts = 0
            while attempts < 10:
                i, j = rng.sample(range(N), 2)
                if i > j:
                    i, j = j, i
                if (i, j) not in existing:
                    break
                attempts += 1
            else:
                T *= cooling
                continue
            candidate = list(current_net) + [(i, j)]
            # Only accept if valid (it will be, since adding a comp can't break sorting... wait, it can't make it worse)
            # Actually adding a comparator always preserves validity if the original was valid
            # So we don't need to check, but let's be safe
            if _is_valid_fast(candidate):
                # This is a "bad" move in terms of count, but might help later
                # Accept with Boltzmann probability
                delta_cost = 1  # we added one
                if rng.random() < min(1.0, np.exp(-delta_cost / T)):
                    current_net = candidate
            T *= cooling
    
    # Phase 3: Final greedy removal pass
    improved = True
    while improved and elapsed() < DEADLINE - 2:
        improved = False
        for i in range(len(current_net)):
            if elapsed() > DEADLINE - 2:
                break
            candidate = current_net[:i] + current_net[i+1:]
            if _is_valid_fast(candidate):
                current_net = candidate
                improved = True
                if len(current_net) < best_len:
                    best_len = len(current_net)
                    best_net = list(current_net)
                break
    
    return best_net
