import time
import numpy as np
import random

N = 16
DEADLINE = time.time() + 80  # safety margin before 90s kill

def _verify_full(network):
    """Full verification using numpy over all 65536 states."""
    states = np.arange(65536, dtype=np.int64)
    for i, j in network:
        bi = (states >> i) & 1
        bj = (states >> j) & 1
        swap = bi > bj
        states = np.where(swap, states ^ ((1 << i) | (1 << j)), states)
    return bool(np.all(((states + 1) & states) == 0))

def _verify_sample(network, n_samples, rng):
    """Verify on random samples."""
    for _ in range(n_samples):
        state = rng.getrandbits(16)
        for i, j in network:
            bi = (state >> i) & 1
            bj = (state >> j) & 1
            if bi > bj:
                state ^= (1 << i) | (1 << j)
        if (state + 1) & state != 0:
            return False
    return True

def _bitonic_sort_network(n):
    """Generate bitonic sort network for n wires."""
    if n <= 1:
        return []
    
    def generate(start, length, direction):
        if length <= 1:
            return []
        
        half = length // 2
        comps = []
        
        # Sort halves in opposite directions
        comps.extend(generate(start, half, 1))
        comps.extend(generate(start + half, half, -1))
        
        # Bitonic Merge
        for i in range(half):
            a = start + i
            b = start + half + i
            if direction == 1:
                comps.append((a, b))
            else:
                comps.append((b, a))
        
        # Recursively merge
        comps.extend(generate(start, half, direction))
        comps.extend(generate(start + half, half, direction))
        
        return comps
    
    return generate(0, n, 1)

def construct():
    start = time.time()
    rng = random.Random(12345)
    
    # Start with Bitonic Sort (56 comparators for N=16)
    net = _bitonic_sort_network(N)
    best = list(net)
    
    # Verify initial network
    if not _verify_full(best):
        # Fallback to odd-even transposition (120 comparators)
        best = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    # Greedy removal with multiple passes
    improved = True
    while improved and time.time() < DEADLINE - 10:
        improved = False
        order = list(range(len(best)))
        rng.shuffle(order)
        for idx in order:
            if time.time() > DEADLINE - 10:
                break
            candidate = best[:idx] + best[idx+1:]
            # Fast verification: check a subset of states
            if _verify_sample(candidate, 200, rng):
                if _verify_full(candidate):
                    best = candidate
                    improved = True
                    # Reset order since best changed
                    order = list(range(len(best)))
                    rng.shuffle(order)
    
    # Try to replace pairs of comparators with single ones
    if time.time() < DEADLINE - 15:
        all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
        
        for _ in range(20):  # Limited attempts
            if time.time() > DEADLINE - 15:
                break
            # Try removing 2 and adding 1
            for i in range(len(best)):
                for j in range(i + 1, len(best)):
                    if time.time() > DEADLINE - 15:
                        break
                    candidate = best[:i] + best[i+1:j] + best[j+1:]
                    # Try adding a new comparator
                    new_comp = rng.choice(all_pairs)
                    if new_comp not in set(candidate):
                        candidate.append(new_comp)
                        if _verify_sample(candidate, 200, rng):
                            if _verify_full(candidate):
                                best = candidate
    
    # Final full verification
    if _verify_full(best):
        return best
    else:
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
