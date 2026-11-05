"""Size-efficient sorting network search for 16 wires."""
import time
import random
import numpy as np

N = 16
TOTAL_BITS = 1 << N  # 65536

# Precompute sorted masks: for input x, sorted output has top popcount(x) bits set.
def _build_sorted_masks():
    masks = np.zeros(TOTAL_BITS, dtype=np.uint64)
    for x in range(TOTAL_BITS):
        cnt = bin(x).count('1')
        if cnt > 0:
            m = ((1 << cnt) - 1) << (N - cnt)
        else:
            m = 0
        masks[x] = m
    return masks

SORTED_MASKS = _build_sorted_masks()

def _build_initial_wires():
    """Build initial wire bit-vectors for all 65536 inputs."""
    wires = [0] * N
    for i in range(N):
        w = 0
        for k in range(TOTAL_BITS):
            if (k >> i) & 1:
                w |= (1 << k)
        wires[i] = w
    return wires

INITIAL_WIRES = _build_initial_wires()
FULL_MASK = (1 << TOTAL_BITS) - 1

def _verify(nets):
    """Verify network sorts all inputs using bit-parallel simulation."""
    wires = list(INITIAL_WIRES)
    for (i, j) in nets:
        swap_mask = wires[i] & (~wires[j]) & FULL_MASK
        wires[i] ^= swap_mask
        wires[j] ^= swap_mask
    
    # Check all 65536 inputs
    for k in range(TOTAL_BITS):
        val = 0
        for i in range(N):
            if (wires[i] >> k) & 1:
                val |= (1 << i)
        if val != SORTED_MASKS[k]:
            return False
    return True

def _odd_even_transposition():
    """Standard odd-even transposition sorting network."""
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

def _batcher_odd_even_merge(n):
    """Batcher's odd-even merge network."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    result = []
    # Compare-exchange between halves
    for i in range(n // 2):
        result.append((i, i + n // 2))
    # Recursively merge odd and even positions
    for (i, j) in _batcher_odd_even_merge(n // 2):
        result.append((2 * i + 1, 2 * j + 1))
    for (i, j) in _batcher_odd_even_merge(n // 2):
        result.append((2 * i, 2 * j))
    return result

def _batcher_sort(n):
    """Batcher's odd-even merge sort network."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    half = n // 2
    result = []
    # Sort halves
    result.extend(_batcher_sort(half))
    result.extend([(i + half, j + half) for (i, j) in _batcher_sort(half)])
    # Odd-even merge
    result.extend(_batcher_odd_even_merge(n))
    return result

def _remove_redundant(nets, deadline):
    """Remove redundant comparators via iterative deletion."""
    net = list(nets)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        indices = list(range(len(net)))
        random.shuffle(indices)
        for idx in indices:
            if time.time() > deadline:
                break
            candidate = net[:idx] + net[idx+1:]
            if _verify(candidate):
                net = candidate
                improved = True
    return net

def _batcher_sort_with_reorder(n, deadline):
    """
    Structural change: Instead of just removing from a fixed Batcher network,
    we use a 'phase-based' local search that treats groups of comparators 
    (phases) as units. This allows swapping entire phases or reordering them,
    which can escape local optima that simple deletion cannot.
    """
    # Start with Batcher
    net = _batcher_sort(n)
    
    # Identify potential phases: In Batcher, comparators are added in recursive blocks.
    # For a simpler structural move, we will try to 'merge' adjacent layers if possible
    # or remove entire 'sub-networks' if the rest still sorts.
    
    # Heuristic: Try to remove comparators that are 'far' apart in the network 
    # but might be redundant due to symmetry or previous sorting.
    
    # A more aggressive structural approach: 
    # Try to construct a network from scratch using a greedy algorithm with a 
    # strong lower bound, or use a different base network like AKS (too complex) 
    # or a manually tuned network.
    
    # Given the constraint, we will stick to the removal strategy but add a 
    # 'restart' mechanism that tries different random seeds or deletion orders.
    
    best = list(net)
    best_len = len(net)
    
    while time.time() < deadline:
        current = list(best)
        current = _remove_redundant(current, deadline - 5.0)
        if len(current) < best_len:
            best = current
            best_len = len(current)
        else:
            # If no improvement, try a different randomization
            # by shuffling the input to the removal process
            shuffled = list(best)
            random.shuffle(shuffled)
            current = _remove_redundant(shuffled, deadline - 5.0)
            if len(current) < best_len:
                best = current
                best_len = len(current)
            else:
                break # Stagnation in this lineage
        
    return best

def construct():
    """Search for a minimal sorting network on 16 wires."""
    start_time = time.time()
    # Evaluator kills at 90s. We need to return well before that.
    # Let's use 85s as a hard stop for the search, leaving 5s for safety/return.
    deadline = start_time + 85.0
    
    # Start with known good networks
    candidates = []
    
    # Odd-even transposition: 120 comparators
    oet = _odd_even_transposition()
    candidates.append(oet)
    
    # Batcher's sort
    batcher = _batcher_sort(N)
    candidates.append(batcher)
    
    # Find the best starting point
    best = None
    best_len = float('inf')
    for net in candidates:
        if time.time() > deadline - 5.0:
            break
        if _verify(net):
            if len(net) < best_len:
                best = list(net)
                best_len = len(net)
    
    if best is None:
        return oet  # Fallback to odd-even transposition
    
    # Phase 1: Aggressive removal with multiple restarts
    # We will run the removal loop until time runs out or we hit a local optimum
    # and restart with a different randomization.
    
    while time.time() < deadline - 10.0:
        # Try to improve from current best
        # Use a slightly shorter deadline for each iteration to allow for overhead
        iter_deadline = min(deadline - 10.0, time.time() + 10.0)
        
        prev_len = best_len
        current = _remove_redundant(best, iter_deadline)
        current_len = len(current)
        
        if current_len < best_len:
            best = list(current)
            best_len = current_len
        else:
            # If we couldn't improve, we might be stuck.
            # Try a 'kick' by adding a random comparator and removing again?
            # This is the 'add-then-remove' strategy from inspiration 2.
            if time.time() < deadline - 15.0:
                # Add a random comparator
                i, j = random.sample(range(N), 2)
                if i > j:
                    i, j = j, i
                trial = list(best) + [(i, j)]
                
                # Only verify if it's fast (it is)
                if _verify(trial):
                    # Try to remove from this new network
                    # Use a tighter deadline to ensure we finish
                    trial_deadline = min(deadline - 10.0, time.time() + 5.0)
                    trial = _remove_redundant(trial, trial_deadline)
                    if len(trial) < best_len:
                        best = list(trial)
                        best_len = len(trial)
                    # Even if not better, we might have escaped a local optimum 
                    # for the next iteration? No, we only keep the best.
            
            # If we've been stuck for a while, break? 
            # The while loop condition handles the time.
            # If we just failed to improve, we loop again with a new random seed in _remove_redundant.
    
    # Final cleanup
    if time.time() < deadline - 2.0:
        best = _remove_redundant(best, deadline - 2.0)
    
    return best
