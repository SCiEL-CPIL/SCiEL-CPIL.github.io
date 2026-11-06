"""Sorting network on 18 wires: Batcher merge-sort base + local search refinement."""
import time
import random
import numpy as np

N = 18
DEADLINE_FRACTION = 0.85  # safety margin before 90s kill

# EVOLVE-BLOCK-START

def _batcher_odd_merge(left, right):
    """Odd-even merge of two sorted sequences (left and right are lists of (i,j) comparators
       that sort their respective sub-networks). Returns comparators to merge."""
    comparators = []
    # Standard Batcher odd-even merge
    # We need to merge two sorted lists of length m
    # The odd-even merge algorithm:
    # For k in range(m-1, 0, -1):
    #   for i in range(0, m-k):
    #     compare (left[i], right[i+k]) if i%2==0 else compare (left[i+k], right[i])
    # Actually let me implement the standard Batcher odd-even merge properly.
    
    m = len(left)
    # left and right are the wire indices for the two halves
    # left = list of wire indices for left half (sorted)
    # right = list of wire indices for right half (sorted)
    
    # Odd-even merge:
    # Step 1: For k = m-1 down to 1:
    #   For i = 0 to m-k-1:
    #     If i is even: compare left[i] with right[i+k]  (i.e., the element at position i in left vs position i+k in right)
    #     If i is odd: compare left[i+k] with right[i]
    # Wait, I need to be more careful. Let me use the standard formulation.
    
    # Actually, the standard Batcher odd-even merge works on two sorted sequences of length m.
    # The merge is done as follows:
    # for k in range(m-1, 0, -1):
    #   for i in range(m - k):
    #     if i % 2 == 0:
    #       compare_exchange(left[i], right[i + k])
    #     else:
    #       compare_exchange(left[i + k], right[i])
    
    for k in range(m - 1, 0, -1):
        for i in range(m - k):
            if i % 2 == 0:
                a = left[i]
                b = right[i + k]
            else:
                a = left[i + k]
                b = right[i]
            if a > b:
                a, b = b, a
            comparators.append((a, b))
    
    return comparators


def _batcher_merge_sort(wires):
    """Build a Batcher merge-sort network for the given wire indices.
       wires is a list of wire indices to sort."""
    n = len(wires)
    if n <= 1:
        return []
    if n == 2:
        a, b = wires[0], wires[1]
        if a > b:
            a, b = b, a
        return [(a, b)]
    
    mid = n // 2
    left_wires = wires[:mid]
    right_wires = wires[mid:]
    
    left_comps = _batcher_merge_sort(left_wires)
    right_comps = _batcher_merge_sort(right_wires)
    
    merge_comps = _batcher_odd_merge(left_wires, right_wires)
    
    return left_comps + right_comps + merge_comps


def _verify_network(comparators, sample_size=None):
    """Verify that the network sorts all 2^N Boolean inputs.
       Returns True if correct, False otherwise."""
    if sample_size is not None:
        # Quick check with random samples
        for _ in range(sample_size):
            inp = np.random.randint(0, 2, N)
            arr = inp.copy()
            for (i, j) in comparators:
                if arr[i] > arr[j]:
                    arr[i], arr[j] = arr[j], arr[i]
            if not np.all(np.diff(arr) >= 0):
                return False
        return True
    else:
        # Full verification over all 2^N inputs
        for mask in range(1 << N):
            arr = np.array([(mask >> i) & 1 for i in range(N)], dtype=np.int8)
            for (i, j) in comparators:
                if arr[i] > arr[j]:
                    arr[i], arr[j] = arr[j], arr[i]
            if not np.all(np.diff(arr) >= 0):
                return False
        return True


def _verify_network_fast(comparators):
    """Faster verification using vectorized numpy operations."""
    num_inputs = 1 << N
    # Process in batches to avoid memory issues
    batch_size = min(1024, num_inputs)
    for start in range(0, num_inputs, batch_size):
        end = min(start + batch_size, num_inputs)
        batch = np.array([(m >> i) & 1 for m in range(start, end) for i in range(N)], dtype=np.int8).reshape(-1, N)
        for (i, j) in comparators:
            # Where batch[:, i] > batch[:, j], swap
            swap_mask = batch[:, i] > batch[:, j]
            batch[swap_mask, i] = 0
            batch[swap_mask, j] = 1
        # Check if each row is sorted
        diffs = np.diff(batch, axis=1)
        if np.any(diffs < 0):
            return False
    return True


def _local_search(comparators, max_iter=500, deadline=None):
    """Local search to reduce comparator count.
       Tries removing comparators and swapping adjacent ones.
       Returns improved list of comparators."""
    comps = list(comparators)
    current_len = len(comps)
    best_len = current_len
    best_comps = comps[:]
    
    random.seed(42)
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        if len(comps) >= best_len:
            # Try removing a random comparator
            idx = random.randint(0, len(comps) - 1)
            trial = comps[:idx] + comps[idx+1:]
            if _verify_network_fast(trial):
                comps = trial
                best_len = len(comps)
                best_comps = comps[:]
        else:
            # Try swapping two random comparators
            i = random.randint(0, len(comps) - 1)
            j = random.randint(0, len(comps) - 1)
            if i != j:
                trial = comps[:]
                trial[i], trial[j] = trial[j], trial[i]
                if _verify_network_fast(trial):
                    comps = trial
                    best_len = len(comps)
                    best_comps = comps[:]
        
        # Also try moving a comparator to a different position
        if random.random() < 0.3:
            i = random.randint(0, len(comps) - 1)
            j = random.randint(0, len(comps) - 1)
            if i != j:
                trial = comps[:]
                c = trial.pop(i)
                trial.insert(j, c)
                if _verify_network_fast(trial):
                    comps = trial
                    best_len = len(comps)
                    best_comps = comps[:]
    
    return best_comps


def construct():
    start_time = time.time()
    deadline = start_time + 90 * DEADLINE_FRACTION
    
    # Start with Batcher merge-sort network
    wires = list(range(N))
    comps = _batcher_merge_sort(wires)
    
    # Verify the base network
    if not _verify_network_fast(comps):
        # Fallback to odd-even transposition
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    best_comps = comps[:]
    best_len = len(comps)
    
    # Local search to reduce comparators
    while time.time() < deadline:
        improved = _local_search(best_comps, max_iter=100, deadline=deadline)
        if len(improved) < best_len:
            best_comps = improved
            best_len = len(improved)
        else:
            # Try a different random seed for more exploration
            random.seed(int(time.time() * 1000) % 100000)
            # Try removing random comparators more aggressively
            trial = list(best_comps)
            for _ in range(10):
                if time.time() > deadline:
                    break
                if not trial:
                    break
                idx = random.randint(0, len(trial) - 1)
                candidate = trial[:idx] + trial[idx+1:]
                if _verify_network_fast(candidate):
                    trial = candidate
            if len(trial) < best_len:
                best_comps = trial
                best_len = len(trial)
            else:
                # Try re-inserting removed comparators in different positions
                if time.time() > deadline:
                    break
    
    return best_comps

# EVOLVE-BLOCK-END
