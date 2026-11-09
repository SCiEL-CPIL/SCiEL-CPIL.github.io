import numpy as np
import time
import random

N = 17
DEADLINE = 50.0 # 50 seconds

def generate_inputs(N):
    n_inputs = 2**N
    inputs = np.zeros((n_inputs, N), dtype=np.uint8)
    for i in range(N):
        inputs[:, i] = (np.arange(n_inputs) >> i) & 1
    return inputs

INPUTS = generate_inputs(N)

def is_valid(comparators):
    # Copy inputs to avoid modification
    state = INPUTS.copy()
    for i, j in comparators:
        a = state[:, i]
        b = state[:, j]
        state[:, i] = np.minimum(a, b)
        state[:, j] = np.maximum(a, b)
    # Check if sorted
    # For each row, check if diff >= 0
    diffs = np.diff(state, axis=1)
    return np.all(diffs >= 0)

def batcher_odd_merge(n):
    if n == 1:
        return []
    k = n // 2
    left = batcher_odd_merge(k)
    right = batcher_odd_merge(k)
    # Shift right by 1
    right_shifted = [(i+1, j+1) for i, j in right]
    
    merge = []
    # Merge step
    # Batcher's merge of two sorted lists of size k
    # Recursive merge
    def merge_network(n, offset1, offset2):
        if n == 1:
            return [(offset1, offset2)]
        half = n // 2
        # Merge first halves
        m1 = merge_network(half, offset1, offset2)
        # Merge second halves
        m2 = merge_network(half, offset1 + half, offset2 + half)
        return m1 + m2

    # The merge of two lists of size k
    # Batcher's odd-even merge:
    # 1. Merge first halves
    # 2. Merge second halves
    # 3. Merge the "middle" parts?
    # Actually, standard Batcher Odd-Merge:
    # 1. Sort first half, sort second half
    # 2. Merge them using a specific pattern
    
    # Let's use a known correct Batcher implementation
    def _batcher(n):
        if n == 1: return []
        if n == 2: return [(0, 1)]
        if n % 2 == 0:
            k = n // 2
            left = _batcher(k)
            right = _batcher(k)
            right = [(i+k, j+k) for i, j in right]
            # Merge
            merge = []
            for i in range(k):
                merge.append((i, i + k))
            # Then sort the result? No, Batcher's merge is more complex.
            # Let's use Bitonic for simplicity and reliability, even if slightly larger.
            return left + right + merge # This is NOT a correct sort for all inputs, just a merge of two sorted lists?
            # Actually, if left and right are sorted, comparing (0, k), (1, k+1)... is NOT sufficient.
            pass
    # Let's just use Bitonic Sort for 16.
    def bitonic_sort(n):
        if n == 1: return []
        if n == 2: return [(0, 1)]
        k = n // 2
        left = bitonic_sort(k)
        right = bitonic_sort(k)
        right = [(i+k, j+k) for i, j in right]
        # Bitonic merge
        merge = []
        # Standard bitonic merge of two sorted lists of size k
        # This part is tricky to implement recursively correctly in a few lines.
        
    # Given the complexity, let's use the Odd-Even Transposition as a VERY safe baseline,
    # but 136 is high.
    # Let's use the "Ladder" merge on top of a simple sort.
    # How about just using the 136 comparator network and pruning?
    # 136 -> Pruning might get it down to ~80-90.
    # But 74 is a better start.
    
    # I will implement a known small network for 16:
    # Batcher's Odd-Merge for 16 is 58.
    # I will hardcode the structure or use a reliable library function if available? No, std lib only.
    # I will implement Batcher's Odd-Merge carefully.
    
    pass

def construct():
    start_time = time.time()
    
    # 1. Baseline: Odd-Even Transposition (136 comparators)
    # This is guaranteed to be valid.
    baseline = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    best = baseline
    best_count = len(baseline)
    
    # 2. Greedy Pruning
    def greedy_prune(comps):
        comps = list(comps)
        improved = True
        while improved:
            improved = False
            for i in range(len(comps)):
                # Try removing comps[i]
                temp = comps[:i] + comps[i+1:]
                if is_valid(temp):
                    comps = temp
                    improved = True
                    break
        return comps

    best = greedy_prune(best)
    best_count = len(best)
    
    # 3. Local Search
    while time.time() - start_time < DEADLINE:
        # Perturb best
        current = best.copy()
        
        # Random moves
        for _ in range(10):
            move = random.choice(['del', 'ins', 'swap', 'mod'])
            if move == 'del' and len(current) > 0:
                idx = random.randint(0, len(current)-1)
                current.pop(idx)
            elif move == 'ins':
                i = random.randint(0, N-2)
                j = random.randint(i+1, N-1)
                current.append((i, j))
            elif move == 'swap' and len(current) > 1:
                i = random.randint(0, len(current)-1)
                j = random.randint(0, len(current)-1)
                current[i], current[j] = current[j], current[i]
            elif move == 'mod' and len(current) > 0:
                idx = random.randint(0, len(current)-1)
                i, j = current[idx]
                if random.random() < 0.5:
                    new_i = random.randint(0, N-1)
                    if new_i != j:
                        if new_i > j:
                            new_i, j = j, new_i
                        current[idx] = (new_i, j)
                else:
                    new_j = random.randint(0, N-1)
                    if new_j != i:
                        if new_j < i:
                            i, new_j = new_j, i
                        current[idx] = (i, new_j)
        
        # Prune the perturbed network
        pruned = greedy_prune(current)
        
        if len(pruned) < best_count:
            best = pruned
            best_count = len(best)
            
        # Check time
        if time.time() - start_time > DEADLINE - 5:
            break
            
    # Final validation: ensure all comparators are valid (i < j) and list is not empty
    # If validation fails, fallback to the known-good baseline
    valid_best = all(0 <= i < j < N for i, j in best)
    if not valid_best or len(best) == 0:
        return baseline
    
    return best
