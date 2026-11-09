import numpy as np
import time
import random

N = 18

def construct():
    # Initial odd-even transposition network
    network = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    # Generate all boolean inputs
    inputs = np.zeros((1 << N, N), dtype=np.uint8)
    for i in range(N):
        inputs[:, i] = (np.arange(1 << N) >> i) & 1
    
    # Precompute sorted mask for faster checking
    # A row is sorted if state[:, i] <= state[:, i+1] for all i
    
    def verify(net):
        state = inputs.copy()
        for i, j in net:
            if i == j:
                continue
            min_val = np.minimum(state[:, i], state[:, j])
            max_val = np.maximum(state[:, i], state[:, j])
            state[:, i] = min_val
            state[:, j] = max_val
        # Check if all rows are sorted
        # Compare adjacent elements
        for i in range(N - 1):
            if not np.all(state[:, i] <= state[:, i + 1]):
                return False
        return True
    
    start_time = time.time()
    deadline = 75.0  # Leave 15s margin for 90s limit
    
    best_network = list(network)
    best_len = len(network)
    
    # Phase 1: Systematic deletion pass - try removing each comparator
    # Go left to right, keep if removal breaks sorting
    current = list(network)
    
    # First, try a full left-to-right deletion sweep
    for sweep in range(5):  # Multiple sweeps
        if time.time() - start_time > deadline:
            break
        changed = False
        i = 0
        while i < len(current):
            if time.time() - start_time > deadline:
                break
            candidate = current[:i] + current[i+1:]
            if verify(candidate):
                current = candidate
                changed = True
                # Don't increment i, check same position again
            else:
                i += 1
        if not changed:
            break
    
    if len(current) < best_len:
        best_network = list(current)
        best_len = len(current)
    
    # Phase 2: Random local search with deletion and swaps
    random.seed(42)
    current = list(best_network)
    
    while time.time() - start_time < deadline:
        # Try deleting a random comparator
        if len(current) > 0:
            idx = random.randint(0, len(current) - 1)
            candidate = current[:idx] + current[idx+1:]
            if verify(candidate):
                current = candidate
                if len(current) < best_len:
                    best_network = list(current)
                    best_len = len(current)
        
        # Try swapping adjacent comparators (if they don't share wires)
        if len(current) > 1:
            idx = random.randint(0, len(current) - 2)
            a, b = current[idx], current[idx + 1]
            # Only swap if they don't conflict (share no wires)
            if len(set([a[0], a[1]]) & set([b[0], b[1]])) == 0:
                candidate = current[:idx] + [b, a] + current[idx+2:]
                if verify(candidate):
                    current = candidate
        
        # Try moving a comparator to a random position
        if len(current) > 2 and random.random() < 0.3:
            src_idx = random.randint(0, len(current) - 1)
            dst_idx = random.randint(0, len(current) - 1)
            if src_idx != dst_idx:
                comp = current[src_idx]
                # Check if moving doesn't create immediate conflict
                candidate = current[:src_idx] + current[src_idx+1:]
                if dst_idx > src_idx:
                    dst_idx -= 1
                candidate = candidate[:dst_idx] + [comp] + candidate[dst_idx:]
                if verify(candidate):
                    current = candidate
                    if len(current) < best_len:
                        best_network = list(current)
                        best_len = len(current)
    
    # Final sweep
    current = list(best_network)
    for sweep in range(3):
        if time.time() - start_time > deadline - 5:
            break
        changed = False
        i = 0
        while i < len(current):
            if time.time() - start_time > deadline - 5:
                break
            candidate = current[:i] + current[i+1:]
            if verify(candidate):
                current = candidate
                changed = True
            else:
                i += 1
        if not changed:
            break
    
    if len(current) < best_len:
        best_network = current
        best_len = len(current)
    
    return best_network

