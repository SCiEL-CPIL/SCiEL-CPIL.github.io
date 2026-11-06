import time
import numpy as np

N = 16
MAX_COMPARATORS = 4096
TIME_LIMIT = 85

def _batcher_oes(n):
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    k = n // 2
    left = _batcher_oes(k)
    right = _batcher_oes(k)
    right_shifted = [(i + k, j + k) for i, j in right]
    
    merge = _batcher_merge(n)
    
    return left + right_shifted + merge

def _batcher_merge(n):
    if n == 2:
        return [(0, 1)]
    
    k = n // 2
    result = []
    depth = int(np.log2(n))
    for d in range(1, depth + 1):
        step = 2 ** (depth - d)
        for i in range(0, n, 2 * step):
            for j in range(step):
                a = i + j
                b = i + j + step
                if a < b:
                    result.append((a, b))
    return result

def _verify_network_fast(network):
    """Verify using NumPy for speed."""
    # Generate all 2^N binary inputs
    # Shape: (2^N, N)
    # Using bit manipulation to create the matrix is faster than string formatting
    # But string formatting is fine for 65536 rows.
    inputs = np.array([list(map(int, f'{i:0{N}b}')) for i in range(1 << N)])
    
    for i, j in network:
        min_val = np.minimum(inputs[:, i], inputs[:, j])
        max_val = np.maximum(inputs[:, i], inputs[:, j])
        inputs[:, i] = min_val
        inputs[:, j] = max_val
        
    # Check if all rows are sorted
    diffs = np.diff(inputs, axis=1)
    return np.all(diffs >= 0)

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # 1. Generate initial network
    network = _batcher_oes(N)
    
    # 2. Verify initial network
    if not _verify_network_fast(network):
        # Fallback to odd-even transposition if Batcher is broken
        network = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
        
    best_network = network[:]
    best_count = len(network)
    
    # 3. Greedy removal of redundant comparators (single)
    while time.time() < deadline:
        improved = False
        for idx in range(len(best_network) - 1, -1, -1):
            if time.time() > deadline:
                break
            
            temp_network = best_network[:idx] + best_network[idx+1:]
            
            if _verify_network_fast(temp_network):
                best_network = temp_network
                best_count = len(best_network)
                improved = True
                break
        
        if not improved:
            break
            
    # 4. Try to remove pairs if we are still above 60 and have time
    if len(best_network) > 60 and time.time() < deadline:
        while time.time() < deadline:
            improved = False
            n = len(best_network)
            for i in range(n):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n):
                    temp_network = best_network[:i] + best_network[i+1:j] + best_network[j+1:]
                    if _verify_network_fast(temp_network):
                        best_network = temp_network
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break

    # 5. Final verification
    if not _verify_network_fast(best_network):
        best_network = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
        
    return best_network
