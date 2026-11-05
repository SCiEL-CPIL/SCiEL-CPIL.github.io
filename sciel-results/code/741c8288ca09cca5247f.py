"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 17
MAX_COMPARATORS = 4096

def _verify(network):
    """Verify sorting network over all 2^N Boolean inputs using zero-one principle."""
    total = 1 << N
    # Precompute all 2^N boolean inputs once
    # Use bit patterns: input x has bit i set if (x >> i) & 1
    inputs = np.empty((total, N), dtype=np.uint8)
    for i in range(N):
        inputs[:, i] = (np.arange(total, dtype=np.uint32) >> i) & 1
    
    for (i, j) in network:
        a = inputs[:, i].copy()
        b = inputs[:, j].copy()
        inputs[:, i] = np.minimum(a, b)
        inputs[:, j] = np.maximum(a, b)
    
    # Check sorted: no 1 before 0 in any row
    for k in range(N - 1):
        if np.any(inputs[:, k] > inputs[:, k + 1]):
            return False
    return True

def _odd_even_transposition(n):
    """Generate odd-even transposition sorting network for n wires."""
    comps = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comps.append((i, i + 1))
    return comps

def _try_remove(network, idx):
    """Try removing comparator at index idx, return (success, new_network)."""
    candidate = network[:idx] + network[idx+1:]
    if _verify(candidate):
        return True, candidate
    return False, None

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 85.0  # Leave 5s safety margin before 90s kill
    
    # Start with odd-even transposition network
    network = _odd_even_transposition(N)
    
    # Verify it's correct
    if not _verify(network):
        return network
    
    best_network = list(network)
    best_count = len(network)
    
    # Phase 1: Greedy removal - repeatedly remove redundant comparators
    improved = True
    while improved:
        improved = False
        for idx in range(len(network) - 1, -1, -1):
            if time.time() > deadline:
                return best_network
            
            success, candidate = _try_remove(network, idx)
            if success:
                network = candidate
                improved = True
                best_network = list(network)
                best_count = len(network)
    
    # Phase 2: Local search - try swapping adjacent comparators to enable removals
    if time.time() < deadline:
        for _pass in range(20):
            if time.time() > deadline:
                break
            any_improvement = False
            for idx in range(len(network) - 1):
                if time.time() > deadline:
                    break
                # Try swapping
                candidate = network[:idx] + [network[idx+1], network[idx]] + network[idx+2:]
                # After swap, try removing each of the two
                for rem in [idx, idx+1]:
                    cand2 = candidate[:rem] + candidate[rem+1:]
                    if _verify(cand2):
                        network = cand2
                        best_network = list(network)
                        best_count = len(network)
                        any_improvement = True
                        break
                if time.time() > deadline:
                    break
            if not any_improvement:
                break
    
    # Phase 3: Try inserting a comparator to enable removal of two others
    if time.time() < deadline:
        for ins_pos in range(len(network) + 1):
            if time.time() > deadline:
                break
            for i in range(N):
                if time.time() > deadline:
                    break
                for j in range(i + 1, N):
                    if time.time() > deadline:
                        break
                    new_comp = (i, j)
                    candidate = network[:ins_pos] + [new_comp] + network[ins_pos:]
                    # Try to remove two comparators
                    for rem1 in range(len(candidate) - 1):
                        if time.time() > deadline:
                            break
                        cand1 = candidate[:rem1] + candidate[rem1+1:]
                        if len(cand1) >= len(network):
                            continue
                        for rem2 in range(len(cand1) - 1):
                            if time.time() > deadline:
                                break
                            cand2 = cand1[:rem2] + cand1[rem2+1:]
                            if len(cand2) < len(network) and _verify(cand2):
                                network = cand2
                                best_network = list(network)
                                best_count = len(network)
                                break
                        if time.time() > deadline:
                            break
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
    
    # Phase 4: More aggressive local search with random restarts
    if time.time() < deadline:
        rng = np.random.default_rng(12345)
        for _restart in range(50):
            if time.time() > deadline:
                break
            
            # Randomly perturb the network
            perturbed = list(network)
            if len(perturbed) > 2:
                # Swap two random comparators
                idx1 = rng.integers(0, len(perturbed) - 1)
                idx2 = rng.integers(0, len(perturbed) - 1)
                perturbed[idx1], perturbed[idx2] = perturbed[idx2], perturbed[idx1]
            
            # Try to improve from this perturbed state
            improved_local = True
            while improved_local and time.time() < deadline:
                improved_local = False
                for idx in range(len(perturbed) - 1, -1, -1):
                    if time.time() > deadline:
                        break
                    candidate = perturbed[:idx] + perturbed[idx+1:]
                    if _verify(candidate):
                        perturbed = candidate
                        improved_local = True
                        if len(perturbed) < best_count:
                            best_network = list(perturbed)
                            best_count = len(perturbed)
                            network = perturbed
    
    return best_network
# EVOLVE-BLOCK-END
