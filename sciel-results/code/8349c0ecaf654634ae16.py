"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 18

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin before 60s kill
    
    # Precompute all 2^N Boolean inputs as a (N, 2^N) uint8 matrix
    total = 1 << N
    all_inputs = np.zeros((N, total), dtype=np.uint8)
    for i in range(N):
        period = 1 << (i + 1)
        half = 1 << i
        pattern = np.zeros(period, dtype=np.uint8)
        pattern[half:] = 1
        all_inputs[i] = np.tile(pattern, total // period)
    
    def _verify(network):
        """Vectorized verification: returns True if network sorts all inputs."""
        x = all_inputs.copy()
        for i, j in network:
            mask = (x[i] > x[j])
            if np.any(mask):
                x[i][mask] = 0
                x[j][mask] = 1
        # Check sorted: no adjacent (1,0) pair
        for i in range(N - 1):
            if np.any((x[i] == 1) & (x[i+1] == 0)):
                return False
        return True
    
    def _verify_fast(network, cached_x=None):
        """Faster verification using cached state if available."""
        if cached_x is not None:
            x = cached_x.copy()
        else:
            x = all_inputs.copy()
        for i, j in network:
            mask = (x[i] > x[j])
            if np.any(mask):
                x[i][mask] = 0
                x[j][mask] = 1
        for i in range(N - 1):
            if np.any((x[i] == 1) & (x[i+1] == 0)):
                return False, x
        return True, x
    
    # Start with odd-even transposition
    network = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    # Verify initial network works
    assert _verify(network), "Initial network is invalid"
    
    # Phase 1: Greedy removal - repeatedly try removing comparators
    # Try multiple passes to find all removable comparators
    max_passes = 100
    for _pass in range(max_passes):
        if (time.time() - start_time) > deadline - 3:
            break
        
        improved = False
        # Try removing from end to start (later comparators often less critical)
        for idx in range(len(network) - 1, -1, -1):
            if (time.time() - start_time) > deadline - 3:
                break
            
            test_network = network[:idx] + network[idx+1:]
            if _verify(test_network):
                network = test_network
                improved = True
        
        if not improved:
            break
    
    # Phase 2: Hill-climbing with add-then-remove moves
    # This allows escaping local minima by temporarily adding a comparator
    # to enable removal of multiple others
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    rounds = 30
    for _round in range(rounds):
        if (time.time() - start_time) > deadline - 5:
            break
        
        found_improvement = False
        existing_set = set(network)
        
        for add_pair in all_pairs:
            if add_pair in existing_set:
                continue
            if (time.time() - start_time) > deadline - 5:
                break
            
            # Add the new comparator
            extended = network + [add_pair]
            
            # Try removing each existing comparator (not the one just added)
            for idx in range(len(extended) - 1, -1, -1):
                if (time.time() - start_time) > deadline - 5:
                    break
                
                removed = extended[:idx] + extended[idx+1:]
                if len(removed) < len(network):
                    if _verify(removed):
                        network = removed
                        found_improvement = True
                        break
            
            if found_improvement:
                break
        
        if not found_improvement:
            break
    
    # Phase 3: More aggressive local search - try swapping comparator order
    # or trying different removal orders
    if (time.time() - start_time) < deadline - 10:
        # Try removing pairs of comparators simultaneously
        for _pair_pass in range(20):
            if (time.time() - start_time) > deadline - 10:
                break
            
            pair_improved = False
            n = len(network)
            for idx1 in range(n - 1, -1, -1):
                if (time.time() - start_time) > deadline - 10:
                    break
                for idx2 in range(idx1 - 1, -1, -1):
                    if (time.time() - start_time) > deadline - 10:
                        break
                    
                    # Remove both idx1 and idx2 (idx2 first since it's earlier)
                    if idx2 < idx1:
                        test = network[:idx2] + network[idx2+1:idx1] + network[idx1+1:]
                    else:
                        test = network[:idx1] + network[idx1+1:idx2] + network[idx2+1:]
                    
                    if len(test) == n - 2 and _verify(test):
                        network = test
                        pair_improved = True
                        break
                
                if pair_improved:
                    break
            
            if not pair_improved:
                break
    
    return network
# EVOLVE-BLOCK-END
