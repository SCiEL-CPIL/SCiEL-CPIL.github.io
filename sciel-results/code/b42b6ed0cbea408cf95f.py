"""Find the largest subset of [1,255] containing 255 with pairwise gcd > 1."""
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin before 90s kill
    
    # Simple baseline: all multiples of 3 (largest prime-factor clique)
    baseline = list(range(3, 256, 3))
    best_result = baseline
    best_size = len(baseline)  # 85
    
    # Build candidate set: numbers in [1,255] sharing a factor with 255
    candidates = [x for x in range(1, 256) if math.gcd(x, 255) > 1]
    n = len(candidates)
    idx_255 = candidates.index(255)
    
    # Build adjacency using bitmasks
    adj = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Branch and bound for maximum clique containing idx_255
    best_clique = [idx_255]
    
    def expand_clique(current_clique, available_mask):
        """Recursively expand clique, updating best_clique."""
        nonlocal best_clique, best_size, deadline
        
        if time.time() > deadline:
            return
        
        # Pruning upper bound
        if len(current_clique) + bin(available_mask).count('1') <= best_size:
            return
        
        # Get available nodes sorted by degree within available set
        nodes = []
        mask = available_mask
        while mask:
            lsb = mask & (-mask)
            i = lsb.bit_length() - 1
            nodes.append(i)
            mask ^= lsb
        nodes.sort(key=lambda i: bin(adj[i] & available_mask).count('1'), reverse=True)
        
        for i in nodes:
            if time.time() > deadline:
                return
            # Check compatibility with all in current_clique
            compatible = True
            for j in current_clique:
                if not ((adj[i] >> j) & 1):
                    compatible = False
                    break
            if not compatible:
                continue
            
            new_clique = current_clique + [i]
            if len(new_clique) > best_size:
                best_size = len(new_clique)
                best_clique = new_clique[:]
            
            # Recurse with reduced available set
            new_available = available_mask & adj[i]
            expand_clique(new_clique, new_available)
        
        # After trying to add each node, we should also try without adding any more
        # This is handled by the recursion structure
    
    # Start the search
    initial_available = adj[idx_255]
    expand_clique([idx_255], initial_available)
    
    if best_size > len(baseline):
        result = sorted([candidates[i] for i in best_clique])
        return result
    else:
        return baseline
