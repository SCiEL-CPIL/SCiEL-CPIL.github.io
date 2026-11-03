import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Pre-compute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    
    best_blocks = None
    best_size = float('inf')
    
    run_id = 0
    
    while time.time() < deadline:
        run_id += 1
        random.seed(run_id)
        
        uncovered = set(all_t_subsets)
        blocks = []
        
        while uncovered:
            # Pick a random uncovered subset
            # list(uncovered) is O(N), but N is small (<= 11628)
            S = random.choice(list(uncovered))
            P = list(S)
            
            while len(P) < k:
                best_x = -1
                best_gain = -1
                candidates = []
                
                for x in range(v):
                    if x in P:
                        continue
                    
                    # Calculate gain
                    g = 0
                    # We need to count how many T in uncovered are of the form (rest + x)
                    # where rest is a (t-1)-subset of P
                    for rest in itertools.combinations(P, t-1):
                        T = tuple(sorted(rest + (x,)))
                        if T in uncovered:
                            g += 1
                    
                    if g > best_gain:
                        best_gain = g
                        candidates = [x]
                    elif g == best_gain:
                        candidates.append(x)
                
                if not candidates:
                    # Should not happen if uncovered is not empty
                    break
                P.append(random.choice(candidates))
            
            B = tuple(sorted(P))
            blocks.append(B)
            
            # Remove covered t-subsets
            for T in itertools.combinations(B, t):
                if T in uncovered:
                    uncovered.discard(T)
        
        # Post-processing: Remove redundant blocks
        # This is a simple greedy removal
        # We iterate and check if a block is redundant
        # To speed up, we can just do it once or twice
        # Given the time, a single pass is good.
        
        # Optimization: Check redundancy
        # For each block, if all its t-subsets are covered by other blocks, remove it.
        # We do this in a loop until no more can be removed, or time is tight.
        
        final_blocks = []
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            for i in range(len(blocks)):
                b = blocks[i]
                is_redundant = True
                for T in itertools.combinations(b, t):
                    T_set = set(T)
                    covered = False
                    for j in range(len(blocks)):
                        if i == j:
                            continue
                        if T_set.issubset(blocks[j]):
                            covered = True
                            break
                    if not covered:
                        is_redundant = False
                        break
                if is_redundant:
                    blocks.pop(i)
                    changed = True
                    break # Restart loop as indices changed
        
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = [list(b) for b in blocks]
            
            if best_size <= 264: # Lower bound
                break

    if best_blocks is None:
        # Fallback if no runs completed (should not happen)
        # Just return a single block? No, must be valid.
        # If we didn't finish even one run, we have a problem.
        # But one run takes ~0.1s. So we will finish many.
        pass
        
    return best_blocks
