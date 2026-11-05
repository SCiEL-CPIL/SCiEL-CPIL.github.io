import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 112.0  # safety margin before 120s limit (and 150s kill)
    
    # Build the direct-sum construction
    # Tetracode is a perfect 1-error-correcting code of length 4 over GF(3)
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    
    # Construct initial code: direct sum with all tails
    code = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            code.append(tuple(h) + tuple(tail))
            
    m = len(code)  # 81 for q=3, n=6
    total_words = q ** n
    
    # Precompute powers for fast index calculation
    q_powers = [q ** (n - 1 - i) for i in range(n)]
    
    def word_to_index(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    # Precompute coverage sets for each codeword
    covers = []
    for c in code:
        c_set = set()
        c_idx = word_to_index(c)
        c_set.add(c_idx)
        c_list = list(c)
        for pos in range(n):
            orig_val = c_list[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                # Fast index update
                new_idx = c_idx + (new_val - orig_val) * q_powers[pos]
                c_set.add(new_idx)
        covers.append(c_set)
        
    # Precompute coverage counts for the full initial set
    coverage_count = [0] * total_words
    for ci in range(m):
        for wi in covers[ci]:
            coverage_count[wi] += 1

    best_code = code[:]
    best_count = m
    
    # Phase 1: Greedy removal with randomization
    # We perform multiple restarts. In each restart, we try to remove redundant codewords.
    max_restarts_phase1 = 1000
    for restart in range(max_restarts_phase1):
        if time.time() > deadline - 30.0:
            break
            
        # Reset coverage counts
        curr_coverage = coverage_count[:]
        active = [True] * m
        
        # Random order for removal attempts
        order = list(range(m))
        random.shuffle(order)
        
        for ci in order:
            if not active[ci]:
                continue
            
            # Check if ci is redundant
            is_redundant = True
            for wi in covers[ci]:
                if curr_coverage[wi] <= 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                # Remove ci
                active[ci] = False
                for wi in covers[ci]:
                    curr_coverage[wi] -= 1
        
        current_code = [code[i] for i in range(m) if active[i]]
        if len(current_code) < best_count:
            best_count = len(current_code)
            best_code = current_code[:]
            
    # Phase 2: Swap-based local search
    # Start from the best solution found in Phase 1
    if best_count < m:
        # Initialize active set from best_code
        best_set = set(best_code)
        active = [c in best_set for c in code]
        
        # Rebuild coverage counts for current active set
        curr_coverage = [0] * total_words
        for ci in range(m):
            if active[ci]:
                for wi in covers[ci]:
                    curr_coverage[wi] += 1
        
        # Precompute active/inactive lists for faster sampling
        # We will update these lists as we go
        
        while time.time() < deadline - 2.0:
            # Build lists of active and inactive indices
            # Optimization: instead of rebuilding every time, we can maintain them,
            # but for simplicity and correctness with swaps, rebuilding is safer.
            # With m=81, this is cheap.
            active_indices = [i for i in range(m) if active[i]]
            inactive_indices = [i for i in range(m) if not active[i]]
            
            if not inactive_indices:
                break
                
            # Pick a random active codeword to remove
            ci = random.choice(active_indices)
            
            # Pick a random inactive codeword to add
            cj = random.choice(inactive_indices)
            
            # Evaluate the swap
            # We need to check if removing ci leaves any word uncovered.
            # A word is uncovered if it was only covered by ci (coverage == 1) 
            # AND it is NOT covered by cj.
            
            valid_swap = True
            # Only need to check words in covers[ci]
            for wi in covers[ci]:
                if curr_coverage[wi] == 1:
                    # This word is uniquely covered by ci
                    # If cj does not cover it, the swap is invalid
                    if wi not in covers[cj]:
                        valid_swap = False
                        break
            
            if valid_swap:
                # Perform the swap
                active[ci] = False
                active[cj] = True
                
                # Update coverage counts
                # Words in covers[ci] lose 1
                # Words in covers[cj] gain 1
                # Intersection is unchanged in net count, but we do it sequentially for simplicity
                # Actually, we must be careful.
                # new_cov[wi] = old_cov[wi] - 1 (if wi in covers[ci]) + 1 (if wi in covers[cj])
                
                for wi in covers[ci]:
                    curr_coverage[wi] -= 1
                for wi in covers[cj]:
                    curr_coverage[wi] += 1
                
                # After a valid swap, the code size is the same.
                # However, the swap might make other codewords redundant.
                # We should try to remove redundant codewords immediately.
                
                # Greedy removal pass
                # We iterate through all active codewords and remove any that are redundant.
                # We repeat this until no more can be removed.
                
                changed = True
                while changed:
                    changed = False
                    # Shuffle order to avoid bias
                    current_active = [i for i in range(m) if active[i]]
                    random.shuffle(current_active)
                    
                    for idx in current_active:
                        if not active[idx]:
                            continue
                        
                        is_redundant = True
                        for wi in covers[idx]:
                            if curr_coverage[wi] <= 1:
                                is_redundant = False
                                break
                        
                        if is_redundant:
                            active[idx] = False
                            for wi in covers[idx]:
                                curr_coverage[wi] -= 1
                            changed = True
                            
                # Check if we improved
                current_count = sum(1 for i in range(m) if active[i])
                if current_count < best_count:
                    best_count = current_count
                    best_code = [code[i] for i in range(m) if active[i]]
    
    # Final verification (optional, but good for safety)
    # The task says "Valid when every word... is within Hamming distance radius".
    # Our greedy removal ensures this invariant is maintained.
    # So best_code should be valid.
    
    return [list(c) for c in best_code]
