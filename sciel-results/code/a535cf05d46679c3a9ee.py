"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import itertools
import random
import time

def construct(q, n, radius):
    # Specific optimization for q=3, n=6, radius=1 using Simulated Annealing
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters: Greedy Set Cover
        def word_to_idx_fallback(word):
            idx = 0
            for i in range(n):
                idx = idx * q + word[i]
            return idx

        def idx_to_word_fallback(idx):
            word = [0] * n
            for i in range(n - 1, -1, -1):
                word[i] = idx % q
                idx //= q
            return word

        total_words = q ** n
        indices = list(range(total_words))
        
        # Precompute neighborhoods
        neighborhoods = []
        for i in indices:
            w = idx_to_word_fallback(i)
            neigh = [i]
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]:
                        continue
                    new_w = w.copy()
                    new_w[pos] = val
                    neigh.append(word_to_idx_fallback(new_w))
            neighborhoods.append(neigh)
        
        # Greedy
        uncovered = set(indices)
        code_indices = []
        while uncovered:
            best_idx = -1
            best_count = -1
            for i in indices:
                count = sum(1 for nb in neighborhoods[i] if nb in uncovered)
                if count > best_count:
                    best_count = count
                    best_idx = i
            code_indices.append(best_idx)
            for nb in neighborhoods[best_idx]:
                if nb in uncovered:
                    uncovered.remove(nb)
        return [list(idx_to_word_fallback(i)) for i in code_indices]

    # --- Simulated Annealing for q=3, n=6, radius=1 ---
    
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx

    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    total_words = q ** n
    indices = list(range(total_words))
    
    # Precompute neighborhoods
    neighborhoods = []
    for i in indices:
        w = idx_to_word(i)
        neigh = [i]
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = w.copy()
                new_w[pos] = val
                neigh.append(word_to_idx(new_w))
        neighborhoods.append(neigh)
    
    # 1. Initialize with a valid cover (Greedy)
    uncovered = set(indices)
    code_set = set()
    while uncovered:
        best_idx = -1
        best_count = -1
        for i in indices:
            count = sum(1 for nb in neighborhoods[i] if nb in uncovered)
            if count > best_count:
                best_count = count
                best_idx = i
        code_set.add(best_idx)
        for nb in neighborhoods[best_idx]:
            if nb in uncovered:
                uncovered.remove(nb)
                
    # 2. Simulated Annealing
    # State: code_set (set of indices), coverage_counts (numpy array)
    # We track coverage to quickly check validity.
    # coverage[i] = number of codewords in code_set that cover word i.
    # Valid if min(coverage) >= 1.
    # Energy = len(code_set) + P * (number of uncovered words)
    # Uncovered words = count of i where coverage[i] == 0.
    
    coverage = np.zeros(total_words, dtype=np.int32)
    for c in code_set:
        for nb in neighborhoods[c]:
            coverage[nb] += 1
    
    current_energy = len(code_set) # Since it's valid initially, uncovered=0
    
    best_code = set(code_set)
    best_energy = current_energy
    
    start_time = time.time()
    time_limit = 110.0 # Leave 10s for overhead
    
    T = 10.0 # Initial temperature
    alpha = 0.999 # Cooling rate
    min_T = 0.01
    
    rng = random.Random(42)
    
    while time.time() - start_time < time_limit and T > min_T:
        # Choose a move
        move_type = rng.random()
        
        if move_type < 0.3:
            # Try to remove a random codeword
            if not code_set:
                continue
            c_out = rng.choice(list(code_set))
            
            # Calculate delta
            # If we remove c_out, coverage of its neighbors decreases by 1.
            # New uncovered = count of neighbors where coverage becomes 0.
            # But we must be careful: if a neighbor was covered only by c_out, it becomes uncovered.
            
            new_uncovered = 0
            for nb in neighborhoods[c_out]:
                if coverage[nb] == 1:
                    new_uncovered += 1
            
            # Delta Energy:
            # Size decreases by 1 (-1)
            # Uncovered increases by new_uncovered (+ P * new_uncovered)
            delta_E = -1 + 100 * new_uncovered
            
            if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
                # Accept move
                code_set.remove(c_out)
                for nb in neighborhoods[c_out]:
                    coverage[nb] -= 1
                current_energy += delta_E
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_code = set(code_set)
                    
        elif move_type < 0.6:
            # Try to add a random codeword (from universe)
            c_in = rng.choice(indices)
            if c_in in code_set:
                continue
            
            # Adding a codeword always keeps it valid (or makes it more valid), but increases size.
            # It might allow subsequent removals, but as a single step, it increases energy.
            # However, in SA, we allow worse states.
            # Delta Energy: +1 (size) + 0 (uncovered, since it only adds coverage)
            delta_E = 1
            
            if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
                code_set.add(c_in)
                for nb in neighborhoods[c_in]:
                    coverage[nb] += 1
                current_energy += delta_E
                
                # Note: This move alone doesn't improve best_energy, but sets up for future removals.
                
        else:
            # Try to swap: remove c_out, add c_in
            if not code_set:
                continue
            c_out = rng.choice(list(code_set))
            c_in = rng.choice(indices)
            if c_in == c_out or c_in in code_set:
                continue
            
            # Calculate delta
            # 1. Effect of removing c_out:
            #    Neighbors of c_out lose 1 coverage.
            # 2. Effect of adding c_in:
            #    Neighbors of c_in gain 1 coverage.
            
            # We need to check the net effect on uncovered count.
            # Let's simulate the coverage changes temporarily or calculate carefully.
            
            # Let's just simulate the change on the coverage array to be safe and fast.
            # Save old state of coverage for affected nodes?
            # Or just apply, check, revert if rejected.
            
            # Apply removal
            for nb in neighborhoods[c_out]:
                coverage[nb] -= 1
            # Apply addition
            for nb in neighborhoods[c_in]:
                coverage[nb] += 1
            
            # Check validity and energy
            # Uncovered count = sum(coverage == 0)
            # This is O(N) which is 729. Acceptable.
            new_uncovered = np.sum(coverage == 0)
            new_size = len(code_set) # Size unchanged in swap
            new_energy = new_size + 100 * new_uncovered
            delta_E = new_energy - current_energy
            
            if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
                # Accept
                code_set.remove(c_out)
                code_set.add(c_in)
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_code = set(code_set)
            else:
                # Revert
                for nb in neighborhoods[c_in]:
                    coverage[nb] -= 1
                for nb in neighborhoods[c_out]:
                    coverage[nb] += 1
            
        T *= alpha
        
        # Restart temperature if stuck for a long time? 
        # Simple restart: if no improvement for X steps, increase T slightly.
        # For simplicity, we just let it cool.
        
    # Final cleanup: Remove any redundant codewords from best_code
    # Reconstruct coverage for best_code
    final_coverage = np.zeros(total_words, dtype=np.int32)
    for c in best_code:
        for nb in neighborhoods[c]:
            final_coverage[nb] += 1
            
    # Greedy removal of redundant nodes
    changed = True
    while changed:
        changed = False
        for c in list(best_code):
            removable = True
            for nb in neighborhoods[c]:
                if final_coverage[nb] <= 1:
                    removable = False
                    break
            if removable:
                best_code.remove(c)
                for nb in neighborhoods[c]:
                    final_coverage[nb] -= 1
                changed = True

    return [list(idx_to_word(i)) for i in best_code]

# EVOLVE-BLOCK-END
