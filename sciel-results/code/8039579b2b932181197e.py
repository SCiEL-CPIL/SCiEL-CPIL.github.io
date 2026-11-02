import random
import math
import time

def construct(q, n, radius):
    # For this task, q=3, n=7, radius=1
    # Total words = 2187
    
    def word_to_idx(w):
        idx = 0
        for d in range(n):
            idx = idx * q + w[d]
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for d in range(n-1, -1, -1):
            w[d] = idx % q
            idx //= q
        return w

    total_words = q ** n
    
    # Pre-compute neighbors for radius 1
    # neighbors[idx] = list of indices covered by idx
    neighbors = [[] for _ in range(total_words)]
    for idx in range(total_words):
        w = idx_to_word(idx)
        neigh = [idx]
        for i in range(n):
            for delta in range(1, q):
                new_val = (w[i] + delta) % q
                pow_q = q ** (n - 1 - i)
                new_idx = idx + (new_val - w[i]) * pow_q
                neigh.append(new_idx)
        neighbors[idx] = neigh

    # 1. Initial Solution: Greedy Set Cover
    uncovered = [1] * total_words
    code_indices = []
    remaining = set(range(total_words))
    
    while remaining:
        best_idx = -1
        max_cov = 0
        for idx in range(total_words):
            cov = 0
            for nb in neighbors[idx]:
                if uncovered[nb]:
                    cov += 1
            if cov > max_cov:
                max_cov = cov
                best_idx = idx
        
        if best_idx == -1:
            break
            
        code_indices.append(best_idx)
        
        for nb in neighbors[best_idx]:
            if uncovered[nb]:
                uncovered[nb] = 0
                remaining.discard(nb)
                
    # Calculate initial coverage counts
    coverage = [0] * total_words
    for idx in code_indices:
        for nb in neighbors[idx]:
            coverage[nb] += 1
            
    # 2. Simulated Annealing
    # State: code_indices
    # Energy: number of codewords if valid, else large penalty
    # We maintain coverage array to quickly calculate uncovered count
    
    # Helper to check validity and get uncovered count
    def get_uncovered_count(coverage):
        # Count how many words have coverage 0
        # Since coverage is an array of ints, we can sum(1 for c in coverage if c == 0)
        # But for speed, we can track it? 
        # Given 2187 size, a loop is fast enough.
        unc = 0
        for c in coverage:
            if c == 0:
                unc += 1
        return unc

    current_uncovered = get_uncovered_count(coverage)
    current_size = len(code_indices)
    
    # Energy function
    def energy(size, uncovered):
        if uncovered == 0:
            return size
        else:
            return size + 100 * uncovered # Heavy penalty for invalid

    current_energy = energy(current_size, current_uncovered)
    
    # SA Parameters
    # Time limit 120s. We have ~10s for greedy. 110s for SA.
    start_time = time.time()
    time_limit = 110.0
    
    T_start = 10.0
    T_end = 0.001
    # Number of iterations per temperature step? Or just run until time limit.
    # Let's run for a fixed number of iterations or until time limit.
    # 2187 words. 
    # Move: Pick random i from code_indices, pick random j from all words.
    # Swap code_indices[i] with j? Or Remove/Insert?
    # Standard: Swap one codeword with another.
    
    best_code = list(code_indices)
    best_energy = current_energy
    
    while time.time() - start_time < time_limit:
        # Pick a random codeword index in the list
        i = random.randint(0, len(code_indices) - 1)
        old_c = code_indices[i]
        
        # Pick a random new word
        new_c = random.randint(0, total_words - 1)
        
        if old_c == new_c:
            continue
            
        # Calculate delta coverage
        # Remove old_c
        for nb in neighbors[old_c]:
            coverage[nb] -= 1
        # Add new_c
        for nb in neighbors[new_c]:
            coverage[nb] += 1
            
        new_uncovered = get_uncovered_count(coverage)
        new_size = len(code_indices) # Size doesn't change in swap
        new_energy = energy(new_size, new_uncovered)
        
        delta_e = new_energy - current_energy
        
        # Accept?
        if delta_e < 0 or (T_start > 0 and random.random() < math.exp(-delta_e / T_start)):
            # Accept move
            code_indices[i] = new_c
            current_uncovered = new_uncovered
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_code = list(code_indices)
        else:
            # Reject move, restore coverage
            for nb in neighbors[new_c]:
                coverage[nb] -= 1
            for nb in neighbors[old_c]:
                coverage[nb] += 1
                
        # Cool down
        # Linear or exponential cooling?
        # Exponential is standard.
        # T = T_start * (T_end/T_start)^(iter/total_iters)
        # Since we don't know total_iters, let's use a simple decay per step.
        # But we need to ensure we explore enough.
        # Let's use a fixed number of iterations per "phase" or just decay T slowly.
        # T = T * 0.999 might be too slow.
        # Let's estimate iterations. 
        # 1 swap takes ~ 2*15 (coverage update) + 2187 (uncovered count) ~ 2200 ops.
        # Python can do ~10^6-10^7 simple ops/sec? Maybe 10^5 complex ones.
        # So ~500-5000 iterations per second.
        # 100 seconds -> 50k - 500k iterations.
        # Let's use a cooling schedule that depends on iteration count.
        # We'll just keep T constant for a while, then lower? 
        # Or better: Use a "restart" strategy?
        # Simple: T decays exponentially with time.
        # T = T_start * exp(-lambda * (time - start_time))
        elapsed = time.time() - start_time
        T = T_start * math.exp(-0.05 * elapsed) # Half life ~ 14s
        
        # If T is very low, we are basically hill climbing.
        # To escape, we can occasionally reset to best and reheat?
        # Let's add a simple restart if no improvement for a long time.
        # For now, just standard SA.

    # Final cleanup: Greedy Removal on the best code found
    # Restore state to best_code
    code_indices = list(best_code)
    coverage = [0] * total_words
    for idx in code_indices:
        for nb in neighbors[idx]:
            coverage[nb] += 1
            
    # Greedy Removal
    improved = True
    while improved:
        improved = False
        random.shuffle(code_indices)
        for i in range(len(code_indices)):
            idx = code_indices[i]
            for nb in neighbors[idx]:
                coverage[nb] -= 1
            if min(coverage) >= 1:
                del code_indices[i]
                improved = True
                break
            else:
                for nb in neighbors[idx]:
                    coverage[nb] += 1
                    
    result = []
    for idx in code_indices:
        result.append(idx_to_word(idx))
    return result
