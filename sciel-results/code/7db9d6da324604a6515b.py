import numpy as np
import time
import itertools

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    N = q ** n
    deadline = time.time() + 115.0
    
    # Precompute neighbors for radius 1
    # neighbors[w] contains w and all words at distance 1
    neighbors = np.zeros((N, 1 + 2 * n), dtype=np.int32)
    powers = [3 ** i for i in range(n)]
    
    for w in range(N):
        neighbors[w, 0] = w
        idx = 1
        temp = w
        for i in range(n):
            d = temp % 3
            temp //= 3
            w1 = w + ((d + 1) % 3 - d) * powers[i]
            neighbors[w, idx] = w1
            idx += 1
            w2 = w + ((d + 2) % 3 - d) * powers[i]
            neighbors[w, idx] = w2
            idx += 1

    # Initial seed: Tetracode direct sum (486 words)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    seed_code_ints = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            word = h + list(tail)
            w_int = 0
            for i in range(n):
                w_int += word[i] * powers[i]
            seed_code_ints.append(w_int)
    
    best_code = seed_code_ints[:]
    best_size = len(best_code)
    
    rng = np.random.default_rng(42)
    
    # Helper to check if a set of words covers the whole space
    def is_covering(code_list):
        if not code_list:
            return False
        covered = np.zeros(N, dtype=np.uint8)
        for w in code_list:
            covered[neighbors[w]] = 1
        return covered.all()
        
    # Local Search Loop
    # We maintain:
    # code: list of current codewords
    # covered_count: array of length N, count of how many codewords cover each position
    # This allows O(1) removal checks.
    
    def local_search(initial_code, max_iters=5000):
        code = list(initial_code)
        if not code:
            return code
            
        covered_count = np.zeros(N, dtype=np.int16)
        for w in code:
            covered_count[neighbors[w]] += 1
            
        improved = True
        iters = 0
        
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            
            # 1. Try to remove a redundant word
            # A word w is redundant if for all v in neighbors[w], covered_count[v] > 1
            # We can check this efficiently: min(covered_count[neighbors[w]]) > 1
            
            # We iterate through current code to find a removable word
            # To avoid O(N^2) in the worst case if code is large, we sample or scan.
            # Since code size is ~400-500, scanning is fast.
            
            removable_idx = -1
            for i, w in enumerate(code):
                if np.min(covered_count[neighbors[w]]) > 1:
                    removable_idx = i
                    break
                    
            if removable_idx != -1:
                w = code[removable_idx]
                covered_count[neighbors[w]] -= 1
                code.pop(removable_idx)
                improved = True
                continue # Restart scan as counts changed
                
            # 2. Try to swap a word for another to reduce size or prepare for removal?
            # Actually, standard local search for covering codes often uses "add-then-remove" or "swap".
            # If no word is removable, we are in a local minimum for size.
            # To escape, we can try:
            # a) Add a random word, then try to remove two words? (Expensive)
            # b) Swap word w with word u if it allows a removal later?
            
            # Let's try a "Perturbation": Pick a random word in code, remove it, 
            # then greedily add words to cover the holes, then try to remove again.
            # This is essentially a restart from a slightly modified state.
            
            if len(code) > 403: # Only perturb if we can still improve
                # Pick a random index to remove
                idx_to_remove = rng.integers(0, len(code))
                w_removed = code[idx_to_remove]
                code.pop(idx_to_remove)
                covered_count[neighbors[w_removed]] -= 1
                
                # Now we have holes. Greedily fill them.
                # Find uncovered positions
                holes = np.where(covered_count == 0)[0]
                
                if len(holes) > 0:
                    # Greedy add
                    # To speed up, we can just pick a word that covers the most holes.
                    # But we must be careful not to add too many.
                    # Simple heuristic: pick random word from holes' neighbors?
                    # Or just pick the word that is in the holes list? No, codeword covers neighbors.
                    
                    # Let's find a good word to add.
                    # Gain of adding w = number of holes covered by w.
                    # We only need to cover all holes.
                    
                    # Optimization: Instead of full greedy, just add one random word that covers at least one hole,
                    # or the one with max coverage.
                    # Given small hole count usually (1-5 words removed), this is fast.
                    
                    # To keep it simple and fast, let's just add random candidates from the neighborhood of holes
                    # until no holes remain.
                    
                    while len(holes) > 0 and time.time() < deadline:
                        # Pick a random hole
                        h = holes[rng.integers(0, len(holes))]
                        # Pick a random neighbor of h to add
                        # Neighbors of h are in neighbors[h]
                        cand = neighbors[h, rng.integers(0, neighbors.shape[1])]
                        
                        # Add cand
                        code.append(cand)
                        covered_count[neighbors[cand]] += 1
                        
                        # Update holes
                        # holes = np.where(covered_count == 0)[0] is expensive.
                        # We can just recompute if holes are few, or track them.
                        # For simplicity, recompute.
                        holes = np.where(covered_count == 0)[0]
                        
                        # If we added a word that doesn't cover the specific hole h? 
                        # Wait, cand is a neighbor of h, so cand covers h. 
                        # So h is no longer a hole.
                        
                        # Check if we went overboard? 
                        # We just want to restore coverage.
                        
                # After restoring coverage, loop back to try removals again.
                improved = True
                continue
            
            # If no removal and no perturbation triggered (or perturbation failed to improve net),
            # we might be stuck.
            # Break if we can't improve?
            # Actually, if we did a perturbation, we set improved=True, so we loop.
            # If we didn't do a perturbation (e.g. len(code) <= 403), we break.
            
            break

    # Run multiple restarts
    # Start with the seed
    current_code = best_code[:]
    
    # We can also try random initializations if we have time, but seed is strong.
    # Let's just run local search on the seed.
    
    while time.time() < deadline:
        # Run local search
        new_code = local_search(current_code, max_iters=1000)
        
        if len(new_code) < best_size:
            best_size = len(new_code)
            best_code = new_code[:]
            # If we hit the lower bound, stop
            if best_size <= 403:
                break
        
        # If we didn't improve, do a random perturbation on the best code to escape local minima
        # for the next round of local search
        if len(best_code) > 1:
            # Pick a random word to remove and replace with a random neighbor?
            # Or just remove and let local search fix it?
            # Let's create a perturbed version of best_code for the next local_search call.
            perturbed = best_code[:]
            if len(perturbed) > 0:
                idx = rng.integers(0, len(perturbed))
                w = perturbed[idx]
                perturbed.pop(idx)
                # Add a random word
                rand_w = rng.integers(0, N)
                perturbed.append(rand_w)
                # Ensure distinctness? 
                # If rand_w is already in perturbed, we have a duplicate.
                # The problem requires distinct codewords.
                # Let's check and fix.
                if rand_w in perturbed[:-1]: # Check if it was added and exists in earlier part
                     # Find another random word
                     while rand_w in perturbed[:-1]:
                         rand_w = rng.integers(0, N)
                
            current_code = perturbed
        else:
            break

    # Final validation and cleanup
    # Ensure distinctness
    unique_best = list(set(best_code))
    
    # Validate coverage one last time to be safe
    if len(unique_best) > 0:
        covered = np.zeros(N, dtype=np.uint8)
        for w in unique_best:
            covered[neighbors[w]] = 1
        if not covered.all():
            # Fallback to seed if something went wrong
            unique_best = seed_code_ints[:]
    else:
        unique_best = seed_code_ints[:]

    # Convert integer codewords back to lists of integers
    final_code = []
    for w in unique_best:
        word = []
        temp = w
        for _ in range(n):
            word.append(temp % 3)
            temp //= 3
        final_code.append(word)
        
    return final_code
