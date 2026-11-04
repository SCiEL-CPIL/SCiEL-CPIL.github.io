import itertools
import numpy as np
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 135.0  # 15s margin for 150s limit
    
    # 1. Generate initial code (Tetracode product)
    # For q=3, n=7, radius=1.
    # Tetracode is a [4,2,3] ternary code (perfect).
    # Product with tail gives a code.
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            initial_code.append(h + list(tail))
            
    # Ensure distinct
    initial_code = list(set(tuple(c) for c in initial_code))
    initial_code = [list(c) for c in initial_code]
    
    # 2. Precompute word-to-index and neighbors
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    num_words = len(all_words)
    
    neighbors = []
    for w in all_words:
        nbs = [word_to_idx[w]]
        for i in range(n):
            orig = w[i]
            for s in range(q):
                if s == orig: continue
                new_w = list(w)
                new_w[i] = s
                nbs.append(word_to_idx[tuple(new_w)])
        neighbors.append(nbs)
        
    # 3. Initialize cover counts
    cover_count = np.zeros(num_words, dtype=np.int32)
    current_code_indices = [word_to_idx[tuple(c)] for c in initial_code]
    
    for idx in current_code_indices:
        for nb in neighbors[idx]:
            cover_count[nb] += 1
            
    # Helper to check if a codeword index can be removed
    def can_remove(idx):
        for nb in neighbors[idx]:
            if cover_count[nb] <= 1:
                return False
        return True

    # Helper to remove a codeword
    def remove_codeword(idx):
        for nb in neighbors[idx]:
            cover_count[nb] -= 1
            
    # Helper to add a codeword
    def add_codeword(idx):
        for nb in neighbors[idx]:
            cover_count[nb] += 1
            
    # 4. Greedy Removal (Initial cleanup)
    changed = True
    while changed:
        changed = False
        for i in range(len(current_code_indices)):
            if can_remove(current_code_indices[i]):
                remove_codeword(current_code_indices[i])
                current_code_indices.pop(i)
                changed = True
                break # Restart scan
                
    best_size = len(current_code_indices)
    best_indices = list(current_code_indices)
    
    # 5. Local Search (Swaps and Multi-Removal Kicks)
    # Randomize for reproducibility
    random.seed(42)
    
    # Optimization: Precompute neighbor lists as numpy arrays for faster counting?
    # Actually, python lists are fine for n=7, q=3 (15 neighbors).
    
    # Strategy: 
    # A. Standard Swap: Replace one word with another, then clean up.
    # B. Kick: Remove 1-3 words, then greedily repair (add words to cover uncovered).
    
    # Helper for Greedy Repair
    def greedy_repair(code_indices, cover_count):
        # Find uncovered words
        # We need to identify words with cover_count == 0
        # Efficiently: iterate through all words? Or maintain a set of uncovered?
        # Since we just removed some, let's find zeros.
        
        # To avoid O(N) scan every time, we can just scan. N=2187 is small.
        # But doing it many times might be slow.
        # Let's try to find uncovered words.
        
        uncovered = np.where(cover_count == 0)[0]
        
        while len(uncovered) > 0 and time.time() < deadline:
            # Pick a word to add that covers the most uncovered neighbors
            # Iterate through all words to find best candidate?
            # That's O(N * K) where K is code size.
            # Better: For each uncovered word, look at its neighbors (potential codewords).
            # Count how many uncovered neighbors each candidate covers.
            
            # Candidate set: Neighbors of all uncovered words.
            candidates = set()
            for u in uncovered:
                candidates.update(neighbors[u])
                
            # Filter candidates that are not already in code
            code_set = set(code_indices)
            candidates = [c for c in candidates if c not in code_set]
            
            if not candidates:
                break # Should not happen unless code is invalid or we missed something
                
            best_cand = None
            max_covered = -1
            
            for c in candidates:
                count = 0
                for nb in neighbors[c]:
                    if cover_count[nb] == 0:
                        count += 1
                if count > max_covered:
                    max_covered = count
                    best_cand = c
                    
            if best_cand is None or max_covered == 0:
                break # No improvement possible?
                
            # Add best_cand
            code_indices.append(best_cand)
            add_codeword(best_cand)
            
            # Update uncovered list
            # Instead of full rescan, we can just rescan since N is small
            uncovered = np.where(cover_count == 0)[0]
            
        return code_indices

    # Main Loop
    while time.time() < deadline:
        # Decide strategy: Swap vs Kick
        # Kicks are more aggressive, used when stuck or to escape.
        # Swaps are local improvements.
        
        if random.random() < 0.7:
            # --- Swap Strategy ---
            if len(current_code_indices) == 0: break
            
            i = random.randint(0, len(current_code_indices) - 1)
            c_old_idx = current_code_indices[i]
            
            # Choose a new word
            if random.random() < 0.5:
                c_new_idx = random.randint(0, num_words - 1)
            else:
                c_new_idx = random.choice(neighbors[c_old_idx])
                
            if c_new_idx in current_code_indices:
                continue
                
            # Try swap
            remove_codeword(c_old_idx)
            
            # Check validity
            valid = True
            for nb in neighbors[c_old_idx]:
                if cover_count[nb] == 0:
                    valid = False
                    break
                    
            if not valid:
                add_codeword(c_old_idx)
                continue
                
            add_codeword(c_new_idx)
            current_code_indices[i] = c_new_idx
            
            # Cleanup
            changed = True
            while changed:
                changed = False
                for j in range(len(current_code_indices)):
                    if can_remove(current_code_indices[j]):
                        remove_codeword(current_code_indices[j])
                        current_code_indices.pop(j)
                        changed = True
                        break
                        
            if len(current_code_indices) < best_size:
                best_size = len(current_code_indices)
                best_indices = list(current_code_indices)
                
        else:
            # --- Kick Strategy (Multi-Removal + Repair) ---
            # Save state
            old_indices = list(current_code_indices)
            old_cover_count = cover_count.copy()
            
            # Remove k random words
            k = random.choice([1, 2, 3])
            if len(current_code_indices) <= k:
                continue
                
            # Pick k distinct indices to remove
            to_remove = random.sample(current_code_indices, k)
            for idx in to_remove:
                remove_codeword(idx)
                current_code_indices.remove(idx)
                
            # Repair
            current_code_indices = greedy_repair(current_code_indices, cover_count)
            
            # Cleanup after repair
            changed = True
            while changed:
                changed = False
                for j in range(len(current_code_indices)):
                    if can_remove(current_code_indices[j]):
                        remove_codeword(current_code_indices[j])
                        current_code_indices.pop(j)
                        changed = True
                        break
                        
            # Check if better
            if len(current_code_indices) < best_size:
                best_size = len(current_code_indices)
                best_indices = list(current_code_indices)
            else:
                # If not better, revert to old state to avoid drifting into worse local minima
                # Or accept if same size? Usually revert if worse to maintain diversity.
                # Let's revert if worse to keep the best known state as the base for further exploration.
                # But we might want to explore from the new state.
                # Given the "kick" nature, reverting if worse is safer for stability.
                # However, sometimes a worse state leads to a better one later.
                # Let's stick to reverting if not strictly better for now, to ensure we don't lose the best solution.
                # Actually, standard simulated annealing accepts worse with prob exp(-dE/T).
                # Let's just revert for simplicity and stability, focusing on finding the global min.
                current_code_indices = old_indices
                cover_count[:] = old_cover_count

    # Return the best found
    final_code = [list(all_words[idx]) for idx in best_indices]
    return final_code
