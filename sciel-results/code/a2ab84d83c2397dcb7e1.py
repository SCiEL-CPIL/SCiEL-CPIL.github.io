"""Rewritten program with structured initial code and efficient local search."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys
import numpy as np

def construct(q, n, radius):
    deadline = time.time() + 115  # Safety margin before 150s limit
    start_time = time.time()
    
    # Fallback for non-standard parameters
    if q != 3 or n != 7 or radius != 1:
        if q == 3 and n >= 4 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        return [list(t) for t in itertools.product(range(q), repeat=n)]
    
    # For q=3, n=7, radius=1
    total_words = q ** n  # 2187
    
    # Precompute all words as numpy array
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    
    # Helper: convert word list to int
    def word_to_int(word):
        result = 0
        for w in word:
            result = result * q + w
        return result
    
    # Helper: get neighbor indices (words within Hamming distance 1)
    # Precompute neighbor structure for efficiency
    def get_neighbors(word_idx):
        """Return array of indices of words within Hamming distance 1 of word_idx."""
        word = all_words[word_idx]
        neighbors = [word_idx]
        for i in range(n):
            orig = word[i]
            for new_val in range(q):
                if new_val != orig:
                    word[i] = new_val
                    neighbors.append(int(np.where((all_words == word).all(axis=1))[0][0]))
                    word[i] = orig
        return np.array(neighbors, dtype=np.int32)
    
    # Precompute neighbors for all words
    print("Precomputing neighbors...", file=sys.stderr)
    neighbors_cache = {}
    for i in range(total_words):
        neighbors_cache[i] = get_neighbors(i)
    print("Neighbors precomputed.", file=sys.stderr)
    
    # Build structured initial code using algebraic construction
    # Use a linear code of dimension 4 over GF(3) as seed, then extend
    # Start with tetracode-based construction
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    # Create initial code: direct sum approach
    # For each tetracode word, create variants for positions 4,5,6
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            initial_code.append(h + list(tail))
    
    # Convert to numpy and get indices
    code_array = np.array(initial_code, dtype=np.int8)
    code_indices = []
    for c in initial_code:
        idx = int(np.where((all_words == c).all(axis=1))[0][0])
        code_indices.append(idx)
    code_indices = np.array(code_indices, dtype=np.int32)
    code_set = set(code_indices.tolist())
    
    # Compute coverage using numpy
    covered_count = np.zeros(total_words, dtype=np.int32)
    for idx in code_indices:
        covered_count[neighbors_cache[idx]] += 1
    
    print(f"Initial code size: {len(code_indices)}, uncovered: {(covered_count == 0).sum()}", file=sys.stderr)
    
    # Local search: try to remove redundant codewords
    def try_remove_redundant():
        """Try to remove redundant codewords. Returns number removed."""
        removed = 0
        # Try multiple passes
        for _ in range(50):
            if time.time() > deadline - 20:
                break
            progress = False
            for idx in code_indices.tolist():
                if time.time() > deadline - 20:
                    break
                # Check if removing this codeword would leave any word uncovered
                can_remove = np.all(covered_count[neighbors_cache[idx]] > 1)
                if can_remove:
                    covered_count[neighbors_cache[idx]] -= 1
                    code_indices = np.delete(code_indices, np.where(code_indices == idx)[0][0])
                    code_set.discard(idx)
                    removed += 1
                    progress = True
            if not progress:
                break
        return removed
    
    # Try swaps: remove one codeword, add another that covers more
    def try_swap():
        """Try to find a beneficial swap. Returns True if a swap was made."""
        if time.time() > deadline - 20:
            return False
        
        # Find codewords that can be removed (all their neighbors covered by others)
        removable = []
        for idx in code_indices.tolist():
            if np.all(covered_count[neighbors_cache[idx]] > 1):
                removable.append(idx)
        
        if not removable:
            return False
        
        # Try removing a random removable codeword
        random.shuffle(removable)
        for old_idx in removable[:20]:  # Try up to 20 candidates
            if time.time() > deadline - 20:
                break
            
            # Temporarily remove
            covered_count[neighbors_cache[old_idx]] -= 1
            uncovered = np.where(covered_count == 0)[0]
            
            if len(uncovered) == 0:
                # Just remove, no need to add
                code_indices = np.delete(code_indices, np.where(code_indices == old_idx)[0][0])
                code_set.discard(old_idx)
                return True
            
            # Find a word that covers the most uncovered words
            best_gain = 0
            best_pos = None
            
            # Sample candidate positions
            samples = min(100, total_words)
            candidates = random.sample(range(total_words), samples)
            for cand in candidates:
                if cand in code_set:
                    continue
                gain = np.sum(covered_count[neighbors_cache[cand]] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_pos = cand
            
            if best_pos is not None and best_gain > 0:
                # Add new word
                covered_count[neighbors_cache[best_pos]] += 1
                code_indices = np.append(code_indices, best_pos)
                code_set.add(best_pos)
                # Remove old one
                code_indices = np.delete(code_indices, np.where(code_indices == old_idx)[0][0])
                code_set.discard(old_idx)
                return True
            
            # Roll back
            covered_count[neighbors_cache[old_idx]] += 1
        
        return False
    
    # Main local search loop
    max_iterations = 100
    for iteration in range(max_iterations):
        if time.time() > deadline - 30:
            break
        
        progress = False
        
        # Try to remove redundant codewords
        if try_remove_redundant() > 0:
            progress = True
        
        # Try swaps
        if not progress:
            if try_swap():
                progress = True
            else:
                # Try a few more swaps
                for _ in range(10):
                    if time.time() > deadline - 30:
                        break
                    if try_swap():
                        progress = True
                        break
        
        if not progress and time.time() > start_time + 60:
            break
    
    # Final verification and repair
    uncovered = np.where(covered_count == 0)[0]
    for w in uncovered:
        idx = int(w)
        if idx not in code_set:
            code_indices = np.append(code_indices, idx)
            code_set.add(idx)
            covered_count[neighbors_cache[idx]] += 1
    
    # One more pass of removal
    for _ in range(50):
        if time.time() > deadline - 10:
            break
        if try_remove_redundant() == 0:
            break
    
    # Convert back to list of lists
    code = [all_words[idx].tolist() for idx in code_indices]
    
    print(f"Final code size: {len(code)}", file=sys.stderr)
    return code
# EVOLVE-BLOCK-END
