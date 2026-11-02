import itertools
import time
import random
from collections import defaultdict

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Total number of words in the space
    total_words = q ** n
    
    # Generate all words as tuples
    all_words = list(itertools.product(range(q), repeat=n))
    
    # Map word tuple to index
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors for each word (words within Hamming distance <= radius)
    neighbors = []
    for i, w in enumerate(all_words):
        nbrs = {i}  # Include self
        if radius >= 1:
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]:
                        continue
                    new_list = list(w)
                    new_list[pos] = val
                    nbrs.add(word_to_idx[tuple(new_list)])
        neighbors.append(nbrs)
    
    def greedy_set_cover(uncovered, code_indices):
        """Greedy set cover to add codewords until all words are covered."""
        uncovered = set(uncovered)
        code_indices = list(code_indices)
        while uncovered and time.time() < deadline - 5:
            best_idx = -1
            best_count = 0
            for i in range(total_words):
                if i in code_indices:
                    continue
                count = len(uncovered & neighbors[i])
                if count > best_count:
                    best_count = count
                    best_idx = i
            if best_idx == -1:
                break
            code_indices.append(best_idx)
            uncovered -= neighbors[best_idx]
        return code_indices, uncovered
    
    def remove_redundant(code_indices):
        """Remove redundant codewords (those whose coverage is fully covered by others)."""
        code_indices = list(code_indices)
        removed_any = True
        while removed_any and time.time() < deadline - 5:
            removed_any = False
            for i in range(len(code_indices) - 1, -1, -1):
                idx = code_indices[i]
                # Check if removing idx keeps all words covered
                other_covered = set()
                for j in code_indices:
                    if j != i:
                        other_covered |= neighbors[j]
                if neighbors[idx].issubset(other_covered):
                    code_indices.pop(i)
                    removed_any = True
                    break
        return code_indices
    
    def is_valid(code_indices):
        """Check if the code covers all words."""
        covered = set()
        for idx in code_indices:
            covered |= neighbors[idx]
        return len(covered) == total_words
    
    # Phase 1: Initial greedy set cover with symmetry breaking
    code_indices = []
    uncovered = set(range(total_words))
    
    # Symmetry breaking: Fix the first codeword to all zeros
    first_word = (0,) * n
    first_idx = word_to_idx[first_word]
    code_indices.append(first_idx)
    uncovered -= neighbors[first_idx]
    
    code_indices, uncovered = greedy_set_cover(uncovered, code_indices)
    
    # Remove redundant codewords
    code_indices = remove_redundant(code_indices)
    
    best_code = list(code_indices)
    best_size = len(best_code)
    
    # Phase 2: Local search with swaps
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        current_code_set = set(code_indices)
        
        for i in range(len(code_indices)):
            if time.time() > deadline - 10:
                break
            c = code_indices[i]
            
            # Find words only covered by c (would be uncovered if c removed)
            other_covered = set()
            for j in code_indices:
                if j != i:
                    other_covered |= neighbors[j]
            only_c = neighbors[c] - other_covered
            
            if not only_c:
                # c is redundant, remove it
                code_indices.pop(i)
                improved = True
                break
            
            # Try to find a swap: replace c with c' that covers only_c
            for c_prime in range(total_words):
                if c_prime in current_code_set or c_prime == c:
                    continue
                if only_c.issubset(neighbors[c_prime]):
                    # Valid swap
                    code_indices[i] = c_prime
                    current_code_set.remove(c)
                    current_code_set.add(c_prime)
                    improved = True
                    break
            
            if improved:
                break
        
        if len(code_indices) < best_size:
            best_code = list(code_indices)
            best_size = len(best_code)
    
    # Phase 3: Randomized restarts to escape local minima
    random.seed(42)
    for restart in range(50):
        if time.time() > deadline - 10:
            break
        
        # Random initial code
        code_indices = []
        uncovered = set(range(total_words))
        
        # Add some random codewords first
        num_initial = random.randint(1, 5)
        for _ in range(num_initial):
            idx = random.randint(0, total_words - 1)
            if idx not in code_indices:
                code_indices.append(idx)
                uncovered -= neighbors[idx]
        
        # Complete with greedy
        code_indices, uncovered = greedy_set_cover(uncovered, code_indices)
        
        # Remove redundant
        code_indices = remove_redundant(code_indices)
        
        # Local search
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            current_code_set = set(code_indices)
            
            for i in range(len(code_indices)):
                if time.time() > deadline - 10:
                    break
                c = code_indices[i]
                
                other_covered = set()
                for j in code_indices:
                    if j != i:
                        other_covered |= neighbors[j]
                only_c = neighbors[c] - other_covered
                
                if not only_c:
                    code_indices.pop(i)
                    improved = True
                    break
                
                for c_prime in range(total_words):
                    if c_prime in current_code_set or c_prime == c:
                        continue
                    if only_c.issubset(neighbors[c_prime]):
                        code_indices[i] = c_prime
                        current_code_set.remove(c)
                        current_code_set.add(c_prime)
                        improved = True
                        break
                
                if improved:
                    break
            
            if len(code_indices) < best_size:
                best_code = list(code_indices)
                best_size = len(best_code)
    
    # Use the best code found
    code_indices = best_code
    
    # Final validation
    final_covered = set()
    for idx in code_indices:
        final_covered |= neighbors[idx]
    
    # If something went wrong, fall back to pure greedy
    if final_covered != set(range(total_words)):
        code_indices = []
        uncovered = set(range(total_words))
        first_word = (0,) * n
        first_idx = word_to_idx[first_word]
        code_indices.append(first_idx)
        uncovered -= neighbors[first_idx]
        
        while uncovered and time.time() < deadline:
            best_idx = -1
            best_count = 0
            for i in range(total_words):
                if i in code_indices:
                    continue
                count = len(uncovered & neighbors[i])
                if count > best_count:
                    best_count = count
                    best_idx = i
            if best_idx == -1:
                break
            code_indices.append(best_idx)
            uncovered -= neighbors[best_idx]
    
    code = [list(all_words[i]) for i in code_indices]
    
    # Ensure distinctness
    seen = set()
    unique_code = []
    for c in code:
        t = tuple(c)
        if t not in seen:
            seen.add(t)
            unique_code.append(c)
    
    return unique_code
