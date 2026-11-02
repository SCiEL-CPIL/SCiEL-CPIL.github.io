import itertools
import time
import random
from collections import defaultdict

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Total number of words in the space
    total_words = q ** n
    
    # Generate all words as tuples
    all_words = list(itertools.product(range(q), repeat=n))
    
    # Map word tuple to index
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors for each word (words within Hamming distance <= radius)
    # For radius=1, this is the word itself plus all words differing in exactly 1 position
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
    
    # Greedy set cover
    code_indices = []
    uncovered = set(range(total_words))
    
    # Symmetry breaking: Fix the first codeword to all zeros
    first_word = (0,) * n
    first_idx = word_to_idx[first_word]
    code_indices.append(first_idx)
    uncovered -= neighbors[first_idx]
    
    # Precompute coverage counts for potential optimization
    while uncovered:
        if time.time() > deadline - 30:  # Leave time for local search
            break
            
        best_idx = -1
        best_count = 0
        
        # Find the word that covers the most uncovered words
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
    
    # Convert to list of codewords
    code = [list(all_words[i]) for i in code_indices]
    
    # Local search: try to remove redundant codewords
    def is_redundant(idx, current_code_indices):
        """Check if removing idx keeps all words covered."""
        if len(current_code_indices) <= 1:
            return False
        # Check if all words covered by idx are also covered by others
        other_covered = set()
        for j in current_code_indices:
            if j != idx:
                other_covered |= neighbors[j]
        # Check if neighbors[idx] is subset of other_covered
        return neighbors[idx].issubset(other_covered)
    
    removed_any = True
    while removed_any and time.time() < deadline - 10:
        removed_any = False
        for i in range(len(code_indices) - 1, -1, -1):
            if is_redundant(i, code_indices):
                code_indices.pop(i)
                removed_any = True
                break
    
    # Local search: try swaps to potentially reduce size or maintain validity
    # Try swapping a codeword with a non-codeword that maintains coverage
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        current_code_set = set(code_indices)
        
        for i in range(len(code_indices)):
            if time.time() > deadline - 5:
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
    
    # Final validation
    final_covered = set()
    for idx in code_indices:
        final_covered |= neighbors[idx]
    
    # If something went wrong, fall back to greedy result
    if final_covered != set(range(total_words)):
        # Regenerate with pure greedy
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
