"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius 'radius'.
    For the specific case q=3, n=6, radius=1, we use fast local search with 1-swap and 2-swap moves.
    """
    if q == 3 and n == 6 and radius == 1:
        total_words = 3 ** n
        all_words = list(itertools.product(range(3), repeat=n))
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        
        # Precompute neighbors as index lists
        neighbor_indices = []
        for w in all_words:
            nb_set = set()
            nb_set.add(w)
            for i in range(n):
                for val in range(3):
                    if val == w[i]:
                        continue
                    new_w = list(w)
                    new_w[i] = val
                    new_w = tuple(new_w)
                    nb_set.add(new_w)
            neighbor_indices.append([word_to_idx[x] for x in nb_set])
        
        # Start with tetracode-based construction (81 words)
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        initial_code = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=n - 4):
                initial_code.append(tuple(h + list(tail)))
        
        code_set = set(initial_code)
        code_indices = set(word_to_idx[c] for c in code_set)
        
        # Coverage count array: cov[i] = number of codewords whose neighbor set includes word i
        cov = np.zeros(total_words, dtype=np.int32)
        for idx in code_indices:
            for nb_idx in neighbor_indices[idx]:
                cov[nb_idx] += 1
        
        def is_covered():
            return np.all(cov > 0)
        
        def add_codeword(idx):
            code_indices.add(idx)
            for nb_idx in neighbor_indices[idx]:
                cov[nb_idx] += 1
        
        def remove_codeword(idx):
            code_indices.remove(idx)
            for nb_idx in neighbor_indices[idx]:
                cov[nb_idx] -= 1
        
        # Phase 1: Try to remove codewords (1-removal)
        improved = True
        while improved:
            improved = False
            for idx in list(code_indices):
                # Try removing idx
                for nb_idx in neighbor_indices[idx]:
                    cov[nb_idx] -= 1
                if is_covered():
                    # Keep removed
                    code_indices.remove(idx)
                    improved = True
                    break
                else:
                    # Restore
                    for nb_idx in neighbor_indices[idx]:
                        cov[nb_idx] += 1
        
        # Phase 2: 1-swap moves (remove one, add one)
        improved = True
        while improved:
            improved = False
            code_list = list(code_indices)
            for idx_remove in code_list:
                # Remove idx_remove temporarily
                for nb_idx in neighbor_indices[idx_remove]:
                    cov[nb_idx] -= 1
                # Try adding each word not in code
                for idx_add in range(total_words):
                    if idx_add in code_indices:
                        continue
                    # Try adding idx_add
                    for nb_idx in neighbor_indices[idx_add]:
                        cov[nb_idx] += 1
                    if is_covered():
                        # Accept swap
                        code_indices.remove(idx_remove)
                        code_indices.add(idx_add)
                        improved = True
                        break
                    else:
                        # Restore idx_add
                        for nb_idx in neighbor_indices[idx_add]:
                            cov[nb_idx] -= 1
                # Restore idx_remove
                for nb_idx in neighbor_indices[idx_remove]:
                    cov[nb_idx] += 1
                if improved:
                    break
        
        # Phase 3: 2-swap moves (remove two, add one) - this is more expensive but can escape local minima
        improved = True
        while improved:
            improved = False
            code_list = list(code_indices)
            for i, idx_remove1 in enumerate(code_list):
                for j in range(i + 1, len(code_list)):
                    idx_remove2 = code_list[j]
                    # Remove both temporarily
                    for nb_idx in neighbor_indices[idx_remove1]:
                        cov[nb_idx] -= 1
                    for nb_idx in neighbor_indices[idx_remove2]:
                        cov[nb_idx] -= 1
                    # Try adding each word not in code
                    for idx_add in range(total_words):
                        if idx_add in code_indices:
                            continue
                        # Try adding idx_add
                        for nb_idx in neighbor_indices[idx_add]:
                            cov[nb_idx] += 1
                        if is_covered():
                            # Accept 2-swap
                            code_indices.remove(idx_remove1)
                            code_indices.remove(idx_remove2)
                            code_indices.add(idx_add)
                            improved = True
                            break
                        else:
                            # Restore idx_add
                            for nb_idx in neighbor_indices[idx_add]:
                                cov[nb_idx] -= 1
                    # Restore both removed
                    for nb_idx in neighbor_indices[idx_remove1]:
                        cov[nb_idx] += 1
                    for nb_idx in neighbor_indices[idx_remove2]:
                        cov[nb_idx] += 1
                    if improved:
                        break
                if improved:
                    break
        
        # Convert back to list of lists
        result = [all_words[idx] for idx in code_indices]
        return [list(c) for c in result]

    # Generic fallback for other parameters
    # Simple greedy algorithm
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors
    neighbors = {}
    for w in all_words:
        nb = set()
        for v in all_words:
            dist = sum(1 for i in range(n) if w[i] != v[i])
            if dist <= radius:
                nb.add(v)
        neighbors[w] = nb
    
    uncovered = set(all_words)
    code = []
    
    while uncovered:
        best_word = None
        max_covered = -1
        for w in all_words:
            if w in code:
                continue
            covered_count = len(neighbors[w] & uncovered)
            if covered_count > max_covered:
                max_covered = covered_count
                best_word = w
        
        if best_word is None:
            break
            
        code.append(best_word)
        uncovered -= neighbors[best_word]
        
    return [list(c) for c in code]

# EVOLVE-BLOCK-END
