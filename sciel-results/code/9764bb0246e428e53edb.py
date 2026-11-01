"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import numpy as np

def construct(q, n, radius):
    if q == 3 and n == 6 and radius == 1:
        total_words = 3 ** n
        all_words = list(itertools.product(range(3), repeat=n))
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        
        # Precompute neighbors as sets of indices for fast membership testing
        neighbors = [set() for _ in range(total_words)]
        for i, w in enumerate(all_words):
            neighbors[i].add(i)
            for pos in range(n):
                orig_val = w[pos]
                for new_val in range(3):
                    if new_val == orig_val:
                        continue
                    new_w = list(w)
                    new_w[pos] = new_val
                    new_w = tuple(new_w)
                    neighbors[i].add(word_to_idx[new_w])
        
        # Initial code: Tetracode based
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        initial_code_indices = set()
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=n - 4):
                w = tuple(h + list(tail))
                initial_code_indices.add(word_to_idx[w])
        
        code = set(initial_code_indices)
        
        counts = np.zeros(total_words, dtype=np.int8)
        
        def add_code(idx):
            for nb in neighbors[idx]:
                counts[nb] += 1
                
        def remove_code(idx):
            for nb in neighbors[idx]:
                counts[nb] -= 1
                
        for idx in code:
            add_code(idx)
            
        def is_covered():
            return np.all(counts > 0)
            
        # Local Search
        for _ in range(500):
            # 1. Try to remove
            removed = False
            for idx in list(code):
                remove_code(idx)
                if is_covered():
                    code.remove(idx)
                    removed = True
                    break
                else:
                    add_code(idx)
            if removed:
                continue
                
            # 2. Try to swap
            swapped = False
            for idx in list(code):
                critical_words = [nb for nb in neighbors[idx] if counts[nb] == 1]
                
                if not critical_words:
                    remove_code(idx)
                    code.remove(idx)
                    swapped = True
                    break
                
                for j in range(total_words):
                    if j in code:
                        continue
                    if all(w in neighbors[j] for w in critical_words):
                        remove_code(idx)
                        code.remove(idx)
                        code.add(j)
                        add_code(j)
                        swapped = True
                        break
                if swapped:
                    break
            if not swapped:
                if len(code) > 73:
                    break
                else:
                    break

        result = [list(all_words[idx]) for idx in code]
        return result

    # Generic fallback
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    neighbors = []
    for w in all_words:
        nb = set()
        for v in all_words:
            dist = sum(1 for i in range(n) if w[i] != v[i])
            if dist <= radius:
                nb.add(v)
        neighbors.append(nb)
        
    uncovered = set(all_words)
    code = []
    
    while uncovered:
        best_word = None
        max_covered = -1
        for w in all_words:
            if w in code:
                continue
            covered_count = len(neighbors[word_to_idx[w]] & uncovered)
            if covered_count > max_covered:
                max_covered = covered_count
                best_word = w
        if best_word is None:
            break
        code.append(best_word)
        uncovered -= neighbors[word_to_idx[best_word]]
        
    return [list(c) for c in code]

# EVOLVE-BLOCK-END
