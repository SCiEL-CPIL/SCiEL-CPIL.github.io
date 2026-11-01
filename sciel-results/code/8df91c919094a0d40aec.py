"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import numpy as np
from scipy.optimize import minimize

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius 'radius'.
    For the specific case q=3, n=6, radius=1, we use an optimized local search with
    precomputed coverage counts to efficiently find a minimal covering code.
    """
    if q == 3 and n == 6 and radius == 1:
        total_words = 3 ** n
        all_words = list(itertools.product(range(3), repeat=n))
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        
        # Precompute neighbors as bitsets for fast coverage computation
        # Each word maps to a bitset of words within distance 1
        neighbor_bitsets = []
        for w in all_words:
            nb = set()
            nb.add(w)
            for i in range(n):
                for val in range(3):
                    if val == w[i]:
                        continue
                    new_w = list(w)
                    new_w[i] = val
                    nb.add(tuple(new_w))
            # Convert to bitset (integer)
            bits = 0
            for v in nb:
                idx = word_to_idx[v]
                bits |= (1 << idx)
            neighbor_bitsets.append(bits)
        
        # Start with tetracode-based construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        initial_code = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=n - 4):
                initial_code.append(tuple(h + list(tail)))
        
        code_indices = set(word_to_idx[c] for c in initial_code)
        
        def compute_coverage(code_set):
            """Compute coverage as a bitset and return count of covered words."""
            covered = 0
            for idx in code_set:
                covered |= neighbor_bitsets[idx]
            return covered, bin(covered).count('1')
        
        # Local search phase 1: try to remove words
        improved = True
        while improved:
            improved = False
            for c_idx in list(code_indices):
                new_code = code_indices - {c_idx}
                _, cov_count = compute_coverage(new_code)
                if cov_count == total_words:
                    code_indices = new_code
                    improved = True
                    break
        
        # Local search phase 2: try swaps (remove one, add another)
        improved = True
        swap_count = 0
        max_swaps = 500  # Limit to avoid infinite loops
        while improved and swap_count < max_swaps:
            improved = False
            code_list = list(code_indices)
            for c_remove_idx in code_list:
                # Try adding each word not in code
                for c_add_idx in range(total_words):
                    if c_add_idx in code_indices:
                        continue
                    new_code = (code_indices - {c_remove_idx}) | {c_add_idx}
                    _, cov_count = compute_coverage(new_code)
                    if cov_count == total_words:
                        code_indices = new_code
                        improved = True
                        swap_count += 1
                        break
                if improved:
                    break
        
        # Convert back to list of lists
        result = [list(all_words[idx]) for idx in sorted(code_indices)]
        return result

    # Generic fallback for other parameters
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
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
