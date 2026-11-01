"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import numpy as np
import random
import time

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius 'radius'.
    Uses a multi-start local search approach with efficient coverage tracking.
    """
    # Handle the specific case q=3, n=6, radius=1 with optimized approach
    if q == 3 and n == 6 and radius == 1:
        return _construct_3_6_1()
    
    # Generic fallback: greedy set cover
    return _greedy_cover(q, n, radius)

def _construct_3_6_1():
    """Optimized construction for q=3, n=6, radius=1."""
    total_words = 3 ** 6  # 729
    all_words = list(itertools.product(range(3), repeat=6))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    idx_to_word = all_words  # list indexed by int
    
    # Precompute neighbors: for each word index, list of neighbor indices (dist <= 1)
    # Each word has 1 + 6*2 = 13 neighbors
    neighbors = []
    for w in all_words:
        nb = [word_to_idx[w]]
        for i in range(6):
            for val in range(3):
                if val == w[i]:
                    continue
                new_w = list(w)
                new_w[i] = val
                nb.append(word_to_idx[tuple(new_w)])
        neighbors.append(nb)
    
    # Precompute neighbor sets as numpy arrays for fast operations
    # coverage_count[idx] = number of codewords covering word idx
    # We'll track which words are covered (coverage_count > 0)
    
    def try_remove_and_optimize(initial_code):
        """Try to minimize the code by removing redundant words and swapping."""
        code_set = set(initial_code)
        coverage_count = np.zeros(total_words, dtype=np.int32)
        
        for c in code_set:
            idx = word_to_idx[c]
            for nb_idx in neighbors[idx]:
                coverage_count[nb_idx] += 1
        
        # Check if valid
        if np.any(coverage_count == 0):
            return None
        
        # Greedy removal: try to remove each word if coverage stays complete
        removed_any = True
        while removed_any:
            removed_any = False
            for c in list(code_set):
                idx = word_to_idx[c]
                # Check if removing this word would leave any word uncovered
                # A word becomes uncovered if its only covering codeword is c
                all_covered = True
                for nb_idx in neighbors[idx]:
                    if coverage_count[nb_idx] == 1:
                        all_covered = False
                        break
                if all_covered:
                    # Safe to remove
                    for nb_idx in neighbors[idx]:
                        coverage_count[nb_idx] -= 1
                    code_set.remove(c)
                    removed_any = True
                    break
        
        # Now try swaps: replace one codeword with another to enable further removals
        # This is more complex; do a limited number of swap attempts
        best_size = len(code_set)
        
        # Try to find a better configuration through local moves
        for _ in range(200):  # limited iterations
            if len(code_set) >= best_size:
                break
            # Pick a random codeword to try replacing
            c_remove = random.choice(list(code_set))
            idx_remove = word_to_idx[c_remove]
            
            # Find words that would become uncovered if we remove c_remove
            at_risk = []
            for nb_idx in neighbors[idx_remove]:
                if coverage_count[nb_idx] == 1:
                    at_risk.append(nb_idx)
            
            if not at_risk:
                # Can just remove
                for nb_idx in neighbors[idx_remove]:
                    coverage_count[nb_idx] -= 1
                code_set.remove(c_remove)
                best_size = len(code_set)
                continue
            
            # Find a replacement word that covers all at_risk words
            # The replacement must be within distance 1 of all at_risk words
            # Actually, the replacement just needs to cover the at_risk words
            # Try random words
            found = False
            for _ in range(50):
                c_add = random.choice(all_words)
                if c_add in code_set:
                    continue
                idx_add = word_to_idx[c_add]
                # Check if c_add covers all at_risk words
                covers_all = True
                for r_idx in at_risk:
                    if r_idx not in neighbors[idx_add]:
                        covers_all = False
                        break
                if covers_all:
                    # Perform swap
                    for nb_idx in neighbors[idx_remove]:
                        coverage_count[nb_idx] -= 1
                    for nb_idx in neighbors[idx_add]:
                        coverage_count[nb_idx] += 1
                    code_set.remove(c_remove)
                    code_set.add(c_add)
                    found = True
                    break
            
            if not found:
                continue
            
            # After swap, try to remove more
            removed_any2 = True
            while removed_any2:
                removed_any2 = False
                for c in list(code_set):
                    idx = word_to_idx[c]
                    all_covered = True
                    for nb_idx in neighbors[idx]:
                        if coverage_count[nb_idx] == 1:
                            all_covered = False
                            break
                    if all_covered:
                        for nb_idx in neighbors[idx]:
                            coverage_count[nb_idx] -= 1
                        code_set.remove(c)
                        removed_any2 = True
                        break
            
            if len(code_set) < best_size:
                best_size = len(code_set)
        
        return list(code_set)
    
    # Start from tetracode-based construction
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=2):
            initial_code.append(tuple(h + list(tail)))
    
    best_code = try_remove_and_optimize(initial_code)
    best_size = len(best_code) if best_code else len(initial_code)
    
    # Try multiple random restarts
    random.seed(42)
    for restart in range(10):
        # Create a random initial covering code
        # Start with all 729 words, then randomly remove
        code_set = set(all_words)
        coverage_count = np.zeros(total_words, dtype=np.int32)
        
        for c in code_set:
            idx = word_to_idx[c]
            for nb_idx in neighbors[idx]:
                coverage_count[nb_idx] += 1
        
        # Randomly remove words while maintaining coverage
        random.shuffle(all_words)
        for c in all_words:
            idx = word_to_idx[c]
            all_covered = True
            for nb_idx in neighbors[idx]:
                if coverage_count[nb_idx] == 1:
                    all_covered = False
                    break
            if all_covered:
                for nb_idx in neighbors[idx]:
                    coverage_count[nb_idx] -= 1
                code_set.remove(c)
        
        # Now do local search
        result = try_remove_and_optimize(list(code_set))
        if result and len(result) < best_size:
            best_code = result
            best_size = len(result)
    
    return [list(c) for c in best_code]

def _greedy_cover(q, n, radius):
    """Generic greedy set cover for covering codes."""
    all_words = list(itertools.product(range(q), repeat=n))
    total = len(all_words)
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors
    neighbors = []
    for w in all_words:
        nb = set()
        for v in all_words:
            dist = sum(1 for i in range(n) if w[i] != v[i])
            if dist <= radius:
                nb.add(v)
        neighbors.append(nb)
    
    uncovered = set(range(total))
    code = []
    
    while uncovered:
        best_idx = -1
        max_covered = -1
        for i, w in enumerate(all_words):
            if any(c == w for c in code):
                continue
            covered_count = len(neighbors[i] & uncovered)
            if covered_count > max_covered:
                max_covered = covered_count
                best_idx = i
        
        if best_idx == -1:
            break
        
        code.append(all_words[best_idx])
        uncovered -= neighbors[best_idx]
    
    return [list(c) for c in code]

# EVOLVE-BLOCK-END
