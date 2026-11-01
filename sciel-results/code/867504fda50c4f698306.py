"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import numpy as np
from scipy.optimize import minimize
import random

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius 'radius'.
    For the specific case q=3, n=6, radius=1, we use a heuristic based on local search
    starting from a known good configuration and iteratively improving it by swapping or replacing codewords.
    Uses efficient incremental coverage tracking to enable effective swap operations.
    """
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters: simple greedy
        all_words = list(itertools.product(range(q), repeat=n))
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
    
    # Specific optimization for q=3, n=6, radius=1
    if q == 3 and n == 6 and radius == 1:
        total_words = 3 ** n
        all_words = list(itertools.product(range(3), repeat=n))
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        
        # Precompute neighbors for each word (distance <= 1)
        neighbors = {}
        for w in all_words:
            nb = set()
            nb.add(w)
            for i in range(n):
                for val in range(3):
                    if val == w[i]:
                        continue
                    new_w = list(w)
                    new_w[i] = val
                    new_w = tuple(new_w)
                    nb.add(new_w)
            neighbors[w] = nb
        
        # Start with the tetracode-based construction (9 * 3^2 = 81 words)
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        initial_code = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=n - 4):
                initial_code.append(tuple(h + list(tail)))
        
        code = set(initial_code)
        
        # Efficient coverage tracking: count how many codewords cover each word
        # coverage_count[v] = number of codewords c such that v is in neighbors[c]
        coverage_count = np.zeros(total_words, dtype=np.int32)
        for c in code:
            idx_c = word_to_idx[c]
            for nb in neighbors[c]:
                idx_nb = word_to_idx[nb]
                coverage_count[idx_nb] += 1
        
        # A word is fully covered if coverage_count[word_to_idx[w]] > 0
        # Fully covered count
        fully_covered = np.sum(coverage_count > 0)
        
        def try_remove(c):
            """Try to remove codeword c. Returns True if removal maintains full coverage."""
            idx_c = word_to_idx[c]
            # Check if any neighbor would become uncovered
            for nb in neighbors[c]:
                idx_nb = word_to_idx[nb]
                if coverage_count[idx_nb] == 1:
                    return False
            # Safe to remove
            for nb in neighbors[c]:
                idx_nb = word_to_idx[nb]
                coverage_count[idx_nb] -= 1
            code.remove(c)
            return True
        
        def try_add(c):
            """Add codeword c to the code."""
            idx_c = word_to_idx[c]
            for nb in neighbors[c]:
                idx_nb = word_to_idx[nb]
                coverage_count[idx_nb] += 1
            code.add(c)
        
        def try_swap(c_remove, c_add):
            """Try to swap c_remove with c_add. Returns True if swap maintains full coverage."""
            idx_rm = word_to_idx[c_remove]
            idx_add = word_to_idx[c_add]
            
            # First check: after removing c_remove, which words would be uncovered?
            # A word v is uncovered after removal if coverage_count[v] == 1 and v in neighbors[c_remove]
            # For the swap to work, c_add must cover all such words.
            
            # Temporarily simulate removal
            would_be_uncovered = []
            for nb in neighbors[c_remove]:
                idx_nb = word_to_idx[nb]
                if coverage_count[idx_nb] == 1:
                    would_be_uncovered.append(idx_nb)
            
            # Check if c_add covers all would-be-uncovered words
            for idx_nb in would_be_uncovered:
                w_nb = all_words[idx_nb]
                if w_nb not in neighbors[c_add]:
                    return False
            
            # Perform the swap
            for nb in neighbors[c_remove]:
                idx_nb = word_to_idx[nb]
                coverage_count[idx_nb] -= 1
            for nb in neighbors[c_add]:
                idx_nb = word_to_idx[nb]
                coverage_count[idx_nb] += 1
            code.remove(c_remove)
            code.add(c_add)
            return True
        
        # Phase 1: Greedy removal - try to remove codewords one by one
        improved = True
        while improved:
            improved = False
            for c in list(code):
                if try_remove(c):
                    improved = True
                    break  # restart as code changed
        
        # Phase 2: Swap search - try to replace codewords to enable further removals
        # Use multiple passes with randomization to escape local minima
        random.seed(42)
        for _ in range(50):  # Multiple passes
            improved = False
            code_list = list(code)
            random.shuffle(code_list)
            for c_remove in code_list:
                if c_remove not in code:
                    continue
                # Try swapping with candidate words
                # Prioritize words that are "useful" - cover many currently singly-covered words
                candidates = []
                for c_add in all_words:
                    if c_add in code:
                        continue
                    # Quick check: does c_add help?
                    # Count how many neighbors of c_remove are singly covered and also neighbors of c_add
                    help_count = 0
                    for nb in neighbors[c_remove]:
                        idx_nb = word_to_idx[nb]
                        if coverage_count[idx_nb] == 1:
                            if nb in neighbors[c_add]:
                                help_count += 1
                    if help_count > 0:
                        candidates.append((help_count, c_add))
                
                if not candidates:
                    continue
                
                # Sort by help_count descending, try top candidates
                candidates.sort(reverse=True)
                for _, c_add in candidates[:20]:  # Try top 20
                    if try_swap(c_remove, c_add):
                        improved = True
                        break
                if improved:
                    break
            
            # After each swap pass, try removals again
            if improved:
                for c in list(code):
                    if try_remove(c):
                        break
        
        # Final cleanup: one more removal pass
        for c in list(code):
            if try_remove(c):
                pass
        
        # Convert back to list of lists
        result = [list(c) for c in code]
        return result

# EVOLVE-BLOCK-END
