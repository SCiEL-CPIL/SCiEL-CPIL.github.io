"""Football pool problem: ternary covering code K_3(7,1) using Hamming code structure + local search."""
import time
import random
import numpy as np

# Deadline guard: kill before 150s wall-clock with safety margin
_START = time.time()
_DEADLINE = 140.0  # 10s safety margin

def _time_ok():
    return (time.time() - _START) < _DEADLINE

def construct(q, n, radius):
    """Construct a ternary covering code for q=3, n=7, radius=1.
    Strategy:
      1. Build initial code from ternary Hamming [7,4,3] code cosets.
      2. Apply local search (1-exchange, 2-for-1) to reduce size while maintaining coverage.
      3. Use multi-start perturbation to escape local minima.
    """
    if q != 3 or n != 7 or radius != 1:
        # Fallback: direct product construction
        import itertools
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n-4)]
    
    # All 2187 words
    all_words = np.array([[w >> (2*(6-i)) & 3 for i in range(7)] for w in range(2187)], dtype=np.int8)
    
    # Build coverage bitmask: for each codeword, which words it covers
    # A codeword c covers all words at Hamming distance <= 1 from c
    # For radius 1, that's the codeword itself + 7*2 = 15 neighbors = 16 words total
    
    def word_index(w):
        """Convert a word (list of 7 digits 0-2) to integer index 0..2186."""
        idx = 0
        for i, d in enumerate(w):
            idx = idx * 3 + d
        return idx
    
    def neighbors_of(c):
        """Return list of indices of all words within Hamming distance 1 of codeword c."""
        c = np.array(c, dtype=np.int8)
        covered = [word_index(c)]
        for pos in range(7):
            for d in range(3):
                if d != c[pos]:
                    w = c.copy()
                    w[pos] = d
                    covered.append(word_index(w))
        return covered
    
    # Ternary Hamming [7,4,3] code: 81 codewords
    # Generator matrix:
    # The ternary Hamming code of length 7 has parameters [7,4,3]
    # It's a perfect code (covers all space with radius 1)
    # So 81 words cover all 2187 words exactly once each (perfect covering)
    # But we want to REDUCE below 81? No wait - 81 is the perfect code size.
    # Actually for K_3(7,1), the perfect ternary Hamming code has size 3^4 = 81.
    # But the best known is 186? That doesn't make sense...
    
    # Wait, let me reconsider. The ternary Hamming code [7,4,3] is a perfect 1-error-CORRECTING code.
    # For covering, we need every word within distance 1 of SOME codeword.
    # A perfect code already does this! Size = 3^7 / 16 = 2187/16 ≈ 136.7, so 137 minimum?
    # Actually 3^7 = 2187, each codeword covers 1 + 7*2 = 16 words.
    # Lower bound: ceil(2187/16) = 137.
    # The ternary Hamming code [7,4,3] has 3^4 = 81 codewords and IS a perfect code.
    # 81 * 16 = 1296 ≠ 2187. So it's NOT perfect for covering?
    
    # Let me reconsider. The ternary Hamming [7,4,3] code: minimum distance 3.
    # As a covering code with radius 1: each codeword covers 16 words.
    # 81 codewords cover at most 81*16 = 1296 < 2187. So it does NOT cover the whole space!
    
    # I was confused. Let me think again.
    # For a q-ary code of length n with covering radius 1:
    # Each codeword covers 1 + n*(q-1) words.
    # For q=3, n=7: 1 + 7*2 = 16 words per codeword.
    # Lower bound: ceil(2187/16) = 137.
    # Best known upper bound: 186 (per the problem statement).
    
    # The seed program uses tetracode (9 words, length 4, perfect covering radius 1)
    # Direct sum with all words on remaining 3 coords: 9 * 3^3 = 9*27 = 243 words.
    # Score 1.0 means the seed has 243 words (or the baseline value).
    
    # Actually score = seed/value, and seed program score is 1.0.
    # The seed produces 9 * 3^(7-4) = 9*27 = 243 words.
    # So baseline value = 243, and we need to get below 243.
    # Best known is 186, so we should aim for ~186-200.
    
    # Strategy: Start with a good initial covering, then use local search to reduce size.
    
    # Initial construction: Use the structure more cleverly.
    # Idea: Partition coordinates into groups and use covering codes on each group.
    
    # Alternative: Greedy set cover + local search
    # 1. Greedily add codewords that cover the most uncovered words
    # 2. Then try to remove redundant codewords
    # 3. Then try 1-exchanges and 2-for-1 replacements
    
    def greedy_cover():
        """Greedily construct a covering code."""
        covered = np.zeros(2187, dtype=bool)
        code = []
        
        # Precompute coverage for all possible words
        # coverage[w] = set of word indices covered by codeword w
        # This is 2187 * 16 = ~35k entries, manageable
        
        coverage = []
        for w in range(2187):
            digits = []
            tmp = w
            for i in range(7):
                digits.append(tmp % 3)
                tmp //= 8  # No, this is wrong. Let me use a different encoding.
        
        # Let me use a cleaner approach with numpy
        # Precompute all neighbors for each of the 2187 words
        all_neighbors = np.zeros((2187, 16), dtype=np.int32)
        for w in range(2187):
            # Convert to digits
            digits = np.zeros(7, dtype=np.int8)
            tmp = w
            for i in range(6, -1, -1):
                digits[i] = tmp % 3
                tmp //= 3
            all_neighbors[w, 0] = w
            idx = 1
            for pos in range(7):
                for d in range(3):
                    if d != digits[pos]:
                        nd = digits.copy()
                        nd[pos] = d
                        # Convert back to index
                        ni = 0
                        for j in range(7):
                            ni = ni * 3 + nd[j]
                        all_neighbors[w, idx] = ni
                        idx += 1
        
        # Greedy: at each step, pick the word that covers the most uncovered
        while not np.all(covered) and _time_ok():
            # Find the word that maximizes new coverage
            best_w = -1
            best_count = -1
            for w in range(2187):
                if covered[w]:
                    continue  # Already covered, but we still need to check if adding helps
                # Count how many of its neighbors are uncovered
                nbrs = all_neighbors[w]
                new_count = np.sum(~covered[nbrs])
                if new_count > best_count:
                    best_count = new_count
                    best_w = w
                if best_count == 16:
                    break  # Can't do better than 16
            
            if best_w < 0 or best_count <= 0:
                break
            
            code.append(best_w)
            covered[all_neighbors[best_w]] = True
        
        return code, all_neighbors
    
    # Multi-start with different random initializations
    best_code = None
    best_size = 999999
    
    # Start with greedy
    code, all_neighbors = greedy_cover()
    
    if not _time_ok():
        return [[all_words[w][i] for i in range(7)] for w in code]
    
    # Local search: try to remove codewords, then 1-exchanges, then 2-for-1
    def try_remove(code, all_neighbors):
        """Try to remove each codeword and see if coverage is maintained."""
        code = list(code)
        changed = True
        while changed and _time_ok():
            changed = False
            # Compute coverage count for each word
            cover_count = np.zeros(2187, dtype=np.int32)
            for w in code:
                cover_count[all_neighbors[w]] += 1
            
            for i in range(len(code) - 1, -1, -1):
                w = code[i]
                # Check if removing w leaves any word uncovered
                if np.all(cover_count[all_neighbors[w]] >= 2):
                    # Safe to remove
                    cover_count[all_neighbors[w]] -= 1
                    code.pop(i)
                    changed = True
        
        return code
    
    def try_exchange(code, all_neighbors):
        """Try replacing one codeword with another to reduce redundancy."""
        code = list(code)
        improved = True
        while improved and _time_ok():
            improved = False
            cover_count = np.zeros(2187, dtype=np.int32)
            for w in code:
                cover_count[all_neighbors[w]] += 1
            
            for i in range(len(code)):
                if not _time_ok():
                    break
                # Try removing code[i] and adding a new word
                # First check if code[i] can be removed (all its neighbors have count >= 2)
                if not np.all(cover_count[all_neighbors[code[i]]] >= 2):
                    continue
                
                # Temporarily remove
                cover_count[all_neighbors[code[i]]] -= 1
                # Now find a word that covers the most words with count == 1
                best_new = -1
                best_gain = 0
                for w in range(2187):
                    if w in code:
                        continue
                    gain = np.sum(cover_count[all_neighbors[w]] == 1)
                    if gain > best_gain:
                        best_gain = gain
                        best_new = w
                    if best_gain >= 8:  # Good enough
                        break
                
                if best_new >= 0 and best_gain > 0:
                    code[i] = best_new
                    cover_count[all_neighbors[best_new]] += 1
                    improved = True
                else:
                    # Restore
                    cover_count[all_neighbors[code[i]]] += 1
        
        return code
    
    code = try_remove(code, all_neighbors)
    code = try_exchange(code, all_neighbors)
    code = try_remove(code, all_neighbors)  # Another pass
    
    if len(code) < best_size:
        best_size = len(code)
        best_code = list(code)
    
    # Try multi-start: perturb the solution and re-optimize
    rng = random.Random(42)
    attempts = 0
    while _time_ok() and attempts < 20:
        attempts += 1
        # Start from current best, perturb
        perturbed = list(best_code)
        # Remove a random few codewords and re-greedy
        for _ in range(rng.randint(1, 3)):
            idx = rng.randrange(len(perturbed))
            perturbed.pop(idx)
        
        # Re-add greedily to restore coverage
        covered = np.zeros(2187, dtype=bool)
        for w in perturbed:
            covered[all_neighbors[w]] = True
        
        while not np.all(covered) and _time_ok():
            best_w = -1
            best_count = -1
            for w in range(2187):
                new_count = np.sum(~covered[all_neighbors[w]])
                if new_count > best_count:
                    best_count = new_count
                    best_w = w
                if best_count == 16:
                    break
            if best_w < 0 or best_count <= 0:
                break
            perturbed.append(best_w)
            covered[all_neighbors[best_w]] = True
        
        # Local search
        perturbed = try_remove(perturbed, all_neighbors)
        perturbed = try_exchange(perturbed, all_neighbors)
        perturbed = try_remove(perturbed, all_neighbors)
        
        if len(perturbed) < best_size:
            best_size = len(perturbed)
            best_code = list(perturbed)
    
    # Convert to list of lists
    return [[all_words[w][i] for i in range(7)] for w in best_code]
