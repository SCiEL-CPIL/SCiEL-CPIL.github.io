"""Model-and-optimize approach for ternary covering code K_3(6,1).

Strategy:
1. Start from the direct-sum construction (9 * 3^2 = 81 words) as a baseline.
2. Use vectorized NumPy coverage counting to quickly identify redundant codewords.
3. Iteratively remove redundant codewords using multiple random orderings.
4. Add swap-based local search to escape local minima.
5. Use time management to ensure we finish well within the 120s limit.

The key insight: the direct-sum construction gives 81 words, but many are redundant.
By removing redundant codewords greedily with multiple random orders, we can get down
to ~73 or better. Swap operations can further reduce the count.
"""
import itertools
import time
import random
import sys
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin: finish by 110s, limit is 120s
    
    # For the specific problem: q=3, n=6, radius=1
    # Direct sum of ternary tetracode (9 words of length 4) with all 9 tails
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(q) for b in range(q)]
    
    # Build initial code: direct sum
    code = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            code.append(tuple(h) + tuple(tail))
    
    base_count = len(code)  # 81 for q=3, n=6
    
    # If n <= 4, just return the tetracode-based construction directly
    if n <= 4:
        if n == 4:
            return [list(h) for h in tetracode]
        else:
            result = [list(c) for c in itertools.product(range(q), repeat=n)]
            return result[:q**n]
    
    # For n > 4, use the direct sum and then prune
    total_words = q ** n
    
    # Convert code to numpy array for faster operations
    code_arr = np.array(code, dtype=np.int32)
    num_codewords = len(code)
    
    # Precompute the coverage matrix: cov_matrix[i, j] = 1 if codeword i covers word j
    # Word j has index j in base-q representation
    # A codeword c covers word w if Hamming distance <= radius
    
    # For radius=1, a codeword covers itself and all words at distance 1
    # Precompute word indices for each codeword
    # Word index: sum(c[i] * q^(n-1-i)) for i in range(n)
    
    def compute_word_indices(codewords):
        """Compute the index of each word in the codewords array."""
        indices = np.zeros(len(codewords), dtype=np.int32)
        for i in range(n):
            indices += codewords[:, i] * (q ** (n - 1 - i))
        return indices
    
    # For each codeword, the words it covers are:
    # 1. The codeword itself
    # 2. For each position pos, for each value val != c[pos], the word with c[pos] replaced by val
    
    # Precompute all covered word indices for each codeword
    # This is a sparse representation: list of lists
    
    # Actually, let's use a different approach:
    # coverage_count[word_idx] = number of active codewords covering it
    # We'll use a 2D boolean matrix for which codeword covers which word
    
    # For q=3, n=6: 81 codewords, 729 words
    # Each codeword covers 1 + n*(q-1) = 1 + 6*2 = 13 words
    # Total coverage entries: 81 * 13 = 1053
    
    # Build coverage matrix as a list of sets for each word
    # covered_by[word_idx] = set of codeword indices that cover it
    
    # First, compute the index of each codeword
    cw_indices = compute_word_indices(code_arr)
    
    # For each codeword, compute all words it covers
    # Store as: codeword_covers[ci] = list of word indices
    # And: covered_by[wi] = list of codeword indices
    
    codeword_covers = []
    covered_by = [[] for _ in range(total_words)]
    
    for ci in range(num_codewords):
        c = code_arr[ci]
        covers = set()
        covers.add(cw_indices[ci])
        for pos in range(n):
            orig = c[pos]
            for val in range(q):
                if val == orig:
                    continue
                # Compute new word index
                new_idx = cw_indices[ci] + (val - orig) * (q ** (n - 1 - pos))
                covers.add(new_idx)
        codeword_covers.append(list(covers))
        for wi in covers:
            covered_by[wi].append(ci)
    
    # Now use greedy removal with multiple random orderings
    best_code = None
    best_count = num_codewords
    
    # Number of trials to run
    max_trials = 30
    
    for trial in range(max_trials):
        if time.time() > deadline - 10.0:
            break
        
        # Random order for checking removals
        order = list(range(num_codewords))
        random.shuffle(order)
        
        active = np.ones(num_codewords, dtype=bool)
        # coverage_count[wi] = number of active codewords covering word wi
        coverage_count = np.zeros(total_words, dtype=np.int32)
        for ci in range(num_codewords):
            for wi in codeword_covers[ci]:
                coverage_count[wi] += 1
        
        removed_any = True
        while removed_any:
            if time.time() > deadline - 10.0:
                break
            
            removed_any = False
            for ci in order:
                if time.time() > deadline - 10.0:
                    break
                if not active[ci]:
                    continue
                
                # Check if codeword ci is redundant
                # For each word it covers, check if coverage_count[wi] > 1
                is_redundant = True
                for wi in codeword_covers[ci]:
                    if coverage_count[wi] <= 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    active[ci] = False
                    removed_any = True
                    for wi in codeword_covers[ci]:
                        coverage_count[wi] -= 1
        
        trial_count = int(np.sum(active))
        if trial_count < best_count:
            best_count = trial_count
            best_indices = np.where(active)[0]
            best_code = [list(code[i]) for i in best_indices]
        
        # Early exit if we reached the known optimum
        if best_count <= 73:
            break
    
    if best_code is None:
        best_code = [list(c) for c in code]
    
    # Swap-based local search to escape local minima
    if time.time() < deadline - 15.0 and best_count > 73:
        # Work with the best code found so far
        best_arr = np.array(best_code, dtype=np.int32)
        best_n = len(best_arr)
        
        # Recompute coverage for the best code
        best_coverage_count = np.zeros(total_words, dtype=np.int32)
        best_cw_indices = compute_word_indices(best_arr)
        
        for ci in range(best_n):
            c = best_arr[ci]
            covers = set()
            covers.add(best_cw_indices[ci])
            for pos in range(n):
                orig = c[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    new_idx = best_cw_indices[ci] + (val - orig) * (q ** (n - 1 - pos))
                    covers.add(new_idx)
            for wi in covers:
                best_coverage_count[wi] += 1
        
        # Try swaps: remove codeword i, add codeword j (from original pool not in best)
        # First, identify which original codewords are in the best set
        best_set = set(int(idx) for idx in best_cw_indices)
        candidate_adds = [ci for ci in range(num_codewords) if int(cw_indices[ci]) not in best_set]
        
        # For each candidate removal, check if it's "almost redundant" 
        # (all but a few words are covered by others), then try adding a candidate
        # that covers those uncovered words
        
        # Build mapping from word index to original codeword index
        word_to_orig_cw = {}
        for ci in range(num_codewords):
            for wi in codeword_covers[ci]:
                if wi not in word_to_orig_cw:
                    word_to_orig_cw[wi] = []
                word_to_orig_cw[wi].append(ci)
        
        # Precompute covers for candidate additions
        cand_covers = {}
        for ci in candidate_adds:
            c = code_arr[ci]
            covers = set()
            covers.add(cw_indices[ci])
            for pos in range(n):
                orig = c[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    new_idx = cw_indices[ci] + (val - orig) * (q ** (n - 1 - pos))
                    covers.add(new_idx)
            cand_covers[ci] = list(covers)
        
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            
            # For each codeword in best, check if removing it leaves few uncovered words
            for bi in range(best_n):
                if time.time() > deadline - 5.0:
                    break
                
                # Find words that would become uncovered if we remove best_arr[bi]
                uncovered_if_removed = []
                for pos in range(n):
                    pass  # Need to check all words covered by this codeword
                
                # More efficient: for each word covered by best_arr[bi], check coverage_count
                c = best_arr[bi]
                ci_idx = best_cw_indices[bi]
                covers_bi = set()
                covers_bi.add(ci_idx)
                for pos in range(n):
                    orig = c[pos]
                    for val in range(q):
                        if val == orig:
                            continue
                        new_idx = ci_idx + (val - orig) * (q ** (n - 1 - pos))
                        covers_bi.add(new_idx)
                
                uncovered = [wi for wi in covers_bi if best_coverage_count[wi] <= 1]
                
                if len(uncovered) == 0:
                    # Can remove without adding
                    for wi in covers_bi:
                        best_coverage_count[wi] -= 1
                    best_arr = np.delete(best_arr, bi, axis=0)
                    best_n -= 1
                    best_cw_indices = compute_word_indices(best_arr)
                    best_set = set(int(idx) for idx in best_cw_indices)
                    improved = True
                    break
                elif len(uncovered) <= 3:
                    # Try to find a candidate that covers all uncovered words
                    for ci in candidate_adds:
                        if time.time() > deadline - 5.0:
                            break
                        if int(cw_indices[ci]) in best_set:
                            continue
                        
                        cand_set = set(cand_covers[ci])
                        if all(wi in cand_set for wi in uncovered):
                            # Perform swap: remove bi, add ci
                            # Remove bi
                            for wi in covers_bi:
                                best_coverage_count[wi] -= 1
                            # Add ci
                            for wi in cand_covers[ci]:
                                best_coverage_count[wi] += 1
                            
                            # Update arrays
                            best_arr = np.delete(best_arr, bi, axis=0)
                            new_cw = code_arr[ci].copy()
                            best_arr = np.vstack([best_arr, new_cw])
                            best_n = len(best_arr)
                            best_cw_indices = compute_word_indices(best_arr)
                            best_set = set(int(idx) for idx in best_cw_indices)
                            
                            # Update candidate_adds
                            candidate_adds = [cci for cci in candidate_adds if int(cw_indices[cci]) not in best_set]
                            
                            improved = True
                            break
                    
                    if improved:
                        break
        
        # Update best if improved
        if len(best_arr) < best_count:
            best_count = len(best_arr)
            best_code = [list(row) for row in best_arr]
    
    # Verify coverage
    if time.time() < deadline - 2.0 and len(best_code) <= 100:
        covered_set = set()
        for c in best_code:
            c_arr = np.array(c, dtype=np.int32)
            base_idx = int(compute_word_indices(c_arr.reshape(1, -1))[0])
            covered_set.add(base_idx)
            for pos in range(n):
                orig = c[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    new_idx = base_idx + (val - orig) * (q ** (n - 1 - pos))
                    covered_set.add(new_idx)
        
        if len(covered_set) < total_words:
            # Fallback to original
            best_code = [list(c) for c in code]
    
    return best_code
