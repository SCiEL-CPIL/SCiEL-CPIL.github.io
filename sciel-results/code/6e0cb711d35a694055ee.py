"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np
import random


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    if q == 3 and n == 6 and radius == 1:
        # Specialized fast path for the known problem instance
        all_words = list(itertools.product(range(3), repeat=6))
        total_words = len(all_words)  # 729
        
        # Precompute coverage: for each word, the set of words it covers (itself + distance 1)
        # Use bitsets for efficiency
        coverage = [0] * total_words
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        
        for i, w in enumerate(all_words):
            mask = 1 << i
            coverage[i] = mask
            for pos in range(n):
                for val in range(3):
                    if val != w[pos]:
                        new_word = list(w)
                        new_word[pos] = val
                        j = word_to_idx[tuple(new_word)]
                        coverage[i] |= (1 << j)
        
        # Greedy initialization
        uncovered_mask = (1 << total_words) - 1
        code = []
        cover_count = np.zeros(total_words, dtype=np.int32)
        
        for i in range(total_words):
            cover_count[i] = bin(coverage[i]).count('1')
        
        # Greedy set cover
        while uncovered_mask and time.time() < deadline - 30:
            best_candidate = -1
            best_count = 0
            for i in range(total_words):
                if cover_count[i] > best_count:
                    best_count = cover_count[i]
                    best_candidate = i
            
            if best_candidate == -1 or best_count == 0:
                break
            
            code.append(all_words[best_candidate])
            uncovered_mask &= ~coverage[best_candidate]
            
            # Update cover counts
            for i in range(total_words):
                new_count = bin(coverage[i] & uncovered_mask).count('1')
                cover_count[i] = new_count
        
        # Simulated annealing to improve
        best_code = list(code)
        best_size = len(code)
        current_temp = 10.0
        min_temp = 0.1
        
        while time.time() < deadline - 5 and current_temp > min_temp:
            # Try random swaps
            for _ in range(500):
                if time.time() >= deadline - 5:
                    break
                    
                # Choose a codeword to potentially remove
                idx_remove = random.randint(0, len(code) - 1)
                removed_word = code[idx_remove]
                removed_idx = word_to_idx[removed_word]
                
                # Check if removing it leaves gaps
                new_uncovered = uncovered_mask | coverage[removed_idx]
                new_uncovered_count = bin(new_uncovered).count('1')
                
                if new_uncovered_count > 0:
                    # Find a replacement that covers more uncovered words
                    # Pick a random word that covers some uncovered words
                    candidates = []
                    for i in range(total_words):
                        if not (code and all_words[i] in code):
                            covers_uncovered = bin(coverage[i] & new_uncovered).count('1')
                            if covers_uncovered > 0:
                                candidates.append((i, covers_uncovered))
                    
                    if candidates:
                        # Pick randomly from top candidates
                        candidates.sort(key=lambda x: x[1], reverse=True)
                        top_candidates = candidates[:10]
                        best_idx, best_cov = random.choice(top_candidates)
                        
                        new_code = code[:idx_remove] + code[idx_remove+1:] + [all_words[best_idx]]
                        
                        # Check if this is valid (all words covered)
                        covered_mask = 0
                        for cw in new_code:
                            covered_mask |= coverage[word_to_idx[cw]]
                        
                        if covered_mask == (1 << total_words) - 1:
                            # Valid improvement or same size
                            if len(new_code) < len(code):
                                code = new_code
                                uncovered_mask = 0
                            elif len(new_code) == len(code):
                                # Same size, accept with probability based on temperature
                                if random.random() < np.exp(0 / current_temp):
                                    code = new_code
                        else:
                            # Invalid, skip
                            pass
                else:
                    # Removing it doesn't create gaps, so we can remove it
                    code = code[:idx_remove] + code[idx_remove+1:]
                    uncovered_mask = new_uncovered
                
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = list(code)
            
            # Cool down
            current_temp *= 0.95
        
        # Final check: ensure all words are covered
        covered_mask = 0
        for cw in best_code:
            covered_mask |= coverage[word_to_idx[cw]]
        
        if covered_mask != (1 << total_words) - 1:
            # Invalid, use greedy result
            return [list(cw) for cw in code]
        
        return [list(cw) for cw in best_code]
    
    # General case: fallback to direct product or simple greedy
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
# EVOLVE-BLOCK-END
