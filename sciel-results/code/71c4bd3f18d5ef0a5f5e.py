import random
import numpy as np

def construct(q, n, radius):
    # Generate all words
    total_words = q ** n
    all_words = []
    for i in range(total_words):
        w = []
        x = i
        for _ in range(n):
            w.append(x % q)
            x //= q
        all_words.append(tuple(w))
    
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute coverage lists
    coverage = []
    for i, w in enumerate(all_words):
        covered = [i]
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                nw = list(w)
                nw[pos] = val
                nw = tuple(nw)
                j = word_to_idx[nw]
                covered.append(j)
        coverage.append(covered)
        
    # cov_matrix for greedy phase speed
    cov_matrix = np.zeros((total_words, total_words), dtype=bool)
    for i in range(total_words):
        cov_matrix[i, coverage[i]] = True
        
    best_solution = None
    best_size = total_words
    
    num_iterations = 50
    
    for _ in range(num_iterations):
        # 1. Greedy Initialization
        uncovered = np.ones(total_words, dtype=bool)
        codewords = []
        is_codeword = np.zeros(total_words, dtype=bool)
        covered_counts = np.zeros(total_words, dtype=int)
        
        while np.any(uncovered):
            counts = cov_matrix @ uncovered
            noise = np.random.uniform(0, 0.1, size=total_words)
            scores = counts + noise
            best_c = np.argmax(scores)
            
            codewords.append(best_c)
            is_codeword[best_c] = True
            covered = coverage[best_c]
            covered_counts[covered] += 1
            uncovered[covered] = False
            
        # 2. Local Search: Reduction and Swaps
        # We will loop until no improvement
        improved = True
        while improved:
            improved = False
            # Shuffle order of codewords to escape local minima
            indices = list(range(len(codewords)))
            random.shuffle(indices)
            
            for idx in indices:
                c = codewords[idx]
                covered_c = coverage[c]
                
                # Check if c is redundant
                # Redundant if min(covered_counts[covered_c]) > 1
                if np.min(covered_counts[covered_c]) > 1:
                    # Remove c
                    covered_counts[covered_c] -= 1
                    is_codeword[c] = False
                    codewords.pop(idx)
                    improved = True
                    break # Restart loop with new set
            
            if improved:
                continue
            
            # If no removal, try swaps
            # We look for a codeword c that is NOT redundant, but can be swapped with w
            # such that the new set allows further reduction or is just a different state.
            # Actually, a direct swap that maintains validity doesn't reduce size.
            # We need a swap that *enables* a reduction.
            # Or, we can just do random valid swaps to explore the space, then reduce.
            
            # Heuristic: 
            # Pick a random codeword c.
            # Find words covered ONLY by c (critical words).
            # If none, c is redundant (should have been caught above, but maybe due to order?).
            # If some, find a word w that covers all critical words.
            # If found, swap c with w.
            
            # Let's try a few random swaps
            for _ in range(10):
                if not codewords:
                    break
                idx = random.randint(0, len(codewords) - 1)
                c = codewords[idx]
                covered_c = coverage[c]
                
                # Find critical words for c
                critical = [j for j in covered_c if covered_counts[j] == 1]
                
                if not critical:
                    # c is redundant
                    covered_counts[covered_c] -= 1
                    is_codeword[c] = False
                    codewords.pop(idx)
                    improved = True
                    break
                
                # Need to cover 'critical' with a new word w
                # w must cover all words in 'critical'
                # Pick a random critical word j*
                j_star = random.choice(critical)
                candidates = coverage[j_star]
                random.shuffle(candidates)
                
                found_swap = False
                for w in candidates:
                    if is_codeword[w]:
                        continue
                    # Check if w covers all critical words
                    valid = True
                    for cr in critical:
                        if cr not in coverage[w]:
                            valid = False
                            break
                    if valid:
                        # Swap c with w
                        covered_counts[covered_c] -= 1
                        is_codeword[c] = False
                        
                        covered_w = coverage[w]
                        covered_counts[covered_w] += 1
                        is_codeword[w] = True
                        
                        codewords[idx] = w
                        improved = True
                        found_swap = True
                        break
                
                if found_swap:
                    break # Restart reduction loop
            
        if len(codewords) < best_size:
            best_size = len(codewords)
            best_solution = list(codewords)
            
    if best_solution is None:
        best_solution = codewords
        
    return [list(all_words[i]) for i in best_solution]
