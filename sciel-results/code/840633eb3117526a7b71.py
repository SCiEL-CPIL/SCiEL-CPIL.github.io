"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # For the specific case q=3, n=6, radius=1
    total_words = q ** n
    
    # Precompute all words as numpy array
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    
    # Precompute balls: for each word, which words are within radius
    # For radius=1, ball size = 1 + n*(q-1) = 1 + 6*2 = 13
    # We'll use a coverage count approach
    
    def compute_balls(words, q, n, radius):
        """Compute ball membership: balls[i] is a boolean array of which words are in ball of words[i]"""
        num_words = len(words)
        balls = []
        for i in range(num_words):
            w = words[i]
            if radius == 1:
                # Self
                ball_set = np.zeros(num_words, dtype=bool)
                ball_set[i] = True
                # Distance 1
                for pos in range(n):
                    for new_val in range(q):
                        if new_val != w[pos]:
                            # Find index of this neighbor
                            # We need a mapping from word to index
                            pass
            balls.append(ball_set)
        return balls
    
    # Better approach: use integer representation and precompute neighbor lists
    def word_to_idx(word, q, n):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx
    
    def idx_to_word(idx, q, n):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word
    
    # Precompute neighbors for each word (for radius=1)
    neighbors = [[] for _ in range(total_words)]
    for idx in range(total_words):
        w = idx_to_word(idx, q, n)
        neighbors[idx].append(idx)  # self
        if radius >= 1:
            for pos in range(n):
                orig = w[pos]
                for new_val in range(q):
                    if new_val != orig:
                        w[pos] = new_val
                        nb_idx = word_to_idx(w, q, n)
                        neighbors[idx].append(nb_idx)
                w[pos] = orig
    
    # Convert neighbors to numpy arrays for faster operations
    neighbors_np = [np.array(nb, dtype=np.int32) for nb in neighbors]
    
    def greedy_cover(restart_seed=0, initial_codewords=None):
        """Greedy set cover with coverage counting."""
        if time.time() > deadline:
            return None
        
        rng = random.Random(restart_seed)
        
        # Coverage count: how many codewords cover each word
        coverage = np.zeros(total_words, dtype=np.int32)
        codewords = []
        covered_mask = np.zeros(total_words, dtype=bool)
        
        # If we have initial codewords, add them
        if initial_codewords:
            for cw in initial_codewords:
                if cw not in codewords:
                    codewords.append(cw)
                    for nb in neighbors_np[cw]:
                        coverage[nb] += 1
                        covered_mask[nb] = True
        
        # Uncovered words
        uncovered = set(range(total_words))
        for cw in codewords:
            for nb in neighbors_np[cw]:
                if nb in uncovered:
                    uncovered.discard(nb)
        
        # Greedy: repeatedly pick the codeword that covers the most uncovered words
        while uncovered and time.time() < deadline:
            best_cw = -1
            best_count = 0
            
            # Only consider candidates that could cover uncovered words
            # Optimization: only consider words that are uncovered or near uncovered
            # For now, check all candidates but use fast counting
            
            for candidate in range(total_words):
                # Count how many neighbors of candidate are in uncovered
                count = 0
                for nb in neighbors_np[candidate]:
                    if nb in uncovered:
                        count += 1
                        if count > best_count:
                            # Early termination if we can't beat best
                            pass
                
                if count > best_count:
                    best_count = count
                    best_cw = candidate
                
                # Early termination: if best possible is 1 and we already have 1
                # Actually, let's just continue for now
            
            if best_cw == -1 or best_count == 0:
                break
            
            # Add best_cw
            codewords.append(best_cw)
            for nb in neighbors_np[best_cw]:
                if nb in uncovered:
                    uncovered.discard(nb)
        
        return codewords
    
    def remove_redundant(codewords):
        """Remove redundant codewords iteratively."""
        if time.time() > deadline:
            return codewords
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(codewords)):
                if time.time() > deadline:
                    break
                
                # Try removing codewords[i]
                cw = codewords[i]
                
                # Check if all words in ball(cw) are still covered by other codewords
                redundant = True
                for w in neighbors_np[cw]:
                    # Check if w is covered by any other codeword
                    covered_by_other = False
                    for j, other_cw in enumerate(codewords):
                        if j == i:
                            continue
                        # Check if w is in ball(other_cw)
                        # Instead of checking neighbors_np[other_cw], compute distance
                        # But that's O(n). Let's use a different approach.
                        # Precompute: for each word, which codewords cover it
                        pass
                    
                    # Alternative: maintain a coverage count
                    # Let's use a different strategy: compute coverage counts
                    break
                
                # Better approach: maintain coverage counts
                # Let me redo this with coverage counts
                
                # Actually, let's use a simpler but correct approach:
                # For each word in ball(cw), check if it's within radius of any other codeword
                redundant = True
                for w in neighbors_np[cw]:
                    w_list = idx_to_word(int(w), q, n)
                    covered_by_other = False
                    for j, other_cw in enumerate(codewords):
                        if j == i:
                            continue
                        other_list = idx_to_word(int(other_cw), q, n)
                        # Compute Hamming distance
                        dist = sum(1 for a, b in zip(w_list, other_list) if a != b)
                        if dist <= radius:
                            covered_by_other = True
                            break
                    if not covered_by_other:
                        redundant = False
                        break
                
                if redundant:
                    codewords.pop(i)
                    changed = True
                    break
        
        return codewords
    
    def local_search(codewords):
        """Try to swap codewords to improve coverage or reduce count."""
        if time.time() > deadline:
            return codewords
        
        # Try removing each codeword and see if we can re-cover with fewer
        # This is essentially the same as remove_redundant but more aggressive
        
        # Try: for each pair (i, j), see if removing i and adding some k reduces count
        # This is expensive. Instead, just do multiple passes of redundancy removal.
        
        return remove_redundant(codewords)
    
    # Strategy:
    # 1. Start with the tetracode-based construction (81 words)
    # 2. Aggressively remove redundant codewords
    # 3. Try multiple random restarts of greedy to find better solutions
    # 4. Combine best results
    
    best_result = None
    best_count = float('inf')
    
    # First, try the tetracode construction with redundancy removal
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_codewords = []
    for tail_a in range(3):
        for tail_b in range(3):
            for h in tetracode:
                word = h + [tail_a, tail_b]
                idx = word_to_idx(word, q, n)
                initial_codewords.append(idx)
    
    # Remove redundant from initial
    reduced = remove_redundant(initial_codewords.copy())
    if len(reduced) < best_count:
        best_count = len(reduced)
        best_result = reduced
    
    # Try greedy from scratch with different seeds
    for seed in range(20):
        if time.time() > deadline:
            break
        
        result = greedy_cover(restart_seed=seed, initial_codewords=None)
        if result is not None:
            result = remove_redundant(result)
            if len(result) < best_count:
                best_count = len(result)
                best_result = result
    
    # Try hybrid: start with partial tetracode, then greedy
    for seed in range(10):
        if time.time() > deadline:
            break
        
        # Random subset of tetracode as initial
        rng = random.Random(seed)
        initial = rng.sample(initial_codewords, k=min(len(initial_codewords), rng.randint(30, 60)))
        result = greedy_cover(restart_seed=seed, initial_codewords=initial)
        if result is not None:
            result = remove_redundant(result)
            if len(result) < best_count:
                best_count = len(result)
                best_result = result
    
    # Final local search on best result
    if best_result is not None:
        best_result = local_search(best_result)
    
    # Convert back to lists
    if best_result is None:
        # Fallback: return the tetracode construction
        return [idx_to_word(idx, q, n) for idx in initial_codewords]
    
    return [idx_to_word(idx, q, n) for idx in best_result]
# EVOLVE-BLOCK-END
