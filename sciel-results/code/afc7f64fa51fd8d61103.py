"""Improved covering code search for K_3(6,1)."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    """Construct a covering code for {0..q-1}^n with covering radius 'radius'.
    
    Uses a genetic/local search hybrid to minimize the number of codewords.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Generate all possible words
    all_words = list(itertools.product(range(q), repeat=n))
    num_words = len(all_words)
    
    # Precompute neighbors: for each word, list of words within Hamming distance <= radius
    # For efficiency, we'll compute coverage on the fly using bit operations or direct distance
    
    # Helper: convert word to integer index
    def word_to_idx(w):
        idx = 0
        for v in w:
            idx = idx * q + v
        return idx
    
    def idx_to_word(idx):
        w = []
        for _ in range(n):
            w.append(idx % q)
            idx //= q
        return w[::-1]
    
    # Precompute all neighbors for each word (words within distance <= radius)
    # This is expensive but done once
    neighbors = []
    for i, w in enumerate(all_words):
        neigh = []
        for j, w2 in enumerate(all_words):
            # Compute Hamming distance
            dist = sum(1 for a, b in zip(w, w2) if a != b)
            if dist <= radius:
                neigh.append(j)
        neighbors.append(neigh)
    
    # Build initial code using greedy approach
    def greedy_initial_code():
        uncovered = set(range(num_words))
        code = []
        # Sort words by some heuristic - start with all words, pick greedily
        # For speed, use a simple greedy: pick word that covers most uncovered
        # To make it faster, use random restarts with different orderings
        
        # Use a faster greedy: iterate through words in random order, add if it covers new words
        # Actually, let's use a smarter greedy: at each step, pick the word that maximizes coverage of uncovered set
        # This is O(num_words^2) per step which is too slow for 729 words if done naively
        # Let's use a randomized greedy with early stopping
        
        random.shuffle(all_words)
        uncovered = [True] * num_words
        uncovered_count = num_words
        code = []
        
        for w_idx, w in enumerate(all_words):
            if uncovered_count <= 0:
                break
            # Check if this word covers any uncovered words
            covered_new = 0
            for n_idx in neighbors[w_idx]:
                if uncovered[n_idx]:
                    covered_new += 1
            
            if covered_new > 0:
                code.append(w)
                for n_idx in neighbors[w_idx]:
                    if uncovered[n_idx]:
                        uncovered[n_idx] = False
                        uncovered_count -= 1
                if uncovered_count <= 0:
                    break
        
        return code
    
    # Try multiple random greedy initializations and keep the best
    best_code = None
    best_size = num_words + 1
    
    for _ in range(50):
        if time.time() > deadline - 5:
            break
        code = greedy_initial_code()
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
    
    if best_code is None:
        best_code = greedy_initial_code()
        best_size = len(best_code)
    
    # Local search: try to remove redundant words or swap words
    def is_valid_code(code):
        """Check if code covers all words."""
        covered = [False] * num_words
        for w in code:
            w_idx = word_to_idx(w)
            for n_idx in neighbors[w_idx]:
                covered[n_idx] = True
        return all(covered)
    
    def coverage_count(code):
        """Return list where count[i] = number of codewords covering word i."""
        counts = [0] * num_words
        for w in code:
            w_idx = word_to_idx(w)
            for n_idx in neighbors[w_idx]:
                counts[n_idx] += 1
        return counts
    
    def try_remove_redundant(code):
        """Try to remove redundant codewords. Return new code if successful."""
        counts = coverage_count(code)
        # Find codewords that only cover words with count == 1
        for i, w in enumerate(code):
            w_idx = word_to_idx(w)
            is_redundant = True
            for n_idx in neighbors[w_idx]:
                if counts[n_idx] == 1:
                    is_redundant = False
                    break
            if is_redundant:
                new_code = code[:i] + code[i+1:]
                return new_code
        return None
    
    def try_swap(code, max_attempts=100):
        """Try to replace a codeword with another to maintain coverage while potentially enabling removal later."""
        counts = coverage_count(code)
        # Find words with count == 1 (critical words)
        critical = [i for i in range(num_words) if counts[i] == 1]
        if not critical:
            return None
        
        # Try removing a codeword and see if we can add a better one
        for i in range(len(code)):
            w_idx = word_to_idx(code[i])
            # Words that would become uncovered if we remove code[i]
            would_lose = [n_idx for n_idx in neighbors[w_idx] if counts[n_idx] == 1]
            if not would_lose:
                # Can remove this word
                return code[:i] + code[i+1:]
        
        # Try swapping: remove one word, add another that covers the lost words
        # This is complex; for now, just try removals
        return None
    
    # Iterative improvement
    current_code = best_code[:]
    current_size = best_size
    
    improvement_count = 0
    while time.time() < deadline - 10 and improvement_count < 1000:
        # Try removing redundant words
        new_code = try_remove_redundant(current_code)
        if new_code is not None:
            current_code = new_code
            current_size = len(current_code)
            improvement_count = 0
            continue
        
        # Try random local moves
        if random.random() < 0.5:
            # Try adding a random word and then removing redundants
            if len(current_code) < num_words:
                candidate = random.choice(all_words)
                if candidate not in current_code:
                    temp_code = current_code + [candidate]
                    # Try to remove redundants from temp_code
                    removed = True
                    while removed:
                        removed = False
                        for i in range(len(temp_code)):
                            counts = coverage_count(temp_code)
                            w_idx = word_to_idx(temp_code[i])
                            is_redundant = True
                            for n_idx in neighbors[w_idx]:
                                if counts[n_idx] == 1:
                                    is_redundant = False
                                    break
                            if is_redundant:
                                temp_code = temp_code[:i] + temp_code[i+1:]
                                removed = True
                                break
                        if len(temp_code) < current_size:
                            current_code = temp_code
                            current_size = len(current_code)
                            improvement_count = 0
        else:
            # Try a random restart of local search
            pass
        
        improvement_count += 1
    
    # Final verification
    if not is_valid_code(current_code):
        # Fall back to a valid code
        current_code = greedy_initial_code()
    
    return current_code

# EVOLVE-BLOCK-END
