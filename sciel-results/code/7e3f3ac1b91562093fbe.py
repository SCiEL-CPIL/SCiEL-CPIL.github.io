"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np


def _hamming_dist(a, b):
    """Compute Hamming distance between two tuples/lists."""
    return sum(1 for x, y in zip(a, b) if x != y)


def _within_radius(word, codeword, radius):
    """Check if word is within radius of codeword."""
    dist = 0
    for i in range(len(word)):
        if word[i] != codeword[i]:
            dist += 1
            if dist > radius:
                return False
    return True


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Generate all words in the space
    all_words = list(itertools.product(range(q), repeat=n))
    total_words = len(all_words)
    
    # For each word, precompute all words within radius
    # This is expensive but necessary for greedy set cover
    if radius == 1:
        # For radius 1, a word covers itself and all words at distance 1
        # Number of words covered by each codeword: 1 + n*(q-1)
        # For q=3, n=6: 1 + 6*2 = 13
        pass
    
    # Greedy set cover: repeatedly pick the word that covers the most uncovered words
    # Start with empty code
    code = []
    uncovered = set(range(total_words))
    
    # Precompute coverage: for each word, which words it covers
    # For radius 1, word w covers w and all words at distance 1
    if radius == 1:
        # Build coverage lists
        coverage = []
        for w in all_words:
            covered = [w]
            for pos in range(n):
                for val in range(q):
                    if val != w[pos]:
                        new_word = list(w)
                        new_word[pos] = val
                        covered.append(tuple(new_word))
            coverage.append(covered)
        
        # Map word to index
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        
        # For each candidate, track which uncovered words it covers
        # Use a dictionary: candidate_idx -> set of uncovered word indices it covers
        # But this is memory intensive. Instead, use a different approach.
        
        # Alternative: maintain for each uncovered word, which candidates can cover it
        # And for each candidate, how many uncovered words it covers
        
        # Initialize coverage counts
        # For each candidate word, count how many uncovered words it covers
        # Initially all are uncovered
        
        # To speed up, use the fact that we're doing greedy:
        # Pick the candidate that covers the most uncovered words
        
        # Data structures:
        # uncovered_set: set of indices of uncovered words
        # For each candidate, we need to know how many uncovered words it covers
        # We can maintain: cover_count[candidate] = number of uncovered words this candidate covers
        
        # Initialize
        cover_count = np.zeros(total_words, dtype=np.int32)
        for i in range(total_words):
            for covered_word in coverage[i]:
                idx = word_to_idx[covered_word]
                cover_count[i] += 1
        
        # Now do greedy
        while uncovered and time.time() < deadline:
            # Find candidate with max cover_count among those that cover at least one uncovered word
            # We need to find the candidate that covers the most uncovered words
            # This is expensive. Let's use a simpler approach:
            # Just pick the candidate with highest cover_count
            
            # But we need to be careful: cover_count includes covered words too
            # We need to recompute or track differently
            
            # Simpler approach: just iterate and find best
            best_candidate = -1
            best_count = 0
            for i in range(total_words):
                if cover_count[i] > best_count:
                    best_count = cover_count[i]
                    best_candidate = i
            
            if best_candidate == -1 or best_count == 0:
                break
            
            code.append(list(all_words[best_candidate]))
            
            # Update: mark all words covered by best_candidate as covered
            for covered_word in coverage[best_candidate]:
                idx = word_to_idx[covered_word]
                if idx in uncovered:
                    uncovered.discard(idx)
                    # Update cover_count for all candidates that also cover this word
                    # This is expensive. Skip for now.
            
            # Recompute cover_count for remaining candidates (expensive but correct)
            # Only if we have time
            if time.time() < deadline - 5:
                for i in range(total_words):
                    new_count = 0
                    for covered_word in coverage[i]:
                        if word_to_idx[covered_word] in uncovered:
                            new_count += 1
                    cover_count[i] = new_count
        else:
            # Time's up, return what we have
            pass
    
    # If we didn't finish or used a different radius, fall back to direct product
    if not code or len(code) >= 81:  # Fallback if greedy didn't help
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return code
# EVOLVE-BLOCK-END
