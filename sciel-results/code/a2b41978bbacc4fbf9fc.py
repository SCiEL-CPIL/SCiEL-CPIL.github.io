"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit

    # Handle trivial cases
    if n == 0:
        return [[]]
    if radius >= n:
        return [[0] * n]
    if q == 2:
        # Binary case: use simple construction
        if n <= 4:
            return [list(c) for c in itertools.product(range(2), repeat=n)]
        # For binary, use a simple covering code construction
        # Start with all-zero and add words to cover
        words = [[0] * n]
        covered = {tuple([0] * n)}
        all_words = list(itertools.product(range(2), repeat=n))
        for w in all_words:
            if tuple(w) not in covered:
                words.append(list(w))
                # Add all words within radius
                for delta in itertools.product(range(2), repeat=n):
                    new_w = tuple((w[i] + delta[i]) % 2 for i in range(n))
                    if sum(delta[i] for i in range(n)) <= radius:
                        covered.add(new_w)
        return words

    # For q=3, n=8, radius=1: use the known optimal construction approach
    # Strategy: Start with a good initial code, then try to remove redundant words
    
    # Initial construction: use the direct sum approach but smarter
    # The ternary tetracode is a perfect (3,4,1) code with 9 words
    # For n=8, we can use two copies: 9*9 = 81 words? No, that's not right.
    # Actually, for covering radius 1, each word covers 1 + n*(q-1) = 1+8*2=17 words
    # Total space = 3^8 = 6561, so minimum words >= 6561/17 ≈ 386
    # Known optimal is 486.
    
    # Let's use a greedy approach with local optimization
    
    # Start with a reasonable initial code
    # Use the product of two tetracodes as initial guess: 9*9 = 81 words
    # But this might not cover everything for n=8, radius=1
    
    # Better: use a random initial code and then optimize
    
    # For efficiency, let's use a structured initial code
    # The direct sum gives 9 * 3^(8-4) = 9*81 = 729 words (the seed approach)
    # We want to do better.
    
    # Approach: Start with 729 words, then greedily remove redundant ones
    
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    # Generate initial code using direct sum
    initial_words = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            initial_words.append(h + list(tail))
    
    # Convert to numpy for faster distance calculations
    words = [np.array(w) for w in initial_words]
    
    # Precompute all words in the space (might be too large for n=8, q=3: 6561 words)
    # For q=3, n=8: 6561 words, manageable
    all_space_words = [np.array(list(c)) for c in itertools.product(range(q), repeat=n)]
    
    # For each word in the code, compute which space words it covers
    # A word covers all words within Hamming distance <= radius
    def compute_coverage(word, all_words, q, n, radius):
        covered = set()
        # For radius 1, a word covers itself and all words differing in exactly 1 position
        covered.add(tuple(word))
        for i in range(n):
            for val in range(q):
                if val != word[i]:
                    new_word = word.copy()
                    new_word[i] = val
                    covered.add(tuple(new_word))
        return covered
    
    # For efficiency with radius=1, precompute coverage for each codeword
    # But this is O(729 * 17) which is fine
    
    # Build coverage map: for each space word, which codewords cover it
    # space_word -> set of codeword indices that cover it
    coverage_map = {}
    for sw in all_space_words:
        coverage_map[tuple(sw)] = []
    
    for i, cw in enumerate(words):
        cov = compute_coverage(cw, all_space_words, q, n, radius)
        for sw_tuple in cov:
            if sw_tuple in coverage_map:
                coverage_map[sw_tuple].append(i)
    
    # Now greedily remove codewords that are redundant
    # A codeword is redundant if all words it covers are also covered by other codewords
    
    # Build reverse map: for each codeword, which space words only it covers (or track counts)
    # For each space word, track how many codewords cover it
    cover_count = {}
    for sw in all_space_words:
        cover_count[tuple(sw)] = len(coverage_map[tuple(sw)])
    
    # Codeword i is redundant if for every space word it covers, cover_count > 1
    # But we need to be careful: removing one codeword changes cover_count for others
    
    # Greedy removal: repeatedly find a redundant codeword and remove it
    active = [True] * len(words)
    removed = 0
    
    # For efficiency, precompute for each codeword the list of space words it covers
    cw_coverage = []
    for i, cw in enumerate(words):
        cov = compute_coverage(cw, all_space_words, q, n, radius)
        cw_coverage.append(cov)
    
    # Rebuild cover_count considering only active codewords
    def rebuild_cover_count(active, cw_coverage, all_space_words):
        cc = {}
        for sw in all_space_words:
            cc[tuple(sw)] = 0
        for i in range(len(words)):
            if active[i]:
                for sw_tuple in cw_coverage[i]:
                    if sw_tuple in cc:
                        cc[sw_tuple] += 1
        return cc
    
    # Greedy: try to remove each codeword if possible
    # For speed, do multiple passes
    
    while time.time() < deadline - 5:  # leave 5s margin
        cover_count = rebuild_cover_count(active, cw_coverage, all_space_words)
        
        # Check if all space words are covered
        min_cover = min(cover_count.values())
        if min_cover < 1:
            break  # Can't remove any more without losing coverage
        
        # Find a codeword that can be removed
        removed_this_pass = False
        for i in range(len(words)):
            if not active[i]:
                continue
            
            # Check if all space words covered by i are also covered by at least one other active codeword
            can_remove = True
            for sw_tuple in cw_coverage[i]:
                if cover_count[sw_tuple] <= 1:
                    can_remove = False
                    break
            
            if can_remove:
                active[i] = False
                removed_this_pass = True
                removed += 1
                break  # Restart the pass after removal
        
        if not removed_this_pass:
            break
    
    # Extract final code
    final_words = [words[i].tolist() for i in range(len(words)) if active[i]]
    
    # Verify the result
    # Check that all space words are covered
    final_cover_count = {}
    for sw in all_space_words:
        final_cover_count[tuple(sw)] = 0
    for cw in final_words:
        cov = compute_coverage(np.array(cw), all_space_words, q, n, radius)
        for sw_tuple in cov:
            if sw_tuple in final_cover_count:
                final_cover_count[sw_tuple] += 1
    
    min_cover = min(final_cover_count.values())
    if min_cover < 1:
        # Fallback to initial code if something went wrong
        return initial_words
    
    return final_words

# EVOLVE-BLOCK-END
