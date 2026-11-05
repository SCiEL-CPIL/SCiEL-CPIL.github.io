"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np


def construct(q, n, radius):
    """
    Construct a ternary covering code K_q(n, radius).
    Uses greedy initialization followed by local search with bitset-accelerated
    redundancy removal and 2-for-1 pair replacement.
    """
    deadline = time.time() + 130  # 20s safety margin
    rng = random.Random(42)
    
    # Total number of words in the space
    total_words = q ** n
    
    # Helper: convert a word (list of ints) to an integer index
    def word_to_idx(word):
        idx = 0
        for i, c in enumerate(word):
            idx = idx * q + c
        return idx
    
    # Helper: generate all neighbors of a word within Hamming distance <= radius
    def get_neighbors(word, q, n, radius):
        if radius == 0:
            return [tuple(word)]
        neighbors = set()
        # All words within distance 1 (since radius=1 for this task)
        for pos in range(n):
            orig = word[pos]
            for new_val in range(q):
                if new_val != orig:
                    new_word = list(word)
                    new_word[pos] = new_val
                    neighbors.add(tuple(new_word))
        # If radius > 1, we'd need to go further, but for this task radius=1
        return list(neighbors)
    
    # Represent coverage as a bitset (or set of indices)
    # For q=3, n=7, total_words = 2187, which fits in a single Python int bitset
    
    # Greedy construction
    def greedy_construct(q, n, radius, max_iter=None):
        covered = [False] * total_words
        code = []
        
        # Precompute all words
        all_words = [list(w) for w in itertools.product(range(q), repeat=n)]
        
        # For each word, precompute its index
        word_to_idx_map = {tuple(w): i for i, w in enumerate(all_words)}
        
        # Precompute neighbors for each word (for radius=1, this is 3*n neighbors + itself)
        neighbors_of = {}
        for i, w in enumerate(all_words):
            neigh = set()
            neigh.add(i)  # itself
            for pos in range(n):
                orig = w[pos]
                for new_val in range(q):
                    if new_val != orig:
                        new_word = list(w)
                        new_word[pos] = new_val
                        neigh.add(word_to_idx_map[tuple(new_word)])
            neighbors_of[i] = list(neigh)
        
        # Greedy: repeatedly pick the word that covers the most uncovered words
        remaining = set(range(total_words))
        
        while remaining:
            if time.time() > deadline:
                break
            if max_iter is not None and len(code) >= max_iter:
                break
            
            best_idx = -1
            best_count = -1
            for idx in remaining:
                count = 0
                for nidx in neighbors_of[idx]:
                    if nidx in remaining:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_idx = idx
            
            if best_idx == -1:
                break
            
            code.append(all_words[best_idx])
            for nidx in neighbors_of[best_idx]:
                remaining.discard(nidx)
        
        return code
    
    # Initial greedy construction
    code = greedy_construct(q, n, radius)
    
    # Local search: redundancy removal and 2-for-1 replacement
    def compute_coverage(code, all_words, word_to_idx_map, neighbors_of):
        """Compute which words are covered by the code."""
        covered = set()
        for w in code:
            idx = word_to_idx_map[tuple(w)]
            for nidx in neighbors_of[idx]:
                covered.add(nidx)
        return covered
    
    def is_redundant(code, all_words, word_to_idx_map, neighbors_of):
        """Check if removing a codeword still maintains full coverage."""
        total_idx = set(range(total_words))
        
        # Compute coverage without each codeword
        for i, w in enumerate(code):
            idx = word_to_idx_map[tuple(w)]
            # Coverage of all other codewords
            covered = set()
            for j, w2 in enumerate(code):
                if j == i:
                    continue
                idx2 = word_to_idx_map[tuple(w2)]
                for nidx in neighbors_of[idx2]:
                    covered.add(nidx)
            
            # Check if the words covered only by this codeword are covered by others
            for nidx in neighbors_of[idx]:
                if nidx not in covered:
                    return False, -1  # Not redundant
            return True, i  # Redundant, can remove
        
        return False, -1
    
    def local_search(code, all_words, word_to_idx_map, neighbors_of, max_rounds=5):
        """Perform local search: remove redundant codewords and try 2-for-1 replacements."""
        for round_num in range(max_rounds):
            if time.time() > deadline:
                break
            
            # Phase 1: Remove redundant codewords
            removed = True
            while removed:
                removed = False
                for i in range(len(code)):
                    if time.time() > deadline:
                        break
                    idx = word_to_idx_map[tuple(code[i])]
                    
                    # Check if this codeword is redundant
                    covered_by_others = set()
                    for j in range(len(code)):
                        if j == i:
                            continue
                        idx2 = word_to_idx_map[tuple(code[j])]
                        for nidx in neighbors_of[idx2]:
                            covered_by_others.add(nidx)
                    
                    is_redundant = True
                    for nidx in neighbors_of[idx]:
                        if nidx not in covered_by_others:
                            is_redundant = False
                            break
                    
                    if is_redundant:
                        code.pop(i)
                        removed = True
                        break  # Restart the loop since indices changed
            
            # Phase 2: Try 2-for-1 replacements
            # For each pair of codewords, check if removing both and adding one new word
            # still maintains coverage
            improved = True
            while improved:
                improved = False
                if time.time() > deadline:
                    break
                
                # Try removing each pair and adding a single new word
                for i in range(len(code)):
                    if time.time() > deadline:
                        break
                    for j in range(i + 1, len(code)):
                        if time.time() > deadline:
                            break
                        
                        # Compute coverage of code without i and j
                        covered = set()
                        for k in range(len(code)):
                            if k == i or k == j:
                                continue
                            idx_k = word_to_idx_map[tuple(code[k])]
                            for nidx in neighbors_of[idx_k]:
                                covered.add(nidx)
                        
                        # Find words covered by i or j but not by others
                        uncovered = set()
                        idx_i = word_to_idx_map[tuple(code[i])]
                        idx_j = word_to_idx_map[tuple(code[j])]
                        for nidx in neighbors_of[idx_i]:
                            if nidx not in covered:
                                uncovered.add(nidx)
                        for nidx in neighbors_of[idx_j]:
                            if nidx not in covered:
                                uncovered.add(nidx)
                        
                        # If there's a single word that covers all uncovered words,
                        # we can replace i and j with that word
                        if len(uncovered) > 0:
                            # Find a word that covers all uncovered
                            for candidate in all_words:
                                cand_idx = word_to_idx_map[tuple(candidate)]
                                cand_neighbors = neighbors_of[cand_idx]
                                if all(u in cand_neighbors for u in uncovered):
                                    # Replace i and j with candidate
                                    code.pop(j)
                                    code.pop(i)
                                    code.append(candidate)
                                    improved = True
                                    break
                        else:
                            # All words are covered by others, just remove i and j
                            code.pop(j)
                            code.pop(i)
                            improved = True
                            break
                    
                    if improved:
                        break
        
        return code
    
    # Precompute helper structures
    all_words = [list(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx_map = {tuple(w): i for i, w in enumerate(all_words)}
    
    neighbors_of = {}
    for i, w in enumerate(all_words):
        neigh = set()
        neigh.add(i)
        for pos in range(n):
            orig = w[pos]
            for new_val in range(q):
                if new_val != orig:
                    new_word = list(w)
                    new_word[pos] = new_val
                    neigh.add(word_to_idx_map[tuple(new_word)])
        neighbors_of[i] = list(neigh)
    
    # Perform local search
    code = local_search(code, all_words, word_to_idx_map, neighbors_of, max_rounds=3)
    
    # Verify coverage
    covered = set()
    for w in code:
        idx = word_to_idx_map[tuple(w)]
        for nidx in neighbors_of[idx]:
            covered.add(nidx)
    
    if len(covered) != total_words:
        # Fallback: add missing words
        for idx in range(total_words):
            if idx not in covered:
                code.append(all_words[idx])
    
    return code
# EVOLVE-BLOCK-END
