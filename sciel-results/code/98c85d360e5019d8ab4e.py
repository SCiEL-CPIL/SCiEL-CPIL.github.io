"""Rewritten program: criticality-based greedy with swap refinement for K_3(7,1)."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def construct(q, n, radius):
    deadline = time.time() + 115.0  # safety margin before 120s limit
    rng = random.Random(42)

    # Precompute all words in mixed-radix integer form
    total = q ** n
    # Convert integer to base-q digits
    def int_to_digits(x):
        digits = [0] * n
        for i in range(n):
            digits[i] = x % q
            x //= q
        return digits

    # Precompute digit arrays for all words (memory: total * n * 1 byte)
    # For q=3, n=7: 2187 * 7 = 15309 bytes, trivial
    all_digits = np.array([int_to_digits(i) for i in range(total)], dtype=np.uint8)

    # Precompute neighbors: for each word, list of words within Hamming distance <= radius
    # For radius=1, n=7, q=3: each word has 1 + 7*2 = 15 neighbors
    # Build neighbor lists
    neighbors = []
    for i in range(total):
        nbrs = set([i])
        digits = all_digits[i]
        for pos in range(n):
            orig = digits[pos]
            for val in range(q):
                if val == orig:
                    continue
                # Compute the integer for the modified word
                new_int = i + (val - orig) * (q ** pos)
                nbrs.add(new_int)
        neighbors.append(sorted(nbrs))

    # Convert to numpy arrays for speed
    neighbor_arrays = [np.array(nb, dtype=np.int32) for nb in neighbors]

    # Coverage count: how many codewords cover each word
    coverage = np.zeros(total, dtype=np.int32)

    # Greedy phase: criticality-based ordering
    # Track uncovered words
    uncovered = np.ones(total, dtype=bool)
    num_uncovered = total

    # For criticality: for each uncovered word, count how many codewords are within distance <= radius
    # That is, for word w, criticality = number of candidate codewords c such that dist(w,c) <= radius
    # But we need to pick a codeword that maximizes minimum criticality among uncovered words it covers.
    
    # Actually, let's think differently. The "coverage deficit" for an uncovered word w is:
    # the number of words c (potential codewords) such that dist(w, c) <= radius.
    # We want to pick c that covers the word w with the minimum such count (most constrained).
    
    # Precompute for each word w, the list of potential codewords c within distance <= radius
    # This is exactly neighbors[w] (since dist(c,w) <= radius iff dist(w,c) <= radius)
    # So criticality of w = len(neighbors[w]) which is constant for all w in a regular space.
    # That's not useful. Let me reconsider.
    
    # Better approach: standard greedy with a twist.
    # At each step, for each uncovered word w, define its "urgency" as the number of 
    # currently-uncovered words in its ball (i.e., how many uncovered words would be 
    # covered if we picked w as a codeword). But that's just the standard greedy.
    
    # The criticality idea: among uncovered words, find the one w* with the fewest 
    # "available" codewords that could cover it. A codeword c is "available" for w if 
    # c is within distance <= radius of w AND c hasn't been selected yet. Since we're 
    # selecting codewords, initially all are available. After selecting some, the 
    # available set shrinks.
    
    # Simpler effective approach: 
    # 1. Greedy: repeatedly pick the word that covers the most uncovered words.
    # 2. Then do local search (swap) to improve.
    
    # Let me implement a fast greedy + swap local search.
    
    # Greedy phase
    codewords = []
    
    # For speed, maintain a count of how many uncovered words each candidate covers
    # candidate_cover_count[c] = number of uncovered words w such that c is in neighbors[w]
    # = number of uncovered words within distance <= radius of c
    # = sum over w in neighbors[c] of uncovered[w]
    
    # Initialize
    cand_cover = np.zeros(total, dtype=np.int32)
    for c in range(total):
        cand_cover[c] = np.sum(uncovered[neighbor_arrays[c]])
    
    # Greedy iterations
    while num_uncovered > 0 and time.time() < deadline - 30:
        # Pick the candidate with maximum coverage of uncovered words
        best_c = int(np.argmax(cand_cover))
        if cand_cover[best_c] == 0:
            break
        
        codewords.append(best_c)
        
        # Update coverage and uncovered
        for w in neighbor_arrays[best_c]:
            if uncovered[w]:
                uncovered[w] = False
                num_uncovered -= 1
                # Decrement cand_cover for all c that cover w
                # i.e., for all c in neighbors[w]
                for c in neighbor_arrays[w]:
                    cand_cover[c] -= 1
    
    # Local search: swap-based improvement
    # Try to replace a codeword with another to reduce total size while maintaining coverage
    # Or try to add a codeword that allows removing two existing ones.
    
    # Strategy: try to find a codeword c' such that adding it allows removing one existing codeword c
    # where c's unique coverage (words only covered by c) is a subset of c''s coverage.
    
    # First, compute for each codeword, which words it uniquely covers
    # unique_words[c] = set of words w where coverage[w] == 1 and c covers w
    
    # Recompute coverage from scratch
    coverage = np.zeros(total, dtype=np.int32)
    for c in codewords:
        coverage[neighbor_arrays[c]] += 1
    
    # For each codeword index, find words it uniquely covers
    unique_count = np.zeros(len(codewords), dtype=np.int32)
    for idx, c in enumerate(codewords):
        for w in neighbor_arrays[c]:
            if coverage[w] == 1:
                unique_count[idx] += 1
    
    # Try swaps: for each codeword c_i, try replacing with c_j (not in codewords)
    # such that all words uniquely covered by c_i are also covered by c_j
    # and c_j covers some words not covered by c_i (to allow further removals later)
    
    # For efficiency, only try swaps for codewords with small unique_count
    # Sort codeword indices by unique_count
    sorted_indices = sorted(range(len(codewords)), key=lambda i: unique_count[i])
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for idx in sorted_indices:
            if time.time() > deadline - 10:
                break
            c = codewords[idx]
            # Words uniquely covered by c
            unique_words_c = [w for w in neighbor_arrays[c] if coverage[w] == 1]
            if not unique_words_c:
                # Can remove this codeword
                codewords.pop(idx)
                for w in neighbor_arrays[c]:
                    coverage[w] -= 1
                unique_count = np.zeros(len(codewords), dtype=np.int32)
                for j, cj in enumerate(codewords):
                    for w in neighbor_arrays[cj]:
                        if coverage[w] == 1:
                            unique_count[j] += 1
                sorted_indices = sorted(range(len(codewords)), key=lambda i: unique_count[i])
                improved = True
                break
            
            # Try to find a replacement c' that covers all unique_words_c
            # c' must be within distance <= radius of all words in unique_words_c
            # Intersection of balls around unique_words_c
            # Start with neighbors of first unique word
            candidate_set = set(neighbor_arrays[unique_words_c[0]])
            for w in unique_words_c[1:]:
                candidate_set &= set(neighbor_arrays[w])
                if not candidate_set:
                    break
            
            if not candidate_set:
                continue
            
            # Remove c from codewords, try each candidate
            codewords.pop(idx)
            for w in neighbor_arrays[c]:
                coverage[w] -= 1
            
            best_cand = None
            best_gain = 0
            for cprime in candidate_set:
                if cprime == c:
                    continue
                # Check if cprime is already in codewords
                if cprime in codewords:
                    continue
                # Count how many words cprime newly covers (were uncovered after removing c)
                gain = np.sum(coverage[neighbor_arrays[cprime]] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_cand = cprime
            
            if best_cand is not None and best_gain > 0:
                codewords.append(best_cand)
                for w in neighbor_arrays[best_cand]:
                    coverage[w] += 1
                improved = True
                # Recompute unique counts
                unique_count = np.zeros(len(codewords), dtype=np.int32)
                for j, cj in enumerate(codewords):
                    for w in neighbor_arrays[cj]:
                        if coverage[w] == 1:
                            unique_count[j] += 1
                sorted_indices = sorted(range(len(codewords)), key=lambda i: unique_count[i])
                break
            else:
                # Revert
                codewords.insert(idx, c)
                for w in neighbor_arrays[c]:
                    coverage[w] += 1
    
    # Final greedy cleanup: remove any codeword whose coverage is redundant
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        for idx in range(len(codewords) - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            c = codewords[idx]
            # Check if all words covered by c are covered by at least one other codeword
            redundant = True
            for w in neighbor_arrays[c]:
                if coverage[w] == 1:
                    redundant = False
                    break
            if redundant:
                codewords.pop(idx)
                for w in neighbor_arrays[c]:
                    coverage[w] -= 1
                changed = True
    
    # Convert back to list of digit lists
    result = [list(all_digits[c]) for c in codewords]
    return result
# EVOLVE-BLOCK-END
