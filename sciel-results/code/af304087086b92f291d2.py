"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import product

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 130  # 130s deadline to be safe within 150s
    
    # Total number of words in the space
    total_words = q ** n
    
    # If the space is small enough, we can use a more direct approach
    # For q=3, n=7, radius=1, total_words = 2187
    
    # Convert words to indices and back
    def word_to_idx(word):
        idx = 0
        for v in word:
            idx = idx * q + v
        return idx
    
    def idx_to_word(idx):
        word = []
        for _ in range(n):
            word.append(idx % q)
            idx //= q
        return word[::-1]
    
    # Precompute neighbors: for each word, list of words within Hamming distance <= radius
    # For radius=1, a word's neighbors are itself plus all words differing in exactly one position
    # Number of neighbors for radius=1: 1 + n*(q-1)
    # For q=3, n=7, radius=1: 1 + 7*2 = 15 neighbors per word
    
    # Build adjacency: for each word index, the set of word indices within distance <= radius
    # This is the "covering" relation
    
    # For efficiency, precompute the list of neighbor indices for each word
    # neighbor_list[i] = list of word indices j such that dist(i, j) <= radius
    
    # Actually, for set cover, we want: for each candidate center c, the set of words it covers
    # Let's precompute coverage sets
    
    # For radius=1 and q=3, n=7:
    # Each center covers 1 + 7*2 = 15 words
    # We need to cover 2187 words
    # Lower bound: ceil(2187/15) = 146
    # Best known: 186
    
    # Strategy: Greedy set cover with local search improvement
    # 1. Greedy: repeatedly pick the center that covers the most uncovered words
    # 2. Local search: try to remove a center and see if coverage is maintained; try swaps
    
    # Precompute for each word index, the list of centers that cover it
    # And for each center, the list of words it covers
    
    # For radius=1:
    # A center c covers c itself and all words at Hamming distance 1 from c
    
    # Let's precompute the coverage sets for all possible centers
    # There are q^n = 2187 possible centers
    
    # For each center c (as an index), compute the set of indices it covers
    # For radius=1: c covers c and all words obtained by changing one coordinate
    
    # Precompute all coverage sets
    # cover_sets[c_idx] = set of word indices covered by center c_idx
    
    # To save memory, we can represent coverage as a bitset (numpy array of uint64 or similar)
    # But for 2187 words, a boolean array of length 2187 is fine
    
    # Actually, let's use a different approach:
    # For each word w, compute the list of centers that cover w
    # For radius=1, word w is covered by:
    #   - w itself
    #   - all words at Hamming distance 1 from w
    # So centers covering w = {w} union {words at dist 1 from w}
    # That's 1 + n*(q-1) centers per word
    
    # Let's precompute:
    # covers_word[w_idx] = list of center indices that cover word w_idx
    
    # And for greedy, we need: for a given set of uncovered words, which center covers the most?
    # We can maintain a count for each center of how many uncovered words it covers
    
    # Precompute the mapping
    # For each word w, the centers that cover it are w and its Hamming-1 neighbors
    
    # Let's build:
    # center_covers[c_idx] = list of word indices that center c_idx covers
    # For radius=1: c_idx covers c_idx and all words at Hamming distance 1
    
    # Generate all centers and their coverage
    all_indices = list(range(total_words))
    
    # For each center, compute which words it covers
    # For radius=1:
    #   The center itself
    #   All words obtained by changing one coordinate to a different value
    
    center_covers = {}
    for c_idx in range(total_words):
        c_word = idx_to_word(c_idx)
        covered = [c_idx]
        for pos in range(n):
            for new_val in range(q):
                if new_val != c_word[pos]:
                    new_word = c_word.copy()
                    new_word[pos] = new_val
                    new_idx = word_to_idx(new_word)
                    covered.append(new_idx)
        center_covers[c_idx] = covered
    
    # Now do greedy set cover
    # uncovered = set of all word indices
    uncovered = set(all_indices)
    
    # For each center, count how many uncovered words it covers
    # Initially, all words are uncovered, so count = len(center_covers[c_idx])
    
    # Use a priority approach: repeatedly pick the center with max coverage of uncovered words
    
    # For efficiency, maintain for each center the number of uncovered words it covers
    # and for each uncovered word, the list of centers that cover it
    
    # centers_covering_word[w_idx] = list of centers that cover word w_idx
    centers_covering_word = {}
    for w_idx in range(total_words):
        # Word w_idx is covered by w_idx itself and its Hamming-1 neighbors
        w_word = idx_to_word(w_idx)
        covering_centers = [w_idx]
        for pos in range(n):
            for new_val in range(q):
                if new_val != w_word[pos]:
                    new_word = w_word.copy()
                    new_word[pos] = new_val
                    new_idx = word_to_idx(new_word)
                    covering_centers.append(new_idx)
        centers_covering_word[w_idx] = covering_centers
    
    # Greedy set cover
    # Maintain:
    #   uncovered: set of uncovered word indices
    #   cover_count[c_idx]: number of uncovered words covered by center c_idx
    #   selected: list of selected centers
    
    cover_count = {}
    for c_idx in range(total_words):
        cover_count[c_idx] = len(center_covers[c_idx])
    
    selected = []
    
    while uncovered:
        if time.time() > deadline - 10:
            break
        
        # Find the center with max cover_count among those that cover at least one uncovered word
        best_center = None
        best_count = -1
        for c_idx in range(total_words):
            if cover_count[c_idx] > best_count:
                best_count = cover_count[c_idx]
                best_center = c_idx
        
        if best_center is None or best_count <= 0:
            break
        
        selected.append(best_center)
        
        # Remove from uncovered all words covered by best_center
        for w_idx in center_covers[best_center]:
            if w_idx in uncovered:
                uncovered.remove(w_idx)
                # Decrease cover_count for all centers that cover w_idx
                for c2 in centers_covering_word[w_idx]:
                    cover_count[c2] -= 1
        
        # Set cover_count for best_center to 0 (it's already selected)
        cover_count[best_center] = 0
    
    # Check if we covered everything
    if uncovered:
        # Add centers to cover remaining words (shouldn't happen if we ran to completion)
        for w_idx in list(uncovered):
            # Pick any center that covers w_idx
            for c_idx in centers_covering_word[w_idx]:
                if c_idx not in selected:
                    selected.append(c_idx)
                    break
            uncovered.discard(w_idx)
    
    # Now do local search to try to reduce the number of centers
    # Try removing each center and see if coverage is maintained
    # Try swapping one center for another
    
    def is_covered(selected_centers):
        """Check if all words are covered by selected_centers."""
        if not selected_centers:
            return total_words == 0
        covered_set = set()
        for c_idx in selected_centers:
            covered_set.update(center_covers[c_idx])
        return len(covered_set) == total_words
    
    # Local search: try to remove centers
    improved = True
    while improved and time.time() < deadline - 15:
        improved = False
        for i in range(len(selected)):
            if time.time() > deadline - 15:
                break
            # Try removing selected[i]
            trial = selected[:i] + selected[i+1:]
            if is_covered(trial):
                selected = trial
                improved = True
                break
    
    # Try swaps: remove one center, add another
    for i in range(len(selected)):
        if time.time() > deadline - 15:
            break
        for j in range(len(selected)):
            if i == j:
                continue
            # Try swapping selected[i] with selected[j]... no, that doesn't make sense
            # Instead: remove selected[i], try adding a new center
            pass
    
    # More sophisticated local search: try to replace a center with a different one
    # For each center in selected, try replacing it with a center not in selected
    # that covers the same or more uncovered words
    
    # Actually, let's do a more effective local search:
    # 1. For each selected center, compute which words it uniquely covers (not covered by other selected centers)
    # 2. If a center doesn't uniquely cover any words, remove it
    # 3. Try to find a center that covers more words than some existing center
    
    # Compute coverage by each selected center, excluding overlap
    selected_set = set(selected)
    # For each word, count how many selected centers cover it
    word_cover_count = np.zeros(total_words, dtype=np.int32)
    for c_idx in selected:
        for w_idx in center_covers[c_idx]:
            word_cover_count[w_idx] += 1
    
    # A center can be removed if all words it covers are covered by at least one other selected center
    # i.e., for all w in center_covers[c], word_cover_count[w] >= 2
    
    improved = True
    while improved and time.time() < deadline - 15:
        improved = False
        for i in range(len(selected)):
            if time.time() > deadline - 15:
                break
            c_idx = selected[i]
            can_remove = True
            for w_idx in center_covers[c_idx]:
                if word_cover_count[w_idx] <= 1:
                    can_remove = False
                    break
            if can_remove:
                # Remove this center
                for w_idx in center_covers[c_idx]:
                    word_cover_count[w_idx] -= 1
                selected.pop(i)
                improved = True
                break
    
    # Now try to improve by replacing centers
    # For each center, see if there's a better center that covers the same words plus more
    # This is complex, so let's try a simpler heuristic:
    # Try adding a random center and then see if we can remove more
    
    # Convert selected indices to words
    result = [idx_to_word(c_idx) for c_idx in selected]
    
    # Ensure distinctness
    seen = set()
    unique_result = []
    for word in result:
        key = tuple(word)
        if key not in seen:
            seen.add(key)
            unique_result.append(word)
    
    return unique_result
# EVOLVE-BLOCK-END
