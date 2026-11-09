"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet {0..q-1} with covering radius `radius`.
    Uses a greedy set cover approach followed by local search (deletion/swap) to minimize size.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Handle trivial cases or specific known structures
    if q == 3 and n == 7 and radius == 1:
        # Known optimal is 186. We try to find a small code.
        
        # Generate all 3^7 = 2187 words
        all_words = list(itertools.product(range(q), repeat=n))
        N = len(all_words)
        
        # Map word to index
        word_to_idx = {}
        for i, w in enumerate(all_words):
            word_to_idx[w] = i
            
        # Precompute neighbors for each index
        # Since n=7, q=3, radius=1, each word has 1 + 7*2 = 15 neighbors.
        neighbors = [set() for _ in range(N)]
        for i, w in enumerate(all_words):
            neighbors[i].add(i)
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]: continue
                    new_w = list(w)
                    new_w[pos] = val
                    idx = word_to_idx[tuple(new_w)]
                    neighbors[i].add(idx)
        
        # Convert neighbors to lists for faster iteration
        neighbors_list = [list(s) for s in neighbors]
        
        # Helper to check if a set of indices covers everything
        def is_covered(indices):
            covered = [0] * N
            for idx in indices:
                for nb in neighbors_list[idx]:
                    covered[nb] += 1
            return all(c > 0 for c in covered)
        
        # Helper to perform local search (reduction) on a set
        def local_search_reduce(indices, time_limit):
            current_set = list(indices)
            random.shuffle(current_set)
            
            # Build cover count
            cover_count = [0] * N
            for idx in current_set:
                for nb in neighbors_list[idx]:
                    cover_count[nb] += 1
            
            removed_any = True
            while removed_any and time.time() < time_limit:
                removed_any = False
                random.shuffle(current_set)
                for idx in current_set:
                    if time.time() > time_limit: break
                    # Check if redundant
                    redundant = True
                    for nb in neighbors_list[idx]:
                        if cover_count[nb] <= 1:
                            redundant = False
                            break
                    if redundant:
                        # Remove idx
                        for nb in neighbors_list[idx]:
                            cover_count[nb] -= 1
                        current_set.remove(idx)
                        removed_any = True
            return current_set

        # Strategy 1: Greedy Set Cover from scratch
        # This often gives a smaller initial set than the tetracode construction (243).
        # Greedy: at each step, pick the word that covers the most uncovered words.
        
        best_size = float('inf')
        best_set = set()
        
        # We will run multiple restarts of the greedy + reduction process.
        # Each restart uses a different randomization in the tie-breaking of the greedy step.
        
        num_restarts = 10 # We will adjust this based on time
        
        for r in range(num_restarts):
            if time.time() > deadline - 5: break
            
            uncovered = set(range(N))
            greedy_set = []
            
            # To speed up greedy, we can just scan. 2187 * 15 is small.
            # But we need to do this ~200 times. 200 * 2187 * 15 = 6.5M ops. Fast.
            
            while uncovered and time.time() < deadline - 2:
                best_idx = -1
                best_count = 0
                # Randomize tie-breaking
                # We iterate through all indices, but if counts are equal, we pick randomly?
                # Or just shuffle the range? Shuffling 2187 items is cheap.
                indices_order = list(range(N))
                random.shuffle(indices_order)
                
                for idx in indices_order:
                    if idx in greedy_set: continue
                    # Count how many uncovered words idx covers
                    cnt = 0
                    for nb in neighbors_list[idx]:
                        if nb in uncovered:
                            cnt += 1
                    if cnt > best_count:
                        best_count = cnt
                        best_idx = idx
                    # Early termination if we cover all?
                    if cnt == len(uncovered):
                        break
                
                if best_idx == -1: break
                greedy_set.append(best_idx)
                for nb in neighbors_list[best_idx]:
                    uncovered.discard(nb)
                
                # If we have covered everything, stop
                if not uncovered:
                    break
            
            # Now reduce the greedy set
            reduced_set = local_search_reduce(greedy_set, deadline - 1)
            
            if len(reduced_set) < best_size:
                best_size = len(reduced_set)
                best_set = set(reduced_set)
                
                # If we found something very good, maybe we can stop early?
                # Best known is 186. If we hit 186, we are done.
                if best_size <= 186:
                    break

        # If we didn't find a good solution, try the tetracode seed as a fallback
        if best_size > 200: # Threshold
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            initial_codewords = [tuple(h + list(tail)) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
            codeword_indices = [word_to_idx[c] for c in initial_codewords]
            reduced_set = local_search_reduce(codeword_indices, deadline - 1)
            if len(reduced_set) < best_size:
                best_size = len(reduced_set)
                best_set = set(reduced_set)

        result = [list(all_words[idx]) for idx in best_set]
        return result

    # Generic fallback for other parameters (not used in this specific task instance but good for robustness)
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    neighbors = []
    for w in all_words:
        nbs = {word_to_idx[w]}
        for pos in range(n):
            for val in range(q):
                if val == w[pos]: continue
                nw = list(w)
                nw[pos] = val
                nbs.add(word_to_idx[tuple(nw)])
        neighbors.append(nbs)
    
    uncovered = set(range(len(all_words)))
    code_set = []
    while uncovered and time.time() < deadline:
        best_idx = -1
        best_count = 0
        for idx in range(len(all_words)):
            if idx in code_set: continue
            cnt = sum(1 for nb in neighbors[idx] if nb in uncovered)
            if cnt > best_count:
                best_count = cnt
                best_idx = idx
        if best_idx == -1: break
        code_set.append(best_idx)
        for nb in neighbors[best_idx]:
            uncovered.discard(nb)
    
    return [list(all_words[idx]) for idx in code_set]
# EVOLVE-BLOCK-END
