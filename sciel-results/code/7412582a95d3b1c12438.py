import numpy as np
import itertools
import time

def construct(q, n, radius):
    # Handle the specific high-interest case
    if q == 3 and n == 6 and radius == 1:
        return _construct_3_6_1()
    
    # Fallback for other parameters
    # The seed logic:
    # tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    # return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # For safety, if not 3,6,1, we can try a generic greedy if n is small, else direct sum (if q=3).
    
    if q == 3:
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Simple fallback: return all words (valid, just not minimal)
    return [list(w) for w in itertools.product(range(q), repeat=n)]

def _construct_3_6_1():
    q = 3
    n = 6
    N = q ** n
    deadline = time.time() + 110.0  # Safety margin before 120s limit

    # Generate all words
    words = list(itertools.product(range(q), repeat=n))
    
    # Precompute index for each word
    # index = w[0]*3^5 + ... + w[5]
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Precompute neighbors for each word (radius 1)
    # neighbors[i] is a list of indices of words within distance 1 of words[i]
    neighbors = []
    for i, c in enumerate(words):
        neigh = [i]
        for pos in range(n):
            for new_val in range(q):
                if new_val == c[pos]:
                    continue
                new_w = list(c)
                new_w[pos] = new_val
                neigh.append(word_to_idx[tuple(new_w)])
        neighbors.append(neigh)
    
    # We want to find a set of codewords such that every word is in the neighborhood of at least one codeword.
    # This is equivalent to finding a dominating set in the graph where vertices are words and edges connect words at distance <= 1.
    # Actually, it's a covering code problem.
    
    # Strategy:
    # 1. Greedy set cover to get an initial solution.
    # 2. Local search / heuristic reduction to improve the size.
    
    # Greedy Set Cover
    # uncovered is a boolean array of size N
    uncovered = np.ones(N, dtype=bool)
    selected = []
    
    # To speed up greedy, we can maintain counts of how many uncovered words each codeword covers.
    # But updating counts after each selection is O(N) per step, and we have ~70 steps. Total O(70*N) which is fine.
    # However, calculating the initial counts and updating them efficiently is key.
    
    # Let's use a simpler approach:
    # In each step, find the codeword that covers the most uncovered words.
    # Since N=729, we can just iterate through all codewords and count uncovered neighbors.
    # 729 codewords * 19 neighbors * 729 iterations? No, 729 iterations * 729 codewords * 19 neighbors = 10M ops. Fast enough.
    
    while uncovered.any():
        best_idx = -1
        best_count = 0
        for j in range(N):
            # Count how many neighbors of j are uncovered
            count = 0
            for nb in neighbors[j]:
                if uncovered[nb]:
                    count += 1
            if count > best_count:
                best_count = count
                best_idx = j
        
        if best_idx == -1:
            break
            
        selected.append(best_idx)
        # Update uncovered
        for nb in neighbors[best_idx]:
            uncovered[nb] = False
            
        if time.time() > deadline:
            break
            
    # Initial solution size: len(selected)
    # Now, try to reduce the size using local search.
    # We can try to remove codewords one by one if they are redundant.
    # A codeword j is redundant if every word in its neighborhood is covered by some other selected codeword.
    
    # To check redundancy efficiently:
    # Maintain a count array: coverage_count[i] = number of selected codewords that cover word i.
    # A codeword j is redundant if for all i in neighbors[j], coverage_count[i] > 1.
    
    coverage_count = np.zeros(N, dtype=np.int32)
    for j in selected:
        for nb in neighbors[j]:
            coverage_count[nb] += 1
            
    # Try to remove redundant codewords
    changed = True
    while changed:
        changed = False
        for idx in range(len(selected)):
            j = selected[idx]
            # Check if j is redundant
            is_redundant = True
            for nb in neighbors[j]:
                if coverage_count[nb] == 1:
                    is_redundant = False
                    break
            if is_redundant:
                # Remove j
                for nb in neighbors[j]:
                    coverage_count[nb] -= 1
                selected.pop(idx)
                changed = True
                break  # Restart loop since indices changed
                
        if time.time() > deadline:
            break
            
    # Further improvement: Try to swap codewords?
    # Or use a more sophisticated local search like simulated annealing or tabu search?
    # Given the time limit, simple removal is often sufficient for covering codes if the greedy start is good.
    # But greedy set cover is not optimal.
    # Let's try a different initial construction or a better heuristic.
    
    # Alternative: Start with a known good code if we have one, or use a different greedy strategy.
    # The greedy strategy above picks the codeword covering the most *uncovered* words.
    # This is standard and usually yields good results.
    
    # Let's also try a "random restart" approach if we have time.
    # But for now, let's stick with the greedy + reduction.
    
    # Convert indices back to words
    codewords = [words[i] for i in selected]
    
    return codewords
