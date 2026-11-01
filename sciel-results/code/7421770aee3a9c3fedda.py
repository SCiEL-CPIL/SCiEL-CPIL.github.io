import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    """
    Construct a covering code for q=3, n=8, radius=1.
    Returns a list of codewords (each a list of n integers).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 115s deadline to be safe
    
    # For this specific problem: q=3, n=8, radius=1
    # Space size: 3^8 = 6561
    # Each codeword covers itself + 8*(q-1) = 8*2 = 16 neighbors = 17 words total
    
    # Precompute all words as integers 0..3^8-1
    # word_to_int: convert tuple to integer
    def word_to_int(w):
        idx = 0
        for val in w:
            idx = idx * q + val
        return idx
    
    def int_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w
    
    all_words_int = list(range(q ** n))
    all_words = [int_to_word(i) for i in all_words_int]
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute the ball (cover set) for each word
    # A word w covers: w itself, and all words differing in exactly 1 position
    # Total: 1 + n*(q-1) = 1 + 8*2 = 17 words
    covers = [set() for _ in range(len(all_words))]
    for idx, w in enumerate(all_words):
        covers[idx].add(idx)
        w_list = list(w)
        for i in range(n):
            orig = w_list[i]
            for new_val in range(q):
                if new_val != orig:
                    w_list[i] = new_val
                    covers[idx].add(word_to_idx[tuple(w_list)])
            w_list[i] = orig
    
    # Convert covers to frozensets for hashing
    covers_frozen = [frozenset(c) for c in covers]
    
    # Vectorized representation for fast operations
    # For each word, store its cover as a sorted list of indices
    covers_list = [sorted(c) for c in covers_frozen]
    
    # Build a numpy array: cover_matrix[i][j] = 1 if word j is in cover of word i
    N = len(all_words)
    cover_matrix = np.zeros((N, N), dtype=np.uint8)
    for i in range(N):
        for j in covers_list[i]:
            cover_matrix[i][j] = 1
    
    # Greedy set cover with vectorized operations
    def greedy_set_cover(seed=0):
        rng = random.Random(seed)
        uncovered = np.ones(N, dtype=np.uint8)
        chosen = []
        
        while uncovered.sum() > 0:
            if time.time() > deadline:
                break
            
            # For each word, compute how many uncovered points it covers
            # count[i] = sum(cover_matrix[i] * uncovered)
            counts = (cover_matrix * uncovered).sum(axis=1)
            
            # Get the best word
            best_word = int(np.argmax(counts))
            
            if counts[best_word] == 0:
                break
            
            chosen.append(best_word)
            uncovered[cover_matrix[best_word]] = 0
        
        return chosen
    
    # Better: randomized greedy with top-k selection
    def randomized_greedy(seed=0):
        rng = random.Random(seed)
        uncovered = np.ones(N, dtype=np.uint8)
        chosen = []
        
        while uncovered.sum() > 0:
            if time.time() > deadline:
                break
            
            # For each word, compute how many uncovered points it covers
            counts = (cover_matrix * uncovered).sum(axis=1)
            
            # Get indices where count > 0
            positive = np.where(counts > 0)[0]
            if len(positive) == 0:
                break
            
            # Pick from top-k candidates
            k = min(15, len(positive))
            # Get the top-k by count
            top_k_indices = positive[np.argsort(counts[positive])[-k:]]
            
            # Weighted random selection: higher count = higher probability
            weights = counts[top_k_indices]
            weights = weights.astype(float)
            weights /= weights.sum()
            
            best_word = int(rng.choices(top_k_indices, weights=weights)[0])
            
            chosen.append(best_word)
            uncovered[cover_matrix[best_word]] = 0
        
        return chosen
    
    # Local search: try to remove words, or swap words
    def local_search(chosen, max_iterations=50):
        if time.time() > deadline:
            return chosen
        
        chosen_set = set(chosen)
        
        # Compute coverage counts using numpy
        coverage = np.zeros(N, dtype=np.int32)
        for w_idx in chosen:
            coverage[cover_matrix[w_idx]] += 1
        
        improved = True
        iteration = 0
        while improved and time.time() < deadline and iteration < max_iterations:
            improved = False
            iteration += 1
            
            # Try to remove words: check if any word is redundant
            # A word i is redundant if for all j in cover[i], coverage[j] > 1
            for i in range(len(chosen)):
                if time.time() > deadline:
                    break
                w_idx = chosen[i]
                # Check if removing w_idx keeps all covered
                if coverage[cover_matrix[w_idx]].min() > 1:
                    coverage[cover_matrix[w_idx]] -= 1
                    chosen.remove(w_idx)
                    chosen_set.discard(w_idx)
                    improved = True
                    break
            
            # Try to swap: remove one, add another
            if not improved and len(chosen) > 0:
                # Try random swaps
                for _ in range(200):
                    if time.time() > deadline:
                        break
                    if len(chosen) == 0:
                        break
                    
                    i = random.randint(0, len(chosen) - 1)
                    w_remove = chosen[i]
                    
                    # Try to find a replacement
                    # We need a word w_add such that:
                    # coverage[cover[w_remove]] - 1 + cover[w_add] >= 1 for all positions
                    # i.e., for all j where coverage[j] == 1 and j in cover[w_remove],
                    # we need j in cover[w_add]
                    
                    # Find critical points: positions covered only by w_remove
                    critical = np.where((coverage == 1) & (cover_matrix[w_remove] == 1))[0]
                    
                    if len(critical) == 0:
                        # w_remove is redundant, just remove it
                        coverage[cover_matrix[w_remove]] -= 1
                        chosen.remove(w_remove)
                        chosen_set.discard(w_remove)
                        improved = True
                        break
                    
                    # Find words that cover all critical points
                    # For each candidate w_add, check if cover[w_add] contains all critical points
                    candidates = []
                    for w_idx in range(N):
                        if w_idx in chosen_set:
                            continue
                        if len(critical) == 0:
                            candidates.append(w_idx)
                        else:
                            # Check if w_add covers all critical points
                            if cover_matrix[w_idx][critical].all():
                                # Also check that after swap, coverage is still >= 1
                                new_coverage = coverage.copy()
                                new_coverage[cover_matrix[w_remove]] -= 1
                                new_coverage[cover_matrix[w_idx]] += 1
                                if new_coverage.min() >= 1:
                                    candidates.append(w_idx)
                    
                    if candidates:
                        w_add = random.choice(candidates)
                        # Perform swap
                        coverage[cover_matrix[w_remove]] -= 1
                        coverage[cover_matrix[w_add]] += 1
                        chosen[i] = w_add
                        chosen_set.discard(w_remove)
                        chosen_set.add(w_add)
                        improved = True
                        break
        
        return chosen
    
    # More aggressive local search with 2-opt style moves
    def aggressive_local_search(chosen):
        if time.time() > deadline:
            return chosen
        
        chosen_set = set(chosen)
        coverage = np.zeros(N, dtype=np.int32)
        for w_idx in chosen:
            coverage[cover_matrix[w_idx]] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try all possible single removals
            for i in range(len(chosen)):
                if time.time() > deadline:
                    break
                w_idx = chosen[i]
                if coverage[cover_matrix[w_idx]].min() > 1:
                    coverage[cover_matrix[w_idx]] -= 1
                    chosen.remove(w_idx)
                    chosen_set.discard(w_idx)
                    improved = True
                    break
            
            # Try all possible single swaps (limited)
            if not improved and len(chosen) > 0:
                # Try a limited number of random swaps
                for _ in range(500):
                    if time.time() > deadline:
                        break
                    if len(chosen) == 0:
                        break
                    
                    i = random.randint(0, len(chosen) - 1)
                    w_remove = chosen[i]
                    
                    critical = np.where((coverage == 1) & (cover_matrix[w_remove] == 1))[0]
                    
                    if len(critical) == 0:
                        coverage[cover_matrix[w_remove]] -= 1
                        chosen.remove(w_remove)
                        chosen_set.discard(w_remove)
                        improved = True
                        break
                    
                    # Find a word that covers all critical points and doesn't create new uncovered
                    found = False
                    for w_add in random.sample(range(N), min(200, N)):
                        if w_add in chosen_set:
                            continue
                        if len(critical) > 0 and not cover_matrix[w_add][critical].all():
                            continue
                        new_cov = coverage.copy()
                        new_cov[cover_matrix[w_remove]] -= 1
                        new_cov[cover_matrix[w_add]] += 1
                        if new_cov.min() >= 1:
                            coverage = new_cov
                            chosen[i] = w_add
                            chosen_set.discard(w_remove)
                            chosen_set.add(w_add)
                            improved = True
                            found = True
                            break
                    
                    if found:
                        break
        
        return chosen
    
    # Main search loop
    best_solution = None
    best_size = float('inf')
    
    # Start with greedy solutions
    for restart in range(30):
        if time.time() > deadline:
            break
        
        # Use different strategies
        if restart < 10:
            solution = greedy_set_cover(seed=restart)
        else:
            solution = randomized_greedy(seed=restart)
        
        if not solution:
            continue
        
        # Local search
        solution = local_search(solution)
        
        if len(solution) < best_size:
            best_size = len(solution)
            best_solution = solution[:]
        
        # If we're getting close to the known best, do more intensive local search
        if best_size <= 520 and time.time() < deadline:
            solution = aggressive_local_search(solution)
            if len(solution) < best_size:
                best_size = len(solution)
                best_solution = solution[:]
        
        # Try more local search iterations
        if time.time() < deadline:
            for _ in range(5):
                if time.time() > deadline:
                    break
                solution = aggressive_local_search(solution)
                if len(solution) < best_size:
                    best_size = len(solution)
                    best_solution = solution[:]
    
    # If we haven't found a good solution, try a direct construction
    if best_solution is None or best_size > 500:
        # Use the known structure: direct sum of tetracode
        # Tetracode: 9 words of length 4
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        
        # Generate all tails of length n-4
        tails = list(itertools.product(range(q), repeat=n - 4))
        
        # Construct initial codewords
        codewords = []
        for h in tetracode:
            for tail in tails:
                codewords.append(h + list(tail))
        
        # Convert to indices
        solution_indices = [word_to_idx[tuple(c)] for c in codewords]
        
        # Local search
        solution_indices = aggressive_local_search(solution_indices)
        
        if len(solution_indices) < best_size:
            best_size = len(solution_indices)
            best_solution = solution_indices
    
    if best_solution is None:
        # Fallback: return all words (trivially valid but large)
        best_solution = list(range(N))
    
    # Convert indices back to word lists
    return [all_words[i] for i in best_solution]


if __name__ == "__main__":
    # Test
    code = construct(3, 8, 1)
    print(f"Code size: {len(code)}")
    # Verify
    all_words = list(itertools.product(range(3), repeat=8))
    word_set = set(tuple(w) for w in code)
    
    def hamming_dist(a, b):
        return sum(1 for x, y in zip(a, b) if x != y)
    
    covered = 0
    for w in all_words:
        for c in word_set:
            if hamming_dist(w, c) <= 1:
                covered += 1
                break
    print(f"Covered: {covered}/{len(all_words)}")
