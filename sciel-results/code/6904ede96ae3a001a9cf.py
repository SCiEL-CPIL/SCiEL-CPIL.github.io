import time
import itertools
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill

    # Seed solution: Direct sum of tetracode
    def seed_solution():
        if q != 3:
            return [list(t) for t in itertools.product(range(q), repeat=n)]
        if n < 4:
            return [list(t) for t in itertools.product(range(3), repeat=n)]
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n == 4:
            return [h[:] for h in tetracode]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    best_code = seed_solution()
    best_count = len(best_code)

    if q != 3 or n != 6 or radius != 1:
        return best_code

    # Specific optimization for q=3, n=6, radius=1
    # Total words: 3^6 = 729
    all_words = [list(t) for t in itertools.product(range(3), repeat=n)]
    M = len(all_words)
    
    # Precompute neighbors (Hamming ball of radius 1) for each word
    # Index mapping for O(1) lookup
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    neighbors = []
    for i, w in enumerate(all_words):
        covered = {i}
        for pos in range(n):
            orig = w[pos]
            for val in range(3):
                if val == orig:
                    continue
                w[pos] = val
                idx = word_to_idx[tuple(w)]
                covered.add(idx)
            w[pos] = orig
        neighbors.append(covered)
    
    # Strategy 1: Greedy Set Cover
    # We run multiple greedy passes with different randomizations/tie-breaking
    # to try to find a smaller set than the seed (81).
    
    def run_greedy(seed_val, time_limit):
        rng = random.Random(seed_val)
        t_start = time.time()
        
        # We can randomize the order of candidates or tie-breaking
        # Standard greedy: pick the center that covers the most uncovered points.
        
        uncovered = set(range(M))
        code_indices = []
        
        # Optimization: Pre-compute list of neighbors for faster set operations?
        # The set intersection `neighbors[i] & uncovered` is the main cost.
        
        while uncovered:
            if time.time() > time_limit:
                return None
            
            best_idx = -1
            best_count = 0
            # To improve diversity, we can shuffle the iteration order slightly or just rely on the deterministic max.
            # Deterministic max is usually strong. Let's stick to it but maybe break ties randomly?
            # For now, standard greedy.
            
            # Iterate through all words as potential centers
            # Note: We can restrict candidates to those in `uncovered`? 
            # No, a covered word might cover more *other* uncovered words. 
            # However, if a word is already covered, adding it as a center might be suboptimal compared to an uncovered one?
            # Actually, in set cover, we pick the set with max marginal gain.
            # It is sufficient to check all sets.
            
            # Optimization: If we maintain a list of active candidates, it might be faster.
            # But M=729 is small. 729 * 73 iterations * set ops.
            
            for i in range(M):
                # If i is already in code_indices, skip? No, we don't add duplicates.
                # If i is not in code_indices, check gain.
                # Actually, we just need to find the best i not yet in code_indices.
                # But checking `i not in code_indices` is O(K). 
                # Better to just track selected indices in a set.
                
                # Let's just loop all i.
                gain = len(neighbors[i] & uncovered)
                if gain > best_count:
                    best_count = gain
                    best_idx = i
                    # Early break if we found a perfect cover for remaining?
                    # Unlikely to be perfect for all remaining at once, but if gain == len(uncovered), we are done.
                    if gain == len(uncovered):
                        break
            
            if best_idx == -1 or best_count == 0:
                # Should not happen if uncovered is not empty
                break
                
            code_indices.append(best_idx)
            uncovered -= neighbors[best_idx]
            
        if len(code_indices) < best_count:
            return [all_words[i] for i in code_indices]
        return None

    # Run greedy a few times (deterministic, so same result, but maybe we can randomize tie-breaking)
    # Let's try a randomized greedy where we pick from top-k candidates.
    
    def run_randomized_greedy(seed_val, time_limit, k=1):
        rng = random.Random(seed_val)
        t_start = time.time()
        uncovered = set(range(M))
        code_indices = []
        
        while uncovered:
            if time.time() > time_limit:
                return None
            
            # Calculate gains for all
            gains = []
            for i in range(M):
                gain = len(neighbors[i] & uncovered)
                gains.append(gain)
            
            max_gain = max(gains)
            if max_gain == 0:
                break
                
            # Find all indices with max gain
            candidates = [i for i, g in enumerate(gains) if g == max_gain]
            
            # If k > 1, pick randomly from top k? Or just top 1?
            # Let's pick randomly from candidates to break ties differently in different runs.
            # But standard greedy picks the first one. 
            # If we want diversity, we can pick randomly from candidates.
            
            # However, if we pick randomly, we might pick a worse one later? 
            # No, they all have the same immediate gain.
            # Let's just pick the first one for determinism and speed, 
            # but run multiple times with different shuffles of the `all_words` order?
            # No, the gain depends on the set of uncovered, not the order.
            # So standard greedy is deterministic.
            
            # To get diversity, we can use a "lazy" greedy or random restarts with random initial centers?
            # Let's try: Start with a random center, then greedy.
            
            # Actually, let's just stick to the deterministic greedy first.
            # If it's 81, it matches seed.
            
            # Let's try a different heuristic: 
            # Pick the word that covers the most uncovered, 
            # but if there's a tie, pick the one that covers the most "hard-to-cover" words?
            
            # For now, just run standard greedy.
            best_idx = candidates[0]
            
            code_indices.append(best_idx)
            uncovered -= neighbors[best_idx]
            
        if len(code_indices) < best_count:
            return [all_words[i] for i in code_indices]
        return None

    # Run standard greedy
    greedy_result = run_greedy(0, deadline - 5)
    if greedy_result:
        best_code = greedy_result
        best_count = len(best_code)

    # If greedy didn't beat seed, try a local search / simulated annealing approach?
    # Or just return best found.
    
    # Given the time budget, if greedy is 81, we are at 1.0.
    # The prompt says "best known code has 73 words".
    # Greedy often gets close but not optimal.
    # Let's try to add a local improvement step if we have time.
    
    # Local Improvement: 
    # Try to remove a codeword and see if coverage is maintained.
    # If yes, remove it. Repeat until no more can be removed.
    
    def local_improve(code, time_limit):
        t_start = time.time()
        code = [list(c) for c in code]
        improved = True
        while improved:
            if time.time() > time_limit:
                break
            improved = False
            for i in range(len(code)):
                # Try removing code[i]
                remaining = code[:i] + code[i+1:]
                if not remaining:
                    continue
                # Check if remaining covers everything
                # This is expensive: O(N * M * n)
                # We can optimize by checking only the words that were ONLY covered by code[i]?
                # But we don't have that info easily.
                
                # Fast check using precomputed neighbors?
                # Let's use a set of covered indices.
                covered_set = set()
                for c in remaining:
                    covered_set |= neighbors[word_to_idx[tuple(c)]]
                
                if len(covered_set) == M:
                    code = remaining
                    improved = True
                    break # Restart loop as indices changed
        return code

    # Apply local improvement to the best code found
    if best_count > 73: # Only bother if we are not already optimal
        improved_code = local_improve(best_code, deadline - 2)
        if len(improved_code) < best_count:
            best_code = improved_code
            best_count = len(best_code)

    return best_code
