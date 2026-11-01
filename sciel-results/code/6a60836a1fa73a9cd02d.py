import itertools
import numpy as np
import time
import random

def construct(q, n, radius):
    if q == 3 and n == 7 and radius == 1:
        return _construct_ils_3_7_1()
    
    # Fallback for other parameters: simple greedy
    return _greedy_cover(q, n, radius)

def _construct_ils_3_7_1():
    """
    Iterated Local Search for K_3(7,1).
    1. Precompute neighbors.
    2. Randomized Greedy to get initial solution.
    3. Local Minimization (remove redundant codewords).
    4. Perturbation (remove random subset).
    5. Repair (Greedy add to cover holes).
    6. Repeat.
    """
    N = 3 ** 7
    all_words = list(itertools.product(range(3), repeat=7))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors for each of the N possible codewords
    # neighbors[i] is a list of indices covered by codeword i
    neighbors = [None] * N
    for i, w in enumerate(all_words):
        covered = [i]
        for pos in range(7):
            for delta in range(1, 3):
                new_w = list(w)
                new_w[pos] = (w[pos] + delta) % 3
                idx = word_to_idx[tuple(new_w)]
                covered.append(idx)
        neighbors[i] = covered
        
    # State variables
    best_code = []
    best_size = float('inf')
    
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer
    
    # Randomized Greedy Initialization
    def randomized_greedy():
        uncovered = set(range(N))
        code = []
        # We can randomize the tie-breaking or the order of evaluation
        # Here we just evaluate all, but pick randomly among the best if tied
        while uncovered:
            best_candidates = []
            max_count = 0
            for i in range(N):
                # Optimization: if i is not in uncovered, it might still cover uncovered words.
                # But usually, greedy picks from all.
                # Count intersection
                count = 0
                for u in neighbors[i]:
                    if u in uncovered:
                        count += 1
                if count > max_count:
                    max_count = count
                    best_candidates = [i]
                elif count == max_count and count > 0:
                    best_candidates.append(i)
            
            if not best_candidates:
                break # Should not happen if uncovered is not empty
                
            # Pick randomly from best candidates to diversify
            chosen = random.choice(best_candidates)
            code.append(chosen)
            for u in neighbors[chosen]:
                uncovered.discard(u)
        return code

    def minimize(code):
        """Remove redundant codewords from the list 'code'. Returns new list."""
        # Use cover_count array for efficiency
        cover_count = np.zeros(N, dtype=np.int32)
        for c in code:
            for u in neighbors[c]:
                cover_count[u] += 1
        
        # Try to remove each codeword
        # We iterate in random order to avoid bias
        indices_to_check = list(range(len(code)))
        random.shuffle(indices_to_check)
        
        new_code = list(code)
        for idx in indices_to_check:
            c = new_code[idx]
            # Temporarily remove
            for u in neighbors[c]:
                cover_count[u] -= 1
            
            # Check if all covered
            # np.any(cover_count == 0) is fast
            if not np.any(cover_count == 0):
                # Keep it removed
                new_code.pop(idx)
            else:
                # Revert
                for u in neighbors[c]:
                    cover_count[u] += 1
        return new_code

    def repair(code):
        """Add codewords to 'code' to ensure full coverage. Returns new list."""
        cover_count = np.zeros(N, dtype=np.int32)
        for c in code:
            for u in neighbors[c]:
                cover_count[u] += 1
        
        # Find uncovered
        uncovered_idx = np.where(cover_count == 0)[0]
        uncovered_set = set(uncovered_idx.tolist())
        
        new_code = list(code)
        
        while uncovered_set:
            best_candidates = []
            max_count = 0
            for i in range(N):
                count = 0
                for u in neighbors[i]:
                    if u in uncovered_set:
                        count += 1
                if count > max_count:
                    max_count = count
                    best_candidates = [i]
                elif count == max_count and count > 0:
                    best_candidates.append(i)
            
            if not best_candidates:
                # This shouldn't happen, but break to avoid infinite loop
                break
                
            chosen = random.choice(best_candidates)
            new_code.append(chosen)
            for u in neighbors[chosen]:
                if u in uncovered_set:
                    uncovered_set.discard(u)
                    # We don't strictly need to update cover_count here for the set logic,
                    # but if we were using cover_count for validity, we would.
                    # Here we just track the set of uncovered.
        
        return new_code

    # Main ILS Loop
    # Initial solution
    current_code = randomized_greedy()
    current_code = minimize(current_code)
    best_code = current_code
    best_size = len(current_code)
    
    iteration = 0
    while time.time() - start_time < time_limit:
        iteration += 1
        
        # Perturbation: Remove k random codewords
        # k can vary, e.g., 5 to 15
        k = random.randint(5, 15)
        if len(current_code) > k:
            # Select k indices to remove
            remove_indices = random.sample(range(len(current_code)), k)
            # Create a set of indices to remove to avoid duplicates in loop logic if needed
            # Actually, we just build a new list excluding these
            # We need to be careful: if we remove them, we must repair.
            
            # To speed up, we can just pick the codewords
            codewords_to_remove = [current_code[i] for i in remove_indices]
            # Build new list
            new_code = [c for i, c in enumerate(current_code) if i not in remove_indices]
            
            # Repair
            new_code = repair(new_code)
            
            # Minimize
            new_code = minimize(new_code)
            
            if len(new_code) < best_size:
                best_size = len(new_code)
                best_code = new_code
                current_code = new_code
            else:
                # Accept with some probability to escape local minima?
                # Or just keep current if not improved?
                # ILS usually keeps the best, but sometimes accepts worse to move.
                # Let's accept if it's not significantly worse, or with probability.
                # For now, let's just keep the best found so far as the anchor,
                # but use the new one as the next starting point?
                # Standard ILS: current = new.
                # But if new is much worse, we might get stuck in a bad region.
                # Let's try: if len(new_code) < len(current_code) + 5: current_code = new_code
                # This allows small degradations.
                if len(new_code) < len(current_code) + 10:
                    current_code = new_code
                else:
                    current_code = best_code # Reset to best if perturbation was too destructive?
                    # Or just keep current?
                    # Let's reset to best to re-perturb from a good state.
        
    return [list(all_words[c]) for c in best_code]

def _greedy_cover(q, n, radius):
    """General greedy set cover for covering codes."""
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    N = len(all_words)
    
    # Precompute covers for each word
    covers = []
    for w in all_words:
        covered = set()
        covered.add(word_to_idx[w])
        for r in range(1, radius + 1):
            for positions in itertools.combinations(range(n), r):
                for deltas in itertools.product(range(1, q), repeat=r):
                    new_w = list(w)
                    for pos, d in zip(positions, deltas):
                        new_w[pos] = (w[pos] + d) % q
                    covered.add(word_to_idx[tuple(new_w)])
        covers.append(covered)
    
    uncovered = set(range(N))
    code = []
    while uncovered:
        best = None
        best_count = -1
        for i in range(N):
            count = len(covers[i] & uncovered)
            if count > best_count:
                best_count = count
                best = i
        code.append(list(all_words[best]))
        uncovered -= covers[best]
    
    return code
