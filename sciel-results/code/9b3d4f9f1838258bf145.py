import time
import random
import numpy as np
from itertools import combinations

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for conversion and cleanup
    
    total_space = q ** n
    
    # Precompute powers for index conversion
    powers = [1] * n
    for i in range(n-1, -1, -1):
        if i == n-1:
            powers[i] = 1
        else:
            powers[i] = powers[i+1] * q
    # Correct powers: index = sum(d_i * q^i)
    # So power[i] = q^i
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # Precompute digits for all words
    digits_all = np.zeros((total_space, n), dtype=np.int8)
    for idx in range(total_space):
        val = idx
        for pos in range(n):
            digits_all[idx, pos] = val % q
            val //= q

    # Precompute coverage masks as lists of indices for each word
    # This is used for fast gain calculation and local search
    # For radius=1, each word covers 1 + n*(q-1) words.
    # For generic radius, we compute neighbors.
    
    def get_neighbors(j):
        # Returns list of indices covered by word j
        # Optimization: generate neighbors directly
        neighbors = [j]
        # Generate neighbors within radius
        # For radius 1:
        if radius >= 1:
            for pos in range(n):
                orig_val = (j // powers[pos]) % q
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    i = j + (new_val - orig_val) * powers[pos]
                    neighbors.append(i)
        if radius >= 2:
            for pos1 in range(n):
                for pos2 in range(pos1 + 1, n):
                    v1 = (j // powers[pos1]) % q
                    v2 = (j // powers[pos2]) % q
                    for nv1 in range(q):
                        if nv1 == v1: continue
                        for nv2 in range(q):
                            if nv2 == v2: continue
                            i = j + (nv1 - v1) * powers[pos1] + (nv2 - v2) * powers[pos2]
                            neighbors.append(i)
        return neighbors

    # Precompute all neighbor lists
    # Note: This takes some time but is O(N * 2^R * N) roughly, which is fine for N=2187
    neighbor_lists = []
    for j in range(total_space):
        neighbor_lists.append(get_neighbors(j))
    
    # Convert to numpy arrays for vectorized operations where possible?
    # Actually, for local search, we need to quickly check if removing a word breaks coverage.
    # We maintain a count array: coverage_count[i] = number of selected words covering i.
    
    def greedy_construct(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
        
        remaining = set(range(total_space))
        selected = []
        
        # Precompute a list of all words for randomization
        all_words = list(range(total_space))
        
        while remaining:
            best_gain = -1
            best_word = -1
            # To speed up, we can sample a subset if time is tight, but for 2187 words, 
            # full scan is ~2187 * (cost of gain calc).
            # Gain calc: sum(1 for i in neighbors[w] if i in remaining)
            # Using a set 'remaining', membership is O(1).
            # So one pass is O(N * avg_neighbors). 2187 * 15 ~ 32k ops. Very fast.
            
            # Random tie-breaking
            candidates = all_words
            # Shuffle for random tie-breaking? No, just pick max gain, break ties randomly.
            # To do this efficiently:
            # 1. Calculate gains for all
            # 2. Find max gain
            # 3. Pick random from those with max gain
            
            # Optimization: We can maintain a list of "active" candidates?
            # No, simple loop is fine.
            
            max_gain = -1
            best_candidates = []
            
            for w in all_words:
                if w in selected: # Should not happen if we track remaining properly, but safe
                    continue
                gain = 0
                for i in neighbor_lists[w]:
                    if i in remaining:
                        gain += 1
                if gain > max_gain:
                    max_gain = gain
                    best_candidates = [w]
                elif gain == max_gain:
                    best_candidates.append(w)
            
            if not best_candidates:
                # Fallback: should not happen
                break
                
            w = random.choice(best_candidates)
            selected.append(w)
            for i in neighbor_lists[w]:
                if i in remaining:
                    remaining.remove(i)
                    
        return selected

    def local_search(selected, time_limit=2.0):
        # Convert to set for fast membership
        selected_set = set(selected)
        
        # Calculate initial coverage counts
        coverage = np.zeros(total_space, dtype=np.int32)
        for w in selected_set:
            for i in neighbor_lists[w]:
                coverage[i] += 1
        
        # A word is "critical" if it is the only one covering some word.
        # We want to remove a word if it is NOT critical (i.e., all words it covers are covered by at least 2 words).
        # And we want to add a word if it reduces the size? No, we want to minimize size.
        # So we look for a word w in selected such that removing it keeps coverage >= 1 for all.
        # This happens iff for all i in neighbors[w], coverage[i] >= 2.
        
        # We also look for swaps: remove w1, add w2, if size stays same but maybe allows further removal?
        # Or just remove w1 if possible.
        
        # Strategy:
        # 1. Try to remove any non-critical word.
        # 2. If no removal possible, try a 1-swap: remove w1, add w2.
        #    This is useful if w1 is "blocking" a better configuration? 
        #    Actually, for set cover, if no single removal works, a swap might allow a future removal.
        #    But simpler: just try to remove. If stuck, try random restarts or more complex moves.
        
        # Let's implement "Remove if possible" loop.
        # To make it faster, we can iterate through selected words and check criticality.
        
        # Precompute a boolean array is_critical?
        # is_critical[w] = any(coverage[i] == 1 for i in neighbors[w])
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Check each selected word
            # To avoid O(N^2) in the worst case, we can just iterate.
            # N is ~180. Neighbors ~15. Check is 180 * 15 = 2700 ops. Very fast.
            
            for w in list(selected_set):
                # Check if w is critical
                critical = False
                for i in neighbor_lists[w]:
                    if coverage[i] == 1:
                        critical = True
                        break
                if not critical:
                    # Remove w
                    selected_set.remove(w)
                    for i in neighbor_lists[w]:
                        coverage[i] -= 1
                    improved = True
                    break # Restart loop after change
        
        # Now try 1-swap: remove w1, add w2.
        # This is harder to evaluate quickly.
        # Condition: After removing w1, coverage[i] >= 1 for all i.
        # This requires that for all i in neighbors[w1], coverage[i] >= 2 (so w1 is not critical) OR
        # w2 covers those i where coverage becomes 0.
        # Actually, if w1 is critical, removing it makes some i have coverage 0.
        # w2 must cover all such i.
        # So: Let U = {i in neighbors[w1] | coverage[i] == 1}.
        # If U is empty, we could just remove w1 (handled above).
        # If U is not empty, we need w2 such that U is a subset of neighbors[w2].
        # And we want to do this if it leads to a better solution later?
        # For now, let's just do removals. Swaps are complex.
        # Given the time, removals are the primary driver of improvement from greedy.
        
        return list(selected_set)

    best_solution = None
    best_size = total_space + 1
    
    # Run multiple greedy restarts
    # Each restart: Greedy -> Local Search
    # Estimate time per restart:
    # Greedy: ~180 steps * 32k ops = ~6M ops. ~0.1s in Python? Maybe 0.5s.
    # Local Search: ~180 words * 15 neighbors * few iterations. ~10ms.
    # Total ~0.5s per restart.
    # 115s / 0.5s = 230 restarts.
    
    # To be safe, let's time the first restart and adjust.
    
    for i in range(300): # Max restarts
        if time.time() > deadline - 5.0:
            break
            
        t0 = time.time()
        sol = greedy_construct(random_seed=i)
        sol = local_search(sol, time_limit=1.0)
        t1 = time.time()
        
        # Verify solution validity (just in case)
        # For efficiency, we assume local_search maintained validity.
        # But let's do a quick check on size.
        if len(sol) < best_size:
            best_size = len(sol)
            best_solution = sol
            
        # Adaptive restart frequency?
        # If we are fast, do more.
        
    if best_solution is None:
        # Fallback: Return all words (valid but bad)
        return [[int(d) for d in digits_all[i]] for i in range(total_space)]
        
    # Convert best_solution indices to lists of digits
    result = []
    for idx in best_solution:
        result.append([int(d) for d in digits_all[idx]])
        
    return result

# Simple self-test if run directly
if __name__ == "__main__":
    # Test with small params
    # q=3, n=3, radius=1
    # Total space 27.
    # Best known?
    # Let's just run the main target
    # q=3, n=7, radius=1
    # res = construct(3, 7, 1)
    # print(len(res))
    pass
