import time
import itertools
import numpy as np
import random

_T1 = time.time()
_HARD = _T1 + 150.0


def _deadline(margin):
    return (time.time() + margin) < _HARD


def _build_neighbors(q, n):
    """
    Precompute neighbors for all words in {0..q-1}^n.
    Returns:
    - all_words: list of tuples
    - word_idx: dict mapping tuple to index
    - neighbors: list of lists of indices
    """
    N = q ** n
    # Generate all words
    all_words = list(itertools.product(range(q), repeat=n))
    word_idx = {w: i for i, w in enumerate(all_words)}
    
    neighbors = [[] for _ in range(N)]
    for i, w in enumerate(all_words):
        neighbors[i].append(i)
        for j in range(n):
            for d in range(1, q):
                new_w = list(w)
                new_w[j] = (new_w[j] + d) % q
                neighbors[i].append(word_idx[tuple(new_w)])
                
    return all_words, word_idx, neighbors


def _greedy_cover(q, n, radius, neighbors):
    """
    Greedy set cover to find an initial valid code.
    Returns list of indices.
    """
    N = q ** n
    uncovered = np.ones(N, dtype=bool)
    code = []
    
    # Precompute neighbor arrays as numpy arrays for faster sum if needed, 
    # but lists are fine for N=6561.
    
    while np.any(uncovered):
        best_idx = -1
        best_gain = 0
        
        # Iterate through all possible centers
        # Optimization: We can skip centers that are already covered? 
        # No, a covered center can still cover *new* uncovered neighbors.
        # However, if a center's neighbors are all covered, gain is 0.
        
        for i in range(N):
            # Quick check: if i is covered and all its neighbors are covered, gain is 0.
            # We just calculate gain directly.
            gain = 0
            for nb in neighbors[i]:
                if uncovered[nb]:
                    gain += 1
                    if gain > best_gain:
                        # Early exit if we found a perfect cover for all uncovered? 
                        # Unlikely in one step, but if gain == total_uncovered, we can break.
                        pass
            
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                if best_gain == 0:
                    # Should not happen if uncovered exists
                    break
        
        if best_idx == -1:
            break # Error or done
            
        code.append(best_idx)
        
        # Update uncovered
        for nb in neighbors[best_idx]:
            uncovered[nb] = False
            
    return code


def _local_search(q, n, radius, initial_code, neighbors, deadline_guard):
    """
    Local search to minimize code size.
    1. Peel redundant codewords.
    2. Swap codewords to enable further peeling.
    """
    N = q ** n
    sel_set = set(initial_code)
    sel_list = list(sel_set)
    
    # Coverage counts
    cov = np.zeros(N, dtype=np.int32)
    for c in sel_list:
        for nb in neighbors[c]:
            cov[nb] += 1
            
    best_size = len(sel_list)
    best_code = list(sel_list)
    
    improved = True
    while improved and time.time() < deadline_guard:
        improved = False
        
        # --- Phase 1: Peeling ---
        # Remove any codeword that is redundant (all neighbors covered by >= 2)
        changed = True
        while changed:
            changed = False
            new_sel_list = []
            for c in sel_list:
                # Check if c is redundant
                redundant = True
                for nb in neighbors[c]:
                    if cov[nb] < 2:
                        redundant = False
                        break
                if redundant:
                    # Remove c
                    for nb in neighbors[c]:
                        cov[nb] -= 1
                    sel_set.discard(c)
                    changed = True
                else:
                    new_sel_list.append(c)
            sel_list = new_sel_list
            
            if changed:
                if len(sel_list) < best_size:
                    best_size = len(sel_list)
                    best_code = list(sel_list)
                # Restart peel loop to catch new redundancies
                continue
        
        # --- Phase 2: Swapping ---
        if time.time() > deadline_guard - 5.0:
            break
            
        # Try to find a valid swap
        # We iterate through current codewords and try to replace them
        # To be efficient, we can randomize the order or just iterate.
        # We'll iterate through a copy of the list to avoid modification issues during iteration
        # But we break on first successful swap to re-evaluate peeling.
        
        swap_found = False
        # Shuffle to avoid bias towards early indices
        candidates_to_try = sel_list[:]
        random.shuffle(candidates_to_try)
        
        for c in candidates_to_try:
            if c not in sel_set:
                continue # Might have been removed? No, we just peeled.
                
            # Identify words that would become uncovered if c is removed
            # U = { nb in N(c) | cov[nb] == 1 }
            U = []
            for nb in neighbors[c]:
                if cov[nb] == 1:
                    U.append(nb)
                    
            if not U:
                # Should have been peeled
                continue
                
            # Find a replacement w not in sel_set such that U subset N(w)
            # We can iterate through all w in 0..N-1 not in sel_set
            # Optimization: 
            # For w to cover U, w must be in the intersection of N(u) for u in U.
            # Let's compute the intersection.
            
            # Start with N(U[0])
            common = set(neighbors[U[0]])
            for u in U[1:]:
                common &= set(neighbors[u])
                if not common:
                    break
            
            if not common:
                continue
                
            # Filter out current codewords
            candidates = [w for w in common if w not in sel_set]
            
            if not candidates:
                continue
                
            # Pick a candidate. 
            # Heuristic: Pick one that covers the most *other* words? 
            # Or just pick the first one.
            # Let's pick the first one for speed.
            w = candidates[0]
            
            # Perform swap
            # Remove c
            for nb in neighbors[c]:
                cov[nb] -= 1
            sel_set.discard(c)
            sel_list.remove(c)
            
            # Add w
            for nb in neighbors[w]:
                cov[nb] += 1
            sel_set.add(w)
            sel_list.append(w)
            
            improved = True
            swap_found = True
            break # Restart loop to re-evaluate peeling
            
    return best_code


def construct(q, n, radius):
    global _HARD
    _HARD = time.time() + 150.0
    
    if n < 0:
        return []
    if n == 0:
        return [[]]
        
    # Specific optimization for the main task: q=3, n=8, radius=1
    if q == 3 and n == 8 and radius == 1:
        all_words, word_idx, neighbors = _build_neighbors(q, n)
        
        # 1. Greedy Construction
        initial_code = _greedy_cover(q, n, radius, neighbors)
        
        # 2. Local Search
        deadline_guard = _HARD - 10.0
        improved_code = _local_search(q, n, radius, initial_code, neighbors, deadline_guard)
        
        # Ensure validity (should be guaranteed by local search, but just in case)
        # The local search maintains validity by construction (only valid swaps/removals).
        # However, if it times out, it returns the best valid code found.
        
        return [list(all_words[i]) for i in improved_code]
        
    # Fallback for other parameters
    # Simple direct sum or greedy
    # For general q, n, radius, greedy is a good start
    all_words, word_idx, neighbors = _build_neighbors(q, n)
    initial_code = _greedy_cover(q, n, radius, neighbors)
    
    # Try local search if time permits
    deadline_guard = _HARD - 10.0
    if time.time() < deadline_guard:
        improved_code = _local_search(q, n, radius, initial_code, neighbors, deadline_guard)
    else:
        improved_code = initial_code
        
    return [list(all_words[i]) for i in improved_code]

# EVOLVE-BLOCK-END
