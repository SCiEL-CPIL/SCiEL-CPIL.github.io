# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

def _get_neighbors(c, q, n):
    """Return set of tuples at Hamming distance <= 1 from c (tuple)."""
    neighbors = {c}
    for i in range(n):
        orig = c[i]
        for val in range(q):
            if val == orig:
                continue
            lst = list(c)
            lst[i] = val
            neighbors.add(tuple(lst))
    return neighbors

def _local_search(code, q, n, radius, deadline, seed=0):
    """
    Local search: iterative removal + swap moves to minimize code size.
    code: list of tuples
    Returns: list of tuples (minimized code)
    """
    rng = random.Random(seed)
    all_words = list(itertools.product(range(q), repeat=n))
    all_words_set = set(all_words)
    
    # Precompute neighbors for all words in the space
    # This is feasible for n <= 8 (3^8 = 6561 words, each with 1+2n neighbors)
    if q**n * (1 + radius * n * (q-1)) > 500000:
        # Too large to precompute all; compute on the fly
        neighbors_map = None
    else:
        neighbors_map = {w: _get_neighbors(w, q, n) for w in all_words}
    
    def get_nb(c):
        if neighbors_map is not None:
            return neighbors_map[c]
        return _get_neighbors(c, q, n)
    
    # Start with the given code
    active = list(code)
    active_set = set(active)
    
    # Compute coverage counts
    coverage = {w: 0 for w in all_words}
    for c in active:
        for w in get_nb(c):
            coverage[w] += 1
    
    # Verify initial validity
    if any(cnt == 0 for cnt in coverage.values()):
        # Not valid; try to fix by adding words
        uncovered = [w for w, cnt in coverage.items() if cnt == 0]
        while uncovered and time.time() < deadline:
            # Pick the word that covers most uncovered
            best = None
            best_cnt = -1
            for w in all_words:
                cnt = sum(1 for x in get_nb(w) if coverage[x] == 0)
                if cnt > best_cnt:
                    best_cnt = cnt
                    best = w
            if best is None or best_cnt == 0:
                break
            if best in active_set:
                break
            active.append(best)
            active_set.add(best)
            for w in get_nb(best):
                coverage[w] += 1
            uncovered = [w for w in uncovered if coverage[w] == 0]
    
    # Local search: removal + swap
    best_size = len(active)
    best_code = list(active)
    
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Greedy removal
        while time.time() < deadline:
            to_remove = None
            for c in active:
                can_remove = True
                for w in get_nb(c):
                    if coverage[w] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    to_remove = c
                    break
            
            if to_remove is None:
                break
            
            # Remove it
            for w in get_nb(to_remove):
                coverage[w] -= 1
            active.remove(to_remove)
            active_set.discard(to_remove)
            improved = True
        
        # Phase 2: Try swap moves (remove one, add another) to escape local optima
        if time.time() < deadline:
            # Try a few random swaps
            for _ in range(50):
                if time.time() > deadline:
                    break
                if len(active) < 2:
                    break
                # Pick a random codeword to remove
                c_to_remove = rng.choice(active)
                # Check if removing it makes some words uncovered
                newly_uncovered = []
                for w in get_nb(c_to_remove):
                    if coverage[w] <= 1:
                        newly_uncovered.append(w)
                
                if not newly_uncovered:
                    # This is just a removal (shouldn't happen if phase 1 converged)
                    continue
                
                # Find a word that covers all newly_uncovered
                best_add = None
                best_score = -1
                for candidate in all_words:
                    if candidate in active_set:
                        continue
                    nb = get_nb(candidate)
                    # Check it covers all newly_uncovered
                    covers_all = all(w in nb for w in newly_uncovered)
                    if not covers_all:
                        continue
                    # Score: how many of its neighbors are already well-covered (minimize disruption)
                    # Or just accept the first valid swap
                    best_add = candidate
                    break
                
                if best_add is not None:
                    # Perform swap
                    for w in get_nb(c_to_remove):
                        coverage[w] -= 1
                    active.remove(c_to_remove)
                    active_set.discard(c_to_remove)
                    
                    for w in get_nb(best_add):
                        coverage[w] += 1
                    active.append(best_add)
                    active_set.add(best_add)
                    improved = True
                    break
        
        # Track best
        if len(active) < best_size:
            best_size = len(active)
            best_code = list(active)
        
        if not improved:
            # No improvement in this iteration; try a different random order
            # Shuffle active to break symmetry
            rng.shuffle(active)
            # If we've been stuck, try a small perturbation: remove a random word and re-add
            if time.time() < deadline - 5:
                if len(active) > 1:
                    # Perturbation: remove 1-2 random words, then re-run removal+add
                    perturb_count = min(2, len(active) - 1)
                    removed_words = []
                    for _ in range(perturb_count):
                        if not active:
                            break
                        c = rng.choice(active)
                        for w in get_nb(c):
                            coverage[w] -= 1
                        active.remove(c)
                        active_set.discard(c)
                        removed_words.append(c)
                    
                    # Now try to restore validity by adding words
                    for _ in range(100):
                        if time.time() > deadline:
                            break
                        uncovered = [w for w in all_words if coverage[w] == 0]
                        if not uncovered:
                            break
                        # Find word covering most uncovered
                        best = None
                        best_cnt = -1
                        for w in all_words:
                            if w in active_set:
                                continue
                            cnt = sum(1 for x in get_nb(w) if coverage[x] == 0)
                            if cnt > best_cnt:
                                best_cnt = cnt
                                best = w
                        if best is None or best_cnt == 0:
                            # Can't fix; revert
                            for c in removed_words:
                                if c not in active_set:
                                    for w in get_nb(c):
                                        coverage[w] += 1
                                    active.append(c)
                                    active_set.add(c)
                            removed_words = []
                            break
                        for w in get_nb(best):
                            coverage[w] += 1
                        active.append(best)
                        active_set.add(best)
                    
                    # Continue local search from here
                    improved = True
    
    # Return the best code found
    return best_code

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius 'radius'.
    Returns a list of codewords (each a list of ints in 0..q-1).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin (evaluator kills at 150s)
    
    if n == 0:
        return [[]]
    
    # For q=3, radius=1, n=4: perfect code (tetracode)
    if q == 3 and radius == 1 and n == 4:
        return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    # Build initial code
    if q == 3 and radius == 1 and n >= 4:
        tetracode = [(a, b, (a + b) % 3, (a + 2 * b) % 3) for a in range(3) for b in range(3)]
        init_code = [h + tail for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    elif q == 3 and radius == 1:
        # Small n: use greedy
        init_code = _greedy_cover(q, n, radius, deadline)
    else:
        # General case: greedy
        init_code = _greedy_cover(q, n, radius, deadline)
    
    # For small enough spaces, run local search with multiple restarts
    total_words = q ** n
    if total_words <= 6561:  # n <= 8 for q=3
        best_code = None
        best_size = float('inf')
        
        # Multiple restarts with different initial codes
        num_restarts = 5
        per_restart_time = (deadline - start_time) / num_restarts
        
        for restart in range(num_restarts):
            restart_deadline = start_time + (restart + 1) * per_restart_time
            if time.time() > restart_deadline - 2:
                break
            
            if restart == 0:
                init = [tuple(c) for c in init_code]
            else:
                # Perturbed initial code: take direct sum and randomly remove some, then fix
                if q == 3 and radius == 1 and n >= 4:
                    tetracode = [(a, b, (a + b) % 3, (a + 2 * b) % 3) for a in range(3) for b in range(3)]
                    full_code = [h + tail for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
                    # Randomly remove 10-20% of words
                    rng = random.Random(restart * 12345 + 678)
                    removal_frac = 0.10 + 0.05 * restart
                    num_remove = int(len(full_code) * removal_frac)
                    indices = list(range(len(full_code)))
                    rng.shuffle(indices)
                    remove_set = set(indices[:num_remove])
                    init = [c for i, c in enumerate(full_code) if i not in remove_set]
                else:
                    # Greedy with random tie-breaking
                    init = _greedy_cover(q, n, radius, restart_deadline, seed=restart)
            
            result = _local_search(init, q, n, radius, restart_deadline, seed=restart * 999 + 1)
            
            if len(result) < best_size:
                best_size = len(result)
                best_code = result
        
        if best_code is not None:
            # Final validation
            all_words = list(itertools.product(range(q), repeat=n))
            covered = set()
            for c in best_code:
                covered.update(_get_neighbors(c, q, n))
            if len(covered) == total_words:
                return [list(c) for c in best_code]
            # If invalid, fall through to return init_code
    
    # Fallback: return initial code (should be valid from construction)
    return [list(c) for c in init_code]

def _greedy_cover(q, n, radius, deadline, seed=0):
    """
    Greedy construction of a covering code.
    Returns list of tuples.
    """
    rng = random.Random(seed)
    all_words = list(itertools.product(range(q), repeat=n))
    
    # Precompute neighbors
    neighbors_map = {w: _get_neighbors(w, q, n) for w in all_words}
    
    uncovered = set(all_words)
    code = []
    
    while uncovered and time.time() < deadline:
        best_word = None
        best_count = -1
        for w in all_words:
            if w in set(code):
                continue
            cnt = sum(1 for x in neighbors_map[w] if x in uncovered)
            if cnt > best_count:
                best_count = cnt
                best_word = w
        
        if best_word is None or best_count == 0:
            break
        
        code.append(best_word)
        uncovered.difference_update(neighbors_map[best_word])
    
    return code
# EVOLVE-BLOCK-END
