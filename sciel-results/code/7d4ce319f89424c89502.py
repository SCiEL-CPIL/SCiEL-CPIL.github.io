import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    if q == 3 and n == 6 and radius == 1:
        return _construct_3_6_1()
    
    # Fallback for other parameters
    if q == 3:
        if n >= 4:
            # Simple product construction for fallback
            # This is not optimal but valid
            return [list(w) for w in itertools.product(range(q), repeat=n)]
    return [list(w) for w in itertools.product(range(q), repeat=n)]

def _construct_3_6_1():
    q = 3
    n = 6
    N = q ** n  # 729
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin

    # Generate all words
    words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    
    # Precompute neighbors for each word (word + all words at distance 1)
    # Shape: (N, 1 + n*(q-1)) = (729, 13)
    neighbors = np.zeros((N, 1 + n * (q - 1)), dtype=np.int32)
    for i in range(N):
        c = words[i]
        neighbors[i, 0] = i
        idx = 1
        for pos in range(n):
            for new_val in range(q):
                if new_val == c[pos]:
                    continue
                new_w = c.copy()
                new_w[pos] = new_val
                # Convert to index: treat word as base-q number
                val = 0
                for k in range(n):
                    val = val * q + new_w[k]
                neighbors[i, idx] = val
                idx += 1

    def get_coverage(selected):
        """Calculate coverage counts for a set of codewords."""
        cov = np.zeros(N, dtype=np.int32)
        for j in selected:
            cov[neighbors[j, :]] += 1
        return cov

    def is_valid_cov(cov):
        return np.all(cov >= 1)

    # 1. Greedy Initialization
    # We use a greedy set cover.
    uncovered = np.ones(N, dtype=bool)
    selected = []
    
    # Symmetry breaking: fix first word to 0
    selected.append(0)
    uncovered[neighbors[0, :]] = False
    
    while uncovered.any():
        if time.time() > deadline - 60:
            break
        # Find word covering most uncovered points
        # Efficiently: for each candidate, count overlaps with uncovered
        # Since N=729 and candidates=729, this is 729*729*13 ~ 6.8M ops per step.
        # Steps ~ 80. Total ~ 500M ops. Might be slow in pure Python loops.
        # Vectorize:
        # counts[j] = sum(uncovered[neighbors[j, :]])
        counts = np.sum(uncovered[neighbors], axis=1)
        best_idx = int(np.argmax(counts))
        if counts[best_idx] == 0:
            break
        selected.append(best_idx)
        uncovered[neighbors[best_idx, :]] = False

    best_selected = list(selected)
    best_cov = get_coverage(best_selected)
    best_size = len(best_selected)

    random.seed(42)

    # Local Search
    # We try to reduce the size.
    # Moves:
    # 1. Remove redundant
    # 2. Swap 1 for 1
    # 3. Swap 2 for 1 (remove u, v, add w)
    
    # Precompute neighbor sets as frozensets for faster intersection? 
    # Or just use numpy arrays.
    
    def try_reduce(selected, cov, time_limit):
        """Try to remove redundant words."""
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            for i in range(len(selected)):
                j = selected[i]
                # Check if all neighbors of j are covered by >1
                if np.all(cov[neighbors[j, :]] > 1):
                    cov[neighbors[j, :]] -= 1
                    selected.pop(i)
                    changed = True
                    break
        return selected

    # Main search loop
    stagnation_counter = 0
    max_stagnation = 100
    
    while time.time() < deadline - 5:
        # Step 1: Reduce
        best_selected = try_reduce(best_selected, best_cov, deadline - 5)
        if len(best_selected) < best_size:
            best_size = len(best_selected)
            stagnation_counter = 0
            # Update cov if we modified it in try_reduce? 
            # try_reduce modifies cov in place.
            # If we broke early, cov is consistent with best_selected.
            continue
        else:
            # If no reduction, cov is still consistent.
            pass

        # Step 2: Try 2-for-1 moves
        # This is the key to breaking through.
        # For each pair (u, v) in selected, see if there is a w not in selected
        # such that removing u and v and adding w keeps coverage valid.
        
        # Optimization: Only consider pairs where u and v have low "criticality".
        # Criticality: number of words covered ONLY by this codeword.
        # If criticality is high, removing it is hard.
        
        # Calculate criticality for all selected
        # unique_cov[j] = number of words w such that cov[w] == 1 and j in neighbors[w]
        # This is expensive to compute from scratch every time.
        # Approximation: count words with cov==1 in neighbors[j].
        
        # Let's try a simpler random swap first to escape local minima.
        # If we haven't improved in a while, do a random perturbation.
        
        if stagnation_counter > 20:
            # Perturb: remove a random word, add a random word that helps
            if len(best_selected) > 1:
                idx = random.randint(0, len(best_selected) - 1)
                u = best_selected[idx]
                # Remove u
                best_cov[neighbors[u, :]] -= 1
                # Find uncovered
                uncovered_mask = best_cov < 1
                if np.any(uncovered_mask):
                    # Find w that covers most uncovered
                    counts = np.sum(uncovered_mask[neighbors], axis=1)
                    # Exclude selected
                    for s in best_selected:
                        counts[s] = 0
                    w = int(np.argmax(counts))
                    if counts[w] > 0:
                        # Add w
                        best_cov[neighbors[w, :]] += 1
                        best_selected[idx] = w
                    else:
                        # Revert
                        best_cov[neighbors[u, :]] += 1
                else:
                    # u was redundant, remove it
                    best_selected.pop(idx)
                    best_size = len(best_selected)
                    stagnation_counter = 0
                    continue
            stagnation_counter = 0
        else:
            # Try 2-for-1
            # Pick a random pair u, v from selected
            if len(best_selected) >= 2:
                i1, i2 = random.sample(range(len(best_selected)), 2)
                u = best_selected[i1]
                v = best_selected[i2]
                
                # Tentatively remove u and v
                best_cov[neighbors[u, :]] -= 1
                best_cov[neighbors[v, :]] -= 1
                
                uncovered_mask = best_cov < 1
                if not np.any(uncovered_mask):
                    # Both were redundant! Remove both
                    # Need to remove from list carefully
                    # Sort indices
                    if i1 > i2:
                        i1, i2 = i2, i1
                    best_selected.pop(i1)
                    best_selected.pop(i2)
                    best_size = len(best_selected)
                    stagnation_counter = 0
                    continue
                
                # Find w that covers all uncovered words
                # w must cover every word in uncovered_mask
                # This is an intersection problem.
                # For each candidate w, check if neighbors[w] covers uncovered_mask
                # i.e., uncovered_mask[neighbors[w, :]] is all True?
                # No, neighbors[w] must be a superset of the set of indices where uncovered_mask is True.
                
                # Efficient check:
                # Let U be the set of indices where uncovered_mask is True.
                # We need w such that U subset of neighbors[w].
                # Equivalent: sum(uncovered_mask[neighbors[w, :]]) == sum(uncovered_mask)
                
                target_count = np.sum(uncovered_mask)
                # Compute for all w
                # counts[w] = sum(uncovered_mask[neighbors[w, :]])
                # We want max counts[w] == target_count
                
                # Vectorized
                # uncovered_mask is boolean array of size N
                # neighbors is (N, 13)
                # counts = np.sum(uncovered_mask[neighbors], axis=1)
                
                # This is 729 * 13 ops. Very fast.
                counts = np.sum(uncovered_mask[neighbors], axis=1)
                
                # Exclude current selected
                for s in best_selected:
                    counts[s] = 0
                
                w = int(np.argmax(counts))
                
                if counts[w] == target_count:
                    # Valid move: remove u, v, add w
                    best_cov[neighbors[w, :]] += 1
                    # Update selected list
                    # Remove u and v, add w
                    # Note: u and v are still in best_selected
                    # We need to replace them with w? No, size reduces by 1.
                    # Remove u and v from list.
                    # If w is already in list? We excluded selected, so w is not in list.
                    
                    # Remove indices i1, i2
                    if i1 > i2:
                        i1, i2 = i2, i1
                    best_selected.pop(i1)
                    best_selected.pop(i2)
                    best_selected.append(w)
                    
                    best_size = len(best_selected)
                    stagnation_counter = 0
                    continue
                else:
                    # Revert
                    best_cov[neighbors[u, :]] += 1
                    best_cov[neighbors[v, :]] += 1
        
        stagnation_counter += 1

    # Final reduction
    best_selected = try_reduce(best_selected, best_cov, deadline - 1)
    
    codewords = [words[i].tolist() for i in best_selected]
    return codewords
