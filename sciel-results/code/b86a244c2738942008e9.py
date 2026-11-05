import numpy as np
import itertools
import time
import random
import math

def construct(q, n, radius):
    # Handle the specific high-interest case
    if q == 3 and n == 6 and radius == 1:
        return _construct_3_6_1()
    
    # Fallback for other parameters
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
    start_time = time.time()
    deadline = start_time + 115.0

    # Generate all words and map to indices
    words = [list(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}
    
    # Precompute neighbors as lists of indices
    neighbors = []
    for i, c in enumerate(words):
        neigh = [i]
        for pos in range(n):
            for new_val in range(q):
                if new_val == c[pos]:
                    continue
                new_w = list(c)
                new_w[pos] = new_val
                idx = word_to_idx[tuple(new_w)]
                neigh.append(idx)
        neighbors.append(neigh)
        
    # Precompute coverers: for each word w, list of codewords c such that w in neighbors[c]
    # This is the transpose of neighbors.
    coverers = [[] for _ in range(N)]
    for c in range(N):
        for w in neighbors[c]:
            coverers[w].append(c)

    # Initial code: 81 words (Tetracode x Identity)
    k4 = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    selected = set()
    for c in k4:
        for suffix in itertools.product(range(3), repeat=2):
            w = c + list(suffix)
            selected.add(word_to_idx[tuple(w)])
    
    # Coverage counts
    coverage_count = np.zeros(N, dtype=np.int32)
    for c in selected:
        for w in neighbors[c]:
            coverage_count[w] += 1
            
    best_selected = set(selected)
    best_size = len(selected)
    
    random.seed(42)
    
    def is_redundant(c, cov):
        for w in neighbors[c]:
            if cov[w] <= 1:
                return False
        return True
        # c is redundant if all its neighbors are covered by > 1 codeword
        # i.e., cov[w] > 1 for all w in neighbors[c]
        # Note: cov includes c itself.
        # So we check if min(cov[neighbors[c]]) > 1
        # Actually, if cov[w] == 1, it means only c covers w (or another codeword, but c is in selected).
        # If we remove c, cov[w] becomes 0.
        # So c is redundant if for all w in neighbors[c], cov[w] > 1.
        # Wait, if cov[w] == 1, it might be covered by c and no one else.
        # So yes, if min(cov[neighbors[c]]) > 1, c is redundant.
        # However, we need to be careful: cov[w] counts the number of selected codewords covering w.
        # If c is in selected, and cov[w]==1, then c is the only one covering w.
        # So removing c would uncover w.
        # Thus, c is redundant iff min(cov[neighbors[c]]) > 1.
        
        # Optimization: use numpy
        # But neighbors[c] is a list.
        # We can just loop.
        for w in neighbors[c]:
            if cov[w] <= 1:
                return False
        return True

    def get_uncovered_if_removed(c, cov):
        # Returns list of words w such that cov[w] == 1 (and c is in selected)
        # These are the words that would become uncovered if c is removed.
        u = []
        for w in neighbors[c]:
            if cov[w] == 1:
                u.append(w)
        return u

    def find_replacement(c, u, selected_set, deadline):
        # Find a codeword c' not in selected_set such that u is subset of neighbors[c']
        # i.e., c' covers all words in u.
        # If u is empty, any c' works (but we handle redundancy separately).
        if not u:
            return None # Handled by reduction
            
        # Optimization: If |u| is large, it's hard to find.
        # Iterate through all possible c'
        # We can prune: c' must be within distance 1 of all w in u.
        # This implies c' must be in the intersection of neighbors[w] for w in u.
        # Let's compute the intersection.
        candidates = set(neighbors[u[0]])
        for w in u[1:]:
            candidates.intersection_update(neighbors[w])
            if not candidates:
                return None
                
        # Now check if any candidate is not in selected_set
        for c_prime in candidates:
            if c_prime not in selected_set:
                return c_prime
        return None

    # Main Loop
    while time.time() < deadline - 1.0:
        # 1. Reduction Phase
        changed = True
        while changed:
            changed = False
            # Iterate over a copy of selected to allow modification
            for c in list(selected):
                if is_redundant(c, coverage_count):
                    # Remove c
                    for w in neighbors[c]:
                        coverage_count[w] -= 1
                    selected.remove(c)
                    changed = True
                    break # Restart reduction loop as coverage changed
            
        current_size = len(selected)
        if current_size < best_size:
            best_size = current_size
            best_selected = set(selected)
            
        # 2. Swap Phase
        if not selected:
            break
            
        # Pick a random codeword to try swapping
        c = random.choice(list(selected))
        
        # Find words that would become uncovered
        u = get_uncovered_if_removed(c, coverage_count)
        
        if not u:
            # Should have been removed in reduction phase
            # But maybe race condition or logic error?
            # Just remove it.
            for w in neighbors[c]:
                coverage_count[w] -= 1
            selected.remove(c)
            continue
            
        # Find replacement
        c_prime = find_replacement(c, u, selected, deadline)
        
        if c_prime is not None:
            # Perform swap
            # Remove c
            for w in neighbors[c]:
                coverage_count[w] -= 1
            selected.remove(c)
            
            # Add c_prime
            for w in neighbors[c_prime]:
                coverage_count[w] += 1
            selected.add(c_prime)
            
            # Check if this swap allowed further reduction?
            # The reduction loop at the top of the next iteration will handle it.
        else:
            # No valid swap found for this c.
            # Try another c? Or break?
            # To avoid getting stuck, we can try a few more random c's.
            # But for now, just continue the loop which picks a new random c.
            pass

    codewords = [words[i] for i in best_selected]
    return codewords
