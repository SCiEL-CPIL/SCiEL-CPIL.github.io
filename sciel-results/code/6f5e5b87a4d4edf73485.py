import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave margin for 120s limit

    # Fallback for other parameters (not expected in this specific task)
    if q != 3 or n != 7 or radius != 1:
        if n <= 4:
            if q == 3 and radius == 1:
                return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(4)]
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(w) for w in itertools.product(range(3), repeat=n)]

    N = 3 ** n  # 2187

    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % 3
            idx //= 3
        return t

    def tuple_to_idx(t):
        idx = 0
        for x in t:
            idx = idx * 3 + x
        return idx

    # Precompute neighbor indices for each word
    # neighbors[i] is a list of indices of words within Hamming distance 1 of word i
    neighbors = []
    for i in range(N):
        w = idx_to_tuple(i)
        nb = [i]
        for pos in range(n):
            val = w[pos]
            for new_val in range(q):
                if new_val != val:
                    new_w = w[:]
                    new_w[pos] = new_val
                    nb.append(tuple_to_idx(new_w))
        neighbors.append(nb)

    # Precompute the Cover Matrix M
    # M is a boolean matrix of shape (N, N)
    # M[i, j] is True if word i is covered by codeword j (i.e., i is in neighbors[j])
    # This allows vectorized gain calculation: gain = M[uncovered, :].sum(axis=0)
    M = np.zeros((N, N), dtype=np.bool_)
    for j in range(N):
        for i in neighbors[j]:
            M[i, j] = True

    # ---------- Vectorized Greedy Construction ----------
    
    def greedy_construct(seed_val):
        rng = random.Random(seed_val)
        # coverage[i] is the number of codewords covering word i
        coverage = np.zeros(N, dtype=np.int32)
        # uncovered_mask[i] is True if word i is not covered
        uncovered_mask = np.ones(N, dtype=np.bool_)
        
        code = []
        code_set = set()
        
        # Random start
        start = rng.randint(0, N - 1)
        code.append(start)
        code_set.add(start)
        coverage[neighbors[start]] += 1
        uncovered_mask[neighbors[start]] = False
        
        while uncovered_mask.any() and time.time() < deadline:
            # Vectorized gain calculation
            # M[uncovered_mask, :] gives a boolean matrix where rows are uncovered words.
            # Summing along axis 0 gives the number of uncovered words covered by each codeword.
            gains = M[uncovered_mask, :].sum(axis=0)
            
            # We only consider codewords not already in the code
            # Mask out existing codewords
            for c in code:
                gains[c] = 0
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            # Find all codewords with max gain
            best_candidates = np.where(gains == max_gain)[0]
            
            # Random tie-breaking
            chosen = best_candidates[rng.randint(0, len(best_candidates) - 1)]
            
            code.append(chosen)
            code_set.add(chosen)
            coverage[neighbors[chosen]] += 1
            uncovered_mask[neighbors[chosen]] = False
            
        return code, code_set, coverage, uncovered_mask

    # Run multiple greedy restarts
    best_code = None
    best_code_set = None
    best_coverage = None
    best_uncovered = None
    best_size = float('inf')
    
    num_restarts = 20
    for i in range(num_restarts):
        if time.time() > deadline - 40:  # Keep time for local search
            break
        c, cs, cov, um = greedy_construct(1000 + i)
        if len(c) < best_size:
            best_size = len(c)
            best_code = c
            best_code_set = cs
            best_coverage = cov
            best_uncovered = um
            
    if best_code is None:
        c, cs, cov, um = greedy_construct(0)
        best_code = c
        best_code_set = cs
        best_coverage = cov
        best_uncovered = um

    code = best_code
    code_set = best_code_set
    coverage = best_coverage
    uncovered_mask = best_uncovered

    # ---------- Vectorized Local Search (Kick & Repair) ----------
    
    def greedy_repair(uncovered_indices, current_code_set):
        """
        Greedily repair uncovered indices by adding codewords.
        Returns list of added codewords.
        """
        if not uncovered_indices:
            return []
        
        added = []
        temp_uncovered = set(uncovered_indices)
        
        while temp_uncovered:
            # Convert to list for indexing M
            unc_list = list(temp_uncovered)
            
            # Vectorized gain: how many of the temp_uncovered does each candidate cover?
            # M[unc_list, :] has shape (len(unc_list), N)
            # Sum along axis 0 gives gains for all N candidates
            gains = M[np.array(unc_list), :].sum(axis=0)
            
            # Mask out current code and already added
            for c in current_code_set:
                gains[c] = 0
            for c in added:
                gains[c] = 0
                
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            best_candidates = np.where(gains == max_gain)[0]
            # Random choice among best
            chosen = best_candidates[random.randint(0, len(best_candidates) - 1)]
            
            added.append(chosen)
            # Update temp_uncovered
            # Remove all words covered by 'chosen' from temp_uncovered
            covered_by_chosen = set(neighbors[chosen])
            temp_uncovered -= covered_by_chosen
            
        return added

    def apply_change(removed, added):
        nonlocal code, code_set, coverage, uncovered_mask
        for c in removed:
            code.remove(c)
            code_set.remove(c)
            coverage[neighbors[c]] -= 1
            # Update uncovered mask: if coverage becomes 0, it's uncovered
            # We re-calculate uncovered_mask from scratch to be safe and simple, 
            # as N is small (2187)
        for c in added:
            if c not in code_set:
                code.append(c)
                code_set.add(c)
                coverage[neighbors[c]] += 1
        # Recalculate uncovered mask efficiently
        uncovered_mask = (coverage == 0)

    def try_kick(k, rng):
        """
        Try to remove k random codewords and repair.
        Returns True if size decreased.
        """
        if len(code) < k:
            return False
            
        # Pick k distinct random indices
        indices = rng.sample(range(len(code)), k)
        removed = [code[i] for i in indices]
        
        # Temporarily remove to check coverage
        temp_coverage = coverage.copy()
        for c in removed:
            temp_coverage[neighbors[c]] -= 1
            
        uncovered = np.where(temp_coverage == 0)[0]
        
        if len(uncovered) == 0:
            # Perfect removal
            apply_change(removed, [])
            return True
            
        uncovered_set = set(uncovered.tolist())
        added = greedy_repair(uncovered_set, code_set)
        
        new_size = len(code) - len(removed) + len(added)
        
        if new_size < len(code):
            apply_change(removed, added)
            return True
        else:
            return False

    # Main Local Search Loop
    rng = random.Random(42)
    
    # Initial cleanup: remove redundant codewords
    changed = True
    while changed:
        changed = False
        for c in list(code):
            # Check if c is redundant: all its neighbors are covered by others
            # i.e., coverage[neighbors[c]] > 1 for all neighbors
            if np.min(coverage[neighbors[c]]) > 1:
                apply_change([c], [])
                changed = True
                break # Restart loop as indices might shift

    # Kick search
    stagnant_count = 0
    max_stagnant = 50
    
    while time.time() < deadline - 5:
        success = False
        
        # Try 1-kick
        if try_kick(1, rng):
            success = True
            stagnant_count = 0
        else:
            # Try 2-kick if 1-kick fails
            if len(code) >= 2:
                if try_kick(2, rng):
                    success = True
                    stagnant_count = 0
        
        if not success:
            stagnant_count += 1
            if stagnant_count > max_stagnant:
                # Restart local search with a new random kick or just break if stuck too long
                # For now, just keep trying with different random seeds
                stagnant_count = 0
                
    # Final cleanup
    changed = True
    while changed:
        changed = False
        for c in list(code):
            if np.min(coverage[neighbors[c]]) > 1:
                apply_change([c], [])
                changed = True
                break

    result = [idx_to_tuple(c) for c in code]
    return result
