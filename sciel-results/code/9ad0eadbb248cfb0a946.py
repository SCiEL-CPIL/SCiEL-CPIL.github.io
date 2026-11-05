import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave margin for 120s limit
    best_code = None
    best_size = float('inf')

    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters, though task is specific
        if n <= 4:
            if q == 3 and radius == 1:
                return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        # General fallback: Direct sum or random
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(w) for w in itertools.product(range(3), repeat=n)]

    # Specific optimization for q=3, n=7, radius=1
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

    # Precompute neighbor lists
    neighbors_list = []
    neighbors_set = []
    for i in range(N):
        w = idx_to_tuple(i)
        nb = [i]
        for pos in range(n):
            val = w[pos]
            for new_val in range(3):
                if new_val != val:
                    new_w = w[:]
                    new_w[pos] = new_val
                    nb.append(tuple_to_idx(new_w))
        neighbors_list.append(np.array(nb, dtype=np.int32))
        neighbors_set.append(set(nb))

    # ---------- Greedy Set-Cover Construction ----------
    
    def run_greedy_init():
        """Run greedy set cover and return (code, coverage)."""
        nonlocal deadline
        cov = np.zeros(N, dtype=np.int32)
        unc_mask = np.ones(N, dtype=bool)
        cod = []
        cod_set = set()
        
        # Greedy initialization
        cod.append(0)
        cod_set.add(0)
        cov[neighbors_list[0]] += 1
        unc_mask[neighbors_list[0]] = False
        
        while unc_mask.any() and time.time() < deadline:
            best_c = -1
            best_count = 0
            
            # Only consider candidates adjacent to uncovered words for speed
            candidate_pool = set()
            uncovered_indices = np.where(unc_mask)[0]
            for w in uncovered_indices:
                candidate_pool.update(neighbors_set[w])
            
            for c in candidate_pool:
                if c in cod_set:
                    continue
                cnt = np.sum(unc_mask[neighbors_list[c]])
                if cnt > best_count:
                    best_count = cnt
                    best_c = c
                
                # Heuristic: If we found a perfect cover (cnt == number of uncovered), break
                if best_count == len(uncovered_indices):
                    break

            if best_c == -1 or best_count == 0:
                break
                
            cod.append(best_c)
            cod_set.add(best_c)
            cov[neighbors_list[best_c]] += 1
            unc_mask[neighbors_list[best_c]] = False
        
        return cod, cov

    code, coverage = run_greedy_init()
    code_set = set(code)
    best_code = list(code)
    best_size = len(code)

    # ---------- Local Search ----------
    
    def is_redundant(c, cov):
        # Check if removing c leaves all its neighbors covered by others
        # i.e., coverage[neighbors] > 1 for all neighbors
        return np.min(cov[neighbors_list[c]]) > 1

    def greedy_repair(uncovered_set, current_code_set):
        """
        Given a set of uncovered word indices, find a minimal set of codewords
        to add to cover them.
        Returns list of added codewords.
        """
        if not uncovered_set:
            return []
        
        added = []
        temp_uncovered = set(uncovered_set)
        
        while temp_uncovered:
            best_c = -1
            best_count = 0
            
            # Candidates are neighbors of uncovered words
            candidate_pool = set()
            for w in temp_uncovered:
                candidate_pool.update(neighbors_set[w])
            
            for c in candidate_pool:
                if c in current_code_set:
                    continue
                cnt = len(neighbors_set[c] & temp_uncovered)
                if cnt > best_count:
                    best_count = cnt
                    best_c = c
                    if best_count == len(temp_uncovered):
                        break
            
            if best_c == -1:
                break
            
            added.append(best_c)
            temp_uncovered.difference_update(neighbors_set[best_c])
            
        return added

    def try_kick(k, code, code_set, coverage):
        """
        Try to remove k random codewords and repair.
        Returns (success, new_code, new_code_set, new_coverage)
        """
        if len(code) < k:
            return False, code, code_set, coverage
            
        # Pick k distinct random indices
        indices = random.sample(range(len(code)), k)
        removed = [code[i] for i in indices]
        
        # Temporarily remove to check coverage
        temp_coverage = coverage.copy()
        for c in removed:
            temp_coverage[neighbors_list[c]] -= 1
            
        uncovered = np.where(temp_coverage == 0)[0]
        
        if len(uncovered) == 0:
            # Perfect removal
            new_code = [c for i, c in enumerate(code) if i not in indices]
            new_code_set = set(new_code)
            return True, new_code, new_code_set, temp_coverage
            
        uncovered_set = set(uncovered.tolist())
        
        # Create a set of removed codewords to exclude from repair candidates
        removed_set = set(removed)
        
        # Temporarily remove from code_set for repair
        temp_code_set = set(code)
        for c in removed:
            temp_code_set.discard(c)
        
        added = greedy_repair(uncovered_set, temp_code_set)
        
        new_size = len(code) - len(removed) + len(added)
        
        if new_size < len(code):
            # Apply change
            new_code = [c for i, c in enumerate(code) if i not in indices] + added
            new_code_set = set(new_code)
            new_coverage = temp_coverage
            for c in added:
                new_coverage[neighbors_list[c]] += 1
            return True, new_code, new_code_set, new_coverage
        else:
            # Even if size doesn't decrease, check if we can find a better configuration
            # by trying different repair strategies
            if len(uncovered_set) <= 20:  # Small uncovered set, try more carefully
                # Try to find a single codeword that covers all uncovered
                candidate_pool = set()
                for w in uncovered_set:
                    candidate_pool.update(neighbors_set[w])
                
                best_single = -1
                best_gain = 0
                for c in candidate_pool:
                    if c in temp_code_set or c in removed_set:
                        continue
                    gain = len(neighbors_set[c] & uncovered_set)
                    if gain > best_gain:
                        best_gain = gain
                        best_single = c
                
                if best_single != -1 and best_gain == len(uncovered_set):
                    # Single codeword covers all uncovered
                    new_code = [c for i, c in enumerate(code) if i not in indices] + [best_single]
                    new_code_set = set(new_code)
                    new_coverage = temp_coverage
                    new_coverage[neighbors_list[best_single]] += 1
                    return True, new_code, new_code_set, new_coverage
            
            return False, code, code_set, coverage

    def try_swap(code, code_set, coverage):
        """
        Try to swap a random codeword with a different one.
        Returns (success, new_code, new_code_set, new_coverage)
        """
        if len(code) < 2:
            return False, code, code_set, coverage
            
        # Pick a random codeword to remove
        idx = random.randint(0, len(code) - 1)
        removed = code[idx]
        
        # Temporarily remove to check coverage
        temp_coverage = coverage.copy()
        temp_coverage[neighbors_list[removed]] -= 1
        
        uncovered = np.where(temp_coverage == 0)[0]
        
        if len(uncovered) == 0:
            # Perfect removal, no need to add
            new_code = [c for i, c in enumerate(code) if i != idx]
            new_code_set = set(new_code)
            return True, new_code, new_code_set, temp_coverage
            
        uncovered_set = set(uncovered.tolist())
        
        # Only consider candidates adjacent to uncovered words (much faster than scanning all N)
        candidate_pool = set()
        for w in uncovered_set:
            candidate_pool.update(neighbors_set[w])
        
        # Find a candidate to add
        best_c = -1
        best_gain = 0
        for c in candidate_pool:
            if c == removed or c in code_set:
                continue
            gain = len(neighbors_set[c] & uncovered_set)
            if gain > best_gain:
                best_gain = gain
                best_c = c
                if best_gain >= len(uncovered_set):
                    break
        
        if best_c == -1:
            return False, code, code_set, coverage
            
        # Apply the swap if it covers all uncovered
        if best_gain >= len(uncovered_set):
            new_code = [c for i, c in enumerate(code) if i != idx] + [best_c]
            new_code_set = set(new_code)
            new_coverage = temp_coverage
            new_coverage[neighbors_list[best_c]] += 1
            return True, new_code, new_code_set, new_coverage
        else:
            return False, code, code_set, coverage

    # Main Loop with multiple restarts
    random.seed(42)
    
    while time.time() < deadline:
        # 1. Exhaustive Removal
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            for c in list(code):
                if is_redundant(c, coverage):
                    # Apply removal
                    code.remove(c)
                    code_set.remove(c)
                    coverage[neighbors_list[c]] -= 1
                    changed = True
                    if time.time() > deadline:
                        break
        
        # Track best
        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)
        
        # 2. Kicks - try multiple attempts per iteration for better mixing
        kick_success = False
        for _ in range(5):  # Try up to 5 kicks/swaps per main loop iteration
            if time.time() > deadline:
                break
            
            # Try 1-removal
            success, new_code, new_code_set, new_coverage = try_kick(1, code, code_set, coverage)
            if success:
                code, code_set, coverage = new_code, new_code_set, new_coverage
                kick_success = True
                break
            # Try 2-removal
            success, new_code, new_code_set, new_coverage = try_kick(2, code, code_set, coverage)
            if success:
                code, code_set, coverage = new_code, new_code_set, new_coverage
                kick_success = True
                break
            # Try 3-removal
            success, new_code, new_code_set, new_coverage = try_kick(3, code, code_set, coverage)
            if success:
                code, code_set, coverage = new_code, new_code_set, new_coverage
                kick_success = True
                break
            # Try swap
            success, new_code, new_code_set, new_coverage = try_swap(code, code_set, coverage)
            if success:
                code, code_set, coverage = new_code, new_code_set, new_coverage
                kick_success = True
                break
        
        if kick_success:
            # Track best
            if len(code) < best_size:
                best_size = len(code)
                best_code = list(code)
        
        # Check time for restart
        if time.time() > deadline - 20:
            break
            
        # Restart if stuck or to explore different local minima
        # Restart after a certain number of iterations or if we've been stuck
        # For now, do a restart every ~20 seconds of computation
        # This allows exploring different starting points
        
        # Simple restart: run greedy init again with different random seed
        # But first, let's do a few more local search steps
        # Actually, let's restart more aggressively to explore different basins
        if random.random() < 0.3:  # 30% chance to restart
            new_code, new_coverage = run_greedy_init()
            code, coverage = new_code, new_coverage
            code_set = set(code)
            if len(code) < best_size:
                best_size = len(code)
                best_code = list(code)

    # Return the best code found
    result = [idx_to_tuple(c) for c in best_code]
    return result
