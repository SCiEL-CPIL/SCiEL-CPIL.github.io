import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave margin for 120s limit

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
    coverage = np.zeros(N, dtype=np.int32)
    uncovered_mask = np.ones(N, dtype=bool)
    code = []
    code_set = set()
    
    # Greedy initialization
    # We can speed this up by maintaining a list of candidates and their current marginal gain
    # But for N=2187, a simple loop is okay for the first few hundred iterations.
    # To be safer, we'll just do the standard greedy.
    
    # Optimization: Precompute neighbor counts for initial selection?
    # Actually, the first pick is always any word (symmetry).
    # Let's just pick 0.
    code.append(0)
    code_set.add(0)
    coverage[neighbors_list[0]] += 1
    uncovered_mask[neighbors_list[0]] = False
    
    while uncovered_mask.any() and time.time() < deadline:
        best_c = -1
        best_count = 0
        
        # Only consider codewords not already in code
        # And ideally, those that cover uncovered words
        # Scanning all N is 2187 * 15 (avg neighbors) operations ~ 30k ops.
        # This is fast in numpy if vectorized, but loop is fine for few iterations.
        # However, as code grows, we should skip codewords in code.
        
        # We can optimize by noting that if a codeword is in code, its marginal gain is 0.
        # So we can just check non-code words.
        
        # To speed up, we can maintain a list of "active" candidates?
        # For now, simple scan with early exit.
        
        for c in range(N):
            if c in code_set:
                continue
            cnt = np.sum(uncovered_mask[neighbors_list[c]])
            if cnt > best_count:
                best_count = cnt
                best_c = c
                if best_count == 0:
                    break # No improvement possible for this c, but maybe others? No, if best is 0, we are stuck? No, we break out of inner if.
            
            # Heuristic: If we found a perfect cover (cnt == number of uncovered), break
            if best_count == np.sum(uncovered_mask):
                break

        if best_c == -1 or best_count == 0:
            break
            
        code.append(best_c)
        code_set.add(best_c)
        coverage[neighbors_list[best_c]] += 1
        uncovered_mask[neighbors_list[best_c]] = False

    # ---------- Local Search ----------
    
    def is_redundant(c):
        # Check if removing c leaves all its neighbors covered by others
        # i.e., coverage[neighbors] > 1 for all neighbors
        return np.min(coverage[neighbors_list[c]]) > 1

    def greedy_repair(uncovered_set):
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
            
            # Filter out codewords already in code
            # Actually, if a codeword is in code, it's not "added", but it might cover things.
            # The repair adds NEW codewords.
            
            for c in candidate_pool:
                if c in code_set:
                    continue
                cnt = 0
                # Count how many of c's neighbors are in temp_uncovered
                # Using set intersection is fast
                cnt = len(neighbors_set[c] & temp_uncovered)
                if cnt > best_count:
                    best_count = cnt
                    best_c = c
                    if best_count == len(temp_uncovered):
                        break
            
            if best_c == -1:
                break
            
            added.append(best_c)
            # Update temp_uncovered
            temp_uncovered.difference_update(neighbors_set[best_c])
            
        return added

    def apply_code_change(removed_list, added_list):
        """
        Apply the change to global state.
        """
        nonlocal code, code_set, coverage
        for c in removed_list:
            code.remove(c)
            code_set.remove(c)
            coverage[neighbors_list[c]] -= 1
        for c in added_list:
            if c not in code_set:
                code.append(c)
                code_set.add(c)
                coverage[neighbors_list[c]] += 1

    def try_kick(k):
        """
        Try to remove k random codewords and repair.
        Returns True if size decreased.
        """
        if len(code) < k:
            return False
            
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
            apply_code_change(removed, [])
            return True
            
        uncovered_set = set(uncovered.tolist())
        added = greedy_repair(uncovered_set)
        
        # Check if we can add the added codewords without conflict (they shouldn't be in code)
        # greedy_repair ensures they are not in code_set at the time of call,
        # but code_set hasn't been updated yet.
        # We must ensure added does not contain any of the removed ones?
        # If it does, it means we removed c and then added c back. Net change 0.
        # If it adds something else, we compare sizes.
        
        new_size = len(code) - len(removed) + len(added)
        
        if new_size < len(code):
            # Apply change
            apply_code_change(removed, added)
            return True
        else:
            return False

    # Main Loop
    no_improvement_count = 0
    max_no_improvement = 50 # Reset counter if we get stuck
    
    while time.time() < deadline:
        # 1. Exhaustive Removal
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            for c in list(code):
                if is_redundant(c):
                    apply_code_change([c], [])
                    changed = True
                    if time.time() > deadline:
                        break
        
        # 2. Kicks
        kick_success = False
        # Try 1-removal
        if try_kick(1):
            kick_success = True
        # Try 2-removal
        elif try_kick(2):
            kick_success = True
        # Try 3-removal
        elif try_kick(3):
            kick_success = True
            
        if kick_success:
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count > max_no_improvement:
                # Perturb the code slightly to escape local minimum?
                # Or just continue, random kicks will eventually find a way.
                # We can also try a "restart" of the greedy phase from a subset?
                # For now, just keep trying random kicks.
                pass

    result = [idx_to_tuple(c) for c in code]
    return result
