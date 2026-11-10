import time
import random
import numpy as np

N = 256

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    best_set = None
    best_size = N + 1

    def _is_sidon(arr):
        """Check if arr is a Sidon set (all pair sums distinct)."""
        if len(arr) == 0:
            return True
        a = np.array(arr, dtype=np.int32)
        n = len(a)
        # Compute all pair sums a[i] + a[j] for i <= j
        # Use broadcasting
        sums = a[:, None] + a[None, :]
        # We only need upper triangle (i <= j)
        # But for checking distinctness, we can check all sums
        # Actually, we need all unordered pairs (i,j) with i<=j to have distinct sums
        # The diagonal (i=i) gives 2*a[i]
        # The off-diagonal (i<j) gives a[i]+a[j]
        # We need all these to be distinct
        # Number of such pairs: n + n*(n-1)/2 = n*(n+1)/2
        # Extract upper triangle
        idx = np.triu_indices(n)
        pair_sums = sums[idx]
        # Check if all distinct
        if len(pair_sums) != len(np.unique(pair_sums)):
            return False
        return True

    def _is_maximal(arr):
        """Check if arr is inclusion-maximal Sidon in [1,N]."""
        if not _is_sidon(arr):
            return False
        a = np.array(arr, dtype=np.int32)
        n = len(a)
        # Compute all pair sums
        sums = a[:, None] + a[None, :]
        idx = np.triu_indices(n)
        used_sums = set(sums[idx].tolist())
        
        # Try each missing element
        present = set(arr)
        for x in range(1, N + 1):
            if x in present:
                continue
            # Check if adding x would violate Sidon property
            # New sums: 2x, x+a[i] for all i
            valid = True
            if 2 * x in used_sums:
                valid = False
            else:
                for ai in a:
                    if x + ai in used_sums:
                        valid = False
                        break
            if valid:
                # x can be added, so not maximal
                return False
        return True

    def greedy_blocking_random(seed=None):
        """Fast greedy: pick element that blocks most other candidates."""
        if seed is not None:
            random.seed(seed)
        
        selected = []
        used_sums = set()
        candidates = set(range(1, N + 1))
        
        while candidates:
            if time.time() > deadline:
                return None
            
            # Find all valid candidates
            valid = []
            for x in candidates:
                # Check if x is valid
                if 2 * x in used_sums:
                    continue
                valid = True
                for s in selected:
                    if x + s in used_sums:
                        valid = False
                        break
                if valid:
                    valid.append(x)
            
            if not valid:
                break
            
            if len(valid) == 1:
                chosen = valid[0]
            else:
                # Compute blocking score for each valid candidate
                best_score = -1
                best_cand = None
                best_list = []
                
                for x in valid:
                    # Sums added by x
                    added = {2 * x}
                    for s in selected:
                        added.add(x + s)
                    
                    # Count how many other valid candidates are blocked
                    blocks = 0
                    for y in valid:
                        if y == x:
                            continue
                        # Check if y would be blocked by adding x
                        # y is blocked if:
                        # 1. 2y is in added
                        # 2. y+s is in added for some s in selected
                        # 3. y+x is in used_sums
                        blocked = False
                        if 2 * y in added:
                            blocked = True
                        else:
                            for s in selected:
                                if y + s in added:
                                    blocked = True
                                    break
                        if not blocked and y + x in used_sums:
                            blocked = True
                        if blocked:
                            blocks += 1
                    
                    if blocks > best_score:
                        best_score = blocks
                        best_list = [x]
                    elif blocks == best_score:
                        best_list.append(x)
                
                chosen = random.choice(best_list)
            
            # Add chosen to selected
            selected.append(chosen)
            used_sums.add(2 * chosen)
            for s in selected[:-1]:
                used_sums.add(chosen + s)
            
            # Remove blocked candidates
            new_candidates = set()
            for y in candidates:
                if y == chosen:
                    continue
                # Check if y is still valid
                if 2 * y in used_sums:
                    continue
                valid_y = True
                for s in selected:
                    if y + s in used_sums:
                        valid_y = False
                        break
                if valid_y:
                    new_candidates.add(y)
            candidates = new_candidates
        
        return selected

    def local_search_swap(arr):
        """Try to reduce size by removing elements and re-adding others, or swapping."""
        if not _is_sidon(arr):
            return arr
        
        current = list(arr)
        current_size = len(current)
        
        # Try removing each element and see if we can get a smaller maximal set
        # Strategy: try to remove element x, then greedily fill in to make it maximal again
        # If the resulting set is smaller, keep it
        
        for _ in range(50):  # Limit iterations
            if time.time() > deadline:
                break
            
            improved = False
            present = set(current)
            
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                
                x = current[i]
                # Try removing x
                trial = [c for c in current if c != x]
                
                if not _is_sidon(trial):
                    continue
                
                # Now try to extend trial to a maximal Sidon set
                # Use greedy to add elements until maximal
                used_sums = set()
                for a in trial:
                    for b in trial:
                        if a <= b:
                            used_sums.add(a + b)
                
                candidates = set(range(1, N + 1)) - set(trial)
                extended = list(trial)
                
                while candidates:
                    if time.time() > deadline:
                        break
                    
                    # Find valid candidates
                    valid = []
                    for y in candidates:
                        if 2 * y in used_sums:
                            continue
                        ok = True
                        for s in extended:
                            if y + s in used_sums:
                                ok = False
                                break
                        if ok:
                            valid.append(y)
                    
                    if not valid:
                        break
                    
                    # Pick one that blocks most
                    best_score = -1
                    best_list = []
                    for y in valid:
                        added = {2 * y}
                        for s in extended:
                            added.add(y + s)
                        blocks = 0
                        for z in valid:
                            if z == y:
                                continue
                            if 2 * z in added:
                                blocks += 1
                                continue
                            blocked = False
                            for s in extended:
                                if z + s in added:
                                    blocked = True
                                    break
                            if not blocked and z + y in used_sums:
                                blocked = True
                            if blocked:
                                blocks += 1
                        if blocks > best_score:
                            best_score = blocks
                            best_list = [y]
                        elif blocks == best_score:
                            best_list.append(y)
                    
                    chosen = random.choice(best_list)
                    extended.append(chosen)
                    used_sums.add(2 * chosen)
                    for s in extended[:-1]:
                        used_sums.add(chosen + s)
                    
                    # Remove blocked
                    new_cands = set()
                    for z in candidates:
                        if z == chosen:
                            continue
                        if 2 * z in used_sums:
                            continue
                        ok = True
                        for s in extended:
                            if z + s in used_sums:
                                ok = False
                                break
                        if ok:
                            new_cands.add(z)
                    candidates = new_cands
                
                if len(extended) < len(current):
                    current = extended
                    improved = True
                    break
            
            if not improved:
                break
        
        return current

    # Run multiple greedy constructions with different random seeds
    for i in range(30):
        if time.time() > deadline - 5.0:
            break
        
        seed = i * 7919 + 1
        res = greedy_blocking_random(seed)
        
        if res is not None and len(res) < best_size:
            # Verify it's actually maximal
            if _is_maximal(res):
                best_size = len(res)
                best_set = res
                
                # Try local search to reduce further
                if time.time() < deadline - 10.0:
                    improved = local_search_swap(res)
                    if len(improved) < best_size and _is_maximal(improved):
                        best_size = len(improved)
                        best_set = improved
    
    if best_set is None:
        # Fallback: simple greedy
        selected = []
        used_sums = set()
        for x in range(1, N + 1):
            if 2 * x in used_sums:
                continue
            ok = True
            for s in selected:
                if x + s in used_sums:
                    ok = False
                    break
            if ok:
                selected.append(x)
                used_sums.add(2 * x)
                for s in selected[:-1]:
                    used_sums.add(x + s)
        return selected
    
    return best_set
