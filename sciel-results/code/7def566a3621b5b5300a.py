import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # Handle non-specific cases with a simple greedy or fallback
    if q != 3 or n != 7 or radius != 1:
        if n <= 4 and q == 3 and radius == 1:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        # General fallback: Product code or random
        # For q=3, n=7, we use specific optimization below.
        # For others, a simple greedy set cover is a safe bet.
        N = q ** n
        def idx_to_tuple(idx):
            t = [0] * n
            for i in range(n - 1, -1, -1):
                t[i] = idx % q
                idx //= q
            return t
        def tuple_to_idx(t):
            idx = 0
            for x in t:
                idx = idx * q + x
            return idx
        neighbors_list = []
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
            neighbors_list.append(nb)
        
        coverage = np.zeros(N, dtype=np.int32)
        uncovered_mask = np.ones(N, dtype=bool)
        code = []
        code_set = set()
        
        while uncovered_mask.any() and time.time() < deadline:
            best_c = -1
            best_count = 0
            # Sample candidates to speed up if N is large
            candidates = range(N) if N < 10000 else random.sample(range(N), 1000)
            for c in candidates:
                if c in code_set:
                    continue
                cnt = np.sum(uncovered_mask[neighbors_list[c]])
                if cnt > best_count:
                    best_count = cnt
                    best_c = c
            if best_c == -1:
                break
            code.append(best_c)
            code_set.add(best_c)
            coverage[neighbors_list[best_c]] += 1
            uncovered_mask[neighbors_list[best_c]] = False
            
        return [idx_to_tuple(c) for c in code]

    # Specific optimization for q=3, n=7, radius=1
    N = 3 ** n  # 2187
    n_dim = n
    
    def idx_to_tuple(idx):
        t = [0] * n_dim
        for i in range(n_dim - 1, -1, -1):
            t[i] = idx % 3
            idx //= 3
        return t

    def tuple_to_idx(t):
        idx = 0
        for x in t:
            idx = idx * 3 + x
        return idx

    # Precompute neighbors
    neighbors_list = []
    neighbors_set = []
    for i in range(N):
        w = idx_to_tuple(i)
        nb = [i]
        for pos in range(n_dim):
            val = w[pos]
            for new_val in range(3):
                if new_val != val:
                    new_w = w[:]
                    new_w[pos] = new_val
                    nb.append(tuple_to_idx(new_w))
        neighbors_list.append(np.array(nb, dtype=np.int32))
        neighbors_set.append(set(nb))

    # State
    coverage = np.zeros(N, dtype=np.int32)
    uncovered_mask = np.ones(N, dtype=bool)
    code = []
    code_set = set()

    # Greedy Initialization
    code.append(0)
    code_set.add(0)
    coverage[neighbors_list[0]] += 1
    uncovered_mask[neighbors_list[0]] = False
    
    while uncovered_mask.any() and time.time() < deadline:
        best_c = -1
        best_count = 0
        
        # Candidate pool: neighbors of uncovered nodes
        # To speed up, we can sample or iterate all if N is small (2187 is small)
        # Iterating all 2187 is fast enough in Python if optimized?
        # Let's use a vectorized approach to find best candidate.
        
        # For each candidate c, gain = sum(uncovered_mask[neighbors_list[c]])
        # We can compute this for all c in parallel using numpy?
        # neighbors_list is a list of arrays. 
        # We can stack them into a 2D array (N x 15) and do matrix multiplication with uncovered_mask?
        # uncovered_mask is (N,). 
        # gain[c] = sum(uncovered_mask[neighbors[c]])
        
        # Let's stick to a fast loop for the first few iterations, then maybe optimize.
        # Actually, 2187 * 15 operations is ~32k ops per candidate scan. 
        # If we scan all 2187 candidates, that's 2187 * 32k ~ 70M ops. Too slow in Python loop.
        
        # Optimization: Only consider candidates that are neighbors of uncovered nodes.
        # The set of such candidates is often much smaller than N.
        candidate_pool = set()
        uncovered_indices = np.where(uncovered_mask)[0]
        if len(uncovered_indices) > 500: # If many uncovered, pool is likely large
            # Sample candidates from pool
            for w in uncovered_indices[:100]: # Sample 100 uncovered to build pool
                candidate_pool.update(neighbors_set[w])
            candidates = list(candidate_pool)
        else:
            for w in uncovered_indices:
                candidate_pool.update(neighbors_set[w])
            candidates = list(candidate_pool)
            
        for c in candidates:
            if c in code_set:
                continue
            cnt = np.sum(uncovered_mask[neighbors_list[c]])
            if cnt > best_count:
                best_count = cnt
                best_c = c
            if best_count == len(uncovered_indices):
                break

        if best_c == -1 or best_count == 0:
            break
            
        code.append(best_c)
        code_set.add(best_c)
        coverage[neighbors_list[best_c]] += 1
        uncovered_mask[neighbors_list[best_c]] = False

    # Local Search
    def is_redundant(c):
        return np.min(coverage[neighbors_list[c]]) > 1

    def apply_change(removed, added):
        nonlocal code, code_set, coverage, uncovered_mask
        for c in removed:
            if c in code_set:
                code.remove(c)
                code_set.remove(c)
                coverage[neighbors_list[c]] -= 1
        for c in added:
            if c not in code_set:
                code.append(c)
                code_set.add(c)
                coverage[neighbors_list[c]] += 1
        
        # Update uncovered mask
        # A node is uncovered if coverage[node] == 0
        uncovered_mask[:] = (coverage == 0)

    def greedy_repair(uncovered_set, max_adds):
        """Find up to max_adds codewords to cover uncovered_set."""
        if not uncovered_set:
            return []
        added = []
        temp_uncovered = set(uncovered_set)
        for _ in range(max_adds):
            if not temp_uncovered:
                break
            best_c = -1
            best_count = 0
            candidate_pool = set()
            # Build pool from a sample of uncovered to save time
            sample = list(temp_uncovered)
            if len(sample) > 50:
                sample = random.sample(sample, 50)
            for w in sample:
                candidate_pool.update(neighbors_set[w])
            
            for c in candidate_pool:
                if c in code_set or c in added:
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

    def try_kick(k):
        """Remove k random codewords, repair. Return True if size decreased."""
        if len(code) < k:
            return False
        indices = random.sample(range(len(code)), k)
        removed = [code[i] for i in indices]
        
        # Temporarily remove
        temp_coverage = coverage.copy()
        for c in removed:
            temp_coverage[neighbors_list[c]] -= 1
            
        uncovered = np.where(temp_coverage == 0)[0]
        
        if len(uncovered) == 0:
            apply_change(removed, [])
            return True
            
        uncovered_set = set(uncovered.tolist())
        
        # Try to repair with < k codewords
        # We want to find a set A such that |A| < k and covers uncovered_set.
        # Greedy repair
        added = greedy_repair(uncovered_set, k - 1)
        
        # Check if added covers all
        # If not, we can't accept this kick directly.
        # But we can check if the current 'added' covers the uncovered set.
        # Actually, greedy_repair returns a list. We need to verify it covers.
        # If it doesn't cover all, we can try to add one more? No, that would be size k, no gain.
        # So we only accept if added covers all AND len(added) < k.
        
        # Verify coverage
        if len(added) < k:
             # Check if added covers uncovered_set
             covered_by_added = set()
             for c in added:
                 covered_by_added.update(neighbors_set[c])
             if uncovered_set.issubset(covered_by_added):
                 apply_change(removed, added)
                 return True
        return False

    def try_swap():
        """Swap 1 codeword for 1 different one. Return True if size decreased (impossible) or just improved structure? 
        Actually, swap keeps size same. We use it to escape local minima for kicks.
        But the objective is size. So swap is only useful if it enables a subsequent kick.
        However, sometimes a swap can lead to a state where a kick works.
        Let's implement 1-for-1 swap that reduces 'potential' or just random.
        Actually, let's focus on kicks.
        """
        if len(code) < 2:
            return False
        idx = random.randint(0, len(code) - 1)
        removed = code[idx]
        
        temp_coverage = coverage.copy()
        temp_coverage[neighbors_list[removed]] -= 1
        uncovered = np.where(temp_coverage == 0)[0]
        
        if len(uncovered) == 0:
            # Redundant, handled by is_redundant pass
            return False
            
        uncovered_set = set(uncovered.tolist())
        candidate_pool = set()
        for w in list(uncovered_set)[:50]:
            candidate_pool.update(neighbors_set[w])
            
        best_c = -1
        best_gain = 0
        for c in candidate_pool:
            if c == removed or c in code_set:
                continue
            gain = len(neighbors_set[c] & uncovered_set)
            if gain > best_gain:
                best_gain = gain
                best_c = c
                if best_gain == len(uncovered_set):
                    break
        
        if best_c != -1 and best_gain == len(uncovered_set):
            # This swap covers all uncovered nodes. Size remains same.
            # Does this help? It might change the structure.
            # Let's accept it to shuffle the code.
            apply_change([removed], [best_c])
            return True
        return False

    # Main Loop
    random.seed(42)
    no_improvement_count = 0
    max_no_improvement = 50
    
    while time.time() < deadline:
        # 1. Exhaustive Removal
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            for c in list(code):
                if is_redundant(c):
                    apply_change([c], [])
                    changed = True
                    if time.time() > deadline:
                        break
        
        # 2. Kicks and Swaps
        kick_success = False
        # Try a mix of kick sizes
        for _ in range(5):
            if time.time() > deadline:
                break
            r = random.random()
            if r < 0.3:
                if try_kick(1):
                    kick_success = True
                    break
            elif r < 0.6:
                if try_kick(2):
                    kick_success = True
                    break
            elif r < 0.8:
                if try_kick(3):
                    kick_success = True
                    break
            else:
                if try_swap():
                    kick_success = True
                    break
        
        if kick_success:
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count > max_no_improvement:
                # Perturb: try a larger kick
                if len(code) > 5:
                    k = random.randint(4, min(8, len(code)))
                    if try_kick(k):
                        no_improvement_count = 0
                else:
                    # Restart local search phase with a new random seed?
                    # Or just continue.
                    pass

    result = [idx_to_tuple(c) for c in code]
    return result
