import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters
        if n <= 4 and q == 3 and radius == 1:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            import itertools
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            import itertools
            return [list(w) for w in itertools.product(range(3), repeat=n)]

    # Specific optimization for q=3, n=7, radius=1
    N = 3 ** n  # 2187
    d = n + 1  # 8 (each word has 8 neighbors including itself)

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

    # Precompute neighbor lists as numpy arrays
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

    # ---------- Fast Greedy Set-Cover ----------
    
    def greedy_construct(seed_val):
        """Run greedy set-cover construction with numpy-optimized gain computation."""
        rng = random.Random(seed_val)
        cov = np.zeros(N, dtype=np.int32)
        unc = np.ones(N, dtype=bool)
        cod = []
        cod_set = set()
        
        # Random start
        start = rng.randint(0, N - 1)
        cod.append(start)
        cod_set.add(start)
        cov[neighbors_list[start]] += 1
        unc[neighbors_list[start]] = False
        
        # Use a simple approach: repeatedly pick the codeword covering most uncovered
        # To speed up, maintain a count of how many uncovered words each candidate covers
        gain_arr = np.zeros(N, dtype=np.int32)
        
        # Initialize gains for all candidates
        for c in range(N):
            gain_arr[c] = int(np.sum(unc[neighbors_list[c]]))
        
        while unc.any() and time.time() < deadline - 5:
            # Find candidate with max gain
            max_gain = int(np.max(gain_arr))
            if max_gain <= 0:
                break
            
            # Pick randomly among candidates with max gain for diversity
            best_candidates = np.where(gain_arr == max_gain)[0]
            best_c = int(rng.choice(best_candidates))
            
            cod.append(best_c)
            cod_set.add(best_c)
            covered = neighbors_list[best_c]
            cov[covered] += 1
            was_unc = unc[covered]
            unc[covered] = False
            
            # Update gains: for each word that just became covered, decrement gain of all its neighbors
            newly_covered = covered[was_unc]
            if len(newly_covered) > 0:
                for w in newly_covered:
                    gain_arr[neighbors_list[w]] -= 1
        
        return cod, cod_set, cov

    # ---------- Local Search with Efficient Delta Evaluation ----------
    
    def is_redundant(c, coverage):
        """Check if removing c leaves all its neighbors covered by others."""
        return np.min(coverage[neighbors_list[c]]) > 1

    def try_remove_one(code, code_set, coverage):
        """Try to remove a single codeword. Returns (success, removed_c)."""
        # First try codewords with minimum coverage (most likely redundant)
        min_covs = np.array([np.min(coverage[neighbors_list[c]]) for c in code])
        if min_covs.max() > 1:
            # Find codeword with minimum coverage value > 1
            min_idx = int(np.argmin(min_covs))
            c = code[min_idx]
            if is_redundant(c, coverage):
                code.remove(c)
                code_set.remove(c)
                coverage[neighbors_list[c]] -= 1
                return True, c
        return False, -1

    def try_remove_pair(code, code_set, coverage):
        """Try to remove two codewords and add one. Returns (success, removed, added)."""
        if len(code) < 2:
            return False, [], []
        
        # Pick two random codewords
        indices = random.sample(range(len(code)), 2)
        c1, c2 = code[indices[0]], code[indices[1]]
        
        # Temporarily remove both
        temp_cov = coverage.copy()
        temp_cov[neighbors_list[c1]] -= 1
        temp_cov[neighbors_list[c2]] -= 1
        
        uncovered = np.where(temp_cov == 0)[0]
        
        if len(uncovered) == 0:
            # Perfect removal of both
            code.remove(c1)
            code.remove(c2)
            code_set.remove(c1)
            code_set.remove(c2)
            coverage[neighbors_list[c1]] -= 1
            coverage[neighbors_list[c2]] -= 1
            return True, [c1, c2], []
        
        uncovered_set = set(uncovered.tolist())
        
        # Try to find a single codeword that covers all uncovered
        candidate_pool = set()
        for w in uncovered_set:
            candidate_pool.update(neighbors_set[w])
        
        best_single = -1
        best_gain = 0
        for c in candidate_pool:
            if c in code_set or c == c1 or c == c2:
                continue
            gain = len(neighbors_set[c] & uncovered_set)
            if gain > best_gain:
                best_gain = gain
                best_single = c
        
        if best_single != -1 and best_gain == len(uncovered_set):
            code.remove(c1)
            code.remove(c2)
            code_set.remove(c1)
            code_set.remove(c2)
            coverage[neighbors_list[c1]] -= 1
            coverage[neighbors_list[c2]] -= 1
            code.append(best_single)
            code_set.add(best_single)
            coverage[neighbors_list[best_single]] += 1
            return True, [c1, c2], [best_single]
        
        return False, [], []

    def try_kick(code, code_set, coverage, k):
        """Try to remove k random codewords and repair. Returns True if size decreased."""
        if len(code) < k:
            return False
        
        # Pick k distinct random indices
        indices = random.sample(range(len(code)), k)
        removed = [code[i] for i in indices]
        
        # Temporarily remove to check coverage
        temp_cov = coverage.copy()
        for c in removed:
            temp_cov[neighbors_list[c]] -= 1
        
        uncovered = np.where(temp_cov == 0)[0]
        
        if len(uncovered) == 0:
            # Perfect removal
            for c in removed:
                code.remove(c)
                code_set.remove(c)
                coverage[neighbors_list[c]] -= 1
            return True
        
        uncovered_set = set(uncovered.tolist())
        
        # Greedy repair
        added = []
        temp_uncovered = set(uncovered_set)
        
        while temp_uncovered and len(added) < k:  # Don't add more than we removed
            best_c = -1
            best_count = 0
            
            # Candidates are neighbors of uncovered words
            candidate_pool = set()
            for w in temp_uncovered:
                candidate_pool.update(neighbors_set[w])
            
            for c in candidate_pool:
                if c in code_set or c in removed:
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
        
        new_size = len(code) - len(removed) + len(added)
        
        if new_size < len(code):
            # Apply change
            for c in removed:
                code.remove(c)
                code_set.remove(c)
                coverage[neighbors_list[c]] -= 1
            for c in added:
                if c not in code_set:
                    code.append(c)
                    code_set.add(c)
                    coverage[neighbors_list[c]] += 1
            return True
        
        return False

    # ---------- Main Construction ----------
    
    # Try multiple greedy restarts
    best_code = None
    best_code_set = None
    best_coverage = None
    best_size = float('inf')
    
    max_restarts = 15
    for restart in range(max_restarts):
        if time.time() > deadline - 60:  # Need time for local search
            break
        cod, cod_set, cov = greedy_construct(100 + restart)
        if len(cod) < best_size:
            best_size = len(cod)
            best_code = cod
            best_code_set = cod_set
            best_coverage = cov
        if best_size <= 160:
            break
    
    code = best_code
    code_set = best_code_set
    coverage = best_coverage

    # ---------- Local Search Phase ----------
    
    no_improvement_count = 0
    max_no_improvement = 30
    random.seed(42)
    
    while time.time() < deadline - 5:
        # 1. Exhaustive removal of redundant codewords
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            for c in list(code):
                if is_redundant(c, coverage):
                    code.remove(c)
                    code_set.remove(c)
                    coverage[neighbors_list[c]] -= 1
                    changed = True
                    if time.time() > deadline:
                        break
        
        # 2. Try various moves
        kick_success = False
        for _ in range(20):
            if time.time() > deadline:
                break
            
            # Try 2-for-1 pair replacement (most important for reduction)
            if try_remove_pair(code, code_set, coverage):
                kick_success = True
                break
            # Try 1-removal
            elif try_remove_one(code, code_set, coverage):
                kick_success = True
                break
            # Try 2-removal with repair
            elif try_kick(code, code_set, coverage, 2):
                kick_success = True
                break
            # Try 3-removal with repair
            elif try_kick(code, code_set, coverage, 3):
                kick_success = True
                break
            # Try 4-removal with repair
            elif try_kick(code, code_set, coverage, 4):
                kick_success = True
                break
        
        if kick_success:
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count > max_no_improvement:
                # Perturb: try larger kicks
                if len(code) > 10:
                    k = random.randint(5, min(12, len(code)))
                    if try_kick(code, code_set, coverage, k):
                        no_improvement_count = 0
                else:
                    # Stuck: try a fresh greedy restart and merge
                    if time.time() < deadline - 20:
                        new_cod, new_cod_set, new_cov = greedy_construct(random.randint(0, 10000))
                        # Try to merge: add new codewords to current code if they help
                        for c in new_cod:
                            if c not in code_set:
                                # Check if adding c allows removal of existing codewords
                                temp_cov = coverage.copy()
                                temp_cov[neighbors_list[c]] += 1
                                # Try to remove redundant codewords
                                for d in list(code):
                                    if np.min(temp_cov[neighbors_list[d]]) > 1:
                                        # This would work, but let's just add c for now
                                        pass
                                # Just add c
                                code.append(c)
                                code_set.add(c)
                                coverage[neighbors_list[c]] += 1
                                no_improvement_count = 0
                                break

    # Final exhaustive removal pass
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        for c in list(code):
            if is_redundant(c, coverage):
                code.remove(c)
                code_set.remove(c)
                coverage[neighbors_list[c]] -= 1
                changed = True
                if time.time() > deadline:
                    break

    result = [idx_to_tuple(c) for c in code]
    return result
