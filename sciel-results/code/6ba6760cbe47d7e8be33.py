import time
import numpy as np

def construct(q, n, radius):
    t_start = time.time()
    deadline = 140.0

    space_size = q ** n
    q_powers = [1] * (n + 1)
    for i in range(1, n + 1):
        q_powers[i] = q_powers[i - 1] * q

    def idx_to_word(idx):
        w = [0] * n
        for pos in range(n):
            w[pos] = (idx // q_powers[n - 1 - pos]) % q
        return w

    # Precompute neighbors (radius=1: self + all 1-change neighbors)
    nb_count = 1 + n * (q - 1)
    neighbors = np.empty((space_size, nb_count), dtype=np.int32)
    for idx in range(space_size):
        w = [0] * n
        for pos in range(n):
            w[pos] = (idx // q_powers[n - 1 - pos]) % q
        nb_list = [idx]
        for pos in range(n):
            for new_d in range(q):
                if new_d == w[pos]:
                    continue
                delta = (new_d - w[pos]) * q_powers[n - 1 - pos]
                nb_list.append(idx + delta)
        neighbors[idx] = nb_list

    def build_cov(code_indices):
        cov = np.zeros(space_size, dtype=np.int32)
        for c in code_indices:
            cov[neighbors[c]] += 1
        return cov

    def greedy_cover(seed=0):
        rng = np.random.default_rng(seed)
        uncovered = np.ones(space_size, dtype=bool)
        code = []
        while np.any(uncovered) and (time.time() - t_start) < deadline - 15.0:
            gains = np.sum(uncovered[neighbors], axis=1)
            perturb = rng.random(space_size) * 0.001
            gains_pert = gains + perturb
            best = np.argmax(gains_pert)
            if gains[best] == 0:
                break
            code.append(int(best))
            uncovered[neighbors[best]] = False
        return code

    def local_search_removal(code, cov, deadline_time):
        """Remove redundant codewords."""
        cur_code = code[:]
        cur_cov = cov.copy()
        changed = True
        while changed and (time.time() - t_start) < deadline_time:
            changed = False
            for i in range(len(cur_code)):
                c = cur_code[i]
                if np.min(cur_cov[neighbors[c]]) >= 2:
                    cur_code.pop(i)
                    cur_cov[neighbors[c]] -= 1
                    changed = True
                    break
        return cur_code, cur_cov

    # Phase 1: Run multiple greedy covers and prune
    best_code = None
    best_size = space_size + 1
    for seed in range(5):
        if (time.time() - t_start) > deadline - 40.0:
            break
        code = greedy_cover(seed)
        cov = build_cov(code)
        if not np.all(cov > 0):
            # repair
            uncovered_mask = cov == 0
            while np.any(uncovered_mask) and (time.time() - t_start) < deadline - 30.0:
                gains = np.sum(uncovered_mask[neighbors], axis=1)
                best = np.argmax(gains)
                if gains[best] == 0:
                    break
                code.append(int(best))
                cov[neighbors[best]] += 1
                uncovered_mask = cov == 0
        code, cov = local_search_removal(code, cov, deadline - 35.0)
        if len(code) < best_size:
            best_code = code[:]
            best_size = len(code)

    # Phase 2: Iterative 1-exchange local search
    if (time.time() - t_start) < deadline - 10.0 and best_code is not None:
        cur_code = best_code[:]
        cur_cov = build_cov(cur_code)
        rng = np.random.default_rng(123)
        # We want to find (i, j) such that removing cur_code[i] and adding j reduces size by 1.
        # This requires:
        # 1. j is not in cur_code
        # 2. min(cur_cov[neighbors[i]]) >= 2 (so that cur_code[i] is not the sole cover for any node)
        # 3. For every node v where cur_cov[v] == 1, v must be in neighbors[j].
        #    i.e., the set of nodes uniquely covered by the current code must be a subset of neighbors[j].
        
        # To speed this up, we can precompute the set of nodes uniquely covered.
        # Actually, the condition "min(cur_cov[neighbors[i]]) >= 2" is necessary but not sufficient.
        # The sufficient condition is: let U = {v | cur_cov[v] == 1}. If U is empty, we can remove any i (handled by local_search_removal).
        # If U is not empty, we need to add j such that U is a subset of neighbors[j].
        # AND we need to remove i such that i is NOT in U (wait, if i is in U, then cur_cov[i] == 1, so removing i would leave i uncovered unless j covers i).
        # Actually, if we remove i, the new coverage for v is cur_cov[v] - (1 if v in neighbors[i] else 0) + (1 if v in neighbors[j] else 0).
        # We need new_cov[v] >= 1 for all v.
        # So: if v in neighbors[i] and v not in neighbors[j], we need cur_cov[v] >= 2.
        # If v not in neighbors[i] and v in neighbors[j], cur_cov[v] >= 0 (always true).
        # If v in neighbors[i] and v in neighbors[j], cur_cov[v] >= 0 (always true).
        # If v not in neighbors[i] and v not in neighbors[j], cur_cov[v] >= 1 (always true if valid before).
        
        # So the condition is: For all v in (neighbors[i] \ neighbors[j]), cur_cov[v] >= 2.
        # Equivalently: Let S_i = neighbors[i]. Let S_j = neighbors[j].
        # We need min(cur_cov[S_i \ S_j]) >= 2.
        
        # This is still O(N^2) to check all pairs. But we can do it smarter.
        # Let's just do a random search over (i, j) pairs.
        
        improved = True
        while improved and (time.time() - t_start) < deadline - 5.0:
            improved = False
            cur_size = len(cur_code)
            if cur_size <= 1:
                break
            # Sample random i and j
            # We can also bias j to be "good" (high coverage of unique nodes)
            # Let's just sample uniformly first.
            for _ in range(500):
                if (time.time() - t_start) > deadline - 5.0:
                    break
                i = rng.integers(0, cur_size)
                j = rng.integers(0, space_size)
                if j in cur_code:
                    continue
                ci = cur_code[i]
                # Check condition: min(cur_cov[neighbors[ci] \ neighbors[j]]) >= 2
                # neighbors[ci] \ neighbors[j] is the set of nodes covered by ci but not by j.
                # We can compute this mask.
                # mask = (np.isin(neighbors[ci], neighbors[j]) == False)
                # This is slow. 
                # Alternative: 
                # new_cov = cur_cov.copy()
                # new_cov[neighbors[ci]] -= 1
                # new_cov[neighbors[j]] += 1
                # if np.min(new_cov) >= 1:
                #    accept
                # This is O(N) per trial. 500 trials * O(N) is fine.
                
                new_cov = cur_cov.copy()
                new_cov[neighbors[ci]] -= 1
                new_cov[neighbors[j]] += 1
                if np.min(new_cov) >= 1:
                    # Accept move
                    cur_code[i] = int(j)
                    cur_cov = new_cov
                    improved = True
                    # We can try to remove redundant words now
                    cur_code, cur_cov = local_search_removal(cur_code, cur_cov, deadline - 5.0)
                    if len(cur_code) < best_size:
                        best_code = cur_code[:]
                        best_size = len(cur_code)
                    break # Restart the while loop to re-evaluate with new code
                
    # Final check
    if best_code is None:
        best_code = greedy_cover(0)
    
    final_cov = build_cov(best_code)
    if not np.all(final_cov > 0):
        # Fallback
        best_code = list(range(space_size))
        
    return [idx_to_word(c) for c in best_code]
