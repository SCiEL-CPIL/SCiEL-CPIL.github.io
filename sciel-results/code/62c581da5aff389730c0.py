import itertools
import random
import time
import numpy as np

def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        return [list(t) for t in itertools.product(range(q), repeat=n)]
    
    total = q ** n  # 729
    
    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n):
            t[n - 1 - i] = idx % q
            idx //= q
        return tuple(t)
    
    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    # Precompute neighbors
    neighbors_list = [None] * total
    neighbors_set = [None] * total
    neighbors_arr = [None] * total
    
    for idx in range(total):
        w = idx_to_tuple(idx)
        nbs = [idx]
        for i in range(n):
            for d in range(1, q):
                new_w = list(w)
                new_w[i] = (new_w[i] + d) % q
                nbs.append(tuple_to_idx(tuple(new_w)))
        neighbors_list[idx] = nbs
        neighbors_set[idx] = set(nbs)
        neighbors_arr[idx] = np.array(nbs, dtype=np.int32)

    def tetracode_full():
        """Full tetracode: all 81 cells filled."""
        result = []
        for a in range(3):
            for b in range(3):
                for x in range(3):
                    for y in range(3):
                        cw = (a, b, (a+b)%3, (a+2*b)%3, x, y)
                        result.append(tuple_to_idx(cw))
        return result

    def greedy_cover():
        uncovered = np.ones(total, dtype=bool)
        code = []
        # Precompute coverage counts for all candidates? No, dynamic is better for greedy.
        # But for speed, we can just pick the one that covers most uncovered.
        # Since total is 729, and code size is ~80, we can afford O(N^2) roughly.
        while np.any(uncovered):
            best_cand = -1
            best_count = -1
            # Randomize order to break ties
            candidates = np.random.permutation(total)
            for cand in candidates:
                if np.all(~uncovered): break
                # Count how many uncovered points this candidate covers
                # neighbors_arr[cand] is a list of indices
                # We need to count how many of those are in 'uncovered'
                # Using numpy boolean mask
                count = np.sum(uncovered[neighbors_arr[cand]])
                if count > best_count:
                    best_count = count
                    best_cand = cand
            if best_cand < 0:
                break
            code.append(best_cand)
            uncovered[neighbors_arr[best_cand]] = False
        return code

    def local_search_vectorized(code, time_limit):
        """
        Fast local search using NumPy.
        1. Remove redundant codewords.
        2. Swap codewords to cover new points or allow removal of others.
        3. 2-for-1 moves.
        """
        code = list(dict.fromkeys(code))
        if not code:
            return code
            
        # Build coverage count array
        # cover_count[i] = number of codewords in code that cover point i
        cover_count = np.zeros(total, dtype=np.int32)
        for c in code:
            cover_count[neighbors_arr[c]] += 1
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            random.shuffle(code)
            
            # Phase 1: Removal
            # Find codewords whose removal doesn't leave any point with cover_count 0
            # A codeword c is removable if for all p in N(c), cover_count[p] > 1
            # i.e., min(cover_count[N(c)]) > 1
            removable_indices = []
            for i, c in enumerate(code):
                if np.min(cover_count[neighbors_arr[c]]) > 1:
                    removable_indices.append(i)
            
            if removable_indices:
                # Remove one (or all?) at a time to be safe with indices
                i = removable_indices[0]
                c = code[i]
                cover_count[neighbors_arr[c]] -= 1
                code.pop(i)
                improved = True
                continue

            # Phase 2: Swaps
            # Try replacing code[i] with a new codeword c'
            # Condition: After removal, the set of uncovered points U (where cover_count becomes 0)
            # must be covered by c'.
            # Also, we prefer c' that covers points where cover_count is low?
            # Simple heuristic: Try to replace with a codeword that covers the "most critical" points.
            
            # To speed up, we can sample a few candidates for replacement.
            # For each i, find U_i = {p in N(c_i) | cover_count[p] == 1}
            # If U_i is empty, it's removable (handled above).
            # If U_i is not empty, we need a c' such that U_i subset of N(c').
            
            # Optimization: Precompute for each point p, the list of codewords covering p?
            # We have neighbors_list which is symmetric.
            
            for i in range(len(code)):
                if time.time() > time_limit:
                    break
                c = code[i]
                nb_arr = neighbors_arr[c]
                # Points that will become uncovered if we remove c
                # These are points in nb_arr where cover_count is exactly 1
                critical_points = nb_arr[cover_count[nb_arr] == 1]
                
                if len(critical_points) == 0:
                    # Should have been caught in Phase 1, but maybe race condition or logic gap?
                    # Actually Phase 1 checks min > 1. If min == 1, it's not removable.
                    # So critical_points should be non-empty if not removable.
                    pass
                
                if len(critical_points) > 0:
                    # We need a replacement c' that covers all critical_points.
                    # The candidates for c' are the intersection of neighbors of all critical points.
                    # Since |critical_points| is usually small (1 or 2), this is fast.
                    
                    # Get candidates from the first critical point
                    u0 = critical_points[0]
                    candidates = neighbors_list[u0]
                    
                    # Filter candidates that cover all other critical points
                    # And are not already in the code (or if they are, it's a no-op swap which is useless)
                    # Actually, if c' is in code, we are just moving it? No, we are replacing c with c'.
                    # If c' is already in code, we would have a duplicate. We should avoid that.
                    
                    valid_candidates = []
                    for c_prime in candidates:
                        if c_prime in code:
                            continue
                        # Check if c_prime covers all critical points
                        covers_all = True
                        for p in critical_points[1:]:
                            if p not in neighbors_set[c_prime]:
                                covers_all = False
                                break
                        if covers_all:
                            valid_candidates.append(c_prime)
                    
                    if valid_candidates:
                        # Pick a random valid candidate
                        # Prefer one that covers many other points?
                        # For now, random
                        new_c = random.choice(valid_candidates)
                        
                        # Apply swap
                        cover_count[neighbors_arr[c]] -= 1
                        cover_count[neighbors_arr[new_c]] += 1
                        code[i] = new_c
                        improved = True
                        break # Restart loop after change
            
            if improved:
                continue

            # Phase 3: 2-for-1 moves
            # Remove c1, c2, add c3
            # Condition: Points in N(c1) U N(c2) that become uncovered (cover_count becomes 0)
            # must be covered by c3.
            
            # This is expensive. Sample random pairs.
            if len(code) >= 2:
                num_trials = 10
                for _ in range(num_trials):
                    if time.time() > time_limit:
                        break
                    i, j = random.sample(range(len(code)), 2)
                    c1, c2 = code[i], code[j]
                    
                    # Simulate removal
                    # We need to be careful with indices if we modify the list.
                    # Let's just compute the set of points that would be uncovered.
                    
                    # Points in N(c1) with count 1 (after removing c2? No, independent first)
                    # Let's compute cover_count if we remove both.
                    # temp_count = cover_count
                    # temp_count[N(c1)] -= 1
                    # temp_count[N(c2)] -= 1
                    # Uncovered = {p | temp_count[p] == 0}
                    
                    # To avoid copying the whole array, we can check specific points.
                    # The only points that can become uncovered are those in N(c1) U N(c2).
                    # And specifically those with low counts.
                    
                    # Let's just do the array ops, it's fast enough for 729 size.
                    temp_count = cover_count.copy()
                    temp_count[neighbors_arr[c1]] -= 1
                    temp_count[neighbors_arr[c2]] -= 1
                    
                    # Find points in N(c1) U N(c2) that are 0 in temp_count
                    # We can just check all points in the union
                    union_nbs = np.union1d(neighbors_arr[c1], neighbors_arr[c2])
                    uncovered_pts = union_nbs[temp_count[union_nbs] == 0]
                    
                    if len(uncovered_pts) == 0:
                        # We can remove both!
                        # Remove from list
                        # Handle indices
                        if i > j:
                            i, j = j, i
                            c1, c2 = c2, c1
                        code.pop(i)
                        code.pop(j)
                        cover_count[neighbors_arr[c1]] -= 1
                        cover_count[neighbors_arr[c2]] -= 1
                        improved = True
                        break
                    else:
                        # Try to find a single c3 that covers all uncovered_pts
                        if len(uncovered_pts) > 0:
                            u0 = uncovered_pts[0]
                            candidates = neighbors_list[u0]
                            for c3 in candidates:
                                if c3 in code:
                                    continue
                                # Check if c3 covers all uncovered_pts
                                covers_all = True
                                for p in uncovered_pts:
                                    if p not in neighbors_set[c3]:
                                        covers_all = False
                                        break
                                if covers_all:
                                    # Apply 2-for-1
                                    # Remove c1, c2, add c3
                                    if i > j:
                                        i, j = j, i
                                        c1, c2 = c2, c1
                                    code.pop(i)
                                    code.pop(j)
                                    code.append(c3)
                                    cover_count[neighbors_arr[c1]] -= 1
                                    cover_count[neighbors_arr[c2]] -= 1
                                    cover_count[neighbors_arr[c3]] += 1
                                    improved = True
                                    break
                            if improved:
                                break

        return code

    def validate_and_record(code, best_state):
        # Remove duplicates
        code = list(dict.fromkeys(code))
        if len(code) >= best_state['size']:
            return
        # Validate coverage
        cover_count = np.zeros(total, dtype=np.int32)
        for c in code:
            cover_count[neighbors_arr[c]] += 1
        if np.any(cover_count == 0):
            return
        best_state['code'] = list(code)
        best_state['size'] = len(code)
        return True

    start_time = time.time()
    deadline = 145.0  # Safety margin (limit is 150s)

    best_state = {'code': None, 'size': float('inf')}

    # Strategy 1: Full Tetracode + Local Search
    tetracode = tetracode_full()
    validate_and_record(tetracode, best_state)
    if best_state['size'] <= 73:
        pass # We found a good one, but let's try to improve more
    
    # Strategy 2: Greedy Cover + Local Search
    greedy_code = greedy_cover()
    validate_and_record(greedy_code, best_state)

    # Run Local Search on the best found so far
    if best_state['code']:
        ls_code = local_search_vectorized(best_state['code'], time.time() + 30)
        validate_and_record(ls_code, best_state)

    # Try multiple random initializations from Tetracode subsets
    # and Greedy variants
    for _ in range(5):
        if time.time() > deadline - 20:
            break
        
        # Random subset of Tetracode
        # Start with full Tetracode, remove random ones, then fix?
        # Or just use Greedy with random tie-breaking?
        
        # Let's try a "randomized greedy"
        uncovered = np.ones(total, dtype=bool)
        code = []
        while np.any(uncovered):
            # Pick 5 random candidates, choose best
            candidates = np.random.choice(total, size=5, replace=False)
            best_cand = -1
            best_count = -1
            for cand in candidates:
                count = np.sum(uncovered[neighbors_arr[cand]])
                if count > best_count:
                    best_count = count
                    best_cand = cand
            code.append(best_cand)
            uncovered[neighbors_arr[best_cand]] = False
        
        validate_and_record(code, best_state)
        
        if best_state['code']:
            ls_code = local_search_vectorized(best_state['code'], time.time() + 10)
            validate_and_record(ls_code, best_state)

    # Final Polish: If we have a good code, try to polish it more
    if best_state['code'] and best_state['size'] < 80:
        # Run local search again with more time
        remaining = deadline - time.time() - 5
        if remaining > 5:
            ls_code = local_search_vectorized(best_state['code'], time.time() + remaining)
            validate_and_record(ls_code, best_state)

    if best_state['code'] is None:
        best_state['code'] = tetracode_full()

    # Final verification
    cover_count = np.zeros(total, dtype=np.int32)
    for c in best_state['code']:
        cover_count[neighbors_arr[c]] += 1
    if np.any(cover_count == 0):
        return [list(t) for t in itertools.product(range(q), repeat=n)]
    
    return [idx_to_tuple(c) for c in best_state['code']]
