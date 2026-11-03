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

    # Tetracode structure: 9x9 grid where each cell (a,b) has 9 possible codewords
    # codeword = (a, b, (a+b)%3, (a+2b)%3, x, y) for x,y in {0,1,2}
    # We'll use SA over the 81 cells, each choosing from 9 options (or "empty")
    
    # Precompute for each cell (a,b) and option (x,y): the 19 covered points
    cell_options = {}  # (a,b) -> list of (codeword_idx, covered_set, covered_arr)
    for a in range(3):
        for b in range(3):
            opts = []
            for x in range(3):
                for y in range(3):
                    cw = (a, b, (a+b)%3, (a+2*b)%3, x, y)
                    cw_idx = tuple_to_idx(cw)
                    covered = neighbors_set[cw_idx]
                    covered_arr = neighbors_arr[cw_idx]
                    opts.append((cw_idx, covered, covered_arr))
            cell_options[(a,b)] = opts
    
    # Also precompute: for each point, which (cell, option) covers it
    point_to_coverers = [set() for _ in range(total)]
    for a in range(3):
        for b in range(3):
            for opt_idx, (cw_idx, covered, covered_arr) in enumerate(cell_options[(a,b)]):
                for pt in covered:
                    point_to_coverers[pt].add(((a,b), opt_idx))
    
    def greedy_cover():
        uncovered = set(range(total))
        code = []
        while uncovered:
            best_cand = -1
            best_count = -1
            candidates = list(range(total))
            random.shuffle(candidates)
            for cand in candidates:
                if not uncovered:
                    break
                covered_here = len(neighbors_set[cand] & uncovered)
                if covered_here > best_count:
                    best_count = covered_here
                    best_cand = cand
            if best_cand < 0:
                break
            code.append(best_cand)
            uncovered -= neighbors_set[best_cand]
        return code

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

    def tetracode_partial(k_cells):
        """Choose k_cells out of 81, fill them completely."""
        all_cells = [(a,b) for a in range(3) for b in range(3)]
        random.shuffle(all_cells)
        chosen = all_cells[:k_cells]
        result = []
        for (a,b) in chosen:
            for x in range(3):
                for y in range(3):
                    cw = (a, b, (a+b)%3, (a+2*b)%3, x, y)
                    result.append(tuple_to_idx(cw))
        return result

    def tetracode_random_options():
        """Each cell gets a random subset of its 9 options."""
        result = []
        for a in range(3):
            for b in range(3):
                num_opts = random.randint(1, 9)
                opts = random.sample(range(9), num_opts)
                for opt_idx in opts:
                    x, y = divmod(opt_idx, 3)
                    cw = (a, b, (a+b)%3, (a+2*b)%3, x, y)
                    result.append(tuple_to_idx(cw))
        return result

    start_time = time.time()
    deadline = 130.0  # Safety margin

    best_code = None
    best_size = float('inf')

    def validate_and_record(code):
        nonlocal best_code, best_size
        # Remove duplicates
        code = list(dict.fromkeys(code))
        if len(code) >= best_size:
            return
        # Validate coverage
        cover_count = np.zeros(total, dtype=np.int32)
        for c in code:
            cover_count[neighbors_arr[c]] += 1
        if np.any(cover_count == 0):
            return
        best_code = list(code)
        best_size = len(code)

    def local_search(code, time_limit):
        """Greedy local search: try removals, swaps, and 2-for-1 moves."""
        code = list(code)
        cover_count = np.zeros(total, dtype=np.int32)
        for c in code:
            cover_count[neighbors_arr[c]] += 1
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            random.shuffle(code)
            
            # Phase 1: removals
            for i in range(len(code)):
                if time.time() > time_limit:
                    break
                nb_arr = neighbors_arr[code[i]]
                old_counts = cover_count[nb_arr].copy()
                cover_count[nb_arr] -= 1
                uncovered = [nb for nb in nb_arr if cover_count[nb] == 0]
                if not uncovered:
                    code.pop(i)
                    improved = True
                    break
                else:
                    cover_count[nb_arr] = old_counts
            
            if improved or time.time() > time_limit:
                continue
            
            # Phase 2: single swaps
            for i in range(len(code)):
                if time.time() > time_limit:
                    break
                c = code[i]
                nb_arr = neighbors_arr[c]
                old_counts = cover_count[nb_arr].copy()
                cover_count[nb_arr] -= 1
                uncovered = [nb for nb in nb_arr if cover_count[nb] == 0]
                if not uncovered:
                    code.pop(i)
                    improved = True
                    break
                else:
                    # Find replacement covering all uncovered
                    u0 = uncovered[0]
                    candidates = list(neighbors_list[u0])
                    random.shuffle(candidates)
                    found = False
                    for new_c in candidates:
                        if all(u in neighbors_set[new_c] for u in uncovered):
                            if new_c not in code:
                                cover_count[neighbors_arr[new_c]] += 1
                                code[i] = new_c
                                improved = True
                                found = True
                                break
                            else:
                                # Already in code, just restore
                                pass
                    if found:
                        break
                    cover_count[nb_arr] = old_counts
            
            if improved or time.time() > time_limit:
                continue
            
            # Phase 3: 2-for-1 moves (remove two, add one)
            if len(code) >= 2:
                # Try random pairs
                num_trials = min(20, len(code) * (len(code) - 1) // 2)
                for _ in range(num_trials):
                    if time.time() > time_limit:
                        break
                    i, j = random.sample(range(len(code)), 2)
                    if i == j:
                        continue
                    c1, c2 = code[i], code[j]
                    if i > j:
                        i, j = j, i
                        c1, c2 = c2, c1
                    nb1 = neighbors_arr[c1]
                    nb2 = neighbors_arr[c2]
                    old_counts = cover_count[nb1].copy()
                    old_counts2 = cover_count[nb2].copy()
                    cover_count[nb1] -= 1
                    cover_count[nb2] -= 1
                    uncovered = [nb for nb in set(nb1.tolist()) | set(nb2.tolist()) if cover_count[nb] == 0]
                    if not uncovered:
                        # Remove both
                        code.pop(i)
                        code.pop(j)
                        improved = True
                        break
                    else:
                        # Try to find a single codeword covering all uncovered
                        u0 = uncovered[0]
                        candidates = list(neighbors_list[u0])
                        random.shuffle(candidates)
                        found = False
                        for new_c in candidates:
                            if all(u in neighbors_set[new_c] for u in uncovered):
                                if new_c not in code:
                                    cover_count[neighbors_arr[new_c]] += 1
                                    code[i] = new_c
                                    code.pop(j)
                                    improved = True
                                    found = True
                                    break
                        if found:
                            break
                        # Restore
                        cover_count[nb1] = old_counts
                        cover_count[nb2] = old_counts
        
        return code

    def simulated_annealing_full_space(time_limit):
        """SA over the full space of 729 codewords."""
        # State: boolean array indicating which codewords are in the code
        state = np.zeros(total, dtype=bool)
        
        # Initialize with the full tetracode
        tetracode_indices = tetracode_full()
        for idx in tetracode_indices:
            state[idx] = True
            
        # Compute initial metrics
        cover_count = np.zeros(total, dtype=np.int32)
        for i in range(total):
            if state[i]:
                cover_count[neighbors_arr[i]] += 1
        uncovered = int(np.sum(cover_count == 0))
        num_cw = int(np.sum(state))
        
        objective = num_cw + 100.0 * uncovered
        
        best_state = state.copy()
        best_obj = objective
        best_num_cw = num_cw
        best_uncovered = uncovered
        
        T = 20.0
        T_min = 0.01
        cooling = 0.999
        
        # Use numpy random for speed
        rng = np.random.default_rng()
        
        while time.time() < time_limit:
            if T < T_min:
                T = T_min
                
            # Pick a random codeword to toggle
            idx = rng.integers(total)
            
            if state[idx]:
                # Remove codeword
                nb_arr = neighbors_arr[idx]
                # Points in nb_arr that have cover_count == 1 will become uncovered
                newly_uncovered = np.sum(cover_count[nb_arr] == 1)
                
                delta_cw = -1
                delta_uncovered = newly_uncovered
                delta_obj = delta_cw + 100.0 * delta_uncovered
                
                # Acceptance probability
                if delta_obj <= 0:
                    accept = True
                else:
                    # exp(-delta/T)
                    prob = np.exp(-delta_obj / T)
                    accept = rng.random() < prob
                
                if accept:
                    state[idx] = False
                    cover_count[nb_arr] -= 1
                    num_cw += delta_cw
                    uncovered += delta_uncovered
                    objective += delta_obj
                    
                    if uncovered == 0 and num_cw < best_num_cw:
                        best_num_cw = num_cw
                        best_uncovered = uncovered
                        best_state = state.copy()
                        best_obj = objective
            else:
                # Add codeword
                nb_arr = neighbors_arr[idx]
                # Points in nb_arr that have cover_count == 0 will become covered
                newly_covered = np.sum(cover_count[nb_arr] == 0)
                
                delta_cw = 1
                delta_uncovered = -newly_covered
                delta_obj = delta_cw + 100.0 * delta_uncovered
                
                if delta_obj <= 0:
                    accept = True
                else:
                    prob = np.exp(-delta_obj / T)
                    accept = rng.random() < prob
                
                if accept:
                    state[idx] = True
                    cover_count[nb_arr] += 1
                    num_cw += delta_cw
                    uncovered += delta_uncovered
                    objective += delta_obj
                    
                    if uncovered == 0 and num_cw < best_num_cw:
                        best_num_cw = num_cw
                        best_uncovered = uncovered
                        best_state = state.copy()
                        best_obj = objective
            
            T *= cooling
            
        # Extract best code
        if best_uncovered == 0:
            code = [i for i in range(total) if best_state[i]]
            return code
        return None

    # Try different strategies
    strategies = []
    
    # Strategy 1: Full tetracode + local search
    strategies.append(('tetracode_full', tetracode_full()))
    
    # Strategy 2: Greedy cover + local search
    strategies.append(('greedy', greedy_cover()))
    
    # Strategy 3: Partial tetracode
    for k in [75, 78, 80]:
        strategies.append((f'tetracode_partial_{k}', tetracode_partial(k)))
    
    # Strategy 4: Random options
    for _ in range(3):
        strategies.append(('random_opts', tetracode_random_options()))
    
    random.shuffle(strategies)
    
    # Track time spent on each strategy
    total_time_budget = deadline - start_time
    time_per_strategy = total_time_budget / (len(strategies) + 2)  # +2 for SA runs
    
    for name, init_code in strategies:
        if time.time() > deadline - 10:
            break
        
        # Ensure validity
        cover_count = np.zeros(total, dtype=np.int32)
        for c in init_code:
            cover_count[neighbors_arr[c]] += 1
        if np.any(cover_count == 0):
            uncovered = set(np.where(cover_count == 0)[0].tolist())
            while uncovered:
                best_cand = -1
                best_count = -1
                for cand in list(uncovered):
                    covered_here = len(neighbors_set[cand] & uncovered)
                    if covered_here > best_count:
                        best_count = covered_here
                        best_cand = cand
                if best_cand < 0:
                    break
                init_code.append(best_cand)
                cover_count[neighbors_arr[best_cand]] += 1
                uncovered -= neighbors_set[best_cand]
        
        init_code = list(dict.fromkeys(init_code))
        validate_and_record(init_code)
        
        # Local search with time budget
        remaining = deadline - time.time() - 10
        if remaining > 5:
            ls_time = min(remaining, time_per_strategy)
            code = local_search(init_code, time.time() + ls_time)
            validate_and_record(code)
        
        if best_size <= 73:
            break
    
    # Strategy 5: Simulated Annealing on full space
    if time.time() < deadline - 15:
        sa_time = min(30, deadline - time.time() - 10)
        sa_code = simulated_annealing_full_space(time.time() + sa_time)
        if sa_code is not None:
            validate_and_record(sa_code)
    
    # Final: try more SA runs if time permits
    while time.time() < deadline - 10:
        sa_time = min(15, deadline - time.time() - 5)
        if sa_time < 5:
            break
        sa_code = simulated_annealing_full_space(time.time() + sa_time)
        if sa_code is not None:
            validate_and_record(sa_code)
        if best_size <= 73:
            break
    
    if best_code is None:
        best_code = tetracode_full()
    
    # Final verification
    cover_count = np.zeros(total, dtype=np.int32)
    for c in best_code:
        cover_count[neighbors_arr[c]] += 1
    if np.any(cover_count == 0):
        return [list(t) for t in itertools.product(range(q), repeat=n)]
    
    return [idx_to_tuple(c) for c in best_code]
