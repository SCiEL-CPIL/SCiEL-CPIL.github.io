import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    n = 128
    target_min = 128
    target_max = 256
    target_sums = list(range(target_min, target_max + 1))
    T = target_max - target_min + 1  # 129 target sums
    
    # Precompute for fast count updates
    # For a given A (as boolean mask of size 129), counts[s] = #{(a,b) in A, a<=b, a+b=s}
    # We maintain counts incrementally.
    
    def compute_counts(A_mask):
        """Compute all pair-sum counts using vectorized approach."""
        A_indices = np.where(A_mask)[0]
        m = len(A_indices)
        counts = np.zeros(257, dtype=np.int32)
        if m > 0:
            i_idx, j_idx = np.triu_indices(m)
            sums = A_indices[i_idx] + A_indices[j_idx]
            valid = (sums >= target_min) & (sums <= target_max)
            if np.any(valid):
                np.add.at(counts, sums[valid], 1)
        return counts
    
    def add_element_effects(A_indices, x):
        """Return array of count changes when adding element x to A."""
        # x pairs with all existing elements a (where a <= x for unordered, but we count all pairs)
        # Actually for unordered pairs with equal allowed:
        # Adding x: new pairs are (x, a) for a in A with a <= x, plus (x, x)
        # But wait - the count is over ALL unordered pairs. When we add x, the new pairs are:
        # (x, a) for each a in A where a <= x (to avoid double counting), plus (x, x)
        # Hmm, but actually the standard way: pairs are {a,b} with a<=b.
        # When adding x to A, new pairs are {x, a} for a in A where a <= x, plus {x, x}.
        # But this is tricky to compute incrementally. Let me use a different approach.
        # 
        # Actually, let me just recompute. For speed, I'll use the following:
        # For each element a in A, it contributes to sums a+a, a+(a+1), ..., a+128 that are in A.
        # Let me use a convolution-like approach.
        pass
    
    def fast_count_update_remove(A_mask, A_indices, x):
        """Compute delta in counts when removing x from A."""
        delta = np.zeros(257, dtype=np.int32)
        # Pairs involving x: (x, a) for a in A_indices where a != x, plus (x, x)
        for a in A_indices:
            if a == x:
                continue
            s = x + a
            if target_min <= s <= target_max:
                delta[s] -= 1
        s = 2 * x
        if target_min <= s <= target_max:
            delta[s] -= 1
        return delta
    
    def fast_count_update_add(A_mask, A_indices, x):
        """Compute delta in counts when adding x to A (x not currently in A)."""
        delta = np.zeros(257, dtype=np.int32)
        for a in A_indices:
            s = x + a
            if target_min <= s <= target_max:
                delta[s] += 1
        s = 2 * x
        if target_min <= s <= target_max:
            delta[s] += 1
        return delta
    
    def get_objective(counts):
        """Return (missing_count, peak, sum_of_squares) for target range."""
        target_counts = counts[target_min:target_max + 1]
        missing = int(np.sum(target_counts == 0))
        if missing > 0:
            return (missing, float('inf'), float('inf'))
        peak = int(np.max(target_counts))
        sum_sq = int(np.sum(target_counts ** 2))
        return (0, peak, sum_sq)
    
    def soft_peak(counts, beta=50.0):
        """Soft surrogate for peak using log-sum-exp."""
        target_counts = counts[target_min:target_max + 1].astype(np.float64)
        max_c = np.max(target_counts)
        if max_c == 0:
            return 0.0
        # log-sum-exp for smooth max
        shifted = target_counts - max_c
        log_sum_exp = max_c + np.log(np.sum(np.exp(beta * shifted))) / beta
        return log_sum_exp
    
    def greedy_construct(k_target):
        """Build a set A of size ~k_target that covers all target sums."""
        A_mask = np.zeros(129, dtype=bool)
        A_indices = []
        counts = np.zeros(257, dtype=np.int32)
        
        # Start with a random seed
        initial = random.sample(range(129), min(k_target // 3, 15))
        for x in initial:
            A_mask[x] = True
            A_indices.append(x)
        
        # Greedily add elements to cover missing sums
        max_iter = 200
        for _ in range(max_iter):
            obj = get_objective(counts)
            if obj[0] == 0:
                break
            
            missing_sums = [s for s in target_sums if counts[s] == 0]
            if not missing_sums:
                break
            
            # Find the best element to add
            best_x = None
            best_covered = -1
            best_delta = None
            
            candidates = [x for x in range(129) if not A_mask[x]]
            random.shuffle(candidates)
            
            for x in candidates[:30]:  # Limit candidates for speed
                # Compute what sums x would cover
                covered = 0
                for s in missing_sums:
                    other = s - x
                    if 0 <= other <= 128 and A_mask[other]:
                        covered += 1
                    elif other == x:
                        # x + x = s, so x covers s if 2*x == s
                        covered += 1
                
                if covered > best_covered:
                    best_covered = covered
                    best_x = x
                    best_delta = fast_count_update_add(A_mask, A_indices, x)
            
            if best_x is None or best_covered == 0:
                break
            
            A_mask[best_x] = True
            A_indices.append(best_x)
            counts += best_delta
        
        # If still missing, force add
        obj = get_objective(counts)
        if obj[0] > 0:
            missing_sums = [s for s in target_sums if counts[s] == 0]
            for s in missing_sums:
                # Find any pair that sums to s
                found = False
                for x in range(129):
                    if A_mask[x] and A_mask[s - x] if 0 <= s - x <= 128 else False:
                        found = True
                        break
                    if 2 * x == s and not A_mask[x]:
                        A_mask[x] = True
                        A_indices.append(x)
                        counts += fast_count_update_add(A_mask, A_indices, x)
                        found = True
                        break
                if not found:
                    # Add a new element
                    x = random.choice([v for v in range(129) if not A_mask[v]])
                    A_mask[x] = True
                    A_indices.append(x)
                    counts += fast_count_update_add(A_mask, A_indices, x)
        
        return A_mask, A_indices, counts
    
    def local_search(A_mask, A_indices, counts, time_limit, beta=50.0):
        """Simulated annealing with soft peak objective."""
        A_mask = A_mask.copy()
        A_indices = list(A_indices)
        counts = counts.copy()
        
        current_obj = get_objective(counts)
        if current_obj[0] > 0:
            return A_mask, A_indices, counts, current_obj
        
        current_peak = current_obj[1]
        current_sum_sq = current_obj[2]
        current_soft = soft_peak(counts, beta)
        
        best_A_mask = A_mask.copy()
        best_A_indices = A_indices.copy()
        best_counts = counts.copy()
        best_peak = current_peak
        best_sum_sq = current_sum_sq
        best_soft = current_soft
        
        T = 5.0
        T_min = 0.01
        no_improve = 0
        patience = 30
        
        while time.time() < time_limit:
            move_type = random.random()
            
            if move_type < 0.5:
                # Swap: remove a, add b
                if len(A_indices) < 2:
                    continue
                a = random.choice(A_indices)
                b = random.randint(0, 128)
                if A_mask[b]:
                    continue
                
                # Compute new counts
                new_counts = counts.copy()
                # Remove a
                for u in A_indices:
                    if u == a:
                        continue
                    s = a + u
                    if target_min <= s <= target_max:
                        new_counts[s] -= 1
                s_aa = 2 * a
                if target_min <= s_aa <= target_max:
                    new_counts[s_aa] -= 1
                
                # Add b
                for u in A_indices:
                    if u == a:
                        continue
                    s = b + u
                    if target_min <= s <= target_max:
                        new_counts[s] += 1
                s_bb = 2 * b
                if target_min <= s_bb <= target_max:
                    new_counts[s_bb] += 1
                
                new_obj = get_objective(new_counts)
                
            elif move_type < 0.8:
                # Add element
                candidates = [x for x in range(129) if not A_mask[x]]
                if not candidates:
                    continue
                b = random.choice(candidates)
                
                new_counts = counts.copy()
                for u in A_indices:
                    s = b + u
                    if target_min <= s <= target_max:
                        new_counts[s] += 1
                s_bb = 2 * b
                if target_min <= s_bb <= target_max:
                    new_counts[s_bb] += 1
                
                new_obj = get_objective(new_counts)
                
            else:
                # Remove element
                if len(A_indices) <= 15:
                    continue
                a = random.choice(A_indices)
                
                new_counts = counts.copy()
                for u in A_indices:
                    if u == a:
                        continue
                    s = a + u
                    if target_min <= s <= target_max:
                        new_counts[s] -= 1
                s_aa = 2 * a
                if target_min <= s_aa <= target_max:
                    new_counts[s_aa] -= 1
                
                new_obj = get_objective(new_counts)
            
            # Accept/reject
            if new_obj[0] > 0:
                # Invalid
                if random.random() < 0.05 and T > 1.0:
                    # Accept to escape
                    if move_type < 0.5:
                        A_mask[a] = False
                        A_mask[b] = True
                        A_indices.remove(a)
                        A_indices.append(b)
                    elif move_type < 0.8:
                        A_mask[b] = True
                        A_indices.append(b)
                    else:
                        A_mask[a] = False
                        A_indices.remove(a)
                    counts = new_counts
                    current_obj = get_objective(counts)
                continue
            
            new_peak = new_obj[1]
            new_sum_sq = new_obj[2]
            new_soft = soft_peak(new_counts, beta)
            
            # Accept if better
            if (new_peak < current_peak or 
                (new_peak == current_peak and new_sum_sq < current_sum_sq) or
                (new_peak == current_peak and new_sum_sq == current_sum_sq and new_soft < current_soft)):
                # Apply move
                if move_type < 0.5:
                    A_mask[a] = False
                    A_mask[b] = True
                    A_indices.remove(a)
                    A_indices.append(b)
                elif move_type < 0.8:
                    A_mask[b] = True
                    A_indices.append(b)
                else:
                    A_mask[a] = False
                    A_indices.remove(a)
                
                counts = new_counts
                current_obj = new_obj
                current_peak = new_peak
                current_sum_sq = new_sum_sq
                current_soft = new_soft
                no_improve = 0
                
                if (new_peak < best_peak or
                    (new_peak == best_peak and new_sum_sq < best_sum_sq) or
                    (new_peak == best_peak and new_sum_sq == best_sum_sq and new_soft < best_soft)):
                    best_peak = new_peak
                    best_sum_sq = new_sum_sq
                    best_soft = new_soft
                    best_A_mask = A_mask.copy()
                    best_A_indices = A_indices.copy()
                    best_counts = counts.copy()
                    
            else:
                # Worse: accept with probability
                delta = (new_peak - current_peak) * 1000 + (new_sum_sq - current_sum_sq) * 0.1 + (new_soft - current_soft)
                if delta > 0:
                    accept_prob = np.exp(-delta / max(T, 1e-10))
                    if random.random() < accept_prob:
                        if move_type < 0.5:
                            A_mask[a] = False
                            A_mask[b] = True
                            A_indices.remove(a)
                            A_indices.append(b)
                        elif move_type < 0.8:
                            A_mask[b] = True
                            A_indices.append(b)
                        else:
                            A_mask[a] = False
                            A_indices.remove(a)
                        counts = new_counts
                        current_obj = new_obj
                        current_peak = new_peak
                        current_sum_sq = new_sum_sq
                        current_soft = new_soft
                no_improve += 1
            
            # Cool
            if no_improve >= patience:
                T = max(T * 0.7, T_min)
                no_improve = 0
            else:
                T = max(T * 0.998, T_min)
            
            # Occasionally increase beta for sharper peak
            if random.random() < 0.01:
                beta = min(beta * 1.2, 200.0)
        
        return best_A_mask, best_A_indices, best_counts, (0, best_peak, best_sum_sq)
    
    def differential_evolution(A_mask, A_indices, counts, time_limit):
        """DE-inspired global search for better configurations."""
        A_mask = A_mask.copy()
        A_indices = list(A_indices)
        counts = counts.copy()
        
        current_obj = get_objective(counts)
        if current_obj[0] > 0:
            return A_mask, A_indices, counts, current_obj
        
        best_A_mask = A_mask.copy()
        best_A_indices = A_indices.copy()
        best_counts = counts.copy()
        best_peak = current_obj[1]
        best_sum_sq = current_obj[2]
        
        # Maintain a population of solutions
        pop_size = 8
        population = [(A_mask.copy(), A_indices.copy(), counts.copy())]
        
        # Initialize population with perturbations
        for _ in range(pop_size - 1):
            new_mask = A_mask.copy()
            new_indices = A_indices.copy()
            # Randomly swap a few elements
            for _ in range(random.randint(2, 5)):
                if not new_indices:
                    break
                a = random.choice(new_indices)
                b = random.randint(0, 128)
                if not new_mask[b]:
                    new_mask[a] = False
                    new_mask[b] = True
                    new_indices.remove(a)
                    new_indices.append(b)
            new_counts = compute_counts(new_mask)
            population.append((new_mask, new_indices, new_counts))
        
        while time.time() < time_limit:
            # Evaluate population
            scored_pop = []
            for mask, indices, cnts in population:
                obj = get_objective(cnts)
                if obj[0] == 0:
                    scored_pop.append((obj[1], obj[2], mask, indices, cnts))
            
            if not scored_pop:
                # All invalid, restart
                break
            
            scored_pop.sort(key=lambda x: (x[0], x[1]))
            
            # Update best
            if scored_pop[0][0] < best_peak or (scored_pop[0][0] == best_peak and scored_pop[0][1] < best_sum_sq):
                best_peak = scored_pop[0][0]
                best_sum_sq = scored_pop[0][1]
                best_A_mask = scored_pop[0][2].copy()
                best_A_indices = scored_pop[0][3].copy()
                best_counts = scored_pop[0][4].copy()
            
            # DE step: create new candidates
            new_candidates = []
            for i in range(pop_size // 2):
                # Pick three distinct individuals
                idxs = random.sample(range(len(scored_pop)), min(3, len(scored_pop)))
                if len(idxs) < 3:
                    continue
                base = scored_pop[idxs[0]][2]
                donor1 = scored_pop[idxs[1]][2]
                donor2 = scored_pop[idxs[2]][2]
                
                new_mask = base.copy()
                new_indices = list(scored_pop[idxs[0]][3])
                
                # Mutate: swap some elements
                n_swaps = random.randint(1, 4)
                for _ in range(n_swaps):
                    if not new_indices:
                        break
                    a = random.choice(new_indices)
                    b = random.randint(0, 128)
                    if not new_mask[b]:
                        new_mask[a] = False
                        new_mask[b] = True
                        new_indices.remove(a)
                        new_indices.append(b)
                
                new_counts = compute_counts(new_mask)
                new_candidates.append((new_mask, new_indices, new_counts))
            
            # Replace worst members with new candidates
            for i, cand in enumerate(new_candidates):
                if i < len(population) - len(new_candidates):
                    continue
                obj = get_objective(cand[2])
                if obj[0] == 0:
                    population[-len(new_candidates) + i] = cand
                    break
            
            # Occasionally add a fresh random solution
            if random.random() < 0.1:
                fresh_mask, fresh_indices, fresh_counts = greedy_construct(random.randint(25, 45))
                if get_objective(fresh_counts)[0] == 0:
                    population[-1] = (fresh_mask, fresh_indices, fresh_counts)
        
        return best_A_mask, best_A_indices, best_counts, (0, best_peak, best_sum_sq)
    
    def final_polish(A_mask, A_indices, counts, time_limit):
        """High-precision local search with tighter acceptance."""
        A_mask = A_mask.copy()
        A_indices = list(A_indices)
        counts = counts.copy()
        
        current_obj = get_objective(counts)
        if current_obj[0] > 0:
            return A_mask, A_indices, counts, current_obj
        
        best_A_mask = A_mask.copy()
        best_A_indices = A_indices.copy()
        best_counts = counts.copy()
        best_peak = current_obj[1]
        best_sum_sq = current_obj[2]
        
        # Try all single-element swaps
        while time.time() < time_limit:
            improved = False
            target_counts = counts[target_min:target_max + 1]
            
            # Find sums with max count
            max_count = np.max(target_counts)
            if max_count <= 1:
                break
            
            # Try to reduce congestion at max-count sums
            congested = [target_min + i for i in range(T) if target_counts[i] == max_count]
            
            for a in A_indices[:]:
                for b in range(129):
                    if A_mask[b]:
                        continue
                    if time.time() > time_limit:
                        break
                    
                    # Compute new counts
                    new_counts = counts.copy()
                    for u in A_indices:
                        if u == a:
                            continue
                        s = a + u
                        if target_min <= s <= target_max:
                            new_counts[s] -= 1
                    s_aa = 2 * a
                    if target_min <= s_aa <= target_max:
                        new_counts[s_aa] -= 1
                    
                    for u in A_indices:
                        if u == a:
                            continue
                        s = b + u
                        if target_min <= s <= target_max:
                            new_counts[s] += 1
                    s_bb = 2 * b
                    if target_min <= s_bb <= target_max:
                        new_counts[s_bb] += 1
                    
                    new_obj = get_objective(new_counts)
                    if new_obj[0] > 0:
                        continue
                    
                    new_peak = new_obj[1]
                    new_sum_sq = new_obj[2]
                    
                    if new_peak < best_peak or (new_peak == best_peak and new_sum_sq < best_sum_sq):
                        A_mask[a] = False
                        A_mask[b] = True
                        A_indices.remove(a)
                        A_indices.append(b)
                        counts = new_counts
                        best_peak = new_peak
                        best_sum_sq = new_sum_sq
                        best_A_mask = A_mask.copy()
                        best_A_indices = A_indices.copy()
                        best_counts = counts.copy()
                        improved = True
                        break
                if improved or time.time() > time_limit:
                    break
            
            if not improved:
                break
        
        return best_A_mask, best_A_indices, best_counts, (0, best_peak, best_sum_sq)
    
    # Main search loop
    best_peak = float('inf')
    best_sum_sq = float('inf')
    best_A_mask = None
    best_A_indices = None
    best_counts = None
    
    # Try multiple random initializations
    for trial in range(20):
        if time.time() > deadline:
            break
        
        k_target = random.randint(28, 48)
        A_mask, A_indices, counts = greedy_construct(k_target)
        
        obj = get_objective(counts)
        if obj[0] > 0:
            continue
        
        peak = obj[1]
        sum_sq = obj[2]
        
        if peak < best_peak or (peak == best_peak and sum_sq < best_sum_sq):
            best_peak = peak
            best_sum_sq = sum_sq
            best_A_mask = A_mask.copy()
            best_A_indices = A_indices.copy()
            best_counts = counts.copy()
        
        time_left = deadline - time.time()
        if time_left < 3:
            break
        
        # Allocate time between different strategies
        if trial < 6:
            # Local search
            t_ls = min(time_left - 2, 15)
            if t_ls > 1:
                A_opt, idx_opt, cnt_opt, obj_opt = local_search(A_mask, A_indices, counts, time.time() + t_ls)
                if obj_opt[0] == 0:
                    if obj_opt[1] < best_peak or (obj_opt[1] == best_peak and obj_opt[2] < best_sum_sq):
                        best_peak = obj_opt[1]
                        best_sum_sq = obj_opt[2]
                        best_A_mask = A_opt.copy()
                        best_A_indices = idx_opt.copy()
                        best_counts = cnt_opt.copy()
        elif trial < 12:
            # DE-inspired search
            t_de = min(time_left - 2, 20)
            if t_de > 1:
                A_opt, idx_opt, cnt_opt, obj_opt = differential_evolution(A_mask, A_indices, counts, time.time() + t_de)
                if obj_opt[0] == 0:
                    if obj_opt[1] < best_peak or (obj_opt[1] == best_peak and obj_opt[2] < best_sum_sq):
                        best_peak = obj_opt[1]
                        best_sum_sq = obj_opt[2]
                        best_A_mask = A_opt.copy()
                        best_A_indices = idx_opt.copy()
                        best_counts = cnt_opt.copy()
        else:
            # Final polish on best found so far
            t_polish = min(time_left - 2, 25)
            if t_polish > 1 and best_A_mask is not None:
                A_opt, idx_opt, cnt_opt, obj_opt = final_polish(best_A_mask, best_A_indices, best_counts, time.time() + t_polish)
                if obj_opt[0] == 0:
                    if obj_opt[1] < best_peak or (obj_opt[1] == best_peak and obj_opt[2] < best_sum_sq):
                        best_peak = obj_opt[1]
                        best_sum_sq = obj_opt[2]
                        best_A_mask = A_opt.copy()
                        best_A_indices = idx_opt.copy()
                        best_counts = cnt_opt.copy()
        
        # After first few trials, do a combined local+DE on the best
        if trial >= 3 and trial < 10 and best_A_mask is not None:
            time_left = deadline - time.time()
            if time_left > 5:
                t1 = min(time_left - 2, 10)
                if t1 > 1:
                    A_opt, idx_opt, cnt_opt, obj_opt = local_search(best_A_mask, best_A_indices, best_counts, time.time() + t1)
                    if obj_opt[0] == 0:
                        if obj_opt[1] < best_peak or (obj_opt[1] == best_peak and obj_opt[2] < best_sum_sq):
                            best_peak = obj_opt[1]
                            best_sum_sq = obj_opt[2]
                            best_A_mask = A_opt.copy()
                            best_A_indices = idx_opt.copy()
                            best_counts = cnt_opt.copy()
    
    # Final polish with remaining time
    if best_A_mask is not None:
        time_left = deadline - time.time()
        if time_left > 3:
            A_opt, idx_opt, cnt_opt, obj_opt = final_polish(best_A_mask, best_A_indices, best_counts, time.time() + time_left - 1)
            if obj_opt[0] == 0:
                if obj_opt[1] < best_peak or (obj_opt[1] == best_peak and obj_opt[2] < best_sum_sq):
                    best_peak = obj_opt[1]
                    best_sum_sq = obj_opt[2]
                    best_A_mask = A_opt.copy()
                    best_A_indices = idx_opt.copy()
                    best_counts = cnt_opt.copy()
    
    # Fallback
    if best_A_mask is None:
        # Construct a valid set as fallback
        A_mask = np.zeros(129, dtype=bool)
        for x in range(64, 129):
            A_mask[x] = True
        A_indices = [x for x in range(64, 129)]
        counts = compute_counts(A_mask)
        obj = get_objective(counts)
        if obj[0] == 0:
            best_peak = obj[1]
            best_sum_sq = obj[2]
            best_A_mask = A_mask.copy()
            best_A_indices = A_indices
            best_counts = counts
        else:
            return list(range(129))
    
    A_list = [i for i in range(129) if best_A_mask[i]]
    return A_list
