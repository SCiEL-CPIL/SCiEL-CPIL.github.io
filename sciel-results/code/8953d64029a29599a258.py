"""Max-SAT formulation with vectorized noise-injected local search and dynamic temperature."""
import time
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 115  # safety margin before 150s kill

    # Precompute all 4-term arithmetic progressions (0-indexed: positions 0..255)
    aps = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    num_aps = len(aps)
    aps_arr = np.array(aps, dtype=np.int32)

    # For each position, which APs does it participate in?
    pos_aps = [[] for _ in range(n)]
    for idx in range(num_aps):
        p0, p1, p2, p3 = aps_arr[idx]
        pos_aps[p0].append((idx, 0))
        pos_aps[p1].append((idx, 1))
        pos_aps[p2].append((idx, 2))
        pos_aps[p3].append((idx, 3))

    pos_ap_indices = []
    pos_ap_offsets = []
    for i in range(n):
        if pos_aps[i]:
            indices, offsets = zip(*pos_aps[i])
            pos_ap_indices.append(np.array(indices, dtype=np.int32))
            pos_ap_offsets.append(np.array(offsets, dtype=np.int8))
        else:
            pos_ap_indices.append(np.array([], dtype=np.int32))
            pos_ap_offsets.append(np.array([], dtype=np.int8))

    def count_mono_4ap(color):
        c = color[aps_arr]
        mono = (c[:, 1] == c[:, 0]) & (c[:, 2] == c[:, 0]) & (c[:, 3] == c[:, 0])
        return int(np.sum(mono))

    def delta_flip(color, pos):
        ap_idx = pos_ap_indices[pos]
        if len(ap_idx) == 0:
            return 0
        sub = color[aps_arr[ap_idx]]
        off = pos_ap_offsets[pos]
        curr_mono = (sub[:, 1] == sub[:, 0]) & (sub[:, 2] == sub[:, 0]) & (sub[:, 3] == sub[:, 0])
        curr_count = int(np.sum(curr_mono))
        new_sub = sub.copy()
        new_sub[np.arange(len(ap_idx)), off] = 1 - new_sub[np.arange(len(ap_idx)), off]
        new_mono = (new_sub[:, 1] == new_sub[:, 0]) & (new_sub[:, 2] == new_sub[:, 0]) & (new_sub[:, 3] == new_sub[:, 0])
        new_count = int(np.sum(new_mono))
        return new_count - curr_count

    def local_search(color, score, time_budget=5.0):
        """Greedy local search: keep flipping if it improves."""
        improved = True
        start = time.time()
        while improved and time.time() - start < time_budget:
            improved = False
            for pos in range(n):
                delta = delta_flip(color, pos)
                if delta < 0:
                    color[pos] = 1 - color[pos]
                    score += delta
                    improved = True
        return color, score

    def fast_maxsat_sa(color, score, time_budget=15.0):
        """
        Fast Max-SAT Simulated Annealing with incremental violation count updates.
        """
        start = time.time()
        T_start = 2.0
        T_end = 0.01
        best_score = score
        best_color = color.copy()
        
        # Initialize is_mono and violation_counts
        c = color[aps_arr]
        is_mono = (c[:, 1] == c[:, 0]) & (c[:, 2] == c[:, 0]) & (c[:, 3] == c[:, 0])
        violation_counts = np.zeros(n, dtype=np.int32)
        
        # Compute initial violation counts
        for ap_idx in np.where(is_mono)[0]:
            p0, p1, p2, p3 = aps_arr[ap_idx]
            violation_counts[p0] += 1
            violation_counts[p1] += 1
            violation_counts[p2] += 1
            violation_counts[p3] += 1
            
        num_violated = int(np.sum(is_mono))
        
        while time.time() - start < time_budget:
            elapsed = time.time() - start
            remaining_frac = max(0.0, (time_budget - elapsed) / time_budget)
            T = T_end + (T_start - T_end) * remaining_frac ** 1.2
            
            if num_violated == 0:
                break
            
            # Choose position
            # 80% biased towards high violation count, 20% random
            if np.random.random() < 0.8:
                candidates = np.where(violation_counts > 0)[0]
                if len(candidates) == 0:
                    candidates = np.arange(n)
                weights = violation_counts[candidates].astype(np.float64)
                # Soft-max or power law to avoid zero probs if some are 0? 
                # violation_counts > 0 ensures weights > 0
                weights = weights ** 1.5
                total_w = weights.sum()
                if total_w > 0:
                    probs = weights / total_w
                    pos = candidates[np.random.choice(len(candidates), p=probs)]
                else:
                    pos = candidates[np.random.randint(len(candidates))]
            else:
                pos = np.random.randint(0, n)
                
            # Calculate delta and update is_mono for affected APs
            ap_idx_arr = pos_ap_indices[pos]
            off_arr = pos_ap_offsets[pos]
            
            if len(ap_idx_arr) == 0:
                continue
                
            delta = 0
            affected_aps = []
            
            # We need to check mono status of these APs
            # Get current colors of these APs
            sub_colors = color[aps_arr[ap_idx_arr]]
            
            for i in range(len(ap_idx_arr)):
                ap_idx = ap_idx_arr[i]
                was_mono = is_mono[ap_idx]
                
                # Simulate flip at offset off_arr[i] for this AP
                # The color at this position in sub_colors is at row i, col off_arr[i]
                new_val = 1 - sub_colors[i, off_arr[i]]
                
                # Check if the AP would be mono with new_val
                # The other 3 values are in sub_colors[i, :]
                # We need to check if all 4 are equal to new_val? 
                # No, we just flip one bit. The other 3 bits are unchanged.
                # So we check if the other 3 bits are equal to new_val.
                
                # Extract the 4 values
                v0, v1, v2, v3 = sub_colors[i, 0], sub_colors[i, 1], sub_colors[i, 2], sub_colors[i, 3]
                # Replace the one at off_arr[i]
                if off_arr[i] == 0:
                    v0 = new_val
                elif off_arr[i] == 1:
                    v1 = new_val
                elif off_arr[i] == 2:
                    v2 = new_val
                elif off_arr[i] == 3:
                    v3 = new_val
                    
                is_now_mono = (v0 == v1) and (v1 == v2) and (v2 == v3)
                
                if was_mono and not is_now_mono:
                    delta -= 1
                elif not was_mono and is_now_mono:
                    delta += 1
                    
                # Store result for potential update
                affected_aps.append((ap_idx, was_mono, is_now_mono))
                
            # Decide accept/reject
            accept = False
            if delta <= 0:
                accept = True
            else:
                if np.random.random() < np.exp(-delta / max(T, 0.001)):
                    accept = True
                    
            if accept:
                color[pos] = 1 - color[pos]
                score += delta
                num_violated += delta
                
                # Update is_mono and violation_counts
                for ap_idx, was_mono, is_now_mono in affected_aps:
                    if was_mono != is_now_mono:
                        is_mono[ap_idx] = is_now_mono
                        # Update violation counts for the 4 positions of this AP
                        p0, p1, p2, p3 = aps_arr[ap_idx]
                        if is_now_mono:
                            violation_counts[p0] += 1
                            violation_counts[p1] += 1
                            violation_counts[p2] += 1
                            violation_counts[p3] += 1
                        else:
                            violation_counts[p0] -= 1
                            violation_counts[p1] -= 1
                            violation_counts[p2] -= 1
                            violation_counts[p3] -= 1
                            
            if score < best_score:
                best_score = score
                best_color = color.copy()
                
        return best_color, best_score

    def multi_flip_sa(color, score, time_budget=10.0):
        """Simulated annealing with multi-flip moves for escaping local minima."""
        start = time.time()
        T_start = 4.0
        T_end = 0.005
        best_score = score
        best_color = color.copy()

        while time.time() - start < time_budget:
            elapsed = time.time() - start
            remaining_frac = max(0.0, (time_budget - elapsed) / time_budget)
            T = T_end + (T_start - T_end) * remaining_frac ** 1.5

            # Choose move type: 60% single flip, 40% multi-flip
            if np.random.random() < 0.6:
                pos = np.random.randint(0, n)
                delta = delta_flip(color, pos)
                if delta <= 0:
                    color[pos] = 1 - color[pos]
                    score += delta
                else:
                    if np.random.random() < np.exp(-delta / max(T, 0.001)):
                        color[pos] = 1 - color[pos]
                        score += delta
            else:
                # Multi-flip: pick 2-4 random distinct positions
                k = np.random.randint(2, 5)
                positions = np.random.choice(n, size=k, replace=False)
                affected_aps = set()
                for p in positions:
                    if len(pos_ap_indices[p]) > 0:
                        affected_aps.update(pos_ap_indices[p].tolist())
                if affected_aps:
                    ap_list = np.array(sorted(affected_aps), dtype=np.int32)
                    sub = color[aps_arr[ap_list]]
                    curr_mono = (sub[:, 1] == sub[:, 0]) & (sub[:, 2] == sub[:, 0]) & (sub[:, 3] == sub[:, 0])
                    curr_count = int(np.sum(curr_mono))
                    new_sub = sub.copy()
                    for p in positions:
                        off_arr = pos_ap_offsets[p]
                        idx_arr = pos_ap_indices[p]
                        for j, ap_i in enumerate(idx_arr):
                            row_in_new = np.where(ap_list == ap_i)[0]
                            if len(row_in_new) > 0:
                                new_sub[row_in_new[0], off_arr[j]] = 1 - new_sub[row_in_new[0], off_arr[j]]
                    new_mono = (new_sub[:, 1] == new_sub[:, 0]) & (new_sub[:, 2] == new_sub[:, 0]) & (new_sub[:, 3] == new_sub[:, 0])
                    new_count = int(np.sum(new_mono))
                    delta = new_count - curr_count
                    if delta <= 0:
                        for p in positions:
                            color[p] = 1 - color[p]
                        score += delta
                    else:
                        if np.random.random() < np.exp(-delta / max(T, 0.001)):
                            for p in positions:
                                color[p] = 1 - color[p]
                            score += delta

            if score < best_score:
                best_score = score
                best_color = color.copy()

        return best_color, best_score

    def genetic_algorithm(population, scores, time_budget=30.0):
        """
        Genetic Algorithm with tournament selection, uniform crossover, and mutation.
        Offspring are refined with a short local search.
        """
        start = time.time()
        pop_size = len(population)
        best_score = min(scores)
        best_idx = scores.index(best_score)
        best_color = population[best_idx].copy()
        
        # Ensure population is list of np arrays
        pop = [p.copy() for p in population]
        scores = list(scores)
        
        # Elitism: keep best
        # We will replace worst individuals with offspring
        
        while time.time() - start < time_budget:
            new_pop = [pop[best_idx].copy()]
            new_scores = [scores[best_idx]]
            
            # Generate offspring to fill the rest of the population
            while len(new_pop) < pop_size:
                # Tournament selection
                k = 3
                idxs = np.random.choice(pop_size, k, replace=False)
                parent1_idx = idxs[np.argmin([scores[i] for i in idxs])]
                idxs2 = np.random.choice(pop_size, k, replace=False)
                parent2_idx = idxs2[np.argmin([scores[i] for i in idxs2])]
                
                p1 = pop[parent1_idx]
                p2 = pop[parent2_idx]
                
                # Uniform Crossover
                mask = np.random.randint(0, 2, n)
                child = np.where(mask == 1, p1, p2).astype(np.int8)
                
                # Mutation: flip a few bits
                num_mutations = np.random.randint(1, 5)
                mut_positions = np.random.choice(n, num_mutations, replace=False)
                child[mut_positions] = 1 - child[mut_positions]
                
                # Refine with short local search
                child_score = count_mono_4ap(child)
                # Short LS budget to ensure progress
                ls_time = max(0.01, (time_budget - (time.time() - start)) * 0.01)
                if ls_time > 0:
                    child, child_score = local_search(child, child_score, time_budget=ls_time)
                
                new_pop.append(child)
                new_scores.append(child_score)
            
            pop = new_pop
            scores = new_scores
            
            # Update global best
            current_best_idx = scores.index(min(scores))
            if scores[current_best_idx] < best_score:
                best_score = scores[current_best_idx]
                best_color = pop[current_best_idx].copy()
                best_idx = current_best_idx
            else:
                # Keep the old best index if it's still in the population? 
                # Actually, we just replace the whole pop, so we need to ensure the best is kept.
                # The elitism above keeps pop[best_idx] as the first element.
                # But indices might shift if we don't track carefully.
                # Simpler: just track best_score and best_color independently.
                pass
                
        return best_color, best_score

    def generate_structured_colorings():
        """Generate a set of structured candidate colorings."""
        candidates = []

        # Thue-Morse
        tm = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)
        candidates.append(tm.copy())

        # Bit-reversal Thue-Morse
        br = np.array([bin(i)[::-1].count('1') % 2 for i in range(n)], dtype=np.int8)
        candidates.append(br.copy())

        # Modular patterns
        for p in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
            for q in range(1, p):
                c = np.array([(i % p) < q for i in range(n)], dtype=np.int8)
                candidates.append(c)

        # Walsh functions
        for k in range(8):
            c = np.array([(i >> k) & 1 for i in range(n)], dtype=np.int8)
            candidates.append(c)

        # XOR combinations of bits
        for mask in range(1, 256):
            c = np.array([bin(i & mask).count('1') % 2 for i in range(n)], dtype=np.int8)
            candidates.append(c)

        # Random starting points
        rng = np.random.default_rng(42)
        for _ in range(30):
            c = rng.integers(0, 2, n).astype(np.int8)
            candidates.append(c)

        return candidates

    best_color = None
    best_score = float('inf')

    # Evaluate all structured candidates
    candidates = generate_structured_colorings()
    scored = []
    for c in candidates:
        s = count_mono_4ap(c)
        scored.append((s, c.copy()))

    scored.sort(key=lambda x: x[0])

    # Take top candidates for refinement
    top_k = min(25, len(scored))
    
    # Initial local search on top candidates to establish a strong baseline
    refined_candidates = []
    for rank in range(top_k):
        if time.time() > deadline - 60:
            break
        
        score, color = scored[rank]
        if score < best_score:
            best_score = score
            best_color = color.copy()

        # Phase 1: Greedy local search
        ls_budget = min(1.0, (deadline - time.time() - 60) * 0.05)
        if ls_budget > 0.1:
            color_ls, score_ls = local_search(color.copy(), score, time_budget=ls_budget)
            refined_candidates.append((score_ls, color_ls))
            if score_ls < best_score:
                best_score = score_ls
                best_color = color_ls.copy()
        else:
            refined_candidates.append((score, color.copy()))

    # Sort refined candidates
    refined_candidates.sort(key=lambda x: x[0])
    
    # Prepare population for GA: take top unique-ish candidates
    # We want diversity, so maybe take top 10 and some random ones
    pop_size = 20
    ga_population = []
    ga_scores = []
    
    # Add top refined
    for s, c in refined_candidates[:10]:
        ga_population.append(c)
        ga_scores.append(s)
        
    # Add random ones to ensure diversity
    rng = np.random.default_rng(123)
    while len(ga_population) < pop_size:
        c = rng.integers(0, 2, n).astype(np.int8)
        s = count_mono_4ap(c)
        ga_population.append(c)
        ga_scores.append(s)
        
    # Run Genetic Algorithm
    if time.time() < deadline - 20:
        ga_budget = deadline - time.time() - 20
        ga_color, ga_score = genetic_algorithm(ga_population, ga_scores, time_budget=ga_budget)
        if ga_score < best_score:
            best_score = ga_score
            best_color = ga_color.copy()
            
    # Final polish on the overall best
    if best_color is not None and time.time() < deadline - 10:
        final_budget = deadline - time.time() - 8
        color_final, score_final = local_search(best_color.copy(), best_score, time_budget=final_budget * 0.5)
        if score_final < best_score:
            best_score = score_final
            best_color = color_final.copy()

        if time.time() < deadline - 12:
            sa_final_budget = min(10.0, deadline - time.time() - 12)
            color_sa_f, score_sa_f = fast_maxsat_sa(best_color.copy(), best_score, time_budget=sa_final_budget)
            if score_sa_f < best_score:
                best_score = score_sa_f
                best_color = color_sa_f.copy()

            if time.time() < deadline - 6:
                color_final2, score_final2 = local_search(best_color.copy(), best_score, time_budget=deadline - time.time() - 6)
                if score_final2 < best_score:
                    best_score = score_final2
                    best_color = color_final2.copy()

    if best_color is None:
        best_color = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)

    return [int(c) for c in best_color]
