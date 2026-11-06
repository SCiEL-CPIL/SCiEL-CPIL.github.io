"""
Two-colourings of [1,256] with few monochromatic four-term progressions.
Combined Strategy:
1. Diverse Structured Initialization (Thue-Morse, Modular, Quadratic, Sine).
2. Aggressive Iterative Greedy Local Search.
3. Simulated Annealing with adaptive temperature.
4. Genetic Algorithm with segment crossover and mutation.
5. Final Polish with Multi-flip SA and Greedy LS.
"""
import time
import numpy as np

def construct():
    n = 256
    # Safety margin: Evaluator kills at 150s. We aim to finish by 145s.
    deadline = time.time() + 145.0

    # Precompute all 4-term arithmetic progressions (0-indexed: positions 0..255)
    # a, a+d, a+2d, a+3d where a+3d < n
    aps = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    num_aps = len(aps)
    aps_arr = np.array(aps, dtype=np.int32)

    # For each position, store the indices of APs it belongs to and its offset in that AP
    # This allows fast delta calculation for a single flip
    pos_ap_indices = [[] for _ in range(n)]
    pos_ap_offsets = [[] for _ in range(n)]
    for idx in range(num_aps):
        p0, p1, p2, p3 = aps_arr[idx]
        pos_ap_indices[p0].append(idx)
        pos_ap_offsets[p0].append(0)
        pos_ap_indices[p1].append(idx)
        pos_ap_offsets[p1].append(1)
        pos_ap_indices[p2].append(idx)
        pos_ap_offsets[p2].append(2)
        pos_ap_indices[p3].append(idx)
        pos_ap_offsets[p3].append(3)
        
    # Convert to numpy arrays for faster access, but keep as lists of lists for variable length
    # Actually, for delta_flip, we just need to iterate over the APs containing 'pos'.
    # We can precompute the "neighbors" for each position to speed up delta calculation.
    
    def count_mono_4ap(color):
        """Vectorized count of monochromatic 4-APs."""
        c = color[aps_arr]
        mono = (c[:, 1] == c[:, 0]) & (c[:, 2] == c[:, 0]) & (c[:, 3] == c[:, 0])
        return int(np.sum(mono))

    def delta_flip(color, pos):
        """
        Compute the change in mono-AP count if we flip position pos.
        Returns new_count - old_count.
        """
        ap_idxs = pos_ap_indices[pos]
        if not ap_idxs:
            return 0
        
        delta = 0
        # For each AP containing pos, check if it was mono before and if it will be mono after
        # If it was mono and becomes non-mono: delta -= 1
        # If it was non-mono and becomes mono: delta += 1
        # If it was mono and stays mono (impossible if we flip one bit in a mono AP? 
        # Wait. If AP is mono, all 4 are same. Flip one -> 3 same, 1 diff. Not mono.
        # So mono -> non-mono always.
        # If AP is non-mono:
        #   - If flipping pos makes it mono: delta += 1
        #   - Else: delta += 0
        
        # Optimization: 
        # 1. Count how many of the APs containing pos are currently mono.
        # 2. For each of those, they will become non-mono. So delta -= (count_mono_containing_pos)
        # 3. For the APs containing pos that are NOT currently mono, check if flipping pos makes them mono.
        #    This requires checking the other 3 positions.
        
        curr_color_pos = color[pos]
        new_color_pos = 1 - curr_color_pos
        
        mono_before = 0
        mono_after = 0
        
        for idx in ap_idxs:
            a, b, c, d = aps_arr[idx]
            # Get colors of the 4 positions
            # We know color[pos]. We need the other 3.
            # To avoid array indexing overhead in a tight loop, we can just index the color array.
            colors = [color[a], color[b], color[c], color[d]]
            
            # Check if mono before
            if colors[0] == colors[1] == colors[2] == colors[3]:
                mono_before += 1
                # After flip, it is NOT mono (since we changed one bit)
                # mono_after += 0
            else:
                # Check if it becomes mono after flip
                # The new color at 'pos' is new_color_pos.
                # The other 3 positions must match new_color_pos.
                # Find the other 3 positions
                # We know which offset pos is at? 
                # We didn't store offsets in a parallel array for easy access in this loop structure.
                # Let's just construct the new colors list temporarily.
                new_colors = colors.copy()
                # Find where pos is in a,b,c,d
                # This is a bit slow. Let's precompute the "other 3" indices for each (pos, ap_idx) pair?
                # That's too much memory.
                
                # Alternative: 
                # The other 3 colors are fixed. Let them be o1, o2, o3.
                # For the AP to be mono after flip, o1 == o2 == o3 == new_color_pos.
                # So we need to check if the other 3 are all equal to new_color_pos.
                
                # How to get other 3 efficiently?
                # We have a,b,c,d. pos is one of them.
                # We can just check:
                # If pos == a: check b,c,d == new_color_pos
                # If pos == b: check a,c,d == new_color_pos
                # etc.
                
                # To avoid branching, we can just sum up the matches.
                # But we need to exclude 'pos' itself.
                
                # Let's just do the simple check. It's only ~10-20 APs per position on average.
                # n=256, num_aps ~ 5000. Avg APs per pos ~ 20.
                # 256 * 20 = 5120 checks per full scan. Very fast.
                
                # Identify the other 3
                # This is inefficient. Let's precompute for each position a list of (ap_idx, offset).
                # We already have pos_ap_indices. Let's assume the order in aps_arr is fixed.
                # We can just check the 4 values.
                
                # Actually, a faster way:
                # If the other 3 are all equal to new_color_pos, then it becomes mono.
                # Let's just check the 3 values that are NOT pos.
                
                # Since we are in a Python loop, let's just be explicit.
                # This function is called thousands of times.
                
                # Let's try a different approach for delta_flip that is vectorized for a single position?
                # No, the current approach is fine if we optimize the inner loop.
                
                # Let's just check the 3 other positions.
                # We can precompute for each position 'pos', a list of the 3 "other" indices for each AP it's in?
                # That would be a list of lists of 3-int tuples.
                # Let's do that precomputation.
                
                pass # Placeholder for logic, will implement in precompute below
                
        # RE-IMPLEMENT delta_flip with precomputed "others"
        
        return 0 # Placeholder

    # Precompute "others" for fast delta_flip
    # pos_others[pos] is a list of tuples (other1, other2, other3) for each AP containing pos
    pos_others = [[] for _ in range(n)]
    for idx in range(num_aps):
        a, b, c, d = aps_arr[idx]
        # For a, others are b,c,d
        pos_others[a].append((b, c, d))
        # For b, others are a,c,d
        pos_others[b].append((a, c, d))
        # For c, others are a,b,d
        pos_others[c].append((a, b, d))
        # For d, others are a,b,c
        pos_others[d].append((a, b, c))
        
    # We also need to know if the AP was mono before.
    # We can just check the 4 colors in delta_flip.
    
    def delta_flip_fast(color, pos):
        """
        Fast delta calculation using precomputed 'others'.
        """
        others_list = pos_others[pos]
        if not others_list:
            return 0
            
        curr_color = color[pos]
        new_color = 1 - curr_color
        
        delta = 0
        
        for o1, o2, o3 in others_list:
            # Get colors of the other 3
            c1 = color[o1]
            c2 = color[o2]
            c3 = color[o3]
            
            # Check if mono before:
            # curr_color == c1 == c2 == c3
            if curr_color == c1 and c1 == c2 and c2 == c3:
                # It was mono. After flip, it is NOT mono.
                delta -= 1
            else:
                # It was NOT mono.
                # Check if it becomes mono:
                # new_color == c1 == c2 == c3
                if new_color == c1 and c1 == c2 and c2 == c3:
                    delta += 1
                    # It was non-mono, becomes mono.
                    
        return delta

    def greedy_local_search(color, score, time_budget=5.0):
        """
        Iterative greedy local search.
        Scans all positions, flips if improvement.
        Repeats until no improvement or time runs out.
        """
        start = time.time()
        improved = True
        while improved and time.time() - start < time_budget:
            improved = False
            for pos in range(n):
                delta = delta_flip_fast(color, pos)
                if delta < 0:
                    color[pos] = 1 - color[pos]
                    score += delta
                    improved = True
        return color, score

    def simulated_annealing(color, score, time_budget=10.0):
        """
        Simulated Annealing with single flips.
        """
        start = time.time()
        T_start = 2.0
        T_end = 0.01
        best_score = score
        best_color = color.copy()
        
        # Estimate number of iterations to set cooling schedule
        # Roughly 10000-50000 iterations per second?
        # Let's just use time-based cooling.
        
        while time.time() - start < time_budget:
            elapsed = time.time() - start
            remaining_frac = max(0.0, (time_budget - elapsed) / time_budget)
            T = T_end + (T_start - T_end) * (remaining_frac ** 1.2)
            
            pos = np.random.randint(0, n)
            delta = delta_flip_fast(color, pos)
            
            if delta <= 0:
                color[pos] = 1 - color[pos]
                score += delta
            else:
                # Accept with probability exp(-delta/T)
                if np.random.random() < np.exp(-delta / max(T, 0.001)):
                    color[pos] = 1 - color[pos]
                    score += delta
            
            if score < best_score:
                best_score = score
                best_color = color.copy()
                
        return best_color, best_score

    def multi_flip_sa(color, score, time_budget=10.0):
        """
        Simulated Annealing with multi-flip moves (2-4 bits) to escape local minima.
        """
        start = time.time()
        T_start = 3.0
        T_end = 0.005
        best_score = score
        best_color = color.copy()
        
        while time.time() - start < time_budget:
            elapsed = time.time() - start
            remaining_frac = max(0.0, (time_budget - elapsed) / time_budget)
            T = T_end + (T_start - T_end) * (remaining_frac ** 1.5)
            
            # Decide move type
            if np.random.random() < 0.6:
                # Single flip
                pos = np.random.randint(0, n)
                delta = delta_flip_fast(color, pos)
                if delta <= 0:
                    color[pos] = 1 - color[pos]
                    score += delta
                else:
                    if np.random.random() < np.exp(-delta / max(T, 0.001)):
                        color[pos] = 1 - color[pos]
                        score += delta
            else:
                # Multi-flip (2-4 positions)
                k = np.random.randint(2, 5)
                positions = np.random.choice(n, size=k, replace=False)
                
                # Calculate delta for multi-flip
                # We need to find all APs affected by any of these positions.
                affected_aps = set()
                for p in positions:
                    affected_aps.update(pos_ap_indices[p])
                    
                if not affected_aps:
                    delta = 0
                else:
                    ap_list = list(affected_aps)
                    curr_mono_count = 0
                    new_mono_count = 0
                    
                    for idx in ap_list:
                        a, b, c, d = aps_arr[idx]
                        colors = [color[a], color[b], color[c], color[d]]
                        
                        # Check mono before
                        if colors[0] == colors[1] == colors[2] == colors[3]:
                            curr_mono_count += 1
                            
                        # Create new colors
                        new_colors = colors.copy()
                        for p in positions:
                            # Find index of p in a,b,c,d
                            # This is inefficient. 
                            # We can just check if p is a, b, c, or d.
                            if p == a: new_colors[0] = 1 - new_colors[0]
                            if p == b: new_colors[1] = 1 - new_colors[1]
                            if p == c: new_colors[2] = 1 - new_colors[2]
                            if p == d: new_colors[3] = 1 - new_colors[3]
                            
                        # Check mono after
                        if new_colors[0] == new_colors[1] == new_colors[2] == new_colors[3]:
                            new_mono_count += 1
                            
                    delta = new_mono_count - curr_mono_count
                
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
        Genetic Algorithm.
        """
        start = time.time()
        pop_size = len(population)
        best_score = min(scores)
        best_idx = int(np.argmin(scores))
        best_color = population[best_idx].copy()
        
        pop = [p.copy() for p in population]
        sc = list(scores)
        
        while time.time() - start < time_budget:
            remaining = time_budget - (time.time() - start)
            if remaining < 0.1:
                break
            
            new_pop = [pop[best_idx].copy()] # Elitism
            new_sc = [sc[best_idx]]
            
            while len(new_pop) < pop_size:
                # Tournament selection
                k = 3
                idxs = np.random.choice(pop_size, k, replace=False)
                p1_idx = int(idxs[np.argmin([sc[i] for i in idxs])])
                idxs2 = np.random.choice(pop_size, k, replace=False)
                p2_idx = int(idxs2[np.argmin([sc[i] for i in idxs2])])
                
                p1 = pop[p1_idx]
                p2 = pop[p2_idx]
                
                # Crossover
                if np.random.random() < 0.5:
                    # Segment crossover
                    seg_start = np.random.randint(0, n - 16)
                    seg_len = np.random.randint(16, 64)
                    child = p1.copy()
                    child[seg_start:seg_start + seg_len] = p2[seg_start:seg_start + seg_len]
                else:
                    # Uniform crossover
                    mask = np.random.randint(0, 2, n)
                    child = np.where(mask == 1, p1, p2).astype(np.int8)
                    
                # Mutation
                num_mutations = np.random.randint(1, 6)
                mut_positions = np.random.choice(n, num_mutations, replace=False)
                child[mut_positions] = 1 - child[mut_positions]
                
                # Refine with short local search
                child_score = count_mono_4ap(child)
                ls_time = max(0.01, remaining * 0.01)
                if ls_time > 0.01:
                    child, child_score = greedy_local_search(child, child_score, time_budget=ls_time)
                    
                new_pop.append(child)
                new_sc.append(child_score)
                
            pop = new_pop
            sc = new_sc
            
            current_best_idx = int(np.argmin(sc))
            if sc[current_best_idx] < best_score:
                best_score = sc[current_best_idx]
                best_color = pop[current_best_idx].copy()
                best_idx = current_best_idx
                
        return best_color, best_score

    def generate_structured_colorings():
        """Generate diverse initial candidates."""
        candidates = []
        
        # Thue-Morse
        tm = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)
        candidates.append(tm.copy())
        
        # Bit-reversal Thue-Morse
        br = np.array([bin(i)[::-1].count('1') % 2 for i in range(n)], dtype=np.int8)
        candidates.append(br.copy())
        
        # Modular patterns
        for p in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
            for q in range(1, p):
                c = np.array([(i % p) < q for i in range(n)], dtype=np.int8)
                candidates.append(c)
                
        # Walsh functions
        for k in range(8):
            c = np.array([(i >> k) & 1 for i in range(n)], dtype=np.int8)
            candidates.append(c)
            
        # XOR combinations (Walsh-Hadamard)
        for mask in range(1, 256):
            c = np.array([bin(i & mask).count('1') % 2 for i in range(n)], dtype=np.int8)
            candidates.append(c)
            
        # Quadratic (from Inspiration 1)
        for a in range(1, 4):
            for b in range(2):
                for c in range(2):
                    col = np.array([(a * i * i + b * i + c) % 2 for i in range(n)], dtype=np.int8)
                    candidates.append(col)
                    
        # Sine-wave quantization (from Inspiration 1)
        for freq in [1, 2, 3, 4, 5, 7, 8, 11]:
            for phase in range(4):
                vals = np.sin(2 * np.pi * freq * np.arange(n) / n + phase * np.pi / 2)
                c = (vals > 0).astype(np.int8)
                candidates.append(c)
                
        # Random
        rng = np.random.default_rng(42)
        for _ in range(20):
            c = rng.integers(0, 2, n).astype(np.int8)
            candidates.append(c)
            
        return candidates

    best_color = None
    best_score = float('inf')
    
    # Phase 1: Generate and Evaluate Initial Candidates
    candidates = generate_structured_colorings()
    scored = []
    for c in candidates:
        s = count_mono_4ap(c)
        scored.append((s, c.copy()))
        
    scored.sort(key=lambda x: x[0])
    
    # Take top K for refinement
    top_k = min(15, len(scored))
    
    # Phase 2: Greedy Local Search on Top Candidates
    refined_candidates = []
    for rank in range(top_k):
        if time.time() > deadline - 60:
            break
            
        score, color = scored[rank]
        if score < best_score:
            best_score = score
            best_color = color.copy()
            
        # Allocate time for LS
        ls_budget = min(2.0, (deadline - time.time() - 60) * 0.05)
        if ls_budget > 0.1:
            color_ls, score_ls = greedy_local_search(color.copy(), score, time_budget=ls_budget)
            refined_candidates.append((score_ls, color_ls))
            if score_ls < best_score:
                best_score = score_ls
                best_color = color_ls.copy()
        else:
            refined_candidates.append((score, color.copy()))
            
    refined_candidates.sort(key=lambda x: x[0])
    
    # Phase 3: Simulated Annealing on Top Refined Candidates
    sa_refined = []
    for rank in range(min(8, len(refined_candidates))):
        if time.time() > deadline - 40:
            break
            
        score, color = refined_candidates[rank]
        sa_budget = min(5.0, (deadline - time.time() - 40) * 0.1)
        if sa_budget > 0.5:
            color_sa, score_sa = simulated_annealing(color.copy(), score, time_budget=sa_budget)
            sa_refined.append((score_sa, color_sa))
            if score_sa < best_score:
                best_score = score_sa
                best_color = color_sa.copy()
                
    sa_refined.sort(key=lambda x: x[0])
    
    # Phase 4: Genetic Algorithm
    pop_size = 24
    ga_population = []
    ga_scores = []
    
    # Seed GA with best results from previous phases
    for s, c in refined_candidates[:6]:
        ga_population.append(c)
        ga_scores.append(s)
    for s, c in sa_refined[:6]:
        ga_population.append(c)
        ga_scores.append(s)
        
    # Add randoms for diversity
    rng = np.random.default_rng(123)
    while len(ga_population) < pop_size:
        c = rng.integers(0, 2, n).astype(np.int8)
        s = count_mono_4ap(c)
        ga_population.append(c)
        ga_scores.append(s)
        
    if time.time() < deadline - 20:
        ga_budget = deadline - time.time() - 20
        ga_color, ga_score = genetic_algorithm(ga_population, ga_scores, time_budget=ga_budget)
        if ga_score < best_score:
            best_score = ga_score
            best_color = ga_color.copy()
            
    # Phase 5: Final Polish
    if best_color is not None and time.time() < deadline - 5:
        # Multi-flip SA for escaping deep local minima
        mf_budget = min(5.0, deadline - time.time() - 5)
        if mf_budget > 1.0:
            color_mf, score_mf = multi_flip_sa(best_color.copy(), best_score, time_budget=mf_budget)
            if score_mf < best_score:
                best_score = score_mf
                best_color = color_mf.copy()
                
        # Final Greedy LS
        if time.time() < deadline - 2:
            ls_budget = deadline - time.time() - 2
            color_final, score_final = greedy_local_search(best_color.copy(), best_score, time_budget=ls_budget)
            if score_final < best_score:
                best_score = score_final
                best_color = color_final.copy()

    if best_color is None:
        # Fallback
        best_color = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)

    return [int(c) for c in best_color]

