import time
import random
import numpy as np
from itertools import product

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    if q == 3 and n == 7 and radius == 1:
        return _solve_ternary_7_1(start_time, deadline)

    return _generic_construct(q, n, radius, start_time, deadline)

def _solve_ternary_7_1(start_time, deadline):
    q, n, radius = 3, 7, 1
    N = q ** n  # 2187

    # Precomputation
    # Map word -> index and idx -> word
    word_to_idx = {}
    idx_to_word = [None] * N
    for i, w in enumerate(product(range(q), repeat=n)):
        word_to_idx[w] = i
        idx_to_word[i] = w

    # Precompute neighbors (Hamming distance <= 1)
    # neighbors[i] is a list of indices
    neighbors = [[] for _ in range(N)]
    # For fast access in local search, also store as numpy arrays? 
    # List of lists is fine for N=2187.
    for i in range(N):
        w = idx_to_word[i]
        neighbors[i].append(i)
        w_list = list(w)
        for pos in range(n):
            orig_val = w_list[pos]
            for val in range(q):
                if val != orig_val:
                    w_list[pos] = val
                    nb_idx = word_to_idx[tuple(w_list)]
                    neighbors[i].append(nb_idx)
            w_list[pos] = orig_val

    # Precompute coverers: coverers[point] = list of codeword indices that cover this point
    coverers = [[] for _ in range(N)]
    for i in range(N):
        for pt in neighbors[i]:
            coverers[pt].append(i)

    # Precompute bitmasks for balls (for fast coverage checks in greedy)
    ball_bitsets = [0] * N
    for i in range(N):
        mask = 0
        for pt in neighbors[i]:
            mask |= (1 << pt)
        ball_bitsets[i] = mask

    all_mask = (1 << N) - 1

    # Shared state for greedy and local search
    is_chosen = [False] * N

    # 1. Greedy Set Cover Initialization
    def greedy_init():
        current_cover_mask = 0
        selected = []
        
        while current_cover_mask != all_mask:
            if time.time() > deadline - 50.0:
                break
                
            best_idx = -1
            best_count = 0
            
            # Scan all candidates
            for i in range(N):
                if is_chosen[i]:
                    continue
                new_bits = ball_bitsets[i] & ~current_cover_mask
                if new_bits:
                    cnt = new_bits.bit_count()
                    if cnt > best_count:
                        best_count = cnt
                        best_idx = i
                        
            if best_idx == -1:
                break
                
            selected.append(best_idx)
            is_chosen[best_idx] = True
            current_cover_mask |= ball_bitsets[best_idx]
            
        # Ensure valid coverage
        if current_cover_mask != all_mask:
            while current_cover_mask != all_mask:
                i = random.randint(0, N-1)
                if not is_chosen[i]:
                    selected.append(i)
                    is_chosen[i] = True
                current_cover_mask |= ball_bitsets[i]
                if time.time() > deadline - 10.0:
                    break
        return selected

    selected = greedy_init()
    
    # Initialize cover counts
    cover_count = np.zeros(N, dtype=np.int16)
    for cw in selected:
        for pt in neighbors[cw]:
            cover_count[pt] += 1
            
    best_selected = list(selected)
    best_size = len(selected)
    
    # 2. Local Search with Simulated Annealing / Randomized Swaps
    # We use a temperature-based approach to accept worse moves occasionally to escape local minima,
    # but primarily we focus on swaps that maintain validity or improve "slack".
    # Objective: Minimize size. Secondary: Maximize slack (sum of cover counts) to allow future removals?
    # Actually, standard heuristic for covering codes: 
    # 1. Try to remove redundant nodes.
    # 2. If stuck, try to swap a node to a position that increases the number of "removable" neighbors or reduces tightness.
    
    random.seed(42)
    
    def try_remove_redundant(selected, cover_count, is_chosen):
        removed = 0
        i = 0
        while i < len(selected):
            if time.time() > deadline - 5.0:
                break
            cw = selected[i]
            b = neighbors[cw]
            # Check if min count in ball > 1
            is_redundant = True
            for pt in b:
                if cover_count[pt] == 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                for pt in b:
                    cover_count[pt] -= 1
                selected.pop(i)
                is_chosen[cw] = False
                removed += 1
            else:
                i += 1
        return removed

    # Main Local Search Loop
    no_improvement = 0
    max_no_improvement = 50
    temperature = 1.0
    
    while time.time() < deadline - 2.0:
        improved = False
        
        # Phase 1: Removals
        removed = try_remove_redundant(selected, cover_count, is_chosen)
        if removed > 0:
            improved = True
            no_improvement = 0
            if len(selected) < best_size:
                best_size = len(selected)
                best_selected = list(selected)
            # Cool down slightly on success? Or reset temp?
            # temperature *= 0.95
            continue # Restart loop after modification
        
        # Phase 2: Swaps
        if selected:
            idx = random.randint(0, len(selected) - 1)
            old_cw = selected[idx]
            
            b_old = neighbors[old_cw]
            critical_points = [pt for pt in b_old if cover_count[pt] == 1]
            
            if not critical_points:
                # It's redundant, force remove
                for pt in b_old:
                    cover_count[pt] -= 1
                selected.pop(idx)
                is_chosen[old_cw] = False
                improved = True
                no_improvement = 0
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_selected = list(selected)
                continue
            
            # Find candidates that cover all critical points
            # Intersection of coverers for each critical point
            cand_set = set(coverers[critical_points[0]])
            for pt in critical_points[1:]:
                cand_set.intersection_update(coverers[pt])
                
            cand_set.discard(old_cw)
            for cw in selected:
                cand_set.discard(cw)
                
            if cand_set:
                candidates_list = list(cand_set)
                if len(candidates_list) > 30:
                    random.shuffle(candidates_list)
                    candidates_list = candidates_list[:30]
                
                best_new_cw = -1
                best_score = float('inf') # We want to minimize "damage" or maximize "slack"
                
                # Heuristic Score:
                # When we swap old_cw -> new_cw:
                # Points in b_old lose 1 count.
                # Points in b_new gain 1 count.
                # We already know b_old critical points (count 1) will be covered by new_cw (so they become count 1 again? No, they were 1, lost 1 -> 0, gained 1 -> 1).
                # Wait, if new_cw covers critical points, those points are in b_new.
                # So for critical points: count goes 1 -> 0 (remove old) -> 1 (add new). Net change 0.
                # For other points in b_old: count decreases by 1. Risk of becoming 0? No, they were >= 2, so >= 1.
                # For other points in b_new: count increases by 1.
                
                # A good heuristic is to minimize the number of points in b_new that have count 1 (excluding criticals)?
                # Or minimize the "tightness" of the new set.
                # Let's try: Minimize the number of points in b_new that will have count == 1 after the swap.
                # Actually, simpler: Minimize the sum of cover_count[pt] for pt in b_new. Lower sum means less "wasted" coverage?
                # Or maximize the number of points in b_old that had count 2 (so they become 1, creating new redundancy opportunities)?
                
                # Let's use a composite score:
                # 1. Count of points in b_new that are currently count 1 (excluding criticals which are forced).
                # 2. Count of points in b_old that are currently count 2 (these become 1, potentially allowing removal of other codewords later).
                
                # Let's just pick the candidate that minimizes the number of "new" count-1 points created in the rest of the space?
                # Actually, the most effective heuristic in covering codes is often:
                # Minimize the number of points in ball(new_cw) that have cover_count == 1 (excluding the criticals we are forced to cover).
                # This keeps the code "loose".
                
                for new_cw in candidates_list:
                    b_new = neighbors[new_cw]
                    # Count points in b_new with count 1, excluding critical_points
                    # critical_points are a subset of b_new? Yes, by definition of candidates.
                    risk_count = 0
                    for pt in b_new:
                        if cover_count[pt] == 1 and pt not in critical_points:
                            risk_count += 1
                            
                    # Also consider how many points in b_old had count 2 (becoming 1)
                    # This is a "benefit" as it creates more redundancy.
                    benefit_count = 0
                    for pt in b_old:
                        if cover_count[pt] == 2 and pt not in critical_points:
                            benefit_count += 1
                            
                    # Score: Minimize risk, Maximize benefit
                    score = risk_count - benefit_count
                    
                    if score < best_score:
                        best_score = score
                        best_new_cw = new_cw
                        
                if best_new_cw != -1:
                    # Perform Swap
                    for pt in b_old:
                        cover_count[pt] -= 1
                    selected[idx] = best_new_cw
                    is_chosen[old_cw] = False
                    is_chosen[best_new_cw] = True
                    
                    for pt in neighbors[best_new_cw]:
                        cover_count[pt] += 1
                    
                    improved = True
                    no_improvement = 0
                else:
                    no_improvement += 1
            else:
                no_improvement += 1
        else:
            no_improvement += 1
            
        if not improved:
            no_improvement += 1
            if no_improvement > max_no_improvement:
                # Try a restart with a slightly different random seed or perturbation?
                # For now, just break if stuck too long.
                # We could add a "kick" here: remove a random non-redundant codeword and re-greedy?
                # That's expensive. Let's just break.
                break
                
    # Re-verify best_selected and try to reduce it further if time permits.
    if time.time() < deadline - 1.0:
        # Check if best_selected is valid
        test_count = np.zeros(N, dtype=np.int16)
        for cw in best_selected:
            for pt in neighbors[cw]:
                test_count[pt] += 1
        
        if (test_count == 0).any():
            # If invalid, fall back to the current 'selected' state which is maintained as valid
            best_selected = list(selected)
            best_size = len(selected)
        else:
            # Try to reduce best_selected further
            # We need to re-initialize cover counts for best_selected
            bc = np.zeros(N, dtype=np.int16)
            for cw in best_selected:
                for pt in neighbors[cw]:
                    bc[pt] += 1
            # Simple removal pass
            i = 0
            while i < len(best_selected):
                cw = best_selected[i]
                b = neighbors[cw]
                redundant = True
                for pt in b:
                    if bc[pt] == 1:
                        redundant = False
                        break
                if redundant:
                    for pt in b:
                        bc[pt] -= 1
                    best_selected.pop(i)
                else:
                    i += 1
            best_size = len(best_selected)

    # Convert tuples to lists to match the required output format
    return [list(idx_to_word[i]) for i in best_selected]

def _generic_construct(q, n, radius, start_time, deadline):
    # Fallback for other parameters
    all_words = list(product(range(q), repeat=n))
    N = len(all_words)
    
    # Simple Greedy
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute balls
    balls = []
    for i, w in enumerate(all_words):
        ball = {i}
        w_list = list(w)
        for pos in range(n):
            orig = w_list[pos]
            for val in range(q):
                if val != orig:
                    w_list[pos] = val
                    nb = tuple(w_list)
                    ball.add(word_to_idx[nb])
            w_list[pos] = orig
        balls.append(ball)
        
    uncovered = set(range(N))
    selected = []
    
    while uncovered:
        if time.time() > deadline - 5.0:
            break
        best_idx = -1
        best_count = 0
        for i in range(N):
            if i in selected:
                continue
            cnt = len(balls[i] & uncovered)
            if cnt > best_count:
                best_count = cnt
                best_idx = i
        if best_idx == -1:
            break
        selected.append(best_idx)
        uncovered -= balls[best_idx]
        
    return [all_words[i] for i in selected]
