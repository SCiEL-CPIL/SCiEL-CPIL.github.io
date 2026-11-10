"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
N = 256

def _has_4ap_fast(selected_bool, x):
    """Check if adding x to selected_bool creates a 4-AP. selected_bool is a boolean array."""
    # x is 1st term: x, x+d, x+2d, x+3d
    max_d1 = (N - x) // 3
    for d in range(1, max_d1 + 1):
        if selected_bool[x+d] and selected_bool[x+2*d] and selected_bool[x+3*d]:
            return True
    
    # x is 2nd term: x-d, x, x+d, x+2d
    max_d2 = min(x - 1, (N - x) // 2)
    for d in range(1, max_d2 + 1):
        a = x - d
        if selected_bool[a] and selected_bool[x+d] and selected_bool[x+2*d]:
            return True
            
    # x is 3rd term: x-2d, x-d, x, x+d
    max_d3 = min((x - 1) // 2, N - x)
    for d in range(1, max_d3 + 1):
        a = x - 2*d
        if selected_bool[a] and selected_bool[x-d] and selected_bool[x+d]:
            return True
            
    # x is 4th term: x-3d, x-2d, x-d, x
    max_d4 = (x - 1) // 3
    for d in range(1, max_d4 + 1):
        a = x - 3*d
        if selected_bool[a] and selected_bool[x-2*d] and selected_bool[x-d]:
            return True
            
    return False

def _count_3aps_fast(selected_bool):
    """Count 3-APs using boolean array."""
    count = 0
    for a in range(1, N+1):
        if not selected_bool[a]:
            continue
        for d in range(1, (N - a) // 2 + 1):
            if selected_bool[a+d] and selected_bool[a+2*d]:
                count += 1
    return count

def _count_3aps_delta(selected_bool, x):
    """Count 3-APs that involve x (as first, middle, or last term)."""
    delta = 0
    # x as first: (x, x+d, x+2d)
    max_d1 = (N - x) // 2
    for d in range(1, max_d1 + 1):
        if selected_bool[x+d] and selected_bool[x+2*d]:
            delta += 1
            
    # x as middle: (x-d, x, x+d)
    max_d2 = min(x - 1, N - x)
    for d in range(1, max_d2 + 1):
        if selected_bool[x-d] and selected_bool[x+d]:
            delta += 1
            
    # x as last: (x-2d, x-d, x)
    max_d3 = (x - 1) // 2
    for d in range(1, max_d3 + 1):
        if selected_bool[x-2*d] and selected_bool[x-d]:
            delta += 1
            
    return delta

def _greedy_weighted_construct(rng):
    """
    Construct a set by greedily adding elements that maximize 3-AP delta 
    without creating 4-APs.
    """
    selected = np.zeros(N+1, dtype=bool)
    remaining = list(range(1, N+1))
    
    while remaining:
        best_x = -1
        best_delta = -1
        # To introduce randomness and allow different paths, we can shuffle remaining
        # or just pick the best. To avoid getting stuck in a specific greedy trap,
        # we might want to consider top-K candidates.
        
        # For speed, we calculate delta for all remaining and pick the best valid one.
        # N=256, so 256 * 256 operations per iteration is ~65k ops. 
        # 256 iterations * 65k = 16M ops. This might be too slow in Python if done naively.
        # Let's optimize: only check validity for top few deltas?
        # Or just check validity for all, as _has_4ap_fast is O(N) ~ 256 ops.
        # Total: 256 (iters) * 256 (cands) * 256 (check) = 16M ops. 
        # In Python, 16M ops might take ~1-2 seconds. This is acceptable for initialization.
        
        valid_candidates = []
        for x in remaining:
            if not _has_4ap_fast(selected, x):
                d = _count_3aps_delta(selected, x)
                valid_candidates.append((d, x))
        
        if not valid_candidates:
            break
            
        # Sort by delta descending
        valid_candidates.sort(key=lambda item: item[0], reverse=True)
        
        # Pick the best, or randomly from top 5 to diversify
        top_k = min(5, len(valid_candidates))
        if rng.random() < 0.8:
            best_d, best_x = valid_candidates[0]
        else:
            best_d, best_x = rng.choice(valid_candidates[:top_k])
        
        selected[best_x] = True
        remaining.remove(best_x)
        
    return selected

def _tabu_search(selected, current_score, deadline, rng):
    """Tabu search with add/remove/swap moves."""
    tabu_list = {}  # x -> iteration when it becomes non-tabu
    tabu_tenure = 10
    best_selected = selected.copy()
    best_score = current_score
    iter_count = 0
    no_improve = 0
    max_no_improve = 30 # Reduced to allow more restarts if stuck
    
    sel_list = [x for x in range(1, N+1) if selected[x]]
    unsel_list = [x for x in range(1, N+1) if not selected[x]]
    
    while time.time() < deadline and no_improve < max_no_improve:
        iter_count += 1
        moves = []
        
        # Add moves
        for x in unsel_list:
            if x in tabu_list and tabu_list[x] > iter_count:
                continue
            if not _has_4ap_fast(selected, x):
                delta = _count_3aps_delta(selected, x)
                moves.append(('add', x, delta))
        
        # Remove moves
        for x in sel_list:
            if x in tabu_list and tabu_list[x] > iter_count:
                continue
            delta = -_count_3aps_delta(selected, x)
            moves.append(('remove', x, delta))
        
        # Swap moves (Sampled)
        if sel_list and unsel_list:
            n_swap = min(len(sel_list), len(unsel_list), 30) # Increased sample size
            if len(sel_list) > n_swap:
                sel_sample = rng.sample(sel_list, n_swap)
            else:
                sel_sample = sel_list
            if len(unsel_list) > n_swap:
                unsel_sample = rng.sample(unsel_list, n_swap)
            else:
                unsel_sample = unsel_list
                
            for x_out in sel_sample:
                if x_out in tabu_list and tabu_list[x_out] > iter_count:
                    continue
                for x_in in unsel_sample:
                    if x_in in tabu_list and tabu_list[x_in] > iter_count:
                        continue
                    selected[x_out] = False
                    if not _has_4ap_fast(selected, x_in):
                        delta_out = -_count_3aps_delta(selected, x_out)
                        delta_in = _count_3aps_delta(selected, x_in)
                        delta = delta_out + delta_in
                        moves.append(('swap', (x_out, x_in), delta))
                    selected[x_out] = True
        
        if not moves:
            break
        
        moves.sort(key=lambda m: m[2], reverse=True)
        best_move = moves[0]
        mtype, mdata, mdelta = best_move
        
        # Apply move
        if mtype == 'add':
            x = mdata
            selected[x] = True
            current_score += mdelta
            tabu_list[x] = iter_count + tabu_tenure
            unsel_list.remove(x)
            sel_list.append(x)
        elif mtype == 'remove':
            x = mdata
            selected[x] = False
            current_score += mdelta
            tabu_list[x] = iter_count + tabu_tenure
            sel_list.remove(x)
            unsel_list.append(x)
        elif mtype == 'swap':
            x_out, x_in = mdata
            selected[x_out] = False
            selected[x_in] = True
            current_score += mdelta
            tabu_list[x_out] = iter_count + tabu_tenure
            tabu_list[x_in] = iter_count + tabu_tenure
            sel_list.remove(x_out)
            sel_list.append(x_in)
            unsel_list.remove(x_in)
            unsel_list.append(x_out)
        
        # Update best
        if current_score > best_score:
            best_score = current_score
            best_selected = selected.copy()
            no_improve = 0
        else:
            no_improve += 1
            
    return best_selected, best_score

def _sa_polish(selected, current_score, deadline, rng):
    """Simulated Annealing polish to escape local minima."""
    temperature = 2.0
    cooling = 0.999
    min_temp = 0.1
    best_selected = selected.copy()
    best_score = current_score
    
    sel_list = [x for x in range(1, N+1) if selected[x]]
    unsel_list = [x for x in range(1, N+1) if not selected[x]]
    
    while time.time() < deadline and temperature > min_temp:
        move_type = rng.random()
        
        if move_type < 0.4:
            if not unsel_list: continue
            x = rng.choice(unsel_list)
            if not _has_4ap_fast(selected, x):
                delta = _count_3aps_delta(selected, x)
                if delta >= 0 or rng.random() < np.exp(delta / temperature):
                    selected[x] = True
                    current_score += delta
                    unsel_list.remove(x)
                    sel_list.append(x)
                    if current_score > best_score:
                        best_score = current_score
                        best_selected = selected.copy()
        elif move_type < 0.8:
            if not sel_list: continue
            x = rng.choice(sel_list)
            delta = -_count_3aps_delta(selected, x)
            if delta >= 0 or rng.random() < np.exp(delta / temperature):
                selected[x] = False
                current_score += delta
                sel_list.remove(x)
                unsel_list.append(x)
                if current_score > best_score:
                    best_score = current_score
                    best_selected = selected.copy()
        else:
            if not sel_list or not unsel_list: continue
            x_out = rng.choice(sel_list)
            x_in = rng.choice(unsel_list)
            selected[x_out] = False
            if not _has_4ap_fast(selected, x_in):
                delta_out = -_count_3aps_delta(selected, x_out)
                delta_in = _count_3aps_delta(selected, x_in)
                total_delta = delta_out + delta_in
                if total_delta >= 0 or rng.random() < np.exp(total_delta / temperature):
                    selected[x_in] = True
                    current_score += total_delta
                    sel_list.remove(x_out)
                    sel_list.append(x_in)
                    unsel_list.remove(x_in)
                    unsel_list.append(x_out)
                    if current_score > best_score:
                        best_score = current_score
                        best_selected = selected.copy()
                else:
                    selected[x_out] = True
            else:
                selected[x_out] = True
                
        temperature *= cooling
        
    return best_selected, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin
    
    rng = random.Random(42)
    
    best_selected = None
    best_score = -1
    
    # Phase 1: Multiple Weighted Greedy Initializations + Tabu
    n_restarts = 15
    time_per_restart = (deadline - start_time) / n_restarts
    
    for restart in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        # 1. Weighted Greedy Construction
        selected = _greedy_weighted_construct(rng)
        current_score = _count_3aps_fast(selected)
        
        # 2. Tabu Search
        tabu_deadline = min(deadline - 10, time.time() + time_per_restart * 0.7)
        selected, current_score = _tabu_search(selected, current_score, tabu_deadline, rng)
        
        # 3. Update Global Best
        if current_score > best_score:
            best_score = current_score
            best_selected = selected.copy()
            
    # Phase 2: SA Polish on the best found so far
    if best_selected is not None and time.time() < deadline - 5:
        polish_deadline = deadline - 2
        best_selected, best_score = _sa_polish(best_selected, best_score, polish_deadline, rng)
        
    # Convert to sorted list
    result = [x for x in range(1, N+1) if best_selected[x]]
    return sorted(result)
# EVOLVE-BLOCK-END
