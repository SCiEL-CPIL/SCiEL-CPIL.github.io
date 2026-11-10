import time
import random

N = 256
DEADLINE = 85.0  # Safety margin from 90s limit

def _precompute():
    """
    Precompute 3-AP and 4-AP partner lists.
    ap3_partners[x] = list of (y, z) such that {x, y, z} is a 3-AP.
    ap4_partners[x] = list of (y, z, w) such that {x, y, z, w} is a 4-AP.
    """
    ap3_partners = [[] for _ in range(N + 1)]
    ap4_partners = [[] for _ in range(N + 1)]
    
    # 3-APs: (a, a+d, a+2d)
    # x can be a, a+d, or a+2d
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            # x=a: partners (b, c)
            ap3_partners[a].append((b, c))
            # x=b: partners (a, c)
            ap3_partners[b].append((a, c))
            # x=c: partners (a, b)
            ap3_partners[c].append((a, b))
            
    # 4-APs: (a, a+d, a+2d, a+3d)
    # x can be a, a+d, a+2d, or a+3d
    for d in range(1, (N - 1) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            # x=a: partners (b, c, e)
            ap4_partners[a].append((b, c, e))
            # x=b: partners (a, c, e)
            ap4_partners[b].append((a, c, e))
            # x=c: partners (a, b, e)
            ap4_partners[c].append((a, b, e))
            # x=e: partners (a, b, c)
            ap4_partners[e].append((a, b, c))
            
    return ap3_partners, ap4_partners

def _is_valid_addition(y, in_set, ap4_partners):
    """Check if adding y creates a 4-AP."""
    for j, k, l in ap4_partners[y]:
        if in_set[j] and in_set[k] and in_set[l]:
            return False
    return True

def _is_valid_swap(x, y, in_set, ap4_partners):
    """Check if swapping x (out) for y (in) is valid."""
    # We remove x, then add y.
    # Check if adding y to S \ {x} creates a 4-AP.
    # A 4-AP involving y might include x. If it does, x is removed, so it's not a problem.
    # If it doesn't include x, then it must be in S.
    for j, k, l in ap4_partners[y]:
        # If the partner is x, it's not in the set S \ {x}, so it doesn't form a 4-AP.
        # If the partner is not x, it must be in in_set.
        if j == x or k == x or l == x:
            continue
        if in_set[j] and in_set[k] and in_set[l]:
            return False
    return True

def _calculate_gain(y, in_set, ap3_partners, exclude_x=None):
    """Calculate the number of 3-APs formed by adding y to in_set (with optional exclusion of x)."""
    count = 0
    for j, k in ap3_partners[y]:
        # If exclude_x is set, we are effectively checking against in_set \ {exclude_x}
        j_ok = in_set[j] and (exclude_x is None or j != exclude_x)
        k_ok = in_set[k] and (exclude_x is None or k != exclude_x)
        if j_ok and k_ok:
            count += 1
    return count

def _calculate_loss(x, in_set, ap3_partners):
    """Calculate the number of 3-APs lost by removing x from in_set."""
    count = 0
    for j, k in ap3_partners[x]:
        if in_set[j] and in_set[k]:
            count += 1
    return count

def _count_3aps(in_set, ap3_partners):
    """Count total 3-APs in the set. Each 3-AP is counted 3 times (once for each element)."""
    total = 0
    for i in range(1, N + 1):
        if in_set[i]:
            total += _calculate_gain(i, in_set, ap3_partners)
    return total // 3

def _grasp_construct(ap3_partners, ap4_partners, rng):
    """Greedy Randomized Adaptive Search Procedure for initialization."""
    in_set = [False] * (N + 1)
    candidates = list(range(1, N + 1))
    
    while candidates:
        valid_candidates = []
        gains = {}
        
        for y in candidates:
            if _is_valid_addition(y, in_set, ap4_partners):
                gain = _calculate_gain(y, in_set, ap3_partners)
                valid_candidates.append(y)
                gains[y] = gain
        
        if not valid_candidates:
            break
            
        # Sort by gain descending
        valid_candidates.sort(key=lambda x: gains[x], reverse=True)
        
        # Select from top K
        K = min(5, len(valid_candidates))
        top_k = valid_candidates[:K]
        
        # Random selection from top K
        y = rng.choice(top_k)
        
        in_set[y] = True
        candidates.remove(y)
        
    return in_set

def _tabu_search(in_set, ap3_partners, ap4_partners, rng, time_limit):
    """Tabu Search to improve the set."""
    current_score = _count_3aps(in_set, ap3_partners)
    best_score = current_score
    best_set = in_set[:]
    
    tabu_list = {}  # x -> iteration when it becomes non-tabu
    tabu_tenure = 15
    iter_count = 0
    no_improve = 0
    max_no_improve = 50
    
    start_time = time.time()
    
    sel_list = [x for x in range(1, N + 1) if in_set[x]]
    unsel_list = [x for x in range(1, N + 1) if not in_set[x]]
    
    while time.time() - start_time < time_limit and no_improve < max_no_improve:
        iter_count += 1
        best_move = None
        best_delta = -10**9
        
        # 1. Add moves
        for y in unsel_list:
            if y in tabu_list and tabu_list[y] > iter_count:
                continue
            if _is_valid_addition(y, in_set, ap4_partners):
                delta = _calculate_gain(y, in_set, ap3_partners)
                if delta > best_delta:
                    best_delta = delta
                    best_move = ('add', y)
        
        # 2. Remove moves
        for x in sel_list:
            if x in tabu_list and tabu_list[x] > iter_count:
                continue
            delta = -_calculate_loss(x, in_set, ap3_partners)
            if delta > best_delta:
                best_delta = delta
                best_move = ('remove', x)
        
        # 3. Swap moves (Sampled)
        if sel_list and unsel_list:
            # Sample to keep it fast
            n_swap_sample = min(len(sel_list), len(unsel_list), 20)
            sel_sample = rng.sample(sel_list, n_swap_sample) if len(sel_list) > n_swap_sample else sel_list
            unsel_sample = rng.sample(unsel_list, n_swap_sample) if len(unsel_list) > n_swap_sample else unsel_list
            
            for x_out in sel_sample:
                if x_out in tabu_list and tabu_list[x_out] > iter_count:
                    continue
                for x_in in unsel_sample:
                    if x_in in tabu_list and tabu_list[x_in] > iter_count:
                        continue
                    if _is_valid_swap(x_out, x_in, in_set, ap4_partners):
                        # Delta = Gain(x_in, S \ {x_out}) - Loss(x_out, S)
                        # Note: Gain calculation needs to exclude x_out
                        gain_in = _calculate_gain(x_in, in_set, ap3_partners, exclude_x=x_out)
                        loss_out = _calculate_loss(x_out, in_set, ap3_partners)
                        delta = gain_in - loss_out
                        if delta > best_delta:
                            best_delta = delta
                            best_move = ('swap', x_out, x_in)
        
        if best_move is None:
            break
            
        mtype = best_move[0]
        
        # Aspiration criterion: if the move is tabu but improves global best, allow it
        # (Already handled by checking tabu before evaluating delta, but we can be more aggressive if needed)
        
        # Apply move
        if mtype == 'add':
            y = best_move[1]
            in_set[y] = True
            current_score += best_delta
            tabu_list[y] = iter_count + tabu_tenure
            unsel_list.remove(y)
            sel_list.append(y)
        elif mtype == 'remove':
            x = best_move[1]
            in_set[x] = False
            current_score += best_delta
            tabu_list[x] = iter_count + tabu_tenure
            sel_list.remove(x)
            unsel_list.append(x)
        elif mtype == 'swap':
            x_out, x_in = best_move[1], best_move[2]
            in_set[x_out] = False
            in_set[x_in] = True
            current_score += best_delta
            tabu_list[x_out] = iter_count + tabu_tenure
            tabu_list[x_in] = iter_count + tabu_tenure
            sel_list.remove(x_out)
            sel_list.append(x_in)
            unsel_list.remove(x_in)
            unsel_list.append(x_out)
            
        # Update best
        if current_score > best_score:
            best_score = current_score
            best_set = in_set[:]
            no_improve = 0
        else:
            no_improve += 1
            
    return best_set, best_score

def construct():
    start_time = time.time()
    ap3_partners, ap4_partners = _precompute()
    
    best_score = -1
    best_set = [False] * (N + 1)
    
    rng = random.Random(42)
    
    # Run multiple GRASP + Tabu iterations
    # We have ~85 seconds. Let's do iterations until time runs out.
    # Each GRASP is fast. Tabu search takes the bulk of the time.
    # Let's allocate ~5 seconds per iteration for Tabu, and see how many fit.
    
    while time.time() - start_time < DEADLINE - 2.0:
        # 1. GRASP Initialization
        in_set = _grasp_construct(ap3_partners, ap4_partners, rng)
        
        # 2. Tabu Search
        # Allocate remaining time minus a safety buffer, but cap it for the first few runs
        # to allow more restarts if early runs are fast or bad.
        time_elapsed = time.time() - start_time
        time_remaining = DEADLINE - 2.0 - time_elapsed
        # Cap tabu time for first few runs to explore more diverse solutions
        tabu_time_limit = min(5.0, time_remaining)
        
        improved_set, score = _tabu_search(in_set, ap3_partners, ap4_partners, rng, tabu_time_limit)
        
        if score > best_score:
            best_score = score
            best_set = improved_set[:]
            
        # Check if we are running out of time
        if time.time() - start_time > DEADLINE - 10.0:
            break
            
    # Final polish with remaining time?
    # If there is significant time left, do a final SA or Tabu on the best found.
    time_left = DEADLINE - 2.0 - (time.time() - start_time)
    if time_left > 5.0 and best_score > 0:
        improved_set, score = _tabu_search(best_set, ap3_partners, ap4_partners, rng, time_left)
        if score > best_score:
            best_score = score
            best_set = improved_set[:]
            
    result = [x for x in range(1, N + 1) if best_set[x]]
    return sorted(result)
