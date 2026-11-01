import time
import numpy as np
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0  # 2s margin
    
    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)
    
    # Fallback
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n
    ball_size = 1 + 2 * n
    
    # Precompute balls as indices
    # weights[i] is the multiplier for position i
    weights = [3 ** (n - 1 - i) for i in range(n)]
    
    balls = np.zeros((total, ball_size), dtype=np.int32)
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                balls[w, idx] = new_w
                idx += 1
    
    # 1. Fast Greedy Initialization using incremental uncovered-neighbor count
    cov = np.zeros(total, dtype=np.int32)
    selected = []
    
    # uncovered_neighbors[w] = number of words in ball[w] that are currently uncovered
    # Initially all words are uncovered, so uncovered_neighbors[w] = ball_size for all w
    uncovered_neighbors = np.full(total, ball_size, dtype=np.int32)
    
    while True:
        if time.time() > deadline - 30.0:
            break
        max_count = np.max(uncovered_neighbors)
        if max_count == 0:
            break
        # Pick smallest index with max count for determinism
        candidates = np.where(uncovered_neighbors == max_count)[0]
        best_idx = candidates[0]
        
        selected.append(best_idx)
        cov[balls[best_idx]] += 1
        
        # Update uncovered_neighbors: for each word v in ball[best_idx],
        # if v was previously uncovered (cov[v] was 0 before this update),
        # then for each w such that v in ball[w], decrement uncovered_neighbors[w]
        # Since ball is symmetric, w in ball[v] iff v in ball[w]
        # So for each v in ball[best_idx] with cov[v] == 1 (just became covered):
        #   for each w in ball[v]: uncovered_neighbors[w] -= 1
        for v in balls[best_idx]:
            if cov[v] == 1:  # just became covered
                uncovered_neighbors[balls[v]] -= 1
    
    # If greedy timed out, we may have uncovered words. Complete them.
    if np.any(cov == 0):
        missing = np.where(cov == 0)[0]
        for m in missing:
            selected.append(m)
            cov[balls[m]] += 1
    
    # 2. Local Search: Remove and Swap, plus Simulated Annealing
    in_code = np.zeros(total, dtype=bool)
    for c in selected:
        in_code[c] = True
        
    def add_code(c):
        cov[balls[c]] += 1
        in_code[c] = True
        selected.append(c)
        
    def remove_code(c):
        cov[balls[c]] -= 1
        in_code[c] = False
        selected.remove(c)
    
    def is_removable(c):
        return np.min(cov[balls[c]]) > 1
    
    def find_valid_swaps(c):
        """Find all words w not in code that can replace c."""
        unique_neighbors = balls[c][cov[balls[c]] == 1]
        if len(unique_neighbors) == 0:
            return []
        valid_w_set = set(balls[unique_neighbors[0]].tolist())
        for v in unique_neighbors[1:]:
            valid_w_set.intersection_update(balls[v].tolist())
        return [w for w in valid_w_set if not in_code[w]]
    
    def do_swap(c, w):
        remove_code(c)
        add_code(w)
    
    # Phase 1: Remove all redundant codewords
    changed = True
    while changed:
        changed = False
        for c in list(selected):
            if is_removable(c):
                remove_code(c)
                changed = True
    
    # Phase 2: Hill-climbing swaps
    improved = True
    while improved and time.time() < deadline - 15.0:
        improved = False
        for c in list(selected):
            if not in_code[c]:
                continue
            valid_swaps = find_valid_swaps(c)
            if not valid_swaps:
                continue
            best_w = -1
            best_score = -1
            for w in valid_swaps:
                score = np.sum(cov[balls[w]] > 1)
                if score > best_score:
                    best_score = score
                    best_w = w
            if best_w != -1:
                do_swap(c, best_w)
                improved = True
                break
    
    # Phase 3: Simulated Annealing
    # Objective: minimize number of codewords. 
    # Moves: (a) remove a redundant code, (b) swap c->w where w is valid.
    # We can also try: remove c, add w (if it doesn't break coverage) to get size -1? No, that's just swap.
    # Actually, the only way to reduce size is to remove a redundant code.
    # So SA should focus on finding sequences of swaps that lead to removable codes.
    
    # SA approach: 
    # - Temperature starts high, decreases.
    # - At each step, pick a random code c, find valid swaps.
    # - If there's a swap that makes c removable after swap, do it (size decreases by 1).
    # - Otherwise, with probability exp(-delta/T), accept a swap that doesn't change size but improves "redundancy potential".
    # - Also try removing any currently removable code.
    
    # Simpler SA: 
    # - Try to find a swap that reduces the minimum coverage over the ball of the new code.
    # - Or use a cost function: sum of min(cov[v]) for v in all balls? Too expensive.
    
    # Let's use: cost = number of codewords. 
    # A move is accepted if it reduces cost, or with probability exp(-(new_cost - old_cost)/T) if it increases cost.
    # But cost only changes when we remove a code. Swaps keep cost same.
    # So we need a secondary cost to guide swaps.
    
    # Secondary cost: total_redundancy = sum over all words v of max(0, cov[v] - 1)
    # Lower redundancy means more codes are "essential", which might make it harder to remove.
    # Actually, we want HIGH redundancy to find removable codes.
    # So we want to MINIMIZE the number of codes with low coverage.
    
    # Let's define: penalty = sum over active codes c of (min(cov[balls[c]]) - 1)
    # Lower penalty means codes are less redundant.
    # We want to find states where some code has min_cov > 1 (removable).
    
    # Simpler: just do random swaps and check if any code becomes removable.
    
    T = 10.0
    T_min = 0.1
    cooling = 0.999
    
    best_size = len(selected)
    best_selected = list(selected)
    best_cov = cov.copy()
    best_in_code = in_code.copy()
    
    sa_steps = 0
    while time.time() < deadline - 5.0 and T > T_min:
        # Try to remove a removable code
        removable = [c for c in selected if is_removable(c)]
        if removable:
            c = random.choice(removable)
            remove_code(c)
            if len(selected) < best_size:
                best_size = len(selected)
                best_selected = list(selected)
                best_cov = cov.copy()
                best_in_code = in_code.copy()
            T *= cooling
            sa_steps += 1
            continue
        
        # Pick a random code to swap
        if not selected:
            break
        c = random.choice(selected)
        valid_swaps = find_valid_swaps(c)
        if not valid_swaps:
            T *= cooling
            sa_steps += 1
            continue
        
        w = random.choice(valid_swaps)
        
        # Compute score before and after
        # Score: number of codes that are "close to removable" 
        # Let's use: score = sum over active codes of min(cov[balls[code]])
        # Higher score = more redundant = easier to remove.
        
        # Before swap:
        min_c = np.min(cov[balls[c]])
        
        # After swap: remove c, add w
        cov[balls[c]] -= 1
        cov[balls[w]] += 1
        
        min_w = np.min(cov[balls[w]])
        
        # If min_w > 1, w is removable after swap
        if min_w > 1:
            # Accept: do the swap, then remove w
            in_code[c] = False
            selected.remove(c)
            in_code[w] = True
            selected.append(w)
            
            # Now remove w since it's removable
            remove_code(w)
            
            if len(selected) < best_size:
                best_size = len(selected)
                best_selected = list(selected)
                best_cov = cov.copy()
                best_in_code = in_code.copy()
        else:
            # Decide whether to keep the swap based on temperature
            # Use a heuristic: accept if the new code covers more redundant words
            old_redundancy = np.sum(cov[balls[c]] > 1)
            new_redundancy = np.sum(cov[balls[w]] > 1)
            
            if new_redundancy > old_redundancy or random.random() < np.exp((old_redundancy - new_redundancy) / T):
                # Keep swap
                in_code[c] = False
                selected.remove(c)
                in_code[w] = True
                selected.append(w)
            else:
                # Revert
                cov[balls[c]] += 1
                cov[balls[w]] -= 1
        
        T *= cooling
        sa_steps += 1
    
    # Restore best state if we found one
    if best_size < len(selected):
        # Rebuild from best state
        selected.clear()
        cov.fill(0)
        in_code.fill(False)
        for c in best_selected:
            add_code(c)
    
    # Final cleanup: remove any remaining redundant codes
    changed = True
    while changed:
        changed = False
        for c in list(selected):
            if is_removable(c):
                remove_code(c)
                changed = True
    
    # Verify coverage
    final_cov = np.zeros(total, dtype=np.int32)
    for c in selected:
        final_cov[balls[c]] += 1
        
    if np.any(final_cov == 0):
        missing = np.where(final_cov == 0)[0]
        for m in missing:
            add_code(m)
            final_cov[balls[m]] += 1
    
    # Convert to lists of digits
    result = []
    for w in selected:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
        
    return result
