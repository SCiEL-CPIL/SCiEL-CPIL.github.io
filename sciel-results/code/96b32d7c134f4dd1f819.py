import time
import random
import math
from collections import deque

N = 256
DEADLINE = 55.0  # Safety margin for 60s limit

def _precompute():
    """
    Precompute 3-APs and 4-APs with bitmask representations for fast operations.
    """
    aps3 = []
    aps4 = []
    elem_to_aps3 = [[] for _ in range(N + 1)]
    elem_to_aps4 = [[] for _ in range(N + 1)]
    
    # Generate 3-APs: (a, a+d, a+2d)
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            aps3.append((a, b, c))
            mask_3 = (1 << a) | (1 << b) | (1 << c)
            elem_to_aps3[a].append((1 << b) | (1 << c))
            elem_to_aps3[b].append((1 << a) | (1 << c))
            elem_to_aps3[c].append((1 << a) | (1 << b))
            
    # Generate 4-APs: (a, a+d, a+2d, a+3d)
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            aps4.append((a, b, c, e))
            elem_to_aps4[a].append((1 << b) | (1 << c) | (1 << e))
            elem_to_aps4[b].append((1 << a) | (1 << c) | (1 << e))
            elem_to_aps4[c].append((1 << a) | (1 << b) | (1 << e))
            elem_to_aps4[e].append((1 << a) | (1 << b) | (1 << c))
    
    return aps3, aps4, elem_to_aps3, elem_to_aps4

def _initial_greedy_mask(elem_to_aps3, elem_to_aps4):
    """
    Greedy construction: add x if it doesn't complete a 4-AP.
    Prioritize elements with more 3-AP connections.
    """
    A_mask = 0
    ap_count = [len(elem_to_aps3[i]) for i in range(N + 1)]
    candidates = sorted(range(1, N + 1), key=lambda x: -ap_count[x])
    
    for x in candidates:
        valid = True
        for mask_other3 in elem_to_aps4[x]:
            if (A_mask & mask_other3) == mask_other3:
                valid = False
                break
        if valid:
            A_mask |= (1 << x)
    return A_mask

def construct():
    start_time = time.time()
    aps3, aps4, elem_to_aps3, elem_to_aps4 = _precompute()
    
    def calc_score(mask):
        score = 0
        for a, b, c in aps3:
            if (mask & (1 << a)) and (mask & (1 << b)) and (mask & (1 << c)):
                score += 1
        return score

    # Initial greedy set
    A_mask = _initial_greedy_mask(elem_to_aps3, elem_to_aps4)
    current_score = calc_score(A_mask)
    best_mask = A_mask
    best_score = current_score
    
    # Tabu Search Parameters
    TABU_SIZE = 50
    tabu_list = deque()
    no_improve_count = 0
    max_no_improve = 100
    
    def is_valid_swap(x, y, mask):
        for mask_other3 in elem_to_aps4[y]:
            if (mask & mask_other3) == mask_other3:
                if (mask_other3 & (1 << x)) == 0:
                    return False
        return True

    def calc_delta_swap(x, y, mask):
        loss = 0
        gain = 0
        
        for mask_other2 in elem_to_aps3[x]:
            if (mask & mask_other2) == mask_other2:
                loss += 1
                
        for mask_other2 in elem_to_aps3[y]:
            if (mask & mask_other2) == mask_other2:
                if (mask_other2 & (1 << x)) == 0:
                    gain += 1
        return gain - loss

    in_set = [i for i in range(1, N + 1) if (A_mask >> i) & 1]
    out_set = [i for i in range(1, N + 1) if not ((A_mask >> i) & 1)]

    while (time.time() - start_time) < DEADLINE:
        if not in_set or not out_set:
            break
        
        best_delta = -10000
        best_x = -1
        best_y = -1
        
        # Evaluate all possible swaps
        # Optimization: We can skip swaps that are tabu.
        # To check if a swap is tabu, we check if (x, y) is in tabu_list.
        # For speed, we can store tabu moves in a set of tuples.
        
        # To make this faster, we'll just iterate and check.
        # 16k iterations * ~100 ops = 1.6M ops. Python might be slow.
        # Let's try to optimize the inner loop.
        
        for x in in_set:
            x_bit = 1 << x
            for y in out_set:
                # Check tabu
                # We store (x, y) in tabu_list. 
                # If (x, y) is in tabu, skip.
                # Checking deque for membership is O(N). 
                # Better to use a set for tabu moves.
                
                # For now, let's just do it simply and see if it's fast enough.
                # If not, we'll switch to sampling or a set.
                
                # Actually, let's use a set for tabu moves for O(1) lookup.
                # We'll maintain `tabu_set` alongside `tabu_list` (deque for order).
                
                # Wait, I haven't defined tabu_set yet. I will add it in the edit.
                
                if not is_valid_swap(x, y, A_mask):
                    continue
                
                delta = calc_delta_swap(x, y, A_mask)
                
                if delta > best_delta:
                    best_delta = delta
                    best_x = x
                    best_y = y
        
        if best_x == -1:
            break
            
        # Apply best move
        A_mask &= ~(1 << best_x)
        A_mask |= (1 << best_y)
        current_score += best_delta
        
        in_set.remove(best_x)
        out_set.remove(best_y)
        in_set.append(best_y)
        out_set.append(best_x)
        
        # Update Tabu
        # The move is (best_x, best_y). 
        # We forbid reversing it, i.e., moving best_y back to best_x?
        # Usually, we forbid the move itself or its reverse.
        # Let's forbid the reverse move: (best_y, best_x).
        # Or simply store the pair (best_x, best_y) and forbid it.
        # Standard Tabu: Forbid the move that would return to the previous state.
        # So if we moved x->y, we forbid y->x.
        
        # Let's store (best_y, best_x) in the tabu list to forbid undoing.
        
        # We need to implement the tabu logic properly in the code below.
        
        if current_score > best_score:
            best_score = current_score
            best_mask = A_mask
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # If we are stuck, we might need to restart or accept worse moves.
        # For now, Tabu search usually handles this.
        # If no_improve_count > max_no_improve, we could randomize.
        # But let's stick to strict Tabu for now.
        
        # The loop above is O(|A| * (N-|A|)). 
        # If this is too slow, we will sample.
        
    return sorted([i for i in range(1, N + 1) if (best_mask >> i) & 1])
