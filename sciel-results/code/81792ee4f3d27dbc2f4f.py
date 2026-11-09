import time
import numpy as np
import random

N = 17
TOTAL_MASKS = 1 << N  # 131072

# EVOLVE-BLOCK-START

def _build_seed():
    """Odd-even transposition network for N wires."""
    net = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            net.append((i, i + 1))
    return net

def _run_network(net, masks):
    """
    Run network `net` on array of masks.
    masks: np.array of int32, shape (M,)
    Returns sorted masks.
    """
    x = masks.astype(np.int32).copy()
    for i, j in net:
        a = (x >> i) & 1
        b = (x >> j) & 1
        # If a > b (a=1, b=0), swap
        swap = a & ~b
        # Clear bit i if swap, set bit j if swap
        x = (x & ~(swap << i)) | (swap << j)
    return x

def _count_unsorted(net, masks, sorted_masks):
    """
    Count how many masks are not sorted.
    """
    result = _run_network(net, masks)
    return int(np.sum(result != sorted_masks))

def construct():
    deadline = time.time() + 85  # Safety margin
    masks = np.arange(TOTAL_MASKS, dtype=np.int32)
    
    # Precompute sorted versions
    sorted_masks = np.zeros(TOTAL_MASKS, dtype=np.int32)
    for m in range(TOTAL_MASKS):
        k = bin(m).count('1')
        if k == 0:
            sorted_masks[m] = 0
        elif k == N:
            sorted_masks[m] = (1 << N) - 1
        else:
            # k ones at the top (higher indices)
            # N-k zeros at the bottom (lower indices)
            sorted_masks[m] = ((1 << k) - 1) << (N - k)
            
    # Initial network
    best_net = _build_seed()
    best_count = _count_unsorted(best_net, masks, sorted_masks)
    if best_count != 0:
        # Fallback if seed is broken
        return best_net
        
    current_net = best_net[:]
    current_count = len(current_net)
    best_score = current_count
    
    # Precompute all possible comparators
    all_comps = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # Search parameters
    temp = 1.0
    cooling = 0.9995
    min_temp = 0.001
    no_improve_time = time.time()
    restart_threshold = 10.0 # seconds
    
    while time.time() < deadline:
        # 1. Try to remove a comparator (Greedy)
        improved = False
        indices = list(range(len(current_net)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline:
                break
            trial_net = current_net[:idx] + current_net[idx+1:]
            cnt = _count_unsorted(trial_net, masks, sorted_masks)
            if cnt == 0:
                # Valid removal
                current_net = trial_net
                current_count -= 1
                if current_count < best_score:
                    best_score = current_count
                    best_net = current_net[:]
                    no_improve_time = time.time()
                improved = True
                break # Restart removal scan
        
        if improved:
            temp = max(min_temp, temp * cooling)
            continue
        
        # 2. If no removal, try a "kick" (lateral move)
        # Try to swap two comparators
        if len(current_net) >= 2:
            i1 = random.randint(0, len(current_net) - 1)
            i2 = random.randint(0, len(current_net) - 1)
            if i1 != i2:
                trial_swap = current_net[:]
                trial_swap[i1], trial_swap[i2] = trial_swap[i2], trial_swap[i1]
                cnt = _count_unsorted(trial_swap, masks, sorted_masks)
                if cnt == 0:
                    current_net = trial_swap
                    temp = max(min_temp, temp * cooling)
                    # Lateral move, doesn't improve score, but explores space
                    continue
        
        # Try to insert and then remove (net 0 change, or -1 if lucky)
        # Insert random comp
        comp_to_add = random.choice(all_comps)
        ins_pos = random.randint(0, len(current_net))
        trial_net = current_net[:ins_pos] + [comp_to_add] + current_net[ins_pos:]
        
        # Try to remove 1 comp from trial_net
        # This results in same size as current_net
        # If we remove the inserted one, we revert.
        # If we remove another, we move laterally.
        
        # To increase chance of finding a *useful* removal, 
        # we can try to remove the one we just added first? No, that's revert.
        # We want to remove a DIFFERENT one.
        
        # Let's try to remove a random comp from trial_net
        # But we must ensure the result is valid.
        
        # Optimization: Only try a few random removals from trial_net
        moves_tried = 0
        max_moves = 5
        while moves_tried < max_moves:
            if time.time() > deadline:
                break
            idx = random.randint(0, len(trial_net) - 1)
            # Avoid removing the inserted one if possible? 
            # Actually, removing the inserted one is a null move.
            # We want a non-null move.
            if idx == ins_pos:
                moves_tried += 1
                continue
                
            sub_trial = trial_net[:idx] + trial_net[idx+1:]
            cnt = _count_unsorted(sub_trial, masks, sorted_masks)
            if cnt == 0:
                # Valid lateral move
                current_net = sub_trial
                temp = max(min_temp, temp * cooling)
                break
            moves_tried += 1
            
        # If no move found, cool down
        temp = max(min_temp, temp * 0.99)
        
        # Check for restart
        if time.time() - no_improve_time > restart_threshold:
            # Restart from best_net with perturbation
            current_net = best_net[:]
            # Perturb: swap a few
            for _ in range(5):
                if len(current_net) >= 2:
                    i1 = random.randint(0, len(current_net) - 1)
                    i2 = random.randint(0, len(current_net) - 1)
                    if i1 != i2:
                        current_net[i1], current_net[i2] = current_net[i2], current_net[i1]
            # Verify perturbed net
            if _count_unsorted(current_net, masks, sorted_masks) != 0:
                current_net = best_net[:]
            no_improve_time = time.time()
            
    return best_net

# EVOLVE-BLOCK-END
