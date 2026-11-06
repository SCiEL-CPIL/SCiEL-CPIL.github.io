import time
import random
import math
from itertools import combinations

V = 16
ALL_TRIPLES = list(combinations(range(V), 3))
N_TRIPLES = len(ALL_TRIPLES)  # 560

# Precompute index mapping
triple_to_idx = {t: i for i, t in enumerate(ALL_TRIPLES)}
idx_to_triple = [None] * N_TRIPLES
for t, i in triple_to_idx.items():
    idx_to_triple[i] = t

# Precompute vertex sets for each triple for faster access
triple_vertices = [set(t) for t in ALL_TRIPLES]

def get_idx(a, b, c):
    """Get index of triple {a,b,c}."""
    # Sort for canonical form
    if a > b: a, b = b, a
    if b > c: b, c = c, b
    if a > b: a, b = b, a
    return triple_to_idx[(a, b, c)]

def is_present(mask, idx):
    return (mask >> idx) & 1

def can_add(mask, t_idx, present_list):
    """
    Check if adding triple t_idx to the set represented by mask is valid.
    present_list is a list of currently present triple indices for faster removal checks if needed, 
    but we mainly rely on mask.
    """
    a, b, c = idx_to_triple[t_idx]
    
    # To form a K(2,2,2) with (a,b,c), we need a', b', c' such that:
    # (a',b,c), (a,b',c), (a',b',c), (a,b,c'), (a',b,c'), (a,b',c'), (a',b',c') are all present.
    
    # Optimization: Precompute indices for the 7 companions for each possible (a', b', c')?
    # No, that's too much memory. Compute on fly.
    
    # We iterate over potential a'
    # a' cannot be a, b, c
    # We check if (a', b, c) is present. If not, this a' is invalid for this K(2,2,2) completion.
    
    # To speed this up, we can precompute for each triple (a,b,c) the list of indices 
    # for (x, b, c) for all x != a,b,c.
    # Let's create a helper structure: 
    # partners[t_idx] = list of (a_prime, idx_ap_bc)
    
    # Actually, let's just do the loops. V=16 is small.
    
    for ap in range(V):
        if ap == a or ap == b or ap == c:
            continue
        
        idx_ap_bc = get_idx(ap, b, c)
        if not is_present(mask, idx_ap_bc):
            continue
            
        for bp in range(V):
            if bp == a or bp == b or bp == c or bp == ap:
                continue
                
            idx_ab_pc = get_idx(a, bp, c)
            if not is_present(mask, idx_ab_pc):
                continue
                
            idx_ap_bp_c = get_idx(ap, bp, c)
            if not is_present(mask, idx_ap_bp_c):
                continue
                
            for cp in range(V):
                if cp == a or cp == b or cp == c or cp == ap or cp == bp:
                    continue
                    
                idx_ab_cp = get_idx(a, b, cp)
                if not is_present(mask, idx_ab_cp):
                    continue
                    
                idx_ap_b_cp = get_idx(ap, b, cp)
                if not is_present(mask, idx_ap_b_cp):
                    continue
                    
                idx_abp_cp = get_idx(a, bp, cp)
                if not is_present(mask, idx_abp_cp):
                    continue
                    
                idx_ap_bp_cp = get_idx(ap, bp, cp)
                if not is_present(mask, idx_ap_bp_cp):
                    continue
                
                # If we reach here, all 7 companions are present.
                # Adding (a,b,c) completes the K(2,2,2).
                return False
                
    return True

def greedy_construct(order, deadline):
    """Greedy construction from a given order."""
    mask = 0
    present = []
    for t_idx in order:
        if time.time() > deadline:
            break
        if is_present(mask, t_idx):
            continue
        if can_add(mask, t_idx, None):
            mask |= (1 << t_idx)
            present.append(t_idx)
    return mask, present

def simulated_annealing(initial_mask, initial_present, deadline):
    """
    Simulated Annealing to maximize the number of triples.
    """
    # State
    mask = initial_mask
    present = set(initial_present)
    absent = [i for i in range(N_TRIPLES) if i not in present]
    
    current_count = len(present)
    best_mask = mask
    best_present = set(present)
    best_count = current_count
    
    T = 10.0
    T_min = 0.01
    cooling = 0.995
    
    # Precompute list of all triples for random selection
    all_indices = list(range(N_TRIPLES))
    
    while T > T_min and time.time() < deadline:
        # Choose move type
        # Bias towards swaps to maintain density
        r = random.random()
        
        if r < 0.2:
            # Add move
            if not absent:
                T *= cooling
                continue
            t_in = random.choice(absent)
            if can_add(mask, t_in, None):
                # Always accept improvement
                mask |= (1 << t_in)
                present.add(t_in)
                absent.remove(t_in)
                current_count += 1
                if current_count > best_count:
                    best_count = current_count
                    best_mask = mask
                    best_present = set(present)
        elif r < 0.4:
            # Remove move
            if not present:
                T *= cooling
                continue
            t_out = random.choice(list(present))
            # Remove
            mask &= ~(1 << t_out)
            present.remove(t_out)
            absent.append(t_out)
            current_count -= 1
        else:
            # Swap move: Remove t_out, Add t_in
            if not present or not absent:
                T *= cooling
                continue
            t_out = random.choice(list(present))
            t_in = random.choice(absent)
            
            # Check if t_in is valid after removing t_out
            # Temporarily remove t_out
            mask_temp = mask & ~(1 << t_out)
            
            if can_add(mask_temp, t_in, None):
                # Accept swap (neutral energy change in terms of count, but explores space)
                # We accept all valid swaps to escape local optima
                mask = mask_temp | (1 << t_in)
                present.remove(t_out)
                present.add(t_in)
                absent.remove(t_in)
                absent.append(t_out)
                # Count remains same, but structure changes
                # Check if this leads to a better state in future? 
                # For now, just accept to explore.
            else:
                # Reject swap, keep state
                pass
        
        T *= cooling
        
    # Restore best
    return best_mask, best_present

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_mask = 0
    best_present = set()
    best_count = 0
    
    # Strategy: Multiple random restarts with Greedy + SA
    # The number of restarts depends on time available per restart.
    # Greedy is fast. SA takes the bulk of the time.
    
    # Initial greedy pass to get a decent baseline
    order = list(range(N_TRIPLES))
    random.shuffle(order)
    base_mask, base_present = greedy_construct(order, deadline)
    best_mask = base_mask
    best_present = set(base_present)
    best_count = len(base_present)
    
    # Run SA for the remaining time
    # We can run multiple SA instances if time permits, or one long one.
    # Given the complexity, one long SA with good cooling is usually better than many short ones.
    # But random restarts help escape global optima.
    
    # Let's do a few restarts.
    # Estimate time per SA iteration: 
    # can_add is O(V^3) worst case ~ 4096 ops. 
    # With bitmasks, it's fast. 
    # Let's try 5 restarts.
    
    time_per_restart = (deadline - time.time()) / 5
    
    for restart in range(5):
        if time.time() > deadline:
            break
        
        # New random greedy start
        random.shuffle(order)
        # Greedy for a short time to get a base
        g_deadline = time.time() + 2.0
        if g_deadline > deadline:
            g_deadline = deadline
        cur_mask, cur_present = greedy_construct(order, g_deadline)
        
        # Run SA for the allocated time
        sa_deadline = time.time() + time_per_restart
        if sa_deadline > deadline:
            sa_deadline = deadline
            
        sa_mask, sa_present = simulated_annealing(cur_mask, cur_present, sa_deadline)
        
        if len(sa_present) > best_count:
            best_count = len(sa_present)
            best_mask = sa_mask
            best_present = sa_present
    
    # Final validation
    # Convert best_present to list of triples
    result = [list(idx_to_triple[i]) for i in best_present]
    
    # Verify validity (optional but good for safety)
    # We trust the can_add logic, but a final check is cheap for 16 vertices.
    # Actually, the check in can_add is rigorous.
    
    return result

# To ensure the function is callable and handles edge cases
if __name__ == "__main__":
    # This block is for local testing, not used by evaluator
    pass
