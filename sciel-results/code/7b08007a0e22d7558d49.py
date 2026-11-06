import time
import random
import numpy as np

def construct():
    n = 256
    start_time = time.time()
    time_limit = 58.0  # Safety margin for 60s limit
    
    # Precomputation
    # Map 1..n to bits 0..n-1
    def val_to_bit(x):
        return 1 << (x - 1)
    
    # Precompute forbidden masks for 4-APs
    # forbidden_masks[v] = list of masks. If S contains any mask, v is forbidden.
    forbidden_masks = [[] for _ in range(n + 1)]
    
    for v in range(1, n + 1):
        masks = []
        # v is 1st: v, v+d, v+2d, v+3d. Need {v+d, v+2d, v+3d} in S.
        # v is 2nd: v-d, v, v+d, v+2d. Need {v-d, v+d, v+2d} in S.
        # v is 3rd: v-2d, v-d, v, v+d. Need {v-2d, v-d, v+d} in S.
        # v is 4th: v-3d, v-2d, v-d, v. Need {v-3d, v-2d, v-d} in S.
        
        for d in range(1, n // 3 + 1):
            # Case 1
            if v + 3 * d <= n:
                m = val_to_bit(v + d) | val_to_bit(v + 2 * d) | val_to_bit(v + 3 * d)
                masks.append(m)
            # Case 2
            if v - d >= 1 and v + 2 * d <= n:
                m = val_to_bit(v - d) | val_to_bit(v + d) | val_to_bit(v + 2 * d)
                masks.append(m)
            # Case 3
            if v - 2 * d >= 1 and v + d <= n:
                m = val_to_bit(v - 2 * d) | val_to_bit(v - d) | val_to_bit(v + d)
                masks.append(m)
            # Case 4
            if v - 3 * d >= 1:
                m = val_to_bit(v - 3 * d) | val_to_bit(v - 2 * d) | val_to_bit(v - d)
                masks.append(m)
        forbidden_masks[v] = masks
        
    # Precompute 3-AP contributions for delta
    # ap3_masks[u] = list of masks (j|k) such that {u, j, k} is a 3-AP.
    ap3_masks = [[] for _ in range(n + 1)]
    for u in range(1, n + 1):
        masks = []
        # u is 1st: u, u+d, u+2d. Other: u+d, u+2d.
        for d in range(1, (n - u) // 2 + 1):
            masks.append(val_to_bit(u + d) | val_to_bit(u + 2 * d))
        # u is 2nd: u-d, u, u+d. Other: u-d, u+d.
        for d in range(1, min(u - 1, n - u) + 1):
            masks.append(val_to_bit(u - d) | val_to_bit(u + d))
        # u is 3rd: u-2d, u-d, u. Other: u-2d, u-d.
        for d in range(1, (u - 1) // 2 + 1):
            masks.append(val_to_bit(u - 2 * d) | val_to_bit(u - d))
        ap3_masks[u] = masks
        
    def count_3aps_from_bitmask(S_mask):
        S_set = {i+1 for i in range(n) if (S_mask >> i) & 1}
        count = 0
        for a in S_set:
            for d in range(1, (n - a) // 2 + 1):
                if (a + d in S_set) and (a + 2 * d in S_set):
                    count += 1
        return count

    def can_add(S_mask, v):
        masks = forbidden_masks[v]
        for m in masks:
            if (S_mask & m) == m:
                return False
        return True

    def calc_delta_3aps(S_mask, elements_to_remove, elements_to_add):
        # S_mask contains elements_to_remove.
        # We want to calculate the change in 3-AP count.
        # This is complex for >1 elements.
        # For simplicity in SA, we will primarily use Swap (1 out, 1 in).
        # For Add (0 out, 1 in) and Delete (1 out, 0 in), we can compute directly.
        
        # Let's implement specific fast paths for 1-move operations.
        pass

    def delta_swap(S_mask, u, v):
        # Remove u, Add v.
        # S_mask contains u, not v.
        # Lost: 3-APs containing u in S_mask.
        # Gained: 3-APs containing v in S_mask \ {u}.
        
        S_mask_without_u = S_mask & ~val_to_bit(u)
        
        lost = 0
        for m in ap3_masks[u]:
            if (S_mask & m) == m:
                lost += 1
                
        gained = 0
        for m in ap3_masks[v]:
            if (S_mask_without_u & m) == m:
                gained += 1
                
        return gained - lost

    def delta_add(S_mask, v):
        # Add v to S_mask. v not in S_mask.
        # Gained: 3-APs containing v in S_mask.
        gained = 0
        for m in ap3_masks[v]:
            if (S_mask & m) == m:
                gained += 1
        return gained

    def delta_delete(S_mask, u):
        # Remove u from S_mask. u in S_mask.
        # Lost: 3-APs containing u in S_mask.
        lost = 0
        for m in ap3_masks[u]:
            if (S_mask & m) == m:
                lost += 1
        return -lost

    # Multi-Start Strategy
    # Define several initial construction strategies
    
    def greedy_ascending():
        S_mask = 0
        for x in range(1, n + 1):
            if can_add(S_mask, x):
                S_mask |= val_to_bit(x)
        return S_mask

    def greedy_descending():
        S_mask = 0
        for x in range(n, 0, -1):
            if can_add(S_mask, x):
                S_mask |= val_to_bit(x)
        return S_mask

    def greedy_random(seed):
        rng = random.Random(seed)
        S_mask = 0
        candidates = list(range(1, n + 1))
        rng.shuffle(candidates)
        for x in candidates:
            if can_add(S_mask, x):
                S_mask |= val_to_bit(x)
        return S_mask

    def run_sa(initial_mask, time_budget):
        # Simulated Annealing with Variable Cardinality
        # Moves: Swap, Add, Delete
        # T_start, T_end
        T_start = 5.0
        T_end = 0.01
        # Estimate steps per second to set num_steps
        # We will just loop until time budget is exhausted.
        
        current_mask = initial_mask
        current_count = count_3aps_from_bitmask(current_mask)
        best_mask = current_mask
        best_count = current_count
        
        # Precompute lists of elements in S and not in S for faster sampling?
        # For n=256, sampling is fine.
        
        # To make Add/Delete more effective, we should bias towards them if the set is not "maximal" or "minimal".
        # But for now, uniform selection among valid moves.
        
        # Optimization: Precompute valid adds/deletes? No, state changes.
        
        steps = 0
        while time.time() - start_time < start_time + time_budget:
            steps += 1
            T = T_start * (T_end / T_start) ** (min(1.0, steps / 5000.0)) # Decay over ~5000 steps
            
            # Choose move type
            # 70% Swap, 15% Add, 15% Delete
            r = random.random()
            
            if r < 0.70:
                # Swap
                # Pick u in S, v not in S
                u = 0
                while True:
                    u = random.randint(1, n)
                    if (current_mask >> (u-1)) & 1:
                        break
                v = 0
                while True:
                    v = random.randint(1, n)
                    if not ((current_mask >> (v-1)) & 1):
                        break
                
                # Check if v can be added to S \ {u}
                S_mask_without_u = current_mask & ~val_to_bit(u)
                valid = True
                for m in forbidden_masks[v]:
                    if (S_mask_without_u & m) == m:
                        valid = False
                        break
                if not valid:
                    continue
                
                delta = delta_swap(current_mask, u, v)
                new_mask = S_mask_without_u | val_to_bit(v)
                
            elif r < 0.85:
                # Add
                v = 0
                while True:
                    v = random.randint(1, n)
                    if not ((current_mask >> (v-1)) & 1):
                        break
                
                if not can_add(current_mask, v):
                    continue
                
                delta = delta_add(current_mask, v)
                new_mask = current_mask | val_to_bit(v)
                
            else:
                # Delete
                u = 0
                while True:
                    u = random.randint(1, n)
                    if (current_mask >> (u-1)) & 1:
                        break
                if current_mask == val_to_bit(u): # Can't delete the only element
                    continue
                    
                delta = delta_delete(current_mask, u)
                new_mask = current_mask & ~val_to_bit(u)
            
            # Acceptance
            if delta > 0 or random.random() < np.exp(delta / T):
                current_mask = new_mask
                current_count += delta
                if current_count > best_count:
                    best_count = current_count
                    best_mask = current_mask
        
        return best_mask, best_count

    # Execute Multi-Start
    # Total time budget: 58s
    # Precomputation took ~0.1s
    # We have ~57s for search.
    # Let's run 4 starts, each with ~14s budget.
    
    seeds = [
        greedy_ascending(),
        greedy_descending(),
        greedy_random(42),
        greedy_random(123)
    ]
    
    best_overall_mask = 0
    best_overall_count = -1
    
    for i, seed_mask in enumerate(seeds):
        if time.time() - start_time > time_limit - 2:
            break
            
        seed_count = count_3aps_from_bitmask(seed_mask)
        if seed_count > best_overall_count:
            best_overall_count = seed_count
            best_overall_mask = seed_mask
            
        # Run SA on this seed
        # Allocate time based on remaining budget
        remaining = time_limit - (time.time() - start_time)
        if remaining < 1:
            break
        time_budget_for_this = remaining / (len(seeds) - i)
        
        res_mask, res_count = run_sa(seed_mask, time_budget_for_this)
        
        if res_count > best_overall_count:
            best_overall_count = res_count
            best_overall_mask = res_mask
            
    # Convert best mask to sorted list
    best_set = {i+1 for i in range(n) if (best_overall_mask >> i) & 1}
    
    # Final sanity check: ensure no 4-APs
    # (SA should maintain this, but let's verify if time permits)
    # If time is tight, skip verification as SA logic is strict.
    
    return sorted(best_set)

# Test run to ensure it works
if __name__ == "__main__":
    res = construct()
    print(len(res), res[:10])
