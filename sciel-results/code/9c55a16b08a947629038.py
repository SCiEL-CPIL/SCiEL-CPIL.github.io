"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    N = 256
    
    # Precompute all 3-APs and 4-APs for fast evaluation
    # 3-AP: (a, a+d, a+2d) with a+2d <= N
    # 4-AP: (a, a+d, a+2d, a+3d) with a+3d <= N
    
    # Precompute for each element x, the list of 3-APs it participates in
    # and the list of 4-APs it participates in
    # This allows O(1) delta evaluation when adding/removing an element
    
    # For 3-APs: for each x, list of (other1, other2) pairs such that {x, other1, other2} forms a 3-AP
    # For 4-APs: for each x, list of (other1, other2, other3) tuples such that {x, other1, other2, other3} forms a 4-AP
    
    # Actually, let's use a different approach:
    # Represent the set as a boolean array.
    # For 3-AP count: sum over all (a,d) of [a in S and a+d in S and a+2d in S]
    # For 4-AP check: for all (a,d), NOT [a in S and a+d in S and a+2d in S and a+3d in S]
    
    # Precompute all 3-APs as tuples (a, b, c)
    all_3aps = []
    for a in range(1, N + 1):
        for d in range(1, N - a + 1):
            b = a + d
            c = a + 2 * d
            if c <= N:
                all_3aps.append((a, b, c))
    
    # Precompute all 4-APs as tuples (a, b, c, d)
    all_4aps = []
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            all_4aps.append((a, b, c, e))
    
    # For each element x, precompute:
    # - threes[x]: list of indices of 3-APs containing x
    # - fours[x]: list of indices of 4-APs containing x
    
    threes = [[] for _ in range(N + 1)]
    for idx, (a, b, c) in enumerate(all_3aps):
        threes[a].append(idx)
        threes[b].append(idx)
        threes[c].append(idx)
    
    fours = [[] for _ in range(N + 1)]
    for idx, (a, b, c, e) in enumerate(all_4aps):
        fours[a].append(idx)
        fours[b].append(idx)
        fours[c].append(idx)
        fours[e].append(idx)
    
    # Use a bitmask-like approach with boolean arrays
    # S_present[x] = True if x is in the current set
    
    def count_3aps(S_present):
        """Count 3-APs using precomputed list."""
        count = 0
        for a, b, c in all_3aps:
            if S_present[a] and S_present[b] and S_present[c]:
                count += 1
        return count
    
    def has_4ap(S_present):
        """Check if any 4-AP is fully present."""
        for a, b, c, e in all_4aps:
            if S_present[a] and S_present[b] and S_present[c] and S_present[e]:
                return True
        return False
    
    def check_4ap_with_x(S_present, x):
        """Check if adding x to S_present would create a 4-AP."""
        for idx in fours[x]:
            a, b, c, e = all_4aps[idx]
            # Check if the other 3 elements are in S_present
            others = [a, b, c, e]
            others.remove(x)
            if all(S_present[o] for o in others):
                return True
        return False
    
    def gain_3aps_with_x(S_present, x):
        """Count how many new 3-APs would be created by adding x to S_present."""
        gain = 0
        for idx in threes[x]:
            a, b, c = all_3aps[idx]
            others = [a, b, c]
            others.remove(x)
            if all(S_present[o] for o in others):
                gain += 1
        return gain
    
    def loss_3aps_with_removal(S_present, x):
        """Count how many 3-APs would be lost by removing x from S_present."""
        loss = 0
        for idx in threes[x]:
            a, b, c = all_3aps[idx]
            others = [a, b, c]
            others.remove(x)
            if all(S_present[o] for o in others):
                loss += 1
        return loss
    
    # Greedy construction with bitmask
    def greedy_construct():
        S_present = [False] * (N + 1)
        S = set()
        remaining = list(range(1, N + 1))
        
        while remaining:
            if time.time() > deadline:
                break
            
            best_x = None
            best_gain = -1
            
            for x in remaining:
                if check_4ap_with_x(S_present, x):
                    continue
                gain = gain_3aps_with_x(S_present, x)
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x is None:
                break
            
            S_present[best_x] = True
            S.add(best_x)
            remaining.remove(best_x)
        
        return S
    
    # Deterministic steepest-ascent local search with bitmask acceleration
    def steepest_ascent(S, max_iter=5000):
        S_present = [False] * (N + 1)
        for x in S:
            S_present[x] = True
        
        current_3ap = count_3aps(S_present)
        best_S = set(S)
        best_3ap = current_3ap
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try adding each element
            for x in range(1, N + 1):
                if S_present[x]:
                    continue
                if check_4ap_with_x(S_present, x):
                    continue
                gain = gain_3aps_with_x(S_present, x)
                if gain > 0:
                    S_present[x] = True
                    S.add(x)
                    current_3ap += gain
                    if current_3ap > best_3ap:
                        best_3ap = current_3ap
                        best_S = set(S)
                    improved = True
            
            # Try removing and re-adding (swap)
            if not improved:
                # Try swaps: remove one element, add another
                for x_remove in list(S):
                    if time.time() > deadline:
                        break
                    loss = loss_3aps_with_removal(S_present, x_remove)
                    S_present[x_remove] = False
                    S.discard(x_remove)
                    current_3ap -= loss
                    
                    best_swap = None
                    best_swap_gain = -1
                    
                    for x_add in range(1, N + 1):
                        if S_present[x_add]:
                            continue
                        if check_4ap_with_x(S_present, x_add):
                            continue
                        gain = gain_3aps_with_x(S_present, x_add)
                        if gain > best_swap_gain:
                            best_swap_gain = gain
                            best_swap = x_add
                    
                    if best_swap is not None and best_swap_gain > loss:
                        S_present[best_swap] = True
                        S.add(best_swap)
                        current_3ap += best_swap_gain
                        if current_3ap > best_3ap:
                            best_3ap = current_3ap
                            best_S = set(S)
                        improved = True
                    else:
                        # Revert
                        S_present[x_remove] = True
                        S.add(x_remove)
                        current_3ap += loss
            
            if not improved:
                break
        
        return best_S, best_3ap
    
    # Run greedy
    S = greedy_construct()
    if not S:
        S = {1}
    
    # Run steepest ascent
    best_S, best_3ap = steepest_ascent(S, max_iter=3000)
    
    # Try multiple random restarts with perturbation
    for trial in range(5):
        if time.time() > deadline:
            break
        
        # Perturb the current best
        perturbed = set(best_S)
        for _ in range(random.randint(2, 5)):
            if perturbed:
                perturbed.discard(random.choice(list(perturbed)))
        
        result_S, result_3ap = steepest_ascent(perturbed, max_iter=2000)
        if result_3ap > best_3ap:
            best_S = result_S
            best_3ap = result_3ap
    
    return sorted(best_S)
# EVOLVE-BLOCK-END
