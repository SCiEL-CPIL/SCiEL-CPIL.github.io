import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 256
    
    # Precompute 4-AP tuples
    four_ap_tuples = []
    for d in range(1, (N - 1) // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            four_ap_tuples.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # Precompute 3-AP tuples
    three_ap_tuples = []
    for d in range(1, (N - 1) // 2 + 1):
        max_a = N - 2 * d
        for a in range(1, max_a + 1):
            three_ap_tuples.append((a, a + d, a + 2 * d))
    
    # Precompute conflict lists: for each x, which 4-APs contain x
    # and which other elements are in those APs
    # conflicts[x] = list of (other1, other2, other3) for each 4-AP containing x
    # Actually, we just need to know if adding x creates a 4-AP with existing elements.
    # Let's precompute for each x, the list of 4-APs it belongs to.
    x_to_4aps = {i: [] for i in range(1, N + 1)}
    for idx, (a, b, c, d) in enumerate(four_ap_tuples):
        for x in (a, b, c, d):
            x_to_4aps[x].append(idx)
    
    # Precompute contribution: for each x, the list of 3-APs it belongs to
    x_to_3aps = {i: [] for i in range(1, N + 1)}
    for idx, (a, b, c) in enumerate(three_ap_tuples):
        for x in (a, b, c):
            x_to_3aps[x].append(idx)

    def count_3aps(S):
        count = 0
        for a, b, c in three_ap_tuples:
            if a in S and b in S and c in S:
                count += 1
        return count

    def check_4ap_violation(S):
        for a, b, c, d in four_ap_tuples:
            if a in S and b in S and c in S and d in S:
                return True
        return False

    # --- Phase 1: Greedy Density Construction ---
    # Start with all elements.
    # While there is a 4-AP, find the element in that AP with the lowest "contribution"
    # (number of 3-APs it is part of, considering only currently selected elements).
    # Remove it.
    selected = set(range(1, N + 1))
    
    # To speed up, we can track current 3-AP count and 4-AP violations
    # But simple iteration is fine for N=256 if we do it smartly.
    
    # We will iterate until no 4-APs remain.
    # To avoid infinite loops or slow convergence, we limit steps.
    max_greedy_steps = 2000
    steps = 0
    
    while steps < max_greedy_steps:
        if time.time() > deadline - 10: break # Leave time for local search
        
        # Find any 4-AP violation
        violation_found = None
        for a, b, c, d in four_ap_tuples:
            if a in selected and b in selected and c in selected and d in selected:
                violation_found = (a, b, c, d)
                break
        
        if not violation_found:
            break
            
        # For this violation, find the element with the lowest marginal 3-AP contribution
        # Marginal contribution of x = number of 3-APs containing x where the other 2 are in selected
        min_contrib = float('inf')
        best_remove = None
        
        for x in violation_found:
            contrib = 0
            for ap_idx in x_to_3aps[x]:
                a, b, c = three_ap_tuples[ap_idx]
                # Check if the other two are in selected
                others = [y for y in (a, b, c) if y != x]
                if others[0] in selected and others[1] in selected:
                    contrib += 1
            
            if contrib < min_contrib:
                min_contrib = contrib
                best_remove = x
                
        if best_remove is not None:
            selected.discard(best_remove)
            steps += 1
        else:
            break

    current_score = count_3aps(selected)
    best_set = set(selected)
    best_score = current_score
    
    # --- Phase 2: Local Search (Add/Swap) ---
    # Use incremental updates for speed.
    # We maintain a set 'selected'.
    # We try to add elements or swap elements to increase 3-AP count.
    
    # Precompute for each x, the list of 3-APs it forms with pairs.
    # Actually, let's just use the precomputed lists and check membership.
    
    def get_marginal_3ap_gain(x, S):
        """Calculate how many new 3-APs are formed by adding x to S."""
        gain = 0
        for ap_idx in x_to_3aps[x]:
            a, b, c = three_ap_tuples[ap_idx]
            others = [y for y in (a, b, c) if y != x]
            if others[0] in S and others[1] in S:
                gain += 1
        return gain

    def get_marginal_3ap_loss(x, S):
        """Calculate how many 3-APs are lost by removing x from S."""
        loss = 0
        for ap_idx in x_to_3aps[x]:
            a, b, c = three_ap_tuples[ap_idx]
            others = [y for y in (a, b, c) if y != x]
            if others[0] in S and others[1] in S:
                loss += 1
        return loss

    def creates_4ap(x, S):
        """Check if adding x to S creates any 4-AP."""
        for ap_idx in x_to_4aps[x]:
            a, b, c, d = four_ap_tuples[ap_idx]
            others = [y for y in (a, b, c, d) if y != x]
            if others[0] in S and others[1] in S and others[2] in S:
                return True
        return False

    # Local Search Loop
    # We will use a simple hill-climbing with random restarts if stuck.
    # Since we started from a dense set, we are likely near a local optimum.
    # We try to add elements first (to increase size and potentially 3-APs), then swap.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding any element
        candidates = [x for x in range(1, N + 1) if x not in selected]
        random.shuffle(candidates)
        
        for x in candidates:
            if creates_4ap(x, selected):
                continue
            
            gain = get_marginal_3ap_gain(x, selected)
            if gain > 0:
                selected.add(x)
                current_score += gain
                if current_score > best_score:
                    best_score = current_score
                    best_set = set(selected)
                improved = True
                # Restart candidate loop to account for new additions enabling more
                # But for efficiency, we just break and restart the outer loop
                break
        
        if improved:
            continue
            
        # If no additions, try swaps
        # Pick a random element to remove, and see if we can replace it with one that gains more 3-APs
        # This is expensive, so we do it sparingly or with a limit.
        
        if len(selected) < 2:
            break
            
        # Try a few random swaps
        swaps_attempted = 0
        max_swaps = 100
        while swaps_attempted < max_swaps and time.time() < deadline:
            x_remove = random.choice(list(selected))
            candidates = [x for x in range(1, N + 1) if x not in selected]
            if not candidates:
                break
            x_add = random.choice(candidates)
            
            # Check if removing x_remove and adding x_add maintains 4-free
            # This requires checking 4-APs involving x_add with (selected - x_remove)
            # and 4-APs involving x_remove (which are being removed, so they don't cause new violations)
            # Wait, removing x_remove might break 4-APs, but adding x_add might create new ones.
            # We only need to check if x_add creates a 4-AP with the *remaining* set.
            
            temp_set = set(selected)
            temp_set.discard(x_remove)
            
            if creates_4ap(x_add, temp_set):
                swaps_attempted += 1
                continue
            
            # Calculate delta
            loss = get_marginal_3ap_loss(x_remove, selected)
            gain = get_marginal_3ap_gain(x_add, temp_set)
            # Note: gain is calculated on temp_set (without x_remove).
            # The net change is gain - loss.
            
            delta = gain - loss
            if delta > 0:
                # Accept swap
                selected.discard(x_remove)
                selected.add(x_add)
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_set = set(selected)
                improved = True
                break # Restart outer loop to try more additions
            
            swaps_attempted += 1

    return sorted(best_set)
