"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    
    # Precompute sizes and masks
    masks = list(range(total_masks))
    
    def is_comparable(a, b):
        return (a & b) == a or (a & b) == b

    # 1. Initialization
    # Start with a chain to ensure high density
    # Chain: 0, 1, 3, 7, 15, ... (2^k - 1)
    initial_chain = []
    mask_val = 0
    for k in range(n + 1):
        initial_chain.append((1 << k) - 1)
        
    selected = initial_chain.copy()
    selected_set = set(selected)
    
    # Greedily add remaining sets
    # We want to maximize comparability with existing selected sets
    # Sort all masks by potential (size close to n/2 are more connected generally, but let's just score them)
    
    # To speed up greedy init, we can just pick the best available mask at each step
    # But O(M * N * M) is too slow for init if done naively.
    # Let's just pick random high-degree masks or use a simpler heuristic.
    # Actually, let's just do a quick greedy on a subset of candidates to keep init fast, then local search will fix it.
    
    # Candidate pool: All masks not in selected
    candidates = [mask for mask in masks if mask not in selected_set]
    
    # Sort candidates by popcount? No, let's just pick the best 28 greedily from the full list, but optimized.
    # We can approximate the gain.
    
    # Let's just do a simple greedy:
    # For each step, find mask in candidates with max comparable count to selected.
    # This is O(remaining * selected * N_candidates) which is 28 * 12 * 4000 ~ 1.3M. Fine.
    
    for _ in range(m - len(selected)):
        best_mask = -1
        best_score = -1
        for mask in candidates:
            score = 0
            for s in selected:
                if is_comparable(mask, s):
                    score += 1
            if score > best_score:
                best_score = score
                best_mask = mask
        
        selected.append(best_mask)
        selected_set.add(best_mask)
        candidates.remove(best_mask)

    # 2. Local Search with Optimized Updates
    # State:
    # selected: list of masks
    # selected_set: set of masks for O(1) lookup
    # unselected: list of masks
    # compat_count: array of size total_masks, compat_count[u] = number of s in selected such that is_comparable(u, s)
    
    unselected = [mask for mask in masks if mask not in selected_set]
    unselected_set = set(unselected)
    
    # Initialize compat_count
    compat_count = [0] * total_masks
    for u in range(total_masks):
        count = 0
        for s in selected:
            if is_comparable(u, s):
                count += 1
        compat_count[u] = count
        
    # Also precompute deg_in_S for each selected item
    # deg_in_S[i] = number of other selected items comparable to selected[i]
    def calculate_deg_in_S(selected):
        degs = [0] * len(selected)
        for i in range(len(selected)):
            for j in range(len(selected)):
                if i != j and is_comparable(selected[i], selected[j]):
                    degs[i] += 1
        return degs

    deg_in_S = calculate_deg_in_S(selected)
    
    improved = True
    iterations = 0
    max_iterations = 5000 # Safety limit
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        best_gain = 0
        best_i = -1
        best_u = -1
        
        # Find best swap
        # Gain of swapping selected[i] with unselected u:
        # New edges for u: compat_count[u] - (1 if is_comparable(u, selected[i]) else 0)
        # Lost edges for selected[i]: deg_in_S[i]
        # Note: deg_in_S[i] counts edges to OTHER selected items.
        # compat_count[u] counts edges to ALL selected items.
        # If u is comparable to selected[i], then in the new set (where selected[i] is removed), 
        # u will have compat_count[u] - 1 edges.
        # If u is NOT comparable to selected[i], u will have compat_count[u] edges.
        
        for i, s_i in enumerate(selected):
            lost = deg_in_S[i]
            for u in unselected:
                # Check if u is comparable to s_i
                if is_comparable(u, s_i):
                    gained = compat_count[u] - 1
                else:
                    gained = compat_count[u]
                
                gain = gained - lost
                if gain > best_gain:
                    best_gain = gain
                    best_i = i
                    best_u = u
        
        if best_gain > 0:
            # Perform swap
            s_old = selected[best_i]
            u_new = best_u
            
            # Update selected list and set
            selected[best_i] = u_new
            selected_set.remove(s_old)
            selected_set.add(u_new)
            
            # Update unselected list and set
            unselected_set.remove(u_new)
            unselected_set.add(s_old)
            # Replace in list (order doesn't matter for logic, but keep list consistent)
            unselected[unselected.index(u_new)] = s_old
            
            # Update compat_count for ALL masks
            # Remove s_old: decrement if comparable
            # Add u_new: increment if comparable
            for w in range(total_masks):
                if is_comparable(w, s_old):
                    compat_count[w] -= 1
                if is_comparable(w, u_new):
                    compat_count[w] += 1
            
            # Update deg_in_S
            # This is O(M^2) = 1600 ops. Acceptable.
            deg_in_S = calculate_deg_in_S(selected)
            
            improved = True

    # Convert to lists
    result = []
    for mask in selected:
        result.append([b for b in range(n) if (mask >> b) & 1])
        
    return result

# EVOLVE-BLOCK-END
