import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 90s limit
    
    n = 12
    m = 40
    full_mask = (1 << n) - 1
    num_masks = 1 << n  # 4096
    
    # Precompute popcount
    popcount = [0] * num_masks
    for mask in range(num_masks):
        popcount[mask] = bin(mask).count('1')
    
    # Precompute comparability: comp[a][b] = 1 if a is strict subset of b or vice versa
    comp = [0] * (num_masks * num_masks)
    for a in range(num_masks):
        for b in range(num_masks):
            if a != b and ((a & b) == a or (a & b) == b):
                comp[a * num_masks + b] = 1
    
    # Precompute for each mask: number of comparable masks among all 4096
    # This gives an upper bound on how many comparables a new mask can add
    comparable_count = [0] * num_masks
    for a in range(num_masks):
        base = a * num_masks
        for b in range(num_masks):
            comparable_count[a] += comp[base + b]
    
    # Precompute subset relationships for branch-and-bound pruning
    # For a given mask, count how many of the remaining candidate masks are comparable to it
    # We'll compute this on-the-fly during B&B
    
    def calc_score(masks):
        s = 0
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                if comp[masks[i] * num_masks + masks[j]]:
                    s += 1
        return s
    
    # 1. Greedy Initialization
    degrees = [0] * num_masks
    current_masks = []
    used_masks = set()
    
    # Start with a middle-layer mask
    seed_mask = 0
    for mask in range(num_masks):
        if popcount[mask] == 6:
            seed_mask = mask
            break
            
    current_masks.append(seed_mask)
    used_masks.add(seed_mask)
    
    base = seed_mask * num_masks
    for k in range(num_masks):
        if comp[base + k]:
            degrees[k] += 1
            
    for _ in range(m - 1):
        best_mask = -1
        best_degree = -1
        best_pop = -1
        
        for mask in range(num_masks):
            if mask in used_masks:
                continue
            deg = degrees[mask]
            if deg > best_degree or (deg == best_degree and popcount[mask] > best_pop):
                best_degree = deg
                best_pop = popcount[mask]
                best_mask = mask
        
        if best_mask == -1:
            best_mask = random.randint(0, full_mask)
            while best_mask in used_masks:
                best_mask = random.randint(0, full_mask)
        
        current_masks.append(best_mask)
        used_masks.add(best_mask)
        
        base = best_mask * num_masks
        for k in range(num_masks):
            if comp[base + k]:
                degrees[k] += 1
                
    best_score = calc_score(current_masks)
    best_solution = list(current_masks)
    
    # 2. Branch-and-Bound with lattice-based pruning
    # Key insight: For a set S of selected masks, the maximum additional comparable pairs
    # we can get by adding mask x is bounded by:
    #   (a) Number of masks in S comparable to x (exact)
    #   (b) For each unselected candidate y comparable to x, at most 1 new pair per y
    # So upper bound = current_pairs + sum over remaining slots of max possible pairs
    
    # Better bound: sort candidates by their "potential" (comparables to existing set)
    # and take the top (m - |S|) to get an optimistic bound
    
    def branch_and_bound(initial_solution, time_limit):
        """Branch-and-bound using lattice structure for pruning."""
        best_bnb_score = calc_score(initial_solution)
        best_bnb_solution = list(initial_solution)
        
        # Start from initial solution
        selected = list(initial_solution)
        used = set(selected)
        current_pairs = best_bnb_score
        
        # Precompute for each mask: which other masks are comparable
        # comp_neighbors[mask] = list of masks comparable to mask
        comp_neighbors = [[] for _ in range(num_masks)]
        for a in range(num_masks):
            base = a * num_masks
            for b in range(num_masks):
                if comp[base + b]:
                    comp_neighbors[a].append(b)
        
        # For B&B: try replacing one selected mask with a better one
        # Use the lattice to compute tight bounds
        
        # Strategy: iterative improvement with B&B-style pruning
        # For each position, find the best replacement considering all interactions
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            for idx in range(m):
                if time.time() > time_limit:
                    break
                
                old_mask = selected[idx]
                old_base = old_mask * num_masks
                
                # Compute contribution of old_mask to total pairs
                old_contrib = 0
                for j in range(m):
                    if j != idx:
                        old_contrib += comp[old_base + selected[j]]
                
                # Try all possible replacements
                best_new_mask = -1
                best_delta = 0
                
                # Sort candidates by potential (comparables to current set minus idx)
                # This gives us a good ordering for B&B
                candidates = []
                for mask in range(num_masks):
                    if mask in used:
                        continue
                    # Quick upper bound: max comparables this mask could have
                    # to the other (m-1) selected masks
                    potential = 0
                    new_base = mask * num_masks
                    for j in range(m):
                        if j != idx:
                            potential += comp[new_base + selected[j]]
                    
                    # Upper bound on delta = potential - old_contrib
                    # If potential - old_contrib <= best_delta, skip
                    if potential - old_contrib > best_delta:
                        candidates.append((potential, mask))
                
                # Sort by potential descending (optimistic)
                candidates.sort(key=lambda x: -x[0])
                
                # Branch and bound: process in order, prune when bound can't improve
                for potential, mask in candidates:
                    if time.time() > time_limit:
                        break
                    
                    # Pruning: if potential - old_contrib <= best_delta, no point
                    if potential - old_contrib <= best_delta:
                        break  # All subsequent have even lower potential
                    
                    # Exact delta calculation
                    new_base = mask * num_masks
                    delta = 0
                    for j in range(m):
                        if j != idx:
                            delta += comp[new_base + selected[j]]
                    delta -= old_contrib
                    
                    if delta > best_delta:
                        best_delta = delta
                        best_new_mask = mask
                
                if best_new_mask != -1 and best_delta > 0:
                    used.remove(old_mask)
                    used.add(best_new_mask)
                    selected[idx] = best_new_mask
                    current_pairs += best_delta
                    improved = True
                    
                    if current_pairs > best_bnb_score:
                        best_bnb_score = current_pairs
                        best_bnb_solution = list(selected)
        
        return best_bnb_solution, best_bnb_score
    
    # Run B&B
    bnb_solution, bnb_score = branch_and_bound(best_solution, deadline - 5.0)
    if bnb_score > best_score:
        best_score = bnb_score
        best_solution = bnb_solution
    
    # 3. Simulated Annealing for further refinement
    random.seed(42)
    
    restarts = 0
    max_restarts = 20
    
    while time.time() < deadline and restarts < max_restarts:
        current_masks = list(best_solution)
        used_masks = set(current_masks)
        current_score = best_score
        
        T = 6.0
        cooling_rate = 0.99998
        min_T = 0.0005
        no_improve_steps = 0
        max_no_improve = 8000
        
        while time.time() < deadline and T > min_T:
            idx = random.randint(0, m - 1)
            old_mask = current_masks[idx]
            
            r = random.random()
            if r < 0.7:
                # Flip one bit
                if old_mask == 0:
                    bit = random.randint(0, n - 1)
                    new_mask = old_mask | (1 << bit)
                elif old_mask == full_mask:
                    bit = random.randint(0, n - 1)
                    new_mask = old_mask ^ (1 << bit)
                else:
                    if random.random() < 0.5:
                        bits = [i for i in range(n) if (old_mask >> i) & 1]
                        bit = random.choice(bits)
                        new_mask = old_mask ^ (1 << bit)
                    else:
                        bits = [i for i in range(n) if not ((old_mask >> i) & 1)]
                        bit = random.choice(bits)
                        new_mask = old_mask | (1 << bit)
            elif r < 0.9:
                # Flip two bits
                if old_mask == 0:
                    b1 = random.randint(0, n - 1)
                    b2 = random.randint(0, n - 1)
                    while b2 == b1:
                        b2 = random.randint(0, n - 1)
                    new_mask = (1 << b1) | (1 << b2)
                elif old_mask == full_mask:
                    b1 = random.randint(0, n - 1)
                    b2 = random.randint(0, n - 1)
                    while b2 == b1:
                        b2 = random.randint(0, n - 1)
                    new_mask = full_mask ^ (1 << b1) ^ (1 << b2)
                else:
                    all_bits = list(range(n))
                    random.shuffle(all_bits)
                    b1 = all_bits[0]
                    b2 = all_bits[1]
                    new_mask = old_mask ^ (1 << b1) ^ (1 << b2)
            else:
                # Random mask
                new_mask = random.randint(0, full_mask)
            
            if new_mask == old_mask or new_mask in used_masks:
                continue

            delta = 0
            old_base = old_mask * num_masks
            new_base = new_mask * num_masks
            for j in range(m):
                if j == idx:
                    continue
                cm = current_masks[j]
                delta += comp[new_base + cm] - comp[old_base + cm]
            
            if delta >= 0 or random.random() < math.exp(delta / T):
                used_masks.remove(old_mask)
                used_masks.add(new_mask)
                current_masks[idx] = new_mask
                current_score += delta
                if delta > 0:
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
                
                if current_score > best_score:
                    best_score = current_score
                    best_solution = list(current_masks)
            else:
                no_improve_steps += 1
            
            T *= cooling_rate
            
            if no_improve_steps > max_no_improve:
                break
        
        restarts += 1
    
    # Final polish: local search with only positive delta moves
    current_masks = list(best_solution)
    used_masks = set(current_masks)
    current_score = best_score
    
    for _ in range(3000):
        if time.time() > deadline:
            break
        improved = False
        for idx in range(m):
            if time.time() > deadline:
                break
            old_mask = current_masks[idx]
            best_new_mask = -1
            best_delta = 0
            old_base = old_mask * num_masks
            
            for mask in range(num_masks):
                if mask == old_mask or mask in used_masks:
                    continue
                new_base = mask * num_masks
                delta = 0
                for j in range(m):
                    if j == idx:
                        continue
                    cm = current_masks[j]
                    delta += comp[new_base + cm] - comp[old_base + cm]
                if delta > best_delta:
                    best_delta = delta
                    best_new_mask = mask
            
            if best_new_mask != -1 and best_delta > 0:
                used_masks.remove(old_mask)
                used_masks.add(best_new_mask)
                current_masks[idx] = best_new_mask
                current_score += best_delta
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_solution = list(current_masks)
        
        if not improved:
            break
    
    # Convert to list of lists
    result = []
    for mask in best_solution:
        s = []
        for i in range(n):
            if (mask >> i) & 1:
                s.append(i)
        result.append(s)
        
    return result
