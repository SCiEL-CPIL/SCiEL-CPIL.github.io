import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 12
    m = 40
    full_mask = (1 << n) - 1
    num_masks = 1 << n  # 4096
    
    # Precompute comparability bitsets
    comp_bits = [0] * num_masks
    
    for a in range(num_masks):
        bits = 0
        if a != 0:
            s = (a - 1) & a
            while True:
                bits |= (1 << s)
                if s == 0:
                    break
                s = (s - 1) & a
        free = (~a) & full_mask
        if free != 0:
            s = (free - 1) & free
            while True:
                sup = a | s
                bits |= (1 << sup)
                if s == 0:
                    break
                s = (s - 1) & free
        comp_bits[a] = bits
    
    def compute_score(masks):
        """Compute the number of comparable pairs."""
        score = 0
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                if (comp_bits[masks[i]] >> masks[j]) & 1:
                    score += 1
        return score
    
    def greedy_init():
        # Start with a chain: 0, 1, 3, 7, ..., full_mask
        chain = [0]
        cur = 0
        for i in range(n):
            cur |= (1 << i)
            chain.append(cur)
        selected = list(chain)
        selected_set = set(selected)
        selected_bits = 0
        for s in selected:
            selected_bits |= (1 << s)
        
        for _ in range(m - 13):
            best_gain = -1
            best_mask = -1
            for candidate in range(num_masks):
                if candidate in selected_set:
                    continue
                gain = (comp_bits[candidate] & selected_bits).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_mask = candidate
            if best_mask == -1:
                break
            selected.append(best_mask)
            selected_set.add(best_mask)
            selected_bits |= (1 << best_mask)
        return selected
    
    def nested_chain_init(seed):
        """Create multiple nested chains for better structure."""
        random.seed(seed)
        selected = []
        selected_set = set()
        
        # Create several chains of different "directions"
        for chain_idx in range(8):
            cur = 0
            selected.append(cur)
            selected_set.add(cur)
            remaining_bits = list(range(n))
            random.shuffle(remaining_bits)
            for bit in remaining_bits:
                cur |= (1 << bit)
                if len(selected) >= m:
                    break
                if cur not in selected_set:
                    selected.append(cur)
                    selected_set.add(cur)
            if len(selected) >= m:
                break
        
        if len(selected) < m:
            selected_bits = 0
            for s in selected:
                selected_bits |= (1 << s)
            for candidate in range(num_masks):
                if len(selected) >= m:
                    break
                if candidate in selected_set:
                    continue
                gain = (comp_bits[candidate] & selected_bits).bit_count()
                if gain > 0:
                    selected.append(candidate)
                    selected_set.add(candidate)
                    selected_bits |= (1 << candidate)
            while len(selected) < m:
                candidate = random.randint(0, full_mask)
                if candidate not in selected_set:
                    selected.append(candidate)
                    selected_set.add(candidate)
        
        return selected[:m]
    
    def local_search(initial_masks, max_steps=200000, stuck_limit=200, use_pair_swaps=False):
        current = list(initial_masks)
        current_set = set(current)
        selected_bits = 0
        for s in current:
            selected_bits |= (1 << s)
        
        cur_score = compute_score(current)
        best_score = cur_score
        best_solution = list(current)
        stuck = 0
        
        for step in range(max_steps):
            if time.time() > deadline - 5.0:
                break
            
            if use_pair_swaps and step % 3 == 0:
                # Try a pair swap
                i = random.randint(0, m - 1)
                j = random.randint(0, m - 1)
                if i == j:
                    continue
                old_i = current[i]
                old_j = current[j]
                others = selected_bits & ~(1 << old_i) & ~(1 << old_j)
                
                new_i = random.randint(0, full_mask)
                new_j = random.randint(0, full_mask)
                if new_i == old_i or new_j == old_j or new_i == new_j:
                    continue
                if new_i in current_set or new_j in current_set:
                    continue
                
                gain_i = (comp_bits[new_i] & others).bit_count()
                gain_j = (comp_bits[new_j] & others).bit_count()
                gain_ij = 1 if ((comp_bits[new_i] >> new_j) & 1) or ((comp_bits[new_j] >> new_i) & 1) else 0
                
                loss_i = (comp_bits[old_i] & others).bit_count()
                loss_j = (comp_bits[old_j] & others).bit_count()
                loss_ij = 1 if ((comp_bits[old_i] >> old_j) & 1) or ((comp_bits[old_j] >> old_i) & 1) else 0
                
                delta = gain_i + gain_j + gain_ij - loss_i - loss_j - loss_ij
                
                if delta > 0:
                    current_set.remove(old_i)
                    current_set.remove(old_j)
                    current_set.add(new_i)
                    current_set.add(new_j)
                    selected_bits = (selected_bits & ~(1 << old_i) & ~(1 << old_j)) | (1 << new_i) | (1 << new_j)
                    current[i] = new_i
                    current[j] = new_j
                    cur_score += delta
                    stuck = 0
                    if cur_score > best_score:
                        best_score = cur_score
                        best_solution = list(current)
                else:
                    stuck += 1
            else:
                # Single element replacement
                idx = random.randint(0, m - 1)
                old = current[idx]
                
                op = random.random()
                new = old
                if op < 0.35:
                    zeros = [i for i in range(n) if not (old & (1 << i))]
                    if zeros:
                        bit = random.choice(zeros)
                        new = old | (1 << bit)
                elif op < 0.70:
                    ones = [i for i in range(n) if (old & (1 << i))]
                    if ones:
                        bit = random.choice(ones)
                        new = old ^ (1 << bit)
                elif op < 0.85:
                    ones = [i for i in range(n) if (old & (1 << i))]
                    zeros = [i for i in range(n) if not (old & (1 << i))]
                    if ones and zeros:
                        bo = random.choice(ones)
                        bz = random.choice(zeros)
                        new = (old ^ (1 << bo)) | (1 << bz)
                else:
                    new = random.randint(0, full_mask)
                
                if new == old or new in current_set:
                    continue
                
                others = selected_bits & ~(1 << old)
                delta = (comp_bits[new] & others).bit_count() - (comp_bits[old] & others).bit_count()
                
                if delta > 0:
                    current_set.remove(old)
                    current_set.add(new)
                    selected_bits = (selected_bits & ~(1 << old)) | (1 << new)
                    current[idx] = new
                    cur_score += delta
                    stuck = 0
                    if cur_score > best_score:
                        best_score = cur_score
                        best_solution = list(current)
                else:
                    stuck += 1
            
            if stuck > stuck_limit:
                idx = random.randint(0, m - 1)
                old_kick = current[idx]
                while True:
                    new_kick = random.randint(0, full_mask)
                    if new_kick not in current_set:
                        break
                current_set.remove(old_kick)
                current_set.add(new_kick)
                selected_bits = (selected_bits & ~(1 << old_kick)) | (1 << new_kick)
                current[idx] = new_kick
                cur_score = compute_score(current)
                stuck = 0
        
        return best_score, best_solution
    
    def simulated_annealing(initial_masks, max_steps=300000):
        current = list(initial_masks)
        current_set = set(current)
        selected_bits = 0
        for s in current:
            selected_bits |= (1 << s)
        
        cur_score = compute_score(current)
        best_score = cur_score
        best_solution = list(current)
        
        T = 10.0
        cooling_rate = 0.9999
        min_T = 0.001
        
        for step in range(max_steps):
            if time.time() > deadline - 5.0:
                break
            if T < min_T:
                break
            
            idx = random.randint(0, m - 1)
            old = current[idx]
            
            r = random.random()
            new = old
            if r < 0.5:
                bit = random.randint(0, n - 1)
                new = old ^ (1 << bit)
            elif r < 0.8:
                b1 = random.randint(0, n - 1)
                b2 = random.randint(0, n - 1)
                while b2 == b1:
                    b2 = random.randint(0, n - 1)
                new = old ^ (1 << b1) ^ (1 << b2)
            else:
                new = random.randint(0, full_mask)
            
            if new == old or new in current_set:
                continue
            
            others = selected_bits & ~(1 << old)
            delta = (comp_bits[new] & others).bit_count() - (comp_bits[old] & others).bit_count()
            
            if delta > 0:
                current_set.remove(old)
                current_set.add(new)
                selected_bits = (selected_bits & ~(1 << old)) | (1 << new)
                current[idx] = new
                cur_score += delta
                if cur_score > best_score:
                    best_score = cur_score
                    best_solution = list(current)
            else:
                if random.random() < math.exp(delta / T):
                    current_set.remove(old)
                    current_set.add(new)
                    selected_bits = (selected_bits & ~(1 << old)) | (1 << new)
                    current[idx] = new
                    cur_score += delta
            
            T *= cooling_rate
        
        return best_score, best_solution
    
    random.seed(42)
    
    best_global_score = -1
    best_global_solution = None
    
    # Multiple different initialization strategies
    init_strategies = []
    
    # Strategy 1: Chain-based greedy
    greedy_solution = greedy_init()
    init_strategies.append(greedy_solution)
    
    # Strategy 2: Nested chains
    for seed in [1, 2, 3, 4]:
        nc_solution = nested_chain_init(seed)
        init_strategies.append(nc_solution)
    
    # Strategy 3: Random with some structure
    for r in range(2):
        random_solution = random.sample(range(num_masks), m)
        init_strategies.append(random_solution)
    
    # Run local search on each initialization
    for init_sol in init_strategies:
        if time.time() > deadline - 15.0:
            break
        
        ls_score, ls_sol = local_search(init_sol, max_steps=150000, stuck_limit=200, use_pair_swaps=True)
        if ls_score > best_global_score:
            best_global_score = ls_score
            best_global_solution = ls_sol
    
    # Simulated annealing on best solution
    if best_global_solution is not None and time.time() < deadline - 10.0:
        sa_score, sa_sol = simulated_annealing(best_global_solution)
        if sa_score > best_global_score:
            best_global_score = sa_score
            best_global_solution = sa_sol
    
    # Final polish with more local search
    if best_global_solution is not None and time.time() < deadline - 5.0:
        ls_score, ls_sol = local_search(best_global_solution, max_steps=100000, stuck_limit=300, use_pair_swaps=True)
        if ls_score > best_global_score:
            best_global_score = ls_score
            best_global_solution = ls_sol
    
    # Convert to list of lists
    result = []
    for mask in best_global_solution:
        s = []
        for i in range(n):
            if (mask >> i) & 1:
                s.append(i)
        result.append(s)
    
    return result
