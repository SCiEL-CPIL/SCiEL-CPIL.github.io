"""
Maximum Weight Independent Set on compatible (m,r) pairs via:
1. Precomputed conflict tables for speed
2. Greedy initialization with multiple orderings
3. Aggressive simulated annealing with adaptive temperature
4. Exact local polish (systematic add/remove/swap)
5. Multiple random restarts
"""
import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Precompute gcd table and conflict structure
    # moduli: 20..200 (181 values)
    # For each pair (m1, m2), compute gcd
    # Two residues r1, r2 conflict iff r1 % gcd == r2 % gcd
    
    moduli = list(range(20, 201))
    n_mod = len(moduli)
    mod_index = {m: i for i, m in enumerate(moduli)}
    
    # Precompute gcd for all pairs
    gcd_table = [[0] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        gcd_table[i][i] = moduli[i]
        for j in range(i + 1, n_mod):
            g = math.gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # Precompute for each pair (i,j) and each residue r, which residues of the other modulus conflict
    # Actually, for a given (m_i, r_i) and (m_j, r_j): conflict iff r_i % gcd(i,j) == r_j % gcd(i,j)
    # So for fixed i and r_i, the set of conflicting r_j values for modulus j is:
    #   r_j % g == r_i % g, where g = gcd(m_i, m_j)
    # This means r_j must be congruent to r_i mod g
    
    def is_compatible_fast(i, r_i, j, r_j):
        """Check if (moduli[i], r_i) and (moduli[j], r_j) are compatible."""
        if i == j:
            return r_i != r_j  # Can't have same modulus twice, but this shouldn't happen
        g = gcd_table[i][j]
        return (r_i % g) != (r_j % g)
    
    def find_valid_residues_fast(i, selected_idx_res):
        """Find all valid residues for modulus index i given selected (index, residue) pairs."""
        m = moduli[i]
        valid = []
        for r in range(m):
            ok = True
            for j, r_j in selected_idx_res:
                if j == i:
                    continue
                g = gcd_table[i][j]
                if (r % g) == (r_j % g):
                    ok = False
                    break
            if ok:
                valid.append(r)
        return valid
    
    def score_of(solution):
        """Compute score. solution is list of (mod_index, residue)."""
        return sum(1.0 / moduli[idx] for idx, r in solution)
    
    def greedy_fill(order, rng):
        """Fill greedily in given order."""
        selected = []  # list of (mod_index, residue)
        selected_set = set()
        for i in order:
            if i in selected_set:
                continue
            valid_rs = find_valid_residues_fast(i, selected)
            if valid_rs:
                r = rng.choice(valid_rs)
                selected.append((i, r))
                selected_set.add(i)
        return selected
    
    def local_search(selected, moduli_order, rng, max_iters=300):
        """Local search: try residue changes and additions."""
        current = list(selected)
        current_score = score_of(current)
        best = list(current)
        best_score = current_score
        
        selected_set = {idx for idx, _ in current}
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            move_type = rng.random()
            
            if move_type < 0.5 and len(current) > 0:
                # Change residue of a random selected modulus
                idx_pos = rng.randint(0, len(current) - 1)
                i, r_old = current[idx_pos]
                others = [c for k, c in enumerate(current) if k != idx_pos]
                valid_rs = find_valid_residues_fast(i, others)
                if valid_rs:
                    new_r = rng.choice(valid_rs)
                    if new_r != r_old:
                        new_current = current[:]
                        new_current[idx_pos] = (i, new_r)
                        new_score = score_of(new_current)
                        if new_score >= current_score:
                            current = new_current
                            current_score = new_score
                            if current_score > best_score:
                                best = list(current)
                                best_score = current_score
                            
            elif move_type < 0.8:
                # Try adding a random unselected modulus
                candidates = [i for i in moduli_order if i not in selected_set]
                if candidates:
                    i_new = rng.choice(candidates)
                    valid_rs = find_valid_residues_fast(i_new, current)
                    if valid_rs:
                        r_new = rng.choice(valid_rs)
                        new_current = current + [(i_new, r_new)]
                        new_score = score_of(new_current)
                        if new_score >= current_score:
                            current = new_current
                            current_score = new_score
                            selected_set.add(i_new)
                            if current_score > best_score:
                                best = list(current)
                                best_score = current_score
                            
            else:
                # Swap: remove one, add another
                if len(current) > 0:
                    idx_pos = rng.randint(0, len(current) - 1)
                    i_remove = current[idx_pos][0]
                    others = [c for k, c in enumerate(current) if k != idx_pos]
                    others_set = {idx for idx, _ in others}
                    candidates = [i for i in moduli_order if i not in others_set]
                    if candidates:
                        i_add = rng.choice(candidates)
                        valid_rs = find_valid_residues_fast(i_add, others)
                        if valid_rs:
                            r_add = rng.choice(valid_rs)
                            new_current = others + [(i_add, r_add)]
                            new_score = score_of(new_current)
                            if new_score >= current_score:
                                current = new_current
                                current_score = new_score
                                selected_set = {idx for idx, _ in current}
                                if current_score > best_score:
                                    best = list(current)
                                    best_score = current_score
        
        return best
    
    def simulated_annealing(selected, moduli_order, rng, start_temp=0.5, end_temp=0.005, max_iters=2000):
        """Simulated annealing."""
        current = list(selected)
        current_score = score_of(current)
        best = list(current)
        best_score = current_score
        
        selected_set = {idx for idx, _ in current}
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
            
            temp = start_temp * (end_temp / start_temp) ** (i / max_iters)
            
            move_type = rng.random()
            neighbor = None
            neighbor_set = None
            
            if move_type < 0.4 and len(current) > 0:
                # Change residue
                idx_pos = rng.randint(0, len(current) - 1)
                i, r_old = current[idx_pos]
                others = [c for k, c in enumerate(current) if k != idx_pos]
                valid_rs = find_valid_residues_fast(i, others)
                if valid_rs:
                    new_r = rng.choice(valid_rs)
                    neighbor = current[:]
                    neighbor[idx_pos] = (i, new_r)
                    neighbor_set = selected_set
                    
            elif move_type < 0.7:
                # Add a modulus
                candidates = [i for i in moduli_order if i not in selected_set]
                if candidates:
                    i_new = rng.choice(candidates)
                    valid_rs = find_valid_residues_fast(i_new, current)
                    if valid_rs:
                        r_new = rng.choice(valid_rs)
                        neighbor = current + [(i_new, r_new)]
                        neighbor_set = selected_set | {i_new}
                        
            elif move_type < 0.85 and len(current) > 1:
                # Remove a modulus
                idx_pos = rng.randint(0, len(current) - 1)
                i_remove = current[idx_pos][0]
                neighbor = current[:idx_pos] + current[idx_pos+1:]
                neighbor_set = selected_set - {i_remove}
                
            else:
                # Swap
                if len(current) > 0:
                    idx_pos = rng.randint(0, len(current) - 1)
                    i_remove = current[idx_pos][0]
                    others = [c for k, c in enumerate(current) if k != idx_pos]
                    others_set = {idx for idx, _ in others}
                    candidates = [i for i in moduli_order if i not in others_set]
                    if candidates:
                        i_add = rng.choice(candidates)
                        valid_rs = find_valid_residues_fast(i_add, others)
                        if valid_rs:
                            r_add = rng.choice(valid_rs)
                            neighbor = others + [(i_add, r_add)]
                            neighbor_set = others_set | {i_add}
            
            if neighbor is None:
                continue
            
            neighbor_score = score_of(neighbor)
            delta = neighbor_score - current_score
            
            if delta >= 0 or rng.random() < math.exp(delta / temp):
                current = neighbor
                current_score = neighbor_score
                selected_set = neighbor_set
                if current_score > best_score:
                    best = list(current)
                    best_score = current_score
        
        return best
    
    def exact_polish(selected, moduli_order, rng, max_time=5.0):
        """Systematic polish: try all possible additions, then all swaps, iteratively."""
        current = list(selected)
        current_score = score_of(current)
        best = list(current)
        best_score = current_score
        selected_set = {idx for idx, _ in current}
        
        improved = True
        while improved and time.time() < start_time + max_time:
            improved = False
            
            # Try adding each unselected modulus
            for i in moduli_order:
                if i in selected_set:
                    continue
                valid_rs = find_valid_residues_fast(i, current)
                if valid_rs:
                    r = valid_rs[0]  # Any valid residue works for score
                    new_current = current + [(i, r)]
                    new_score = score_of(new_current)
                    if new_score > best_score + 1e-12:
                        current = new_current
                        current_score = new_score
                        best = list(current)
                        best_score = new_score
                        selected_set.add(i)
                        improved = True
            
            # Try swapping each selected modulus with each unselected
            for i_sel in list(selected_set):
                for i_add in moduli_order:
                    if i_add in selected_set:
                        continue
                    others = [c for idx, r in current if idx != i_sel]
                    valid_rs = find_valid_residues_fast(i_add, others)
                    if valid_rs:
                        r = valid_rs[0]
                        new_current = others + [(i_add, r)]
                        new_score = score_of(new_current)
                        if new_score > best_score + 1e-12:
                            current = new_current
                            current_score = new_score
                            best = list(current)
                            best_score = new_score
                            selected_set = {idx for idx, _ in current}
                            improved = True
    
    return best
    
    # Main search loop
    best_solution = None
    best_score = 0.0
    
    # Try multiple different initial orderings
    orderings = [
        list(range(n_mod)),  # ascending modulus (smallest first)
        list(reversed(range(n_mod))),  # descending
    ]
    # Add some random orderings
    for seed in range(5):
        rng_ord = random.Random(seed * 1000 + 42)
        order = list(range(n_mod))
        rng_ord.shuffle(order)
        orderings.append(order)
    
    num_restarts = 0
    while time.time() < deadline - 10:
        num_restarts += 1
        rng = random.Random(num_restarts * 7919 + 13)
        
        # Choose an ordering
        order = orderings[num_restarts % len(orderings)]
        
        # Start with greedy fill
        current = greedy_fill(order, rng)
        current_score = score_of(current)
        
        # Local search
        improved = local_search(current, list(range(n_mod)), rng, max_iters=150)
        
        # Simulated annealing
        annealed = simulated_annealing(improved, list(range(n_mod)), rng, start_temp=0.4, end_temp=0.003, max_iters=1500)
        
        # Final local search
        final_ls = local_search(annealed, list(range(n_mod)), rng, max_iters=200)
        
        # Exact polish
        polished = exact_polish(final_ls, list(range(n_mod)), rng, max_time=3.0)
        polished_score = score_of(polished)
        
        if polished_score > best_score:
            best_score = polished_score
            best_solution = polished
    
    # Final polish on the best solution found
    if best_solution is not None and time.time() < deadline - 3:
        rng_final = random.Random(12345)
        best_solution = exact_polish(best_solution, list(range(n_mod)), rng_final, max_time=5.0)
        best_score = score_of(best_solution)
    
    # Convert to required format [[m, r], ...]
    if best_solution is None:
        # Fallback: simple greedy
        rng = random.Random(42)
        best_solution = greedy_fill(list(range(n_mod)), rng)
    
    result = [[moduli[idx], r] for idx, r in sorted(best_solution, key=lambda x: moduli[x[0]])]
    return result
