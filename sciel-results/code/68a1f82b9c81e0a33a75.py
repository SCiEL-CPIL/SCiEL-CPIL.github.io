import math
import time
import random

def construct():
    deadline = time.time() + 115
    
    # Precompute gcd table
    max_m = 200
    gcd_table = [[0] * (max_m + 1) for _ in range(max_m + 1)]
    for i in range(20, max_m + 1):
        for j in range(20, max_m + 1):
            gcd_table[i][j] = math.gcd(i, j)
    
    # Valid moduli: 20..200
    all_moduli = list(range(20, 201))
    num_moduli = len(all_moduli)
    # Map modulus to index
    mod_index = {m: i for i, m in enumerate(all_moduli)}
    
    def calc_score(selected):
        return sum(1.0 / m for m, _ in selected)
    
    def is_valid_add(m, r, selected):
        for m2, r2 in selected:
            g = gcd_table[m][m2]
            if r % g == r2 % g:
                return False
        return True
    
    def find_valid_r(m, selected):
        """Find first valid residue for m given selected set."""
        for r in range(m):
            if is_valid_add(m, r, selected):
                return r
        return None
    
    def find_all_valid_rs(m, selected):
        """Find all valid residues for m given selected set."""
        return [r for r in range(m) if is_valid_add(m, r, selected)]
    
    def greedy_fill(order, randomize_residue=False, rng=None):
        selected = []
        for m in order:
            if time.time() > deadline:
                break
            valid_rs = find_all_valid_rs(m, selected)
            if valid_rs:
                if randomize_residue and rng:
                    selected.append([m, rng.choice(valid_rs)])
                else:
                    selected.append([m, valid_rs[0]])
        return selected
    
    def local_search_1swap(selected, max_iter=50):
        """1-swap local search: remove one large m, try to add a smaller m."""
        improved = True
        iterations = 0
        while improved and iterations < max_iter and time.time() < deadline:
            improved = False
            iterations += 1
            current_score = calc_score(selected)
            sorted_sel = sorted(selected, key=lambda x: -x[0])
            
            for m_remove, r_remove in sorted_sel:
                if time.time() > deadline:
                    break
                new_selected = [x for x in selected if x[0] != m_remove]
                present_ms = set(m for m, _ in new_selected)
                
                for m_add in range(20, m_remove):
                    if m_add in present_ms:
                        continue
                    found_r = find_valid_r(m_add, new_selected)
                    if found_r is not None:
                        new_selected.append([m_add, found_r])
                        new_score = calc_score(new_selected)
                        if new_score > current_score + 1e-9:
                            selected = new_selected
                            improved = True
                            break
                if improved:
                    break
        return selected
    
    def local_search_2swap(selected, max_iter=20):
        """2-swap local search: remove two large m's, try to add two smaller m's."""
        improved = True
        iterations = 0
        while improved and iterations < max_iter and time.time() < deadline:
            improved = False
            iterations += 1
            current_score = calc_score(selected)
            sorted_sel = sorted(selected, key=lambda x: -x[0])
            n = len(sorted_sel)
            
            for i in range(n):
                if time.time() > deadline:
                    break
                m1_remove = sorted_sel[i][0]
                for j in range(i + 1, n):
                    m2_remove = sorted_sel[j][0]
                    new_selected = [x for x in selected if x[0] != m1_remove and x[0] != m2_remove]
                    present_ms = set(m for m, _ in new_selected)
                    
                    # Try to add two smaller moduli
                    added = []
                    for m_add in range(20, min(m1_remove, m2_remove)):
                        if m_add in present_ms:
                            continue
                        found_r = find_valid_r(m_add, new_selected)
                        if found_r is not None:
                            new_selected.append([m_add, found_r])
                            added.append(m_add)
                            present_ms.add(m_add)
                            if len(added) == 2:
                                break
                    
                    if len(added) == 2:
                        new_score = calc_score(new_selected)
                        if new_score > current_score + 1e-9:
                            selected = new_selected
                            improved = True
                            break
                if improved:
                    break
        return selected
    
    def backtracking_refine(selected, depth_limit=3):
        """Try to improve by backtracking: for each element, try alternative residues."""
        if len(selected) > 50:
            return selected  # Too large for backtracking
        
        best = selected
        best_score = calc_score(selected)
        
        def backtrack(idx, current, current_score):
            nonlocal best, best_score
            if time.time() > deadline:
                return
            if idx >= len(current):
                if current_score > best_score:
                    best_score = current_score
                    best = [x[:] for x in current]
                return
            
            m, r = current[idx]
            # Try keeping current r
            backtrack(idx + 1, current, current_score)
            
            # Try alternative residues
            for alt_r in range(m):
                if alt_r == r:
                    continue
                # Check if alt_r is valid with all other elements
                valid = True
                for i, (m2, r2) in enumerate(current):
                    if i == idx:
                        continue
                    g = gcd_table[m][m2]
                    if alt_r % g == r2 % g:
                        valid = False
                        break
                if valid:
                    current[idx][1] = alt_r
                    backtrack(idx + 1, current, current_score)
                    current[idx][1] = r  # restore
        
        backtrack(0, [x[:] for x in selected], calc_score(selected))
        return best
    
    # Strategy 1: Deterministic increasing order
    rng = random.Random(42)
    
    order = list(range(20, 201))
    sol = greedy_fill(order, randomize_residue=False)
    sol = local_search_1swap(sol, max_iter=100)
    sol = local_search_2swap(sol, max_iter=30)
    score = calc_score(sol)
    
    best_score = score
    best_solution = sol
    
    # Strategy 2: Even-only moduli (observed to work well)
    even_order = list(range(20, 201, 2))
    sol = greedy_fill(even_order, randomize_residue=False)
    sol = local_search_1swap(sol, max_iter=100)
    sol = local_search_2swap(sol, max_iter=30)
    score = calc_score(sol)
    if score > best_score:
        best_score = score
        best_solution = sol
    
    # Strategy 3: Random restarts with various orderings
    restart_count = 0
    while time.time() < deadline - 10:
        restart_count += 1
        
        # Different ordering strategies
        strategy = restart_count % 5
        
        if strategy == 0:
            # Noisy increasing
            order = sorted(range(20, 201), key=lambda m: m + rng.uniform(0, 80))
        elif strategy == 1:
            # Even moduli with noise
            order = sorted(range(20, 201, 2), key=lambda m: m + rng.uniform(0, 60))
        elif strategy == 2:
            # Small moduli first with heavy randomization
            order = list(range(20, 201))
            rng.shuffle(order)
            # But bias towards small: sort by (m//20, random)
            order = sorted(order, key=lambda m: (m // 20, rng.random()))
        elif strategy == 3:
            # Pure random
            order = list(range(20, 201))
            rng.shuffle(order)
        else:
            # Weighted random: prefer small m
            order = list(range(20, 201))
            weights = [1.0 / m for m in order]
            # Sample without replacement using weights
            remaining = list(order)
            weighted_order = []
            for _ in range(len(remaining)):
                w = [1.0 / m for m in remaining]
                total = sum(w)
                r = rng.random() * total
                cum = 0
                for idx, m in enumerate(remaining):
                    cum += w[idx]
                    if r <= cum:
                        weighted_order.append(m)
                        remaining.pop(idx)
                        break
            order = weighted_order
        
        sol = greedy_fill(order, randomize_residue=True, rng=rng)
        sol = local_search_1swap(sol, max_iter=50)
        
        # Occasionally do 2-swap and backtracking
        if restart_count % 3 == 0:
            sol = local_search_2swap(sol, max_iter=15)
        
        if restart_count % 5 == 0 and len(sol) <= 40:
            sol = backtracking_refine(sol)
        
        score = calc_score(sol)
        if score > best_score:
            best_score = score
            best_solution = sol
    
    # Final refinement on best solution
    best_solution = local_search_1swap(best_solution, max_iter=100)
    best_solution = local_search_2swap(best_solution, max_iter=30)
    
    return best_solution

if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    score = sum(1.0 / m for m, _ in res)
    print(f"Score: {score}")
    print(f"Time: {end - start:.2f}s")
    print(f"Classes: {len(res)}")
    valid = True
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            m1, r1 = res[i]
            m2, r2 = res[j]
            g = math.gcd(m1, m2)
            if r1 % g == r2 % g:
                valid = False
                print(f"Invalid: {res[i]} and {res[j]}")
    print(f"Valid: {valid}")
