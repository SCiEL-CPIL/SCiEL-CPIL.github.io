import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    moduli = list(range(2, 129))
    N = len(moduli)  # 127 moduli

    # Precompute gcd table
    gcd_table = [[0]*129 for _ in range(129)]
    for i in range(2, 129):
        for j in range(i, 129):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    def check_pair_disjoint(r1, m1, r2, m2):
        """Check if residue class r1 mod m1 is disjoint from r2 mod m2.
        They are disjoint iff r1 mod g != r2 mod g where g = gcd(m1, m2).
        If gcd=1, they always intersect (by CRT)."""
        g = gcd_table[m1][m2]
        if g == 1:
            return False  # gcd=1 means they intersect
        return (r1 % g) != (r2 % g)

    def is_valid(r, m, selected):
        """Check if (r, m) is disjoint from all in selected."""
        for r_k, m_k in selected:
            if not check_pair_disjoint(r, m, r_k, m_k):
                return False
        return True

    def find_valid_residues(m, selected):
        """Find all valid residues for modulus m given selected classes."""
        valid = []
        for r in range(m):
            if is_valid(r, m, selected):
                valid.append(r)
        return valid

    def greedy_build(order, deterministic=True, rng=None):
        """Greedy: add moduli in given order, picking a valid residue."""
        selected = []
        for m in order:
            if not selected:
                selected.append((0, m))
                continue
            if deterministic:
                for r in range(m):
                    if is_valid(r, m, selected):
                        selected.append((r, m))
                        break
            else:
                valid_r = find_valid_residues(m, selected)
                if valid_r:
                    r = rng.choice(valid_r)
                    selected.append((r, m))
        return selected

    def try_add_one(selected, unselected_mods):
        """Try to add one unselected modulus with a valid residue."""
        for m in unselected_mods:
            valid_r = find_valid_residues(m, selected)
            if valid_r:
                return m, valid_r[0]
        return None, None

    def local_search(selected, max_rounds=100, rng=None):
        """Hill climbing: try to add more classes via 1-for-1 swaps."""
        selected = list(selected)
        selected_set = set(m for _, m in selected)
        
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            improved = False
            
            # Try adding any unselected modulus
            unselected = [m for m in moduli if m not in selected_set]
            if unselected:
                m, r = try_add_one(selected, unselected)
                if m is not None:
                    selected.append((r, m))
                    selected_set.add(m)
                    improved = True
            
            if not improved:
                # Try 1-for-1 swap: remove one, try to add two
                swapped = False
                for i in range(len(selected)):
                    if time.time() > deadline:
                        break
                    removed = selected[i]
                    sol_copy = selected[:i] + selected[i+1:]
                    sol_copy_set = set(m for _, m in sol_copy)
                    
                    # Try to add two unselected moduli
                    added = []
                    for m in moduli:
                        if m in sol_copy_set or m == removed[1]:
                            continue
                        valid_r = find_valid_residues(m, sol_copy + added)
                        if valid_r:
                            added.append((valid_r[0], m))
                            sol_copy_set.add(m)
                            if len(added) == 2:
                                break
                    
                    if len(added) >= 1:
                        new_sol = sol_copy + added
                        if len(new_sol) > len(selected):
                            selected = new_sol
                            selected_set = set(m for _, m in selected)
                            improved = True
                            swapped = True
                            break
            
            if not improved:
                break
        
        return selected

    def local_search_v2(selected, max_rounds=200, rng=None):
        """More aggressive local search with multiple swap strategies."""
        selected = list(selected)
        selected_set = set(m for _, m in selected)
        
        for round_num in range(max_rounds):
            if time.time() > deadline:
                break
            improved = False
            
            # Strategy 1: Try adding any unselected modulus
            unselected = [m for m in moduli if m not in selected_set]
            if unselected:
                # Sort by number of valid residues (fewest first = most constrained)
                def count_valid(m):
                    return len(find_valid_residues(m, selected))
                
                unselected.sort(key=count_valid)
                for m in unselected:
                    valid_r = find_valid_residues(m, selected)
                    if valid_r:
                        selected.append((valid_r[0], m))
                        selected_set.add(m)
                        improved = True
                        break
            
            # Strategy 2: Try 1-for-1 swaps
            if not improved:
                for i in range(len(selected)):
                    if time.time() > deadline:
                        break
                    removed_m = selected[i][1]
                    sol_copy = selected[:i] + selected[i+1:]
                    sol_copy_set = set(m for _, m in sol_copy)
                    
                    # Try to add one unselected modulus
                    best_add = None
                    best_add_len = 0
                    for m in moduli:
                        if m in sol_copy_set:
                            continue
                        valid_r = find_valid_residues(m, sol_copy)
                        if valid_r:
                            # Prefer adding a modulus that opens up more options
                            new_sol = sol_copy + [(valid_r[0], m)]
                            new_set = sol_copy_set | {m}
                            # Count how many more could be added
                            potential = 0
                            for m2 in moduli:
                                if m2 in new_set:
                                    continue
                                if find_valid_residues(m2, new_sol):
                                    potential += 1
                            if potential > best_add_len:
                                best_add_len = potential
                                best_add = (m, valid_r[0])
                    
                    if best_add is not None:
                        new_sol = sol_copy + [(best_add[1], best_add[0])]
                        if len(new_sol) > len(selected):
                            selected = new_sol
                            selected_set = set(m for _, m in selected)
                            improved = True
                            break
            
            # Strategy 3: Try removing two, adding three
            if not improved and len(selected) >= 2:
                for i in range(len(selected)):
                    if time.time() > deadline:
                        break
                    for j in range(i+1, len(selected)):
                        sol_copy = selected[:i] + selected[i+1:j] + selected[j+1:]
                        sol_copy_set = set(m for _, m in sol_copy)
                        
                        added = []
                        for m in moduli:
                            if m in sol_copy_set:
                                continue
                            valid_r = find_valid_residues(m, sol_copy + added)
                            if valid_r:
                                added.append((valid_r[0], m))
                                sol_copy_set.add(m)
                                if len(added) == 3:
                                    break
                        
                        if len(added) >= 2:
                            new_sol = sol_copy + added
                            if len(new_sol) > len(selected):
                                selected = new_sol
                                selected_set = set(m for _, m in selected)
                                improved = True
                                break
                    if improved:
                        break
            
            if not improved:
                break
        
        return selected

    # ---- Main optimization ----
    
    best_solution = []
    best_len = 0

    # 1. Quick deterministic constructions
    # Try greedy with various orderings
    def num_divisors(n):
        count = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count

    # Precompute some properties
    divisor_counts = {m: num_divisors(m) for m in moduli}
    
    orders = [
        list(range(2, 129)),
        list(range(128, 1, -1)),
        sorted(moduli, key=lambda x: -divisor_counts[x]),
        sorted(moduli, key=lambda x: divisor_counts[x]),
        sorted(moduli, key=lambda x: -x.bit_count()),
        sorted(moduli, key=lambda x: x.bit_count()),
    ]

    for order in orders:
        if time.time() > deadline:
            break
        sol = greedy_build(order, deterministic=True)
        if len(sol) > best_len:
            best_len = len(sol)
            best_solution = sol

    # 2. Randomized restarts with greedy
    rng = random.Random(42)
    num_restarts = 300
    for i in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        random_order = moduli[:]
        rng.shuffle(random_order)
        sol = greedy_build(random_order, deterministic=False, rng=rng)
        if len(sol) > best_len:
            best_len = len(sol)
            best_solution = sol

    # 3. Local search from best solution
    if time.time() < deadline - 5.0:
        improved_sol = local_search(best_solution, max_rounds=50, rng=rng)
        if len(improved_sol) > best_len:
            best_len = len(improved_sol)
            best_solution = improved_sol

    # 4. More randomized restarts with local search
    for i in range(150):
        if time.time() > deadline - 8.0:
            break
        random_order = moduli[:]
        rng.shuffle(random_order)
        sol = greedy_build(random_order, deterministic=False, rng=rng)
        sol = local_search(sol, max_rounds=30, rng=rng)
        if len(sol) > best_len:
            best_len = len(sol)
            best_solution = sol

    # 5. Final aggressive local search on best
    if time.time() < deadline - 3.0:
        improved_sol = local_search_v2(best_solution, max_rounds=100, rng=rng)
        if len(improved_sol) > best_len:
            best_len = len(improved_sol)
            best_solution = improved_sol

    # Final safety: ensure non-empty
    if not best_solution:
        best_solution = [[0, 2]]

    return [[r, m] for r, m in best_solution]
