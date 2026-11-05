import math
import random
import time
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    MAX_M = 128
    candidates = list(range(2, MAX_M + 1))
    n = len(candidates)

    # Precompute gcd table
    gcd_tab = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                gcd_tab[i][j] = math.gcd(candidates[i], candidates[j])
    
    # Precompute for each pair (i,j), the set of residues r_j that are incompatible with any r_i
    # For fixed i, j, and r_i, the incompatible r_j are those where (r_j - r_i) % gcd(i,j) == 0
    # So for each pair (i,j) and each possible r_i, we can precompute the forbidden r_j values

    def get_forbidden_residues(selected, idx):
        """Get set of residues forbidden for idx given current selection."""
        forbidden = set()
        m = candidates[idx]
        for (sel_idx, sel_r) in selected:
            g = gcd_tab[idx][sel_idx]
            base = sel_r % g
            for r in range(base, m, g):
                forbidden.add(r)
        return forbidden

    def get_valid_residues(selected, idx):
        """Get all valid residues for idx given current selection."""
        m = candidates[idx]
        forbidden = get_forbidden_residues(selected, idx)
        valid = [r for r in range(m) if r not in forbidden]
        return valid

    def is_compatible(selected, new_idx, new_r):
        """Check if (new_idx, new_r) is disjoint from all in selected."""
        for (idx, r) in selected:
            g = gcd_tab[new_idx][idx]
            if (new_r - r) % g == 0:
                return False
        return True

    def greedy_with_order(order, residue_strategy='random'):
        """Greedy assignment with specified order and residue strategy."""
        selected = []
        for idx in order:
            m = candidates[idx]
            if residue_strategy == 'random':
                residues = list(range(m))
                random.shuffle(residues)
            elif residue_strategy == 'first_valid':
                residues = list(range(m))
            elif residue_strategy == 'middle':
                residues = list(range(m))
                # Start from middle
                mid = m // 2
                residues = list(range(mid, m)) + list(range(mid))
            else:
                residues = list(range(m))
            
            for r in residues:
                if is_compatible(selected, idx, r):
                    selected.append((idx, r))
                    break
        return selected

    def local_search_improve(solution, max_iters=200):
        """Improve solution by trying to add missing moduli or swap."""
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 12:
            improved = False
            iters += 1
            selected_idx_set = set(idx for (idx, _) in solution)

            # Try to add any missing modulus
            for idx in range(n):
                if idx in selected_idx_set:
                    continue
                valid_residues = get_valid_residues(solution, idx)
                if valid_residues:
                    # Choose residue that maximizes future flexibility
                    best_r = None
                    best_score = -1
                    for r in valid_residues:
                        # Score: count how many other unselected moduli would still have valid residues
                        test_solution = solution + [(idx, r)]
                        future_options = 0
                        for other_idx in range(n):
                            if other_idx in selected_idx_set or other_idx == idx:
                                continue
                            if get_valid_residues(test_solution, other_idx):
                                future_options += 1
                        if future_options > best_score:
                            best_score = future_options
                            best_r = r
                    
                    solution.append((idx, best_r))
                    selected_idx_set.add(idx)
                    improved = True
                    break

            if improved:
                continue

            # Try 1-for-1 swap: remove one, add another
            for i in range(len(solution)):
                removed = solution[i]
                reduced = [x for k, x in enumerate(solution) if k != i]
                reduced_idx_set = set(idx for (idx, _) in reduced)
                
                # Try to add a different modulus
                for idx in range(n):
                    if idx in reduced_idx_set:
                        continue
                    valid_residues = get_valid_residues(reduced, idx)
                    if valid_residues:
                        # Check if this gives us a net gain (shouldn't for 1-for-1, but try)
                        new_solution = reduced + [(idx, valid_residues[0])]
                        if len(new_solution) > len(solution):
                            solution = new_solution
                            improved = True
                            break
                
                if improved:
                    break

            # Try 1-for-2 swap: remove one, try to add two
            for i in range(len(solution)):
                reduced = [x for k, x in enumerate(solution) if k != i]
                reduced_idx_set = set(idx for (idx, _) in reduced)
                
                added = 0
                new_solution = list(reduced)
                for idx in range(n):
                    if idx in reduced_idx_set:
                        continue
                    valid_residues = get_valid_residues(new_solution, idx)
                    if valid_residues:
                        new_solution.append((idx, valid_residues[0]))
                        reduced_idx_set.add(idx)
                        added += 1
                        if added >= 2:
                            break
                
                if len(new_solution) > len(solution):
                    solution = new_solution
                    improved = True
                    break

        return solution

    def backtracking_add(selected, available_indices, depth_limit=3):
        """Try to add more moduli via backtracking with limited depth."""
        if time.time() > deadline - 10:
            return selected
        if not available_indices or depth_limit <= 0:
            return selected

        # MRV: pick index with fewest valid residues
        best_idx = None
        best_residues = None
        for idx in available_indices:
            residues = get_valid_residues(selected, idx)
            if not residues:
                continue
            if best_residues is None or len(residues) < len(best_residues):
                best_idx = idx
                best_residues = residues

        if best_idx is None:
            return selected

        # Try residues in order of increasing constraint on others
        def count_future_options(sel, idx, r):
            test_sel = sel + [(idx, r)]
            count = 0
            for other_idx in available_indices:
                if other_idx == idx:
                    continue
                if get_valid_residues(test_sel, other_idx):
                    count += 1
            return count

        # Sort residues by how much they constrain others (less constraining first)
        scored_residues = [(r, count_future_options(selected, best_idx, r)) for r in best_residues]
        scored_residues.sort(key=lambda x: -x[1])  # Less constraining first

        for r, _ in scored_residues:
            new_selected = selected + [(best_idx, r)]
            remaining = [idx for idx in available_indices if idx != best_idx]
            result = backtracking_add(new_selected, remaining, depth_limit - 1)
            if len(result) > len(selected):
                return result

        return selected

    # Phase 1: Multiple greedy strategies with random restarts
    random.seed(42)
    best_solution = []
    best_len = 0

    # Try different ordering strategies
    ordering_strategies = [
        'random',
        'ascending',
        'descending',
        'prime_first',
        'composite_first'
    ]

    for restart in range(300):
        if time.time() > deadline - 40:
            break
        
        strategy = random.choice(ordering_strategies)
        order = list(range(n))
        
        if strategy == 'random':
            random.shuffle(order)
        elif strategy == 'ascending':
            pass  # already ascending
        elif strategy == 'descending':
            order.reverse()
        elif strategy == 'prime_first':
            def is_prime(x):
                if x < 2:
                    return False
                for p in range(2, int(x**0.5) + 1):
                    if x % p == 0:
                        return False
                return True
            primes = [i for i in range(n) if is_prime(candidates[i])]
            composites = [i for i in range(n) if i not in primes]
            random.shuffle(primes)
            random.shuffle(composites)
            order = primes + composites
        elif strategy == 'composite_first':
            def is_prime(x):
                if x < 2:
                    return False
                for p in range(2, int(x**0.5) + 1):
                    if x % p == 0:
                        return False
                return True
            primes = [i for i in range(n) if is_prime(candidates[i])]
            composites = [i for i in range(n) if i not in primes]
            random.shuffle(composites)
            random.shuffle(primes)
            order = composites + primes
        
        residue_strategy = random.choice(['random', 'first_valid', 'middle'])
        sol = greedy_with_order(order, residue_strategy)
        
        if len(sol) > best_len:
            best_solution = sol
            best_len = len(sol)

    # Phase 2: Local search improvement
    if best_solution:
        improved_sol = local_search_improve(best_solution)
        if len(improved_sol) > best_len:
            best_solution = improved_sol
            best_len = len(best_solution)

    # Phase 3: Backtracking to add more
    if time.time() < deadline - 20 and best_solution:
        selected_idx_set = set(idx for (idx, _) in best_solution)
        available = [idx for idx in range(n) if idx not in selected_idx_set]
        
        bt_result = backtracking_add(list(best_solution), available, depth_limit=4)
        if len(bt_result) > best_len:
            best_solution = bt_result
            best_len = len(best_solution)

    # Phase 4: More local search rounds
    if time.time() < deadline - 15 and best_solution:
        for _ in range(5):
            if time.time() > deadline - 15:
                break
            improved_sol = local_search_improve(best_solution, max_iters=50)
            if len(improved_sol) > best_len:
                best_solution = improved_sol
                best_len = len(best_solution)
            else:
                break

    # Final validation: ensure all pairs are truly disjoint
    def validate(solution):
        for i in range(len(solution)):
            for j in range(i + 1, len(solution)):
                idx1, r1 = solution[i]
                idx2, r2 = solution[j]
                g = gcd_tab[idx1][idx2]
                if (r1 - r2) % g == 0:
                    return False
        return True

    if not validate(best_solution):
        # Repair: remove conflicting pairs
        repaired = []
        for item in best_solution:
            if all(is_compatible(repaired, item[0], item[1]) for _ in [1]):
                repaired.append(item)
        best_solution = repaired

    # Convert to required format [[r, m], ...]
    result = [[r, candidates[idx]] for (idx, r) in best_solution]
    return result
