import math
import time
import random

def construct():
    # Deadline guard: 120s limit, use 100s to be safe
    deadline = time.time() + 100

    max_m = 200
    min_m = 20

    # Precompute gcd table for fast lookup
    gcd_table = [[0] * (max_m + 1) for _ in range(max_m + 1)]
    for i in range(min_m, max_m + 1):
        for j in range(min_m, max_m + 1):
            gcd_table[i][j] = math.gcd(i, j)

    # Precompute prime factors
    prime_factors = [[] for _ in range(max_m + 1)]
    for p in range(2, max_m + 1):
        if not prime_factors[p]:
            for multiple in range(p, max_m + 1, p):
                prime_factors[multiple].append(p)

    def is_valid(m, r, selected, skip_idx=None):
        """Check if (m, r) is valid against selected classes."""
        for idx, (m2, r2) in enumerate(selected):
            if skip_idx is not None and idx == skip_idx:
                continue
            g = gcd_table[m][m2]
            if r % g == r2 % g:
                return False
        return True

    def calc_score(selected):
        return sum(1.0 / m for m, _ in selected)

    def ac3_prune(moduli, selected_residues):
        """
        Given a set of moduli and their assigned residues, use AC-3 style
        propagation to determine which residues are still possible for
        unassigned moduli. Returns dict: m -> set of valid residues.
        """
        # For each unassigned modulus, compute valid residues
        domains = {}
        for m in moduli:
            valid = set(range(m))
            for m2, r2 in selected_residues:
                g = gcd_table[m][m2]
                # Remove residues r where r % g == r2 % g
                bad = r2 % g
                to_remove = [r for r in valid if r % g == bad]
                valid -= set(to_remove)
            domains[m] = valid
        return domains

    def greedy_assign(moduli, domains, deadline):
        """
        Greedily assign residues to moduli in order of decreasing 1/m (smallest m first).
        Uses the pruned domains.
        """
        selected = []
        remaining = list(moduli)
        remaining.sort(key=lambda m: 1.0 / m, reverse=True)  # smallest m first
        
        for m in remaining:
            if m not in domains:
                continue
            if not domains[m]:
                continue
            # Pick the first valid residue
            r = min(domains[m])
            selected.append([m, r])
            # Update domains for remaining moduli
            for m2 in list(remaining):
                if m2 == m or m2 not in domains:
                    continue
                g = gcd_table[m][m2]
                bad = r % g
                domains[m2] = {r2 for r2 in domains[m2] if r2 % g != bad}
            remaining.remove(m)
            if time.time() > deadline:
                break
        return selected

    def local_search_polish(selected, deadline):
        """Fast local search: add missing small moduli, 1-swaps, and 2-swaps."""
        current_sol = [list(x) for x in selected]
        current_score = calc_score(current_sol)
        improved = True
        
        max_iters = 30
        iters = 0
        while improved and time.time() < deadline and iters < max_iters:
            improved = False
            iters += 1
            
            # 1. Try adding each missing modulus (prioritize small m)
            for m in range(min_m, max_m + 1):
                if any(m == m_sel for m_sel, _ in current_sol):
                    continue
                for r in range(m):
                    if is_valid(m, r, current_sol):
                        current_sol.append([m, r])
                        current_score += 1.0 / m
                        improved = True
                        break
            
            # 2. 1-swap: remove one class, add another with higher weight
            for idx in range(len(current_sol)):
                m_rem, r_rem = current_sol[idx]
                current_sol.pop(idx)
                current_score -= 1.0 / m_rem
                
                best_add = None
                best_add_score = -1
                for m_add in range(min_m, max_m + 1):
                    if m_add == m_rem:
                        continue
                    if any(m_add == m_sel for m_sel, _ in current_sol):
                        continue
                    if 1.0 / m_add <= 1.0 / m_rem:
                        continue
                    for r_add in range(m_add):
                        if is_valid(m_add, r_add, current_sol):
                            best_add = [m_add, r_add]
                            best_add_score = 1.0 / m_add
                            break
                
                if best_add:
                    current_sol.append(best_add)
                    current_score += best_add_score
                    improved = True
                else:
                    current_sol.insert(idx, [m_rem, r_rem])
                    current_score += 1.0 / m_rem
            
            # 3. 2-swap: remove two classes, add two different ones
            n = len(current_sol)
            if n >= 2:
                for i in range(n):
                    if time.time() > deadline:
                        break
                    for j in range(i + 1, n):
                        if time.time() > deadline:
                            break
                        m_rem1, r_rem1 = current_sol[i]
                        m_rem2, r_rem2 = current_sol[j]
                        removed_score = 1.0 / m_rem1 + 1.0 / m_rem2
                        
                        # Try to find two replacements
                        best_pair = None
                        best_pair_score = -1
                        for m_add1 in range(min_m, max_m + 1):
                            if m_add1 in (m_rem1, m_rem2):
                                continue
                            if any(m_add1 == m_sel for m_sel, _ in current_sol if m_sel not in (m_rem1, m_rem2)):
                                continue
                            for m_add2 in range(min_m, max_m + 1):
                                if m_add2 in (m_rem1, m_rem2, m_add1):
                                    continue
                                if any(m_add2 == m_sel for m_sel, _ in current_sol if m_sel not in (m_rem1, m_rem2)):
                                    continue
                                pair_score = 1.0 / m_add1 + 1.0 / m_add2
                                if pair_score <= removed_score:
                                    continue
                                # Check validity
                                temp_sol = [x for k, x in enumerate(current_sol) if k not in (i, j)]
                                for r1 in range(m_add1):
                                    if is_valid(m_add1, r1, temp_sol):
                                        temp_sol2 = temp_sol + [[m_add1, r1]]
                                        for r2 in range(m_add2):
                                            if is_valid(m_add2, r2, temp_sol2):
                                                best_pair = [[m_add1, r1], [m_add2, r2]]
                                                best_pair_score = pair_score
                                                break
                                        if best_pair:
                                            break
                                    if best_pair:
                                        break
                            if best_pair:
                                break
                        
                        if best_pair:
                            # Apply the 2-swap
                            new_sol = [x for k, x in enumerate(current_sol) if k not in (i, j)]
                            new_sol.extend(best_pair)
                            new_score = calc_score(new_sol)
                            if new_score > current_score:
                                current_sol = new_sol
                                current_score = new_score
                                improved = True
                            break  # Only do one 2-swap per iteration to keep it fast
            if time.time() > deadline:
                break
        
        return current_sol, current_score

    def simulated_annealing(initial_solution, deadline):
        current_sol = [list(x) for x in initial_solution]
        current_score = calc_score(current_sol)
        best_sol = [list(x) for x in current_sol]
        best_score = current_score

        T = 0.5
        cooling = 0.9995
        min_T = 1e-8

        while time.time() < deadline:
            if T < min_T:
                break

            move_type = random.random()
            new_sol = [list(x) for x in current_sol]
            new_score = current_score
            valid_move = False

            if move_type < 0.40:
                # Add a modulus - prefer smaller moduli for higher contribution
                m_add = random.randint(min_m, max_m)
                if random.random() < 0.3:
                    m_add = random.randint(min_m, min_m + 30)
                if any(m == m_add for m, _ in new_sol):
                    continue
                for _ in range(8):
                    r_add = random.randint(0, m_add - 1)
                    if is_valid(m_add, r_add, new_sol):
                        new_sol.append([m_add, r_add])
                        new_score += 1.0 / m_add
                        valid_move = True
                        break
            elif move_type < 0.60:
                # Remove a modulus - prefer removing large moduli
                if not new_sol:
                    continue
                weights = [1.0 / new_sol[i][0] for i in range(len(new_sol))]
                total_w = sum(weights)
                r = random.random() * total_w
                cumulative = 0
                idx = len(new_sol) - 1
                for i in range(len(new_sol)):
                    cumulative += weights[i]
                    if r <= cumulative:
                        idx = i
                        break
                m_rem, r_rem = new_sol[idx]
                new_sol.pop(idx)
                new_score -= 1.0 / m_rem
                valid_move = True
            elif move_type < 0.75:
                # Change residue of a random class
                if not new_sol:
                    continue
                idx = random.randint(0, len(new_sol) - 1)
                m_chg, old_r = new_sol[idx]
                for _ in range(10):
                    new_r = random.randint(0, m_chg - 1)
                    if new_r == old_r:
                        continue
                    if is_valid(m_chg, new_r, new_sol, skip_idx=idx):
                        new_sol[idx] = [m_chg, new_r]
                        valid_move = True
                        break
            elif move_type < 0.90:
                # Swap: remove one class, try to add another (different m)
                if not new_sol:
                    continue
                idx = random.randint(0, len(new_sol) - 1)
                m_old, r_old = new_sol[idx]
                new_sol.pop(idx)
                new_score -= 1.0 / m_old
                for _ in range(15):
                    m_new = random.randint(min_m, max_m)
                    if m_new == m_old:
                        continue
                    if any(m == m_new for m, _ in new_sol):
                        continue
                    r_new = random.randint(0, m_new - 1)
                    if is_valid(m_new, r_new, new_sol):
                        new_sol.append([m_new, r_new])
                        new_score += 1.0 / m_new
                        valid_move = True
                        break
            else:
                # 2-swap: remove two, add two
                if len(new_sol) < 2:
                    continue
                idx1 = random.randint(0, len(new_sol) - 1)
                idx2 = random.randint(0, len(new_sol) - 1)
                if idx1 == idx2:
                    continue
                m_old1, r_old1 = new_sol[idx1]
                m_old2, r_old2 = new_sol[idx2]
                # Remove both (handle index shift)
                if idx1 > idx2:
                    idx1 -= 1
                new_sol.pop(idx1)
                new_sol.pop(idx2)
                new_score -= 1.0 / m_old1 + 1.0 / m_old2
                
                added = 0
                for _ in range(20):
                    m_new = random.randint(min_m, max_m)
                    if m_new in (m_old1, m_old2):
                        continue
                    if any(m == m_new for m, _ in new_sol):
                        continue
                    r_new = random.randint(0, m_new - 1)
                    if is_valid(m_new, r_new, new_sol):
                        new_sol.append([m_new, r_new])
                        new_score += 1.0 / m_new
                        added += 1
                        if added == 2:
                            break
                
                if added == 2:
                    valid_move = True

            if not valid_move:
                continue

            delta = new_score - current_score
            if delta > 0 or (delta > -5.0 and random.random() < math.exp(delta / T)):
                current_sol = new_sol
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_sol = [list(x) for x in current_sol]

            T *= cooling

        return best_sol

    def initial_solution_csp(deadline):
        """
        Build initial solution using CSP with AC-3 pruning.
        Strategy: select all moduli, prune domains, then greedily assign.
        """
        all_moduli = list(range(min_m, max_m + 1))
        
        # Start with empty assignment, try to add all moduli
        # Use AC-3 to check feasibility
        selected = []
        remaining = list(all_moduli)
        
        # Sort by decreasing 1/m (smallest m first)
        remaining.sort(key=lambda m: 1.0 / m, reverse=True)
        
        for m in remaining:
            # Check if we can assign a residue to m given current selections
            found = False
            for r in range(m):
                if is_valid(m, r, selected):
                    selected.append([m, r])
                    found = True
                    break
            if not found:
                # Try with AC-3: see if any residue works after considering all constraints
                # This is already done above, so skip
                pass
            if time.time() > deadline:
                break
        
        return selected

    def initial_solution_prime_aware(deadline):
        """
        Build initial solution using prime-power structure.
        For each prime, try to use the largest power in [20,200] first.
        """
        selected = []
        selected_set = set()
        
        # Group moduli by their "core" (product of distinct prime factors)
        # Priority: moduli with fewer prime factors are easier to place
        moduli_by_parity = []
        for m in range(min_m, max_m + 1):
            pf = prime_factors[m]
            moduli_by_parity.append((m, len(pf), pf))
        
        # Sort by number of prime factors (fewer = easier), then by m
        moduli_by_parity.sort(key=lambda x: (x[1], x[0]))
        
        for m, n_pf, pf in moduli_by_parity:
            for r in range(m):
                if is_valid(m, r, selected):
                    selected.append([m, r])
                    break
            if time.time() > deadline:
                break
        
        return selected

    # Run initial solution (CSP-based)
    best_solution = initial_solution_csp(deadline - 30)
    best_score = calc_score(best_solution)
    
    # Also try prime-aware initial
    if time.time() < deadline - 30:
        sol_pa = initial_solution_prime_aware(deadline - 30)
        score_pa = calc_score(sol_pa)
        if score_pa > best_score:
            best_score = score_pa
            best_solution = sol_pa

    # Local search polish first (fast, deterministic)
    if time.time() < deadline - 20:
        polished, polished_score = local_search_polish(best_solution, deadline - 20)
        if polished_score > best_score:
            best_score = polished_score
            best_solution = polished

    # Run simulated annealing from polished solution
    sa_deadline = deadline - 10
    if time.time() < sa_deadline:
        result = simulated_annealing(best_solution, sa_deadline)
        score = calc_score(result)
        if score > best_score:
            best_score = score
            best_solution = result

    # Final local search polish
    if time.time() < deadline - 2:
        polished, polished_score = local_search_polish(best_solution, deadline - 2)
        if polished_score > best_score:
            best_score = polished_score
            best_solution = polished

    # Try a few more restarts if time permits
    restart_deadline = deadline - 5
    for _ in range(3):
        if time.time() > restart_deadline:
            break
        # Alternate between CSP and prime-aware initializations
        if _ % 2 == 0:
            sol = initial_solution_csp(deadline - 25)
        else:
            sol = initial_solution_prime_aware(deadline - 25)
        if time.time() < deadline - 20:
            sol, _ = local_search_polish(sol, deadline - 20)
        if time.time() < deadline - 10:
            sol = simulated_annealing(sol, deadline - 10)
        score = calc_score(sol)
        if score > best_score:
            best_score = score
            best_solution = sol

    return best_solution

# Test locally
if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    score = sum(1.0 / m for m, _ in res)
    print(f"Score: {score}")
    print(f"Time: {end - start:.2f}s")
    print(f"Classes: {len(res)}")
    # Verify validity
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
