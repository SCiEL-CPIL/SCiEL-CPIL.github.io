import math
import time
import random

def construct():
    # Deadline guard: 150s limit, use 140s to be safe
    deadline = time.time() + 140

    max_m = 200
    min_m = 20

    # Precompute gcd table
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

    def greedy_fill(selected, deadline):
        """Try to add more classes to the selected list."""
        for m in range(min_m, max_m + 1):
            if any(m == m_sel for m_sel, _ in selected):
                continue
            for r in range(m):
                if is_valid(m, r, selected):
                    selected.append([m, r])
                    break
            if time.time() > deadline:
                break
        return selected

    def local_polish(sol, deadline):
        """Exhaustively try all possible additions and residue changes."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try adding every possible (m, r) pair
            for m in range(min_m, max_m + 1):
                if any(m == m_sel for m_sel, _ in sol):
                    continue
                for r in range(m):
                    if is_valid(m, r, sol):
                        sol.append([m, r])
                        improved = True
                        break
                if time.time() > deadline:
                    return sol
            
            # Try changing residue of each class
            for idx in range(len(sol)):
                m_chg, old_r = sol[idx]
                for new_r in range(m_chg):
                    if new_r == old_r:
                        continue
                    if is_valid(m_chg, new_r, sol, skip_idx=idx):
                        # Check if this change enables adding more classes
                        new_sol = [list(x) for x in sol]
                        new_sol[idx] = [m_chg, new_r]
                        # Try to add more after this change
                        added = False
                        for m_add in range(min_m, max_m + 1):
                            if any(m_add == m_sel for m_sel, _ in new_sol):
                                continue
                            for r_add in range(m_add):
                                if is_valid(m_add, r_add, new_sol):
                                    new_sol.append([m_add, r_add])
                                    added = True
                                    break
                            if added:
                                break
                        if added:
                            sol = new_sol
                            improved = True
                            break
                if time.time() > deadline:
                    return sol
        
        return sol

    def simulated_annealing(initial_solution, deadline):
        current_sol = [list(x) for x in initial_solution]
        current_score = calc_score(current_sol)
        best_sol = [list(x) for x in current_sol]
        best_score = current_score

        T = 0.3
        cooling = 0.9997
        min_T = 1e-8

        while time.time() < deadline:
            if T < min_T:
                break

            move_type = random.random()
            new_sol = [list(x) for x in current_sol]
            new_score = current_score
            valid_move = False

            if move_type < 0.55:
                # Add a modulus - prefer smaller moduli for higher contribution
                # Choose m biased towards smaller values
                m_add = random.randint(min_m, max_m)
                if random.random() < 0.4:
                    m_add = random.randint(min_m, min_m + 25)
                if any(m == m_add for m, _ in new_sol):
                    continue
                # Try several random residues
                for _ in range(10):
                    r_add = random.randint(0, m_add - 1)
                    if is_valid(m_add, r_add, new_sol):
                        new_sol.append([m_add, r_add])
                        new_score += 1.0 / m_add
                        valid_move = True
                        break
            elif move_type < 0.80:
                # Remove a modulus - prefer removing large moduli (low contribution)
                if not new_sol:
                    continue
                # Weight towards larger moduli
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
            elif move_type < 0.92:
                # Change residue of a random class
                if not new_sol:
                    continue
                idx = random.randint(0, len(new_sol) - 1)
                m_chg, old_r = new_sol[idx]
                for _ in range(12):
                    new_r = random.randint(0, m_chg - 1)
                    if new_r == old_r:
                        continue
                    if is_valid(m_chg, new_r, new_sol, skip_idx=idx):
                        new_sol[idx] = [m_chg, new_r]
                        valid_move = True
                        break
            else:
                # Swap: remove one class, try to add another (different m)
                if not new_sol:
                    continue
                idx = random.randint(0, len(new_sol) - 1)
                m_old, r_old = new_sol[idx]
                new_sol.pop(idx)
                new_score -= 1.0 / m_old

                # Try to add a new class with different m
                for _ in range(18):
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

            if not valid_move:
                continue

            # Acceptance criterion
            delta = new_score - current_score
            if delta > 0 or (delta > -5.0 and random.random() < math.exp(delta / T)):
                current_sol = new_sol
                current_score = new_score

                if current_score > best_score:
                    best_score = current_score
                    best_sol = [list(x) for x in current_sol]

            T *= cooling

        # Final local polish
        best_sol = local_polish(best_sol, deadline)
        return best_sol

    def initial_solution(deadline):
        """Build initial solution: add all coprime-friendly classes, then fill by prime powers."""
        selected = []
        selected_set = set()

        # Phase 1: Add classes with moduli that have no prime factor in common with existing
        for m in range(min_m, max_m + 1):
            for r in range(m):
                if is_valid(m, r, selected):
                    selected.append([m, r])
                    break
            if time.time() > deadline:
                break

        # Phase 2: Fill remaining - for each prime, try to add at its largest power
        # Then try to add at smaller moduli
        selected = greedy_fill(selected, deadline)

        return selected

    # Run initial solution
    best_solution = initial_solution(deadline - 15)
    best_score = calc_score(best_solution)

    # Run simulated annealing from initial solution
    result = simulated_annealing(best_solution, deadline - 5)
    score = calc_score(result)
    if score > best_score:
        best_score = score
        best_solution = result

    # Try a few more restarts if time permits
    restart_deadline = deadline - 10
    for _ in range(8):
        if time.time() > restart_deadline:
            break
        sol = initial_solution(deadline - 8)
        sol = simulated_annealing(sol, deadline - 5)
        score = calc_score(sol)
        if score > best_score:
            best_score = score
            best_solution = sol

    # Final polish on the best solution
    best_solution = local_polish(best_solution, deadline - 1)

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
    # Print moduli for inspection
    moduli = sorted([m for m, _ in res])
    print(f"Moduli count: {len(moduli)}")
    print(f"Smallest 20 moduli: {moduli[:20]}")
    print(f"Largest 20 moduli: {moduli[-20:]}")
