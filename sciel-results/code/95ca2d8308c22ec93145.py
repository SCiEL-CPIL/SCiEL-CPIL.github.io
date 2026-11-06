import math
import time
import random

def construct():
    deadline = time.time() + 140
    max_m = 200
    min_m = 20

    # Precompute gcd table
    gcd_table = [[0] * (max_m + 1) for _ in range(max_m + 1)]
    for i in range(min_m, max_m + 1):
        for j in range(min_m, max_m + 1):
            gcd_table[i][j] = math.gcd(i, j)

    def is_valid(m, r, selected, skip_idx=None):
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

    def branch_and_bound(selected, deadline):
        """Branch and bound on remaining moduli with fractional knapsack upper bound."""
        current_sol = [list(x) for x in selected]
        current_score = calc_score(current_sol)
        best_sol = [list(x) for x in current_sol]
        best_score = current_score

        selected_moduli = set(m for m, _ in current_sol)
        remaining_moduli = [m for m in range(min_m, max_m + 1) if m not in selected_moduli]
        remaining_moduli.sort(reverse=True)  # larger moduli first (lower weight)

        def try_add(m, sol, score):
            """Try to add modulus m to sol, return (success, new_sol, new_score)."""
            for r in range(m):
                if is_valid(m, r, sol):
                    new_sol = sol + [[m, r]]
                    new_score = score + 1.0 / m
                    return True, new_sol, new_score
            return False, sol, score

        def upper_bound(sol, score, remaining):
            """Fractional knapsack upper bound: assume all remaining moduli can be added."""
            ub = score
            for m in remaining:
                ub += 1.0 / m
            return ub

        def bb_search(sol, score, remaining, depth, deadline):
            nonlocal best_sol, best_score

            if time.time() > deadline:
                return

            if not remaining:
                if score > best_score:
                    best_score = score
                    best_sol = [list(x) for x in sol]
                return

            # Upper bound check
            ub = upper_bound(sol, score, remaining)
            if ub <= best_score:
                return

            # Try to add the first remaining modulus
            m = remaining[0]
            rest = remaining[1:]

            # Branch 1: try to add m
            success, new_sol, new_score = try_add(m, sol, score)
            if success:
                if new_score > best_score:
                    best_score = new_score
                    best_sol = [list(x) for x in new_sol]
                bb_search(new_sol, new_score, rest, depth + 1, deadline)

            # Branch 2: skip m
            bb_search(sol, score, rest, depth + 1, deadline)

        # Use a limited depth to keep runtime manageable
        bb_search(current_sol, current_score, remaining_moduli[:30], 0, deadline)
        return best_sol

    def local_search_polish(selected, deadline):
        current_sol = [list(x) for x in selected]
        current_score = calc_score(current_sol)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding each missing modulus
            for m in range(min_m, max_m + 1):
                if any(m == m_sel for m_sel, _ in current_sol):
                    continue
                for r in range(m):
                    if is_valid(m, r, current_sol):
                        current_sol.append([m, r])
                        current_score += 1.0 / m
                        improved = True
                        break
            # Try changing residues to enable additions
            for idx in range(len(current_sol)):
                m_chg, old_r = current_sol[idx]
                for new_r in range(m_chg):
                    if new_r == old_r:
                        continue
                    if is_valid(m_chg, new_r, current_sol, skip_idx=idx):
                        current_sol[idx] = [m_chg, new_r]
                        added = False
                        for m_add in range(min_m, max_m + 1):
                            if any(m_add == m_sel for m_sel, _ in current_sol):
                                continue
                            for r_add in range(m_add):
                                if is_valid(m_add, r_add, current_sol):
                                    current_sol.append([m_add, r_add])
                                    current_score += 1.0 / m_add
                                    added = True
                                    break
                            if added:
                                break
                        if not added:
                            current_sol[idx] = [m_chg, old_r]
                        else:
                            improved = True
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

            if move_type < 0.50:
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
            elif move_type < 0.75:
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
            elif move_type < 0.90:
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
            else:
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

    def initial_solution(deadline):
        selected = []
        for m in range(min_m, max_m + 1):
            for r in range(m):
                if is_valid(m, r, selected):
                    selected.append([m, r])
                    break
            if time.time() > deadline:
                break
        selected = greedy_fill(selected, deadline)
        return selected

    # Run initial solution
    best_solution = initial_solution(deadline - 20)
    best_score = calc_score(best_solution)

    # Branch and bound
    if time.time() < deadline - 10:
        bb_result = branch_and_bound(best_solution, deadline - 5)
        bb_score = calc_score(bb_result)
        if bb_score > best_score:
            best_score = bb_score
            best_solution = bb_result

    # Simulated annealing
    if time.time() < deadline - 10:
        sa_result = simulated_annealing(best_solution, deadline - 5)
        sa_score = calc_score(sa_result)
        if sa_score > best_score:
            best_score = sa_score
            best_solution = sa_result

    # Local search polish
    if time.time() < deadline - 3:
        polished, polished_score = local_search_polish(best_solution, deadline - 1)
        if polished_score > best_score:
            best_score = polished_score
            best_solution = polished

    # Restart if time permits
    restart_deadline = deadline - 2
    for _ in range(3):
        if time.time() > restart_deadline:
            break
        sol = initial_solution(deadline - 10)
        sol = simulated_annealing(sol, deadline - 5)
        score = calc_score(sol)
        if score > best_score:
            best_score = score
            best_solution = sol

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
        for j in range(i + 1, len(res)):
            m1, r1 = res[i]
            m2, r2 = res[j]
            g = math.gcd(m1, m2)
            if r1 % g == r2 % g:
                valid = False
                print(f"Invalid: {res[i]} and {res[j]}")
    print(f"Valid: {valid}")
