import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    moduli_list = list(range(20, 201))
    gcd_table = {}
    for i, m1 in enumerate(moduli_list):
        for m2 in moduli_list[i:]:
            g = math.gcd(m1, m2)
            gcd_table[(m1, m2)] = g
            gcd_table[(m2, m1)] = g

    def is_valid_pair(m1, r1, m2, r2):
        g = gcd_table.get((m1, m2), math.gcd(m1, m2))
        return (r1 - r2) % g != 0

    def find_valid_residue(m, selected, randomize=True):
        if not selected:
            return 0
        forbidden = set()
        for m_prime, r_prime in selected:
            g = gcd_table.get((m, m_prime), math.gcd(m, m_prime))
            target = r_prime % g
            r = target
            while r < m:
                forbidden.add(r)
                r += g
        if len(forbidden) >= m:
            return None
        if randomize:
            candidates = [r for r in range(m) if r not in forbidden]
            random.shuffle(candidates)
            return candidates[0] if candidates else None
        else:
            for r in range(m):
                if r not in forbidden:
                    return r
            return None

    def greedy_build(order):
        selected = []
        for m in order:
            if time.time() > deadline:
                break
            r = find_valid_residue(m, selected, randomize=True)
            if r is not None:
                selected.append([m, r])
        return selected

    def try_add_missing(solution, max_attempts=200):
        selected_moduli = set(m for m, r in solution)
        missing = [m for m in range(20, 201) if m not in selected_moduli]
        missing.sort()
        added = 0
        for m in missing:
            if time.time() > deadline:
                break
            if added >= max_attempts:
                break
            r = find_valid_residue(m, solution, randomize=True)
            if r is not None:
                solution.append([m, r])
                selected_moduli.add(m)
                added += 1
        return solution

    def local_search_add(solution):
        selected_moduli = set(m for m, r in solution)
        missing = [m for m in range(20, 201) if m not in selected_moduli]
        missing.sort()
        improved = True
        iterations = 0
        while improved and iterations < 100 and time.time() < deadline - 5:
            improved = False
            iterations += 1
            for m in missing[:]:
                if m in selected_moduli:
                    continue
                r = find_valid_residue(m, solution, randomize=True)
                if r is not None:
                    solution.append([m, r])
                    selected_moduli.add(m)
                    missing.remove(m)
                    improved = True
                    break
        return solution

    def local_search_swap(solution, max_swaps=50):
        selected_moduli = set(m for m, r in solution)
        missing = [m for m in range(20, 201) if m not in selected_moduli]
        missing.sort()
        for _ in range(max_swaps):
            if time.time() > deadline - 5:
                break
            best_gain = 0
            best_swap = None
            for idx in range(len(solution)):
                m_remove, r_remove = solution[idx]
                remaining = solution[:idx] + solution[idx+1:]
                for m_add in missing[:30]:
                    r_add = find_valid_residue(m_add, remaining, randomize=False)
                    if r_add is not None:
                        gain = 1.0/m_add - 1.0/m_remove
                        if gain > best_gain:
                            best_gain = gain
                            best_swap = (idx, m_remove, r_remove, m_add, r_add)
            if best_swap is not None:
                idx, m_remove, r_remove, m_add, r_add = best_swap
                solution.pop(idx)
                solution.append([m_add, r_add])
                selected_moduli.discard(m_remove)
                selected_moduli.add(m_add)
                if m_add in missing:
                    missing.remove(m_add)
                if m_remove not in missing:
                    missing.append(m_remove)
                    missing.sort()
            else:
                break
        return solution

    def score_solution(solution):
        if not solution:
            return 0.0
        return sum(1.0 / m for m, r in solution)

    best_result = []
    best_score = 0.0
    all_moduli = list(range(20, 201))

    num_restarts = 0
    while time.time() < deadline - 10:
        num_restarts += 1
        if num_restarts % 3 == 1:
            order = all_moduli[:]
        elif num_restarts % 3 == 2:
            order = sorted(all_moduli, key=lambda m: (-1.0/m, m))
        else:
            order = all_moduli[:]
            random.shuffle(order)

        solution = greedy_build(order)
        solution = try_add_missing(solution, max_attempts=100)
        solution = local_search_add(solution)
        solution = local_search_swap(solution, max_swaps=30)

        current_score = score_solution(solution)
        if current_score > best_score:
            best_score = current_score
            best_result = [list(x) for x in solution]

        if time.time() > deadline - 5:
            break

    if best_result:
        for i in range(len(best_result)):
            for j in range(i + 1, len(best_result)):
                m1, r1 = best_result[i]
                m2, r2 = best_result[j]
                if not is_valid_pair(m1, r1, m2, r2):
                    return [[20, 0]]

    return best_result
