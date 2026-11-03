# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def _ball_indices(q, n, radius):
    total = q ** n
    weights = [q ** (n - 1 - pos) for pos in range(n)]
    balls = np.empty(total, dtype=object)
    for idx in range(total):
        covered = [idx]
        for pos in range(n):
            weight = weights[pos]
            digit = (idx // weight) % q
            base = idx - digit * weight
            for val in range(q):
                if val != digit:
                    covered.append(base + val * weight)
        balls[idx] = np.array(covered, dtype=np.int64)
    return balls


def _valid_cover(cover):
    return not np.any(cover == 0)


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145
    total = q ** n
    balls = _ball_indices(q, n, radius)

    def greedy_init():
        uncovered = np.ones(total, dtype=np.int32)
        chosen = []
        chosen_set = set()
        while uncovered.any():
            if time.time() > deadline - 20:
                return list(range(total))
            best_idx = -1
            best_count = -1
            for idx in range(total):
                if idx in chosen_set:
                    continue
                cnt = np.count_nonzero(uncovered[balls[idx]])
                if cnt > best_count:
                    best_count = cnt
                    best_idx = idx
            if best_idx == -1:
                break
            chosen.append(best_idx)
            chosen_set.add(best_idx)
            uncovered[balls[best_idx]] = 0
        return chosen

    init = greedy_init()
    best_solution = list(init)
    best_solution_len = len(best_solution)

    def solution_to_cover(sol):
        cover = np.zeros(total, dtype=np.int32)
        for ci in sol:
            cover[balls[ci]] += 1
        return cover

    def try_add(sol, cover, cand, sol_set):
        cover[balls[cand]] += 1
        if _valid_cover(cover):
            return cover
        cover[balls[cand]] -= 1
        return None

    def try_remove(sol, cover, pos):
        ci = sol[pos]
        ball = balls[ci]
        if np.any(cover[ball] == 1):
            return None
        cover[ball] -= 1
        return cover

    def try_swap(sol, cover, pos, cand):
        ci = sol[pos]
        ball_i = balls[ci]
        ball_c = balls[cand]
        cover[ball_i] -= 1
        cover[ball_c] += 1
        if _valid_cover(cover):
            return cover
        cover[ball_c] -= 1
        cover[ball_i] += 1
        return None

    def anneal(sol, sol_set, max_time):
        cover = solution_to_cover(sol)
        local_best = list(sol)
        local_best_len = len(sol)
        T = 1.0
        cool = 0.999
        min_T = 1e-6
        no_improve = 0
        while time.time() < max_time:
            r = random.random()
            if r < 0.4:
                cand = random.randrange(total)
                if cand in sol_set:
                    continue
                new_cover = try_add(sol, cover, cand, sol_set)
                if new_cover is not None:
                    sol.append(cand)
                    sol_set.add(cand)
                    cover = new_cover
                    no_improve += 1
            elif r < 0.7:
                if len(sol) <= 1:
                    continue
                pos = random.randrange(len(sol))
                new_cover = try_remove(sol, cover, pos)
                if new_cover is not None:
                    removed = sol.pop(pos)
                    sol_set.discard(removed)
                    cover = new_cover
                    if len(sol) < local_best_len:
                        local_best = list(sol)
                        local_best_len = len(sol)
                        no_improve = 0
                    else:
                        no_improve += 1
            else:
                if len(sol) <= 1:
                    continue
                pos = random.randrange(len(sol))
                cand = random.randrange(total)
                if cand in sol_set:
                    continue
                new_cover = try_swap(sol, cover, pos, cand)
                if new_cover is not None:
                    old = sol[pos]
                    sol[pos] = cand
                    sol_set.discard(old)
                    sol_set.add(cand)
                    cover = new_cover
                    no_improve += 1

            if no_improve > 200:
                if len(sol) > 1:
                    pos = random.randrange(len(sol))
                    removed = sol.pop(pos)
                    sol_set.discard(removed)
                    cover = solution_to_cover(sol)
                    while not _valid_cover(cover):
                        if time.time() > max_time:
                            break
                        uncov = np.where(cover == 0)[0]
                        if len(uncov) == 0:
                            break
                        target = uncov[random.randrange(len(uncov))]
                        added = False
                        for _ in range(200):
                            cand = random.randrange(total)
                            if cand in sol_set:
                                continue
                            if target in balls[cand]:
                                sol.append(cand)
                                sol_set.add(cand)
                                cover[balls[cand]] += 1
                                added = True
                                break
                        if not added:
                            for cand in range(total):
                                if cand in sol_set:
                                    continue
                                if target in balls[cand]:
                                    sol.append(cand)
                                    sol_set.add(cand)
                                    cover[balls[cand]] += 1
                                    break
                    no_improve = 0
                    T = 1.0
                else:
                    no_improve = 0

            T *= cool
            if T < min_T:
                T = min_T

        return local_best, local_best_len

    t0 = time.time()
    starts = 0
    while time.time() < deadline - 5:
        if starts == 0:
            sol = list(init)
        else:
            sol = list(best_solution)
            for _ in range(min(5, len(sol))):
                if random.random() < 0.5 and len(sol) > 1:
                    pos = random.randrange(len(sol))
                    removed = sol.pop(pos)
                else:
                    cand = random.randrange(total)
                    if cand not in sol:
                        sol.append(cand)
        sol = list(dict.fromkeys(sol))
        sol_set = set(sol)
        cover = solution_to_cover(sol)
        while not _valid_cover(cover):
            if time.time() > deadline - 5:
                break
            uncov = np.where(cover == 0)[0]
            if len(uncov) == 0:
                break
            target = uncov[0]
            for cand in range(total):
                if cand in sol_set:
                    continue
                if target in balls[cand]:
                    sol.append(cand)
                    sol_set.add(cand)
                    cover[balls[cand]] += 1
                    break
        max_time = min(deadline - 3, time.time() + 30)
        best_sol, best_len = anneal(sol, sol_set, max_time)
        if best_len < best_solution_len:
            best_solution = best_sol
            best_solution_len = best_len
        starts += 1
        if best_solution_len <= 186:
            break

    cover = solution_to_cover(best_solution)
    changed = True
    while changed and time.time() < deadline - 2:
        changed = False
        for i in range(len(best_solution)):
            ci = best_solution[i]
            ball = balls[ci]
            if not np.any(cover[ball] == 1):
                cover[ball] -= 1
                best_solution.pop(i)
                changed = True
                break

    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    codewords = [idx_to_word(ci) for ci in best_solution]
    return codewords

# EVOLVE-BLOCK-END
