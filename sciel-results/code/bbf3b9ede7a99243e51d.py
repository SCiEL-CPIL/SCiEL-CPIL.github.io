"""Simulated annealing for ternary covering code K_3(7,1)."""
import random
import time
import numpy as np


def construct(q, n, radius):
    deadline = time.time() + 110  # 10s safety margin before 120s limit
    rng = random.Random(42)
    np_rng = np.random.default_rng(42)

    total_words = q ** n  # 2187
    # Precompute all words as indices 0..2186
    # Each word is a base-q number; we'll use numpy for speed

    # Convert index to word and back
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx % q
            idx //= q
        return w

    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    # Precompute neighbors: for each word, list all words at distance <= radius
    # For radius=1, n=7, q=3: each word has 1 + 7*2 = 15 neighbors (itself + 14)
    neighbors = [[] for _ in range(total_words)]
    for idx in range(total_words):
        w = idx_to_word(idx)
        neighbors[idx].append(idx)
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                nw = w[:]
                nw[pos] = val
                neighbors[idx].append(word_to_idx(nw))

    # Precompute for each codeword candidate, which words it covers
    # covers[candidate_idx] = list of word indices covered
    covers = [neighbors[i] for i in range(total_words)]

    # Greedy set cover to get initial solution
    # uncovered: set of word indices not yet covered
    uncovered = set(range(total_words))
    solution = []
    coverage_count = np.zeros(total_words, dtype=np.int32)  # how many times each word is covered

    # For greedy, we need to find the word that covers the most uncovered words
    # We'll use a priority approach
    while uncovered:
        best_candidate = -1
        best_gain = -1
        # Sample candidates to speed up (or check all)
        # For 2187 words, checking all is fast enough
        for cand in range(total_words):
            gain = 0
            for w in covers[cand]:
                if coverage_count[w] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_candidate = cand
            elif gain == best_gain and gain > 0:
                # Random tie-breaking
                if rng.random() < 0.1:
                    best_candidate = cand

        if best_candidate == -1 or best_gain == 0:
            break

        solution.append(best_candidate)
        for w in covers[best_candidate]:
            coverage_count[w] += 1
            if coverage_count[w] == 1:
                uncovered.discard(w)

        if time.time() > deadline:
            break

    # Remove any redundant codewords from greedy solution
    def remove_redundant(sol, cov_cnt):
        changed = True
        while changed:
            changed = False
            for i in range(len(sol)):
                cand = sol[i]
                # Check if removing this codeword leaves all words still covered
                can_remove = True
                for w in covers[cand]:
                    if cov_cnt[w] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    for w in covers[cand]:
                        cov_cnt[w] -= 1
                    sol.pop(i)
                    changed = True
                    break
        return sol

    remove_redundant(solution, coverage_count)
    best_solution = solution[:]
    best_size = len(solution)

    # Simulated annealing
    current_solution = solution[:]
    current_cov = coverage_count.copy()
    T = 5.0
    T_min = 0.01
    cooling = 0.995

    # Precompute: for each candidate, its covered words
    # We'll do moves: add, remove, swap

    while T > T_min and time.time() < deadline:
        # Pick a move type
        move_type = rng.random()
        accepted = False
        old_size = len(current_solution)

        if move_type < 0.3:
            # Remove a codeword
            if len(current_solution) > 1:
                i = rng.randrange(len(current_solution))
                cand = current_solution[i]
                # Check if removal is valid (all words still covered)
                valid = True
                for w in covers[cand]:
                    if current_cov[w] <= 1:
                        valid = False
                        break
                if valid:
                    # Accept
                    for w in covers[cand]:
                        current_cov[w] -= 1
                    current_solution.pop(i)
                    accepted = True
                    new_size = len(current_solution)
                    if new_size < best_size:
                        best_size = new_size
                        best_solution = current_solution[:]
        elif move_type < 0.6:
            # Add a codeword
            # Pick a random candidate not in solution
            cand = rng.randrange(total_words)
            if cand not in current_solution:
                # Compute gain: how many new words covered
                gain = 0
                for w in covers[cand]:
                    if current_cov[w] == 0:
                        gain += 1
                # Cost: +1 codeword, benefit: gain new words covered
                # Delta in objective: we want to minimize size, so adding increases size by 1
                # But it might allow future removals. Use simple acceptance.
                # Accept with probability based on temperature
                delta = 1.0  # size increases by 1
                if gain > 0:
                    # This helps coverage, might enable removals later
                    # Accept more readily
                    p = min(1.0, np.exp(-delta / T) * (1 + gain * 0.1))
                else:
                    p = np.exp(-delta / T)
                if rng.random() < p:
                    current_solution.append(cand)
                    for w in covers[cand]:
                        current_cov[w] += 1
                    accepted = True
        else:
            # Swap: remove one, add another
            if len(current_solution) > 1:
                i = rng.randrange(len(current_solution))
                remove_cand = current_solution[i]
                add_cand = rng.randrange(total_words)
                if add_cand not in current_solution:
                    # Check if after swap, all words are still covered
                    # Words only covered by remove_cand must be covered by add_cand
                    valid = True
                    for w in covers[remove_cand]:
                        if current_cov[w] <= 1:
                            # This word is only covered by remove_cand
                            # Check if add_cand covers it
                            if add_cand not in [c for c in current_solution if w in covers[c]] and \
                               w not in covers[add_cand]:
                                valid = False
                                break
                    if valid:
                        # Perform swap
                        for w in covers[remove_cand]:
                            current_cov[w] -= 1
                        current_solution[i] = add_cand
                        for w in covers[add_cand]:
                            current_cov[w] += 1
                        accepted = True
                        if len(current_solution) < best_size:
                            best_size = len(current_solution)
                            best_solution = current_solution[:]
                    else:
                        # Try to accept with some probability if close
                        # Count how many words would become uncovered
                        uncovered_count = 0
                        for w in covers[remove_cand]:
                            if current_cov[w] <= 1 and w not in covers[add_cand]:
                                uncovered_count += 1
                        if uncovered_count <= 2:
                            delta = 0.0  # size stays same
                            p = np.exp(-uncovered_count * 0.5 / T)
                            if rng.random() < p:
                                # Temporarily allow uncovered words (they'll be fixed later)
                                for w in covers[remove_cand]:
                                    current_cov[w] -= 1
                                current_solution[i] = add_cand
                                for w in covers[add_cand]:
                                    current_cov[w] += 1
                                accepted = True

        # Cool down
        T *= cooling

        # Periodically try to remove redundant codewords
        if not accepted and rng.random() < 0.1:
            remove_redundant(current_solution, current_cov)
            if len(current_solution) < best_size:
                best_size = len(current_solution)
                best_solution = current_solution[:]

        # If solution became invalid (some words uncovered), try to fix
        # Check coverage
        if rng.random() < 0.05:
            min_cov = np.min(current_cov)
            if min_cov < 1:
                # Some words uncovered, add a codeword that covers them
                uncovered_words = [i for i in range(total_words) if current_cov[i] == 0]
                if uncovered_words:
                    # Find candidate that covers most uncovered words
                    best_cand = -1
                    best_gain = -1
                    for cand in range(total_words):
                        gain = 0
                        for w in covers[cand]:
                            if current_cov[w] == 0:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_cand = cand
                    if best_cand >= 0 and best_gain > 0:
                        current_solution.append(best_cand)
                        for w in covers[best_cand]:
                            current_cov[w] += 1

    # Final cleanup: remove redundant
    remove_redundant(best_solution, np.array([0]*total_words, dtype=np.int32))
    # Recompute coverage for best_solution to verify
    final_cov = np.zeros(total_words, dtype=np.int32)
    for cand in best_solution:
        for w in covers[cand]:
            final_cov[w] += 1
    # Remove any that are truly redundant
    changed = True
    while changed:
        changed = False
        for i in range(len(best_solution)):
            cand = best_solution[i]
            can_remove = True
            for w in covers[cand]:
                if final_cov[w] <= 1:
                    can_remove = False
                    break
            if can_remove:
                for w in covers[cand]:
                    final_cov[w] -= 1
                best_solution.pop(i)
                changed = True
                break

    # Convert back to list of words
    result = [idx_to_word(idx) for idx in best_solution]
    return result
