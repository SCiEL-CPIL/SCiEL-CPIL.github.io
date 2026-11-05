import time
import random
import sys
import itertools
import numpy as np

def construct(q, n, radius):
    # Handle the specific task: K_3(6,1)
    # The evaluator passes q, n, radius.
    # If not the specific case, fall back to a simple greedy or direct sum.
    if not (q == 3 and n == 6 and radius == 1):
        # Fallback: Direct sum of tetracode if q=3, n>=4
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Generic greedy for other cases (slow, but safe)
            all_words = list(itertools.product(range(q), repeat=n))
            code = []
            covered = set()
            for w in all_words:
                if w not in covered:
                    code.append(list(w))
                    for i in range(n):
                        for v in range(q):
                            if v == w[i]:
                                continue
                            new_w = list(w)
                            new_w[i] = v
                            covered.add(tuple(new_w))
                    covered.add(w)
            return code

    # Specific case: q=3, n=6, radius=1
    start_time = time.time()
    deadline = 140.0  # 10s safety margin

    total = q ** n
    powers = [q ** p for p in range(n)]

    # Precompute neighbors: neighbors[i] is a list of indices j such that dist(i,j) <= 1
    # There are 1 + n*(q-1) = 1 + 6*2 = 13 neighbors per word.
    neighbors = [[] for _ in range(total)]
    for i in range(total):
        digits = [(i // powers[p]) % q for p in range(n)]
        neighbors[i].append(i)
        for p in range(n):
            for alt in range(q):
                if alt != digits[p]:
                    new_i = i + (alt - digits[p]) * powers[p]
                    neighbors[i].append(new_i)

    # Convert to sets for fast membership testing
    neighbor_sets = [set(nei) for nei in neighbors]

    # Phase 1: Greedy Set Cover to get an initial code
    uncovered = np.ones(total, dtype=np.int8)
    code_indices = set()
    counts = np.zeros(total, dtype=np.int32)

    while uncovered.sum() > 0:
        if time.time() > deadline:
            break

        best_candidate = -1
        best_score = 0

        for c in range(total):
            score = 0
            for u in neighbors[c]:
                if uncovered[u]:
                    score += 1
            if score > best_score:
                best_score = score
                best_candidate = c

        if best_score == 0:
            break

        code_indices.add(best_candidate)
        for u in neighbors[best_candidate]:
            counts[u] += 1
            uncovered[u] = 0

    # Check if we have a valid cover
    if uncovered.sum() != 0:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    best_size = len(code_indices)
    best_code = list(code_indices)

    def can_remove(c_idx):
        for nb in neighbors[c_idx]:
            if counts[nb] <= 1:
                return False
        return True

    def get_max_removability_score():
        max_score = 0
        for c in code_indices:
            min_count = min(counts[nb] for nb in neighbors[c])
            if min_count > max_score:
                max_score = min_count
        return max_score

    # Local Search Loop with Simulated Annealing
    T = 1.0
    cooling_rate = 0.9995

    # Helper for smooth surrogate: log-sum-exp based soft-min
    def soft_min(values, beta=10.0):
        if len(values) == 0:
            return 0
        max_v = max(values)
        if max_v == 0:
            return 0
        exp_vals = [np.exp(beta * (v - max_v)) for v in values]
        log_sum = np.log(sum(exp_vals))
        return max_v - np.log(len(exp_vals)) / beta

    while time.time() < deadline:
        # Phase 1: Greedy Removal
        removed_any = False
        for c in list(code_indices):
            if can_remove(c):
                for nb in neighbors[c]:
                    counts[nb] -= 1
                code_indices.remove(c)
                removed_any = True
                if len(code_indices) < best_size:
                    best_size = len(code_indices)
                    best_code = list(code_indices)
                break

        if removed_any:
            continue

        # Phase 2: Swap with Simulated Annealing
        current_score = get_max_removability_score()

        # Identify critical words
        critical_words = [u for u in range(total) if counts[u] == 1]

        if not critical_words:
            break

        found_improvement = False
        code_list = list(code_indices)

        for c_out in code_list:
            if time.time() > deadline:
                break

            # Find promising c_in candidates
            # Score c_in by how many critical words it covers that c_out doesn't
            c_out_neighbors_set = neighbor_sets[c_out]
            candidate_scores = []
            for c_in in range(total):
                if c_in in code_indices:
                    continue
                # Count critical words covered by c_in but not by c_out
                score = 0
                for u in neighbors[c_in]:
                    if counts[u] == 1 and u not in c_out_neighbors_set:
                        score += 1
                if score > 0:
                    candidate_scores.append((score, c_in))

            if not candidate_scores:
                continue

            # Sort by score and take top candidates
            candidate_scores.sort(reverse=True)
            top_candidates = [c for _, c in candidate_scores[:30]]

            # Shuffle for diversity
            random.shuffle(top_candidates)

            for c_in in top_candidates:
                if time.time() > deadline:
                    break

                # Check validity of swap
                valid = True
                for u in neighbors[c_out]:
                    if u not in neighbor_sets[c_in]:
                        if counts[u] < 2:
                            valid = False
                            break

                if not valid:
                    continue

                # Apply swap temporarily
                for nb in neighbors[c_out]:
                    counts[nb] -= 1
                for nb in neighbors[c_in]:
                    counts[nb] += 1
                code_indices.remove(c_out)
                code_indices.add(c_in)

                new_score = get_max_removability_score()

                # Use smooth surrogate for acceptance
                # Calculate a smooth version of the min-count landscape
                # For the swapped-in codeword, compute soft-min of its neighbor counts
                soft_curr = soft_min([counts[nb] for nb in neighbors[c_out]], beta=5.0)
                soft_new = soft_min([counts[nb] for nb in neighbors[c_in]], beta=5.0)

                accept = False
                if new_score > current_score:
                    accept = True
                elif new_score == current_score:
                    # Neutral move: accept if it improves the soft surrogate
                    if soft_new > soft_curr + 0.1:
                        accept = True
                    else:
                        accept = random.random() < 0.3
                else:
                    # Worse discrete score: accept with low probability (annealing)
                    delta = current_score - new_score
                    accept_prob = np.exp(-delta / T) * 0.5
                    accept = random.random() < accept_prob

                if accept:
                    found_improvement = True
                    # Try immediate removal
                    for c in list(code_indices):
                        if can_remove(c):
                            for nb in neighbors[c]:
                                counts[nb] -= 1
                            code_indices.remove(c)
                            if len(code_indices) < best_size:
                                best_size = len(code_indices)
                                best_code = list(code_indices)
                            break
                    break
                else:
                    # Revert swap
                    for nb in neighbors[c_in]:
                        counts[nb] -= 1
                    for nb in neighbors[c_out]:
                        counts[nb] += 1
                    code_indices.remove(c_in)
                    code_indices.add(c_out)

            if found_improvement:
                break

        if not found_improvement:
            T *= cooling_rate
            if T < 0.05:
                # Perturbation: apply a random valid swap
                perturbed = False
                for _ in range(20):
                    c_out = random.choice(list(code_indices))
                    c_in = random.randint(0, total - 1)
                    if c_in in code_indices:
                        continue
                    valid = True
                    for u in neighbors[c_out]:
                        if u not in neighbor_sets[c_in]:
                            if counts[u] < 2:
                                valid = False
                                break
                    if valid:
                        for nb in neighbors[c_out]:
                            counts[nb] -= 1
                        for nb in neighbors[c_in]:
                            counts[nb] += 1
                        code_indices.remove(c_out)
                        code_indices.add(c_in)
                        perturbed = True
                        break
                if not perturbed:
                    # Restore best and break
                    code_indices = set(best_code)
                    counts = np.zeros(total, dtype=np.int32)
                    for c in code_indices:
                        for nb in neighbors[c]:
                            counts[nb] += 1
                    break
                T = 1.0

    # Convert best_code indices back to words
    return [list((best_code[i] // powers[p]) % q for p in range(n)) for i in range(len(best_code))]
