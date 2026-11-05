import time
import random
import sys
import itertools
import numpy as np

def construct(q, n, radius):
    # Handle the specific task: K_3(6,1)
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
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    total = q ** n
    powers = [q ** p for p in range(n)]

    # Precompute neighbors: neighbors[i] is a list of indices j such that dist(i,j) <= 1
    # There are 1 + n*(q-1) = 1 + 6*2 = 13 neighbors per word.
    neighbors = np.empty((total, 13), dtype=np.int32)
    for i in range(total):
        digits = [(i // powers[p]) % q for p in range(n)]
        neighbors[i, 0] = i
        k = 1
        for p in range(n):
            for alt in range(q):
                if alt != digits[p]:
                    new_i = i + (alt - digits[p]) * powers[p]
                    neighbors[i, k] = new_i
                    k += 1

    # Convert to sets for fast membership testing
    neighbor_sets = [set(neighbors[i].tolist()) for i in range(total)]

    # Phase 1: Greedy Set Cover to get an initial code
    uncovered = np.ones(total, dtype=np.int8)
    code_indices = set()
    counts = np.zeros(total, dtype=np.int32)

    while uncovered.sum() > 0:
        if time.time() > deadline - 80:
            break

        # Vectorized greedy: find word covering most uncovered neighbors
        scores = uncovered[neighbors].sum(axis=1)
        best_candidate = int(np.argmax(scores))
        best_score = scores[best_candidate]

        if best_score == 0:
            break

        code_indices.add(best_candidate)
        counts[neighbors[best_candidate]] += 1
        uncovered[neighbors[best_candidate]] = 0

    # Check if we have a valid cover
    if uncovered.sum() != 0:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    best_size = len(code_indices)
    best_code = list(code_indices)

    def can_remove(c_idx):
        return counts[neighbors[c_idx]].min() >= 2

    def get_max_removability_score():
        # For each codeword, compute min coverage of its neighbors
        # The max of these mins is the "removability" score
        code_list = list(code_indices)
        if not code_list:
            return 0
        mins = [counts[neighbors[c]].min() for c in code_list]
        return max(mins)

    # Local Search Loop with Simulated Annealing
    T = 1.0
    cooling_rate = 0.9995
    T_min = 0.01

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

    random.seed(42)
    np.random.seed(42)

    while time.time() < deadline - 2:
        # Phase 1: Greedy Removal
        removed_any = False
        for c in list(code_indices):
            if can_remove(c):
                counts[neighbors[c]] -= 1
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
        random.shuffle(code_list)

        # Limit number of removal candidates per iteration for speed
        max_removal_tries = min(len(code_list), 5)
        
        for c_out in code_list[:max_removal_tries]:
            if time.time() > deadline - 2:
                break

            # Find promising c_in candidates
            # A valid c_in must cover all critical words that c_out uniquely covers
            # i.e., for all w in neighbors[c_out] with counts[w]==1, w must be in neighbors[c_in]
            
            c_out_crit = [u for u in neighbors[c_out] if counts[u] == 1]
            
            if not c_out_crit:
                # c_out has no critical words, so it's redundant but can_remove should have caught it
                # This shouldn't happen, but handle gracefully
                continue
            
            # Candidates must be in intersection of neighbors of all critical words of c_out
            # Start with neighbors of first critical word
            w0 = int(c_out_crit[0])
            candidates = neighbors[w0]
            
            valid_candidates = []
            for cand in candidates:
                cand_int = int(cand)
                if cand_int == c_out:
                    continue
                if cand_int in code_indices:
                    continue
                # Check if cand covers all critical words
                # i.e., all c_out_crit are in neighbors[cand]
                # Equivalent: c_out_crit is subset of neighbor_sets[cand]
                valid = True
                for w in c_out_crit:
                    if int(w) not in neighbor_sets[cand_int]:
                        valid = False
                        break
                if valid:
                    # Also check that removing c_out doesn't leave other words uncovered
                    # This is guaranteed if c_in covers all critical words of c_out
                    # because non-critical words have count >= 2, so removing c_out leaves them with count >= 1
                    valid_candidates.append(cand_int)
            
            if not valid_candidates:
                continue

            # Score candidates by how much redundancy they add
            # Prefer candidates that cover words with high current count (adding more redundancy)
            # Or simply pick randomly for diversity
            random.shuffle(valid_candidates)
            top_candidates = valid_candidates[:20]

            for c_in in top_candidates:
                if time.time() > deadline - 2:
                    break

                # Apply swap temporarily
                for nb in neighbors[c_out]:
                    counts[nb] -= 1
                for nb in neighbors[c_in]:
                    counts[nb] += 1
                code_indices.remove(c_out)
                code_indices.add(c_in)

                new_score = get_max_removability_score()

                # Use smooth surrogate for acceptance
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
                            counts[neighbors[c]] -= 1
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
            T = max(T * cooling_rate, T_min)
            if T <= T_min + 1e-9:
                # Perturbation: apply a random valid swap
                perturbed = False
                for _ in range(30):
                    c_out = random.choice(list(code_indices))
                    c_in = random.randint(0, total - 1)
                    if c_in in code_indices:
                        continue
                    
                    # Check validity: c_in must cover all critical words of c_out
                    c_out_crit = [u for u in neighbors[c_out] if counts[u] == 1]
                    valid = True
                    for w in c_out_crit:
                        if int(w) not in neighbor_sets[c_in]:
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
                        counts[neighbors[c]] += 1
                    break
                T = 1.0

    # Final cleanup: iteratively remove redundant words
    improved = True
    while improved:
        improved = False
        for c in list(code_indices):
            if can_remove(c):
                counts[neighbors[c]] -= 1
                code_indices.remove(c)
                improved = True
                if len(code_indices) < best_size:
                    best_size = len(code_indices)
                    best_code = list(code_indices)
                break

    # Use the best found
    if len(best_code) < len(code_indices):
        code_indices = set(best_code)

    # Convert best_code indices back to words
    return [list((c // powers[p]) % q for p in range(n)) for c in code_indices]
