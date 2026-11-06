"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # safety margin before 60s kill

    n = 20
    max_val = 10000

    # Generate 2,3-smooth numbers (Hamming numbers) up to max_val
    smooths = []
    for a in range(0, 15):
        for b in range(0, 15):
            val = (2 ** a) * (3 ** b)
            if 1 <= val <= max_val:
                smooths.append(val)
    smooths = list(set(smooths))
    smooths.sort()

    def compute_objective(A):
        """Compute max(|A+A|, |A*A|) for set A"""
        A_set = set(A)
        sumset = set()
        prodset = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                sumset.add(A[i] + A[j])
                prodset.add(A[i] * A[j])
        return max(len(sumset), len(prodset))

    def compute_score_fast(A):
        """Fast score computation using sorted arrays"""
        n = len(A)
        sumset = set()
        prodset = set()
        for i in range(n):
            ai = A[i]
            for j in range(i, n):
                aj = A[j]
                sumset.add(ai + aj)
                prodset.add(ai * aj)
        return max(len(sumset), len(prodset))

    # Build a candidate pool: smooth numbers + structured families
    candidates = set(smooths)

    # Add geometric sequences
    for b in range(2, 12):
        val = b
        for i in range(1, 21):
            val *= b
            if val <= max_val:
                candidates.add(val)
            else:
                break

    # Add arithmetic-geometric hybrids a*b^i
    for a in range(1, 10):
        for b in range(2, 8):
            val = a
            for i in range(20):
                val = a * (b ** i)
                if val <= max_val:
                    candidates.add(val)

    # Add small consecutive integers
    for v in range(1, 101):
        candidates.add(v)

    candidates = sorted(candidates)

    # Precompute collision-aware edge weights between candidate pairs
    # Edge weight = number of shared sums/products with other pairs
    # This approximates how "clustered" two elements are

    def build_collision_graph(cand_list, max_pairs=200):
        """Build a weighted adjacency based on sum/product collisions."""
        m = len(cand_list)
        if m > 400:
            # Sample a subset for speed
            indices = random.sample(range(m), 400)
            cand_list = [cand_list[i] for i in indices]
            m = 400

        # For each pair, count how many other pairs produce the same sum or product
        # This is expensive, so we use a cheaper proxy:
        # weight(i,j) = number of k such that cand[i]+cand[j] == cand[k]+cand[l] for some l
        # Simplified: use the number of distinct sums/products that pair (i,j) creates
        # that are also created by other pairs

        # Cheaper approach: for each element, compute its "collision density"
        # with nearby elements in value space
        weights = np.zeros((m, m))
        for i in range(m):
            for j in range(i + 1, m):
                si = cand_list[i] + cand_list[j]
                pi = cand_list[i] * cand_list[j]
                # Count how many other pairs produce this sum or product
                # This is O(m^2) per pair, too slow. Use a heuristic instead:
                # Elements that are close in log-space tend to have more collisions
                log_diff = abs(np.log(cand_list[i]) - np.log(cand_list[j]))
                weights[i, j] = np.exp(-log_diff)
                weights[j, i] = weights[i, j]

        return weights, cand_list

    # Greedy marginal-cost construction with collision awareness
    def greedy_construct(candidates, target_size=20, num_restarts=5):
        best_set = None
        best_score = float('inf')

        for restart in range(num_restarts):
            if time.time() > deadline - 20:
                break

            current = []
            current_set = set()
            current_score = 0

            # Start with a random seed element
            seed_idx = random.randint(0, len(candidates) - 1)
            current.append(candidates[seed_idx])
            current_set.add(candidates[seed_idx])

            for step in range(target_size - 1):
                if time.time() > deadline - 20:
                    break

                best_candidate = None
                best_marginal = float('inf')

                # Sample a subset of candidates to consider
                if len(candidates) > 500:
                    sample_size = min(500, len(candidates))
                    sample_indices = random.sample(range(len(candidates)), sample_size)
                else:
                    sample_indices = range(len(candidates))

                for idx in sample_indices:
                    cand = candidates[idx]
                    if cand in current_set:
                        continue
                    if cand < 1 or cand > max_val:
                        continue

                    # Compute marginal cost: how much does adding cand increase the score?
                    # Approximate by counting new sums and products
                    new_sums = 0
                    new_prods = 0
                    for elem in current:
                        s = elem + cand
                        p = elem * cand
                        # Check if this sum/product is already in the set
                        # We don't track the full set, so approximate
                        new_sums += 1
                        new_prods += 1

                    # Prefer candidates that create fewer new values
                    # Use a heuristic: prefer candidates close to existing ones in log-space
                    if len(current) > 0:
                        log_cand = np.log(cand)
                        avg_log = np.mean([np.log(e) for e in current])
                        dist = abs(log_cand - avg_log)
                    else:
                        dist = 0

                    marginal = new_sums + new_prods + 0.1 * dist

                    if marginal < best_marginal:
                        best_marginal = marginal
                        best_candidate = cand

                if best_candidate is None:
                    # Fallback: pick a random candidate
                    best_candidate = random.choice(candidates)
                    while best_candidate in current_set:
                        best_candidate = random.choice(candidates)

                current.append(best_candidate)
                current_set.add(best_candidate)

            if len(current) == target_size:
                current.sort()
                score = compute_objective(current)
                if score < best_score:
                    best_score = score
                    best_set = current[:]

        return best_set, best_score

    # Initial population from multiple strategies
    best_set = None
    best_score = float('inf')

    # Strategy 1: Smallest smooths
    if len(smooths) >= n:
        candidate = smooths[:n]
        score = compute_objective(candidate)
        if score < best_score:
            best_score = score
            best_set = candidate[:]

    # Strategy 2: Greedy collision-aware construction
    greedy_result, greedy_score = greedy_construct(candidates, n, num_restarts=3)
    if greedy_result is not None and greedy_score < best_score:
        best_score = greedy_score
        best_set = greedy_result[:]

    # Strategy 3: Random subsets of smooths
    random.seed(42)
    for _ in range(30):
        if len(smooths) < n:
            break
        candidate = random.sample(smooths, n)
        candidate.sort()
        score = compute_objective(candidate)
        if score < best_score:
            best_score = score
            best_set = candidate[:]

        if time.time() > deadline - 15:
            break

    # Strategy 4: Geometric sequences
    for b in range(2, 12):
        A = [b ** i for i in range(1, 21)]
        if max(A) <= max_val and len(set(A)) == 20:
            score = compute_objective(A)
            if score < best_score:
                best_score = score
                best_set = A[:]

    if best_set is None:
        best_set = list(range(1, n + 1))
        best_score = compute_objective(best_set)

    # Simulated Annealing refinement
    current_set = best_set[:]
    current_set_for_membership = set(current_set)
    current_score = best_score
    best_set_sa = current_set[:]
    best_score_sa = current_score

    T = 10.0
    T_min = 0.1
    cooling = 0.995

    sa_deadline = deadline - 10

    while time.time() < sa_deadline:
        if T < T_min:
            T = T_min

        i = random.randint(0, n - 1)
        move_type = random.random()

        if move_type < 0.5:
            new_val = random.choice(candidates)
        elif move_type < 0.75:
            new_val = current_set[i] + random.randint(-5, 5)
        else:
            j = random.randint(0, n - 1)
            if j == i:
                continue
            current_set[i], current_set[j] = current_set[j], current_set[i]
            new_score = compute_objective(current_set)
            delta = new_score - current_score
            if delta < 0 or random.random() < np.exp(-delta / T):
                if new_score < best_score_sa:
                    best_score_sa = new_score
                    best_set_sa = current_set[:]
                current_score = new_score
            else:
                current_set[i], current_set[j] = current_set[j], current_set[i]
            T *= cooling
            continue

        if not (1 <= new_val <= max_val) or new_val in current_set_for_membership:
            T *= cooling
            continue

        old_val = current_set[i]
        current_set[i] = new_val
        current_set_for_membership.remove(old_val)
        current_set_for_membership.add(new_val)

        new_score = compute_objective(current_set)
        delta = new_score - current_score

        if delta < 0 or random.random() < np.exp(-delta / T):
            current_score = new_score
            if new_score < best_score_sa:
                best_score_sa = new_score
                best_set_sa = current_set[:]
        else:
            current_set[i] = old_val
            current_set_for_membership.remove(new_val)
            current_set_for_membership.add(old_val)

        T *= cooling

    # Final local search with tighter tolerance
    if time.time() < deadline - 2:
        current_set_for_membership = set(best_set_sa)
        for _ in range(30):
            improved = False
            for i in range(n):
                for delta_val in range(-5, 6):
                    new_val = best_set_sa[i] + delta_val
                    if not (1 <= new_val <= max_val) or new_val in current_set_for_membership:
                        continue
                    old_val = best_set_sa[i]
                    best_set_sa[i] = new_val
                    current_set_for_membership.remove(old_val)
                    current_set_for_membership.add(new_val)

                    new_score = compute_objective(best_set_sa)
                    if new_score < best_score_sa:
                        best_score_sa = new_score
                        improved = True
                    else:
                        best_set_sa[i] = old_val
                        current_set_for_membership.remove(new_val)
                        current_set_for_membership.add(old_val)
            if not improved:
                break

    best_set_sa = sorted(set(best_set_sa))
    return best_set_sa

# EVOLVE-BLOCK-END
