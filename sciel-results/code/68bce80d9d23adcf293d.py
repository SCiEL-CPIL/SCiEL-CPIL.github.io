import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit

    n = 12
    m = 40
    num_masks = 1 << n
    full_mask = num_masks - 1

    # Precompute comparability bitsets.
    # comp_bits[i] is an integer where bit j is 1 iff i and j are comparable (i subset j or j subset i).
    # We include self-comparability (i is comparable to i).
    # Total iterations: sum_{i=0}^{2^n-1} 2^{popcount(i)} = 3^n = 531,441.
    # This is very fast.
    comp_bits = [0] * num_masks

    for i in range(num_masks):
        sub = i
        while True:
            # sub is a submask of i, so sub <= i in the partial order.
            # They are comparable.
            comp_bits[i] |= (1 << sub)
            comp_bits[sub] |= (1 << i)

            if sub == 0:
                break
            sub = (sub - 1) & i

    # Helper to compute degrees for a given set of masks
    def get_degrees(masks, S_mask_val):
        degs = []
        for mask in masks:
            # Count comparable masks in S_mask_val, excluding self
            deg = (comp_bits[mask] & S_mask_val).bit_count() - 1
            degs.append(deg)
        return degs

    def score_from_degs(degs):
        return sum(degs) // 2

    def perform_swap(current_masks, degs, S_mask, used_masks, u_idx, v):
        """
        Perform a swap: replace current_masks[u_idx] with v.
        Returns updated degs, S_mask, used_masks, and score_delta.
        """
        old_u = current_masks[u_idx]
        if old_u == v:
            return 0

        # Update S_mask
        S_mask ^= (1 << old_u)
        S_mask |= (1 << v)

        # Update used_masks
        used_masks.remove(old_u)
        used_masks.add(v)

        # Update current_masks
        current_masks[u_idx] = v

        # Update degs incrementally
        # For all j != u_idx:
        #   if old_u was comparable to current_masks[j], degs[j] -= 1
        #   if v is comparable to current_masks[j], degs[j] += 1
        for j in range(m):
            if j == u_idx:
                continue
            j_mask = current_masks[j]
            if (comp_bits[old_u] >> j_mask) & 1:
                degs[j] -= 1
            if (comp_bits[v] >> j_mask) & 1:
                degs[j] += 1

        # Update degs[u_idx]
        # It is the number of comparables of v in the new S_mask (which includes v)
        # So (comp_bits[v] & S_mask).bit_count() - 1
        degs[u_idx] = (comp_bits[v] & S_mask).bit_count() - 1

        # Calculate delta
        # The score change is half the change in sum of degrees.
        # Alternatively, we can calculate it directly.
        # Old contribution of u to score: degs_old[u_idx] / 2 (roughly, since each pair counted twice)
        # Actually, let's just return the new score difference.
        # The function doesn't track score, so let's compute delta here.
        # Delta = (new_deg_sum - old_deg_sum) / 2
        # We didn't store old_deg_sum.
        # But we know:
        # Old degs[u_idx] was degs_old_u.
        # New degs[u_idx] is degs[u_idx].
        # For j != u_idx, degs[j] changed by (is_comparable(v, j) - is_comparable(old_u, j)).
        # So sum_delta = (degs[u_idx] - degs_old_u) + sum_{j!=u} (delta_j)
        # We don't have degs_old_u here anymore because we overwrote it.
        # Let's restructure: calculate delta BEFORE updating degs?
        # Or just return the new degs and let the caller update score?
        # The caller needs the delta to update current_score.
        # Let's calculate delta in the caller or return it.
        # It's easier to calculate delta in the caller using the old degs if we pass them or store them.
        # Let's modify perform_swap to return the delta.
        # But we already modified degs.
        # Let's just return the new degs and the delta.
        # To get delta, we need the old degs[u_idx].
        # Let's pass old_deg_u.

        # Wait, I can just compute the delta in the main loop before calling perform_swap?
        # Yes, the main loop already calculates delta to decide if it's an improvement.
        # So perform_swap just needs to update the state.
        # The main loop will do: current_score += delta.
        # So this function should just update the state and return nothing (or the new state if not using references).
        # Since lists and sets are mutable, we can just modify in place.
        # But we need to return the delta? No, the caller has it.
        # However, the caller calculates delta based on `degs[u_idx]` which is the OLD value.
        # So the caller must store old_deg_u = degs[u_idx] before calling perform_swap.
        # Then perform_swap updates degs.
        # Then caller does current_score += delta.

        # This function modifies current_masks, degs, S_mask, used_masks in place.
        # It returns None.
        pass

    # Greedy Initialization
    # Start with a random set of m masks, or a structured one?
    # Let's start with a random set to avoid bias, then hill climb.
    # Or start with all subsets of a small size?
    # Let's try a mix: start with a random set, then do a few greedy improvements.

    random.seed(42)
    all_masks_list = list(range(num_masks))
    random.shuffle(all_masks_list)

    current_masks = all_masks_list[:m]
    used_masks = set(current_masks)

    S_mask = 0
    for mask in current_masks:
        S_mask |= (1 << mask)

    degs = get_degrees(current_masks, S_mask)
    current_score = score_from_degs(degs)
    best_score = current_score
    best_solution = list(current_masks)

    # Local Search Loop
    # We will use a simulated annealing approach to escape local optima.
    # But first, let's do a simple hill-climbing pass to get a good starting point.

    no_improvement_count = 0
    temperature = 5.0
    cooling_rate = 0.999

    while time.time() < deadline:
        max_delta = 0
        best_u_idx = -1
        best_v = -1

        # Find the best swap
        # To speed up, we can iterate u_idx and then v.
        # v must not be in used_masks.
        # We can iterate over all v in range(num_masks) and check membership.
        # Or maintain a list of unused masks?
        # Checking set membership is O(1).
        # 40 * 4096 = 163,840 checks per iteration.
        # Each check: 1 XOR, 1 AND, 1 bit_count.
        # This is fast enough.

        for u_idx in range(m):
            u_mask = current_masks[u_idx]
            u_deg = degs[u_idx]
            S_mask_without_u = S_mask ^ (1 << u_mask)

            for v in range(num_masks):
                if v in used_masks:
                    continue

                v_deg_in_S_minus_u = (comp_bits[v] & S_mask_without_u).bit_count()
                delta = v_deg_in_S_minus_u - u_deg

                if delta > max_delta:
                    max_delta = delta
                    best_u_idx = u_idx
                    best_v = v

        if max_delta > 0:
            # Perform the best swap
            old_u = current_masks[best_u_idx]
            new_v = best_v

            # Update state
            S_mask ^= (1 << old_u)
            S_mask |= (1 << new_v)
            used_masks.remove(old_u)
            used_masks.add(new_v)
            current_masks[best_u_idx] = new_v

            # Update degs
            for j in range(m):
                if j == best_u_idx:
                    continue
                j_mask = current_masks[j]
                if (comp_bits[old_u] >> j_mask) & 1:
                    degs[j] -= 1
                if (comp_bits[new_v] >> j_mask) & 1:
                    degs[j] += 1

            degs[best_u_idx] = (comp_bits[new_v] & S_mask).bit_count() - 1

            current_score += max_delta
            if current_score > best_score:
                best_score = current_score
                best_solution = list(current_masks)
            no_improvement_count = 0
            temperature = max(0.1, temperature * cooling_rate)
        else:
            # No improvement found
            no_improvement_count += 1

            # Simulated Annealing Step
            # Pick a random swap and accept with probability exp(delta / T)
            if random.random() < 0.5 or no_improvement_count > 50:
                u_idx = random.randint(0, m - 1)
                # Pick a random unused v
                # To avoid O(n) scan for unused, we can just pick random v from range(num_masks)
                # and check if it's unused. If it is, use it.
                # But we need to ensure we pick an unused one.
                # Let's just iterate a few times to find an unused one.
                v = -1
                for _ in range(100):
                    v = random.randint(0, num_masks - 1)
                    if v not in used_masks:
                        break
                if v == -1:
                    continue

                old_u = current_masks[u_idx]
                S_mask_without_u = S_mask ^ (1 << old_u)
                v_deg_in_S_minus_u = (comp_bits[v] & S_mask_without_u).bit_count()
                delta = v_deg_in_S_minus_u - degs[u_idx]

                # Accept if delta >= 0 or with probability exp(delta / T)
                if delta >= 0 or random.random() < math.exp(delta / temperature):
                    # Perform swap
                    S_mask ^= (1 << old_u)
                    S_mask |= (1 << v)
                    used_masks.remove(old_u)
                    used_masks.add(v)
                    current_masks[u_idx] = v

                    for j in range(m):
                        if j == u_idx:
                            continue
                        j_mask = current_masks[j]
                        if (comp_bits[old_u] >> j_mask) & 1:
                            degs[j] -= 1
                        if (comp_bits[v] >> j_mask) & 1:
                            degs[j] += 1

                    degs[u_idx] = (comp_bits[v] & S_mask).bit_count() - 1

                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_solution = list(current_masks)

                    temperature = max(0.1, temperature * cooling_rate)

            if no_improvement_count > 100:
                # Restart with a new random solution?
                # Or just reset temperature?
                temperature = 5.0
                no_improvement_count = 0

    # Convert best_solution to list of lists
    result = []
    for mask in best_solution:
        s = []
        for i in range(n):
            if (mask >> i) & 1:
                s.append(i)
        result.append(s)

    return result
