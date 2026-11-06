import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin under 150s

    N = 511
    # Weights: w[a] = 1/(512-a)
    w = [0.0] * (N + 1)
    for a in range(1, N + 1):
        w[a] = 1.0 / (512.0 - a)

    # Precompute coprime bitmasks
    # mask[a] has bit b set if gcd(a, b) == 1
    mask = [0] * (N + 1)
    for a in range(1, N + 1):
        m = 0
        for b in range(1, N + 1):
            if math.gcd(a, b) == 1:
                m |= (1 << b)
        mask[a] = m

    def can_add(a, chosen_mask):
        """Check if a is coprime to all elements in chosen_mask."""
        return (mask[a] & chosen_mask) == chosen_mask

    def calc_score(chosen_list):
        return sum(w[a] for a in chosen_list)

    def fill_greedy(chosen_list, chosen_mask, score):
        """Try to add all compatible elements greedily."""
        improved = True
        while improved and (time.time() < deadline - 10):
            improved = False
            for a in range(N, 0, -1):
                if (chosen_mask >> a) & 1:
                    continue
                if can_add(a, chosen_mask):
                    chosen_list.append(a)
                    chosen_mask |= (1 << a)
                    score += w[a]
                    improved = True
        return chosen_list, chosen_mask, score

    # Greedy initialization: pick from largest weight to smallest
    chosen = []
    chosen_mask = 0
    for a in range(N, 0, -1):
        if can_add(a, chosen_mask):
            chosen.append(a)
            chosen_mask |= (1 << a)

    # Fill greedily
    chosen, chosen_mask, score = fill_greedy(chosen, chosen_mask, 0.0)
    best_score = score
    best_chosen = list(chosen)
    best_mask = chosen_mask

    random.seed(42)

    # Local search with multiple move types
    def local_search(chosen_list, chosen_mask, score):
        improved = True
        while improved and (time.time() < deadline - 10):
            improved = False

            # Try adding any compatible element
            for a in range(N, 0, -1):
                if (chosen_mask >> a) & 1:
                    continue
                if can_add(a, chosen_mask):
                    new_score = score + w[a]
                    if new_score > score + 1e-12:
                        chosen_list.append(a)
                        chosen_mask |= (1 << a)
                        score = new_score
                        improved = True
                        break

            if time.time() > deadline - 10:
                break

            # Try 1-for-1 swaps
            for s in list(chosen_list):
                if time.time() > deadline - 10:
                    break
                for u in range(N, 0, -1):
                    if u == s or (chosen_mask >> u) & 1:
                        continue
                    # Check if u is coprime to all in chosen_list except s
                    temp_mask = chosen_mask & ~(1 << s)
                    if can_add(u, temp_mask):
                        new_score = score - w[s] + w[u]
                        if new_score > score + 1e-12:
                            chosen_list.remove(s)
                            chosen_list.append(u)
                            chosen_mask = (chosen_mask & ~(1 << s)) | (1 << u)
                            score = new_score
                            improved = True
                            break

            if time.time() > deadline - 10:
                break

            # Try 1-for-2 swaps: remove s, add u1 and u2
            for s in list(chosen_list):
                if time.time() > deadline - 10:
                    break
                temp_mask = chosen_mask & ~(1 << s)
                # Find all candidates coprime to temp_mask
                candidates = [u for u in range(N, 0, -1)
                              if not ((chosen_mask >> u) & 1) and can_add(u, temp_mask)]
                # Try pairs from candidates
                for i in range(len(candidates)):
                    if time.time() > deadline - 10:
                        break
                    u1 = candidates[i]
                    for j in range(i + 1, len(candidates)):
                        u2 = candidates[j]
                        if can_add(u2, temp_mask | (1 << u1)):
                            new_score = score - w[s] + w[u1] + w[u2]
                            if new_score > score + 1e-12:
                                chosen_list.remove(s)
                                chosen_list.append(u1)
                                chosen_list.append(u2)
                                chosen_mask = temp_mask | (1 << u1) | (1 << u2)
                                score = new_score
                                improved = True
                                break
                    if improved:
                        break

            if time.time() > deadline - 10:
                break

            # Try 2-for-1 swaps: remove s1 and s2, add u
            for i in range(len(chosen_list)):
                if time.time() > deadline - 10:
                    break
                s1 = chosen_list[i]
                for j in range(i + 1, len(chosen_list)):
                    s2 = chosen_list[j]
                    temp_mask = chosen_mask & ~(1 << s1) & ~(1 << s2)
                    for u in range(N, 0, -1):
                        if (chosen_mask >> u) & 1:
                            continue
                        if can_add(u, temp_mask):
                            new_score = score - w[s1] - w[s2] + w[u]
                            if new_score > score + 1e-12:
                                chosen_list.remove(s1)
                                chosen_list.remove(s2)
                                chosen_list.append(u)
                                chosen_mask = temp_mask | (1 << u)
                                score = new_score
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break

        # Final fill pass
        chosen_list, chosen_mask, score = fill_greedy(chosen_list, chosen_mask, score)
        return chosen_list, chosen_mask, score

    # Run local search
    chosen, chosen_mask, score = local_search(list(best_chosen), best_mask, best_score)
    if score > best_score:
        best_score = score
        best_chosen = list(chosen)
        best_mask = chosen_mask

    # Random restarts with perturbation
    for iteration in range(200):
        if time.time() > deadline - 15:
            break

        # Vary perturbation strategy
        strategy = iteration % 3
        if strategy == 0:
            # Remove 1-3 random elements
            perturb_size = random.randint(1, min(3, len(best_chosen) - 1))
            remove_indices = random.sample(range(len(best_chosen)), perturb_size)
            remove_set = set(best_chosen[i] for i in remove_indices)
            perturbed = [a for a in best_chosen if a not in remove_set]
        elif strategy == 1:
            # Remove 1-2 and add 1-2 random compatible elements
            perturb_size = random.randint(1, min(2, len(best_chosen) - 1))
            remove_indices = random.sample(range(len(best_chosen)), perturb_size)
            remove_set = set(best_chosen[i] for i in remove_indices)
            perturbed = [a for a in best_chosen if a not in remove_set]
            # Add some random compatible elements
            pert_mask = 0
            for a in perturbed:
                pert_mask |= (1 << a)
            for _ in range(random.randint(1, 2)):
                candidates = [a for a in range(1, N + 1) if not ((pert_mask >> a) & 1) and can_add(a, pert_mask)]
                if candidates:
                    u = random.choice(candidates)
                    perturbed.append(u)
                    pert_mask |= (1 << u)
        else:
            # Swap 1-2 random elements with random compatible ones
            perturb_size = random.randint(1, min(2, len(best_chosen) - 1))
            swap_indices = random.sample(range(len(best_chosen)), perturb_size)
            perturbed = list(best_chosen)
            pert_mask = best_mask
            for idx in swap_indices:
                s = perturbed[idx]
                temp_mask = pert_mask & ~(1 << s)
                candidates = [a for a in range(1, N + 1) if not ((pert_mask >> a) & 1) and can_add(a, temp_mask)]
                if candidates:
                    u = random.choice(candidates)
                    perturbed[idx] = u
                    pert_mask = temp_mask | (1 << u)

        perturbed_mask = 0
        for a in perturbed:
            perturbed_mask |= (1 << a)
        perturbed_score = sum(w[a] for a in perturbed)

        # Re-add greedily
        perturbed, perturbed_mask, perturbed_score = fill_greedy(perturbed, perturbed_mask, perturbed_score)

        # Run local search on perturbed solution
        perturbed, perturbed_mask, perturbed_score = local_search(
            perturbed, perturbed_mask, perturbed_score
        )

        if perturbed_score > best_score:
            best_score = perturbed_score
            best_chosen = list(perturbed)
            best_mask = perturbed_mask

    # Final high-precision polish: try all 1-for-1 swaps with tighter tolerance
    if time.time() < deadline - 5:
        for _ in range(3):  # Multiple polish passes
            improved = False
            for s in list(best_chosen):
                if time.time() > deadline - 5:
                    break
                for u in range(N, 0, -1):
                    if u == s or (best_mask >> u) & 1:
                        continue
                    temp_mask = best_mask & ~(1 << s)
                    if can_add(u, temp_mask):
                        new_score = best_score - w[s] + w[u]
                        if new_score > best_score + 1e-15:  # Tighter tolerance
                            best_chosen.remove(s)
                            best_chosen.append(u)
                            best_mask = temp_mask | (1 << u)
                            best_score = new_score
                            improved = True
                            break
                if time.time() > deadline - 5:
                    break
            if not improved:
                break
        # Final fill
        best_chosen, best_mask, best_score = fill_greedy(best_chosen, best_mask, best_score)

    return best_chosen
