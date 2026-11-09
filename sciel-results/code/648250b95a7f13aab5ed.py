"""Rewritten program: bitmask-accelerated 4-AP-free construction + deep SA."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    N = 256
    MASK_ALL = (1 << (N + 1)) - 2  # bits 1..N

    # Precompute 4-AP masks and per-element 4-AP lists
    four_ap_masks = []
    elem_4ap = [0] * (N + 1)
    for d in range(1, (N - 1) // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            m = (1 << a) | (1 << (a + d)) | (1 << (a + 2 * d)) | (1 << (a + 3 * d))
            four_ap_masks.append(m)
            elem_4ap[a] += 1
            elem_4ap[a + d] += 1
            elem_4ap[a + 2 * d] += 1
            elem_4ap[a + 3 * d] += 1

    # Precompute 3-AP masks and per-element 3-AP lists
    three_ap_masks = []
    elem_3ap = [[] for _ in range(N + 1)]
    for d in range(1, (N - 1) // 2 + 1):
        max_a = N - 2 * d
        for a in range(1, max_a + 1):
            m = (1 << a) | (1 << (a + d)) | (1 << (a + 2 * d))
            three_ap_masks.append(m)
            elem_3ap[a].append(m)
            elem_3ap[a + d].append(m)
            elem_3ap[a + 2 * d].append(m)

    def count_3aps(mask):
        c = 0
        for m in three_ap_masks:
            if (mask & m) == m:
                c += 1
        return c

    def has_4ap(mask):
        for m in four_ap_masks:
            if (mask & m) == m:
                return True
        return False

    def is_4ap_free(mask):
        return not has_4ap(mask)

    # ---------- Greedy seed ----------
    def greedy_seed():
        mask = 0
        elems = list(range(1, N + 1))
        random.shuffle(elems)
        while elems:
            best_e = None
            best_s = -1
            for x in elems:
                if mask | (1 << x):
                    nm = mask | (1 << x)
                    if has_4ap(nm):
                        continue
                    # quick 3-AP gain
                    gain = 0
                    for m in elem_3ap[x]:
                        if (mask & m) == (m ^ (1 << x)):
                            gain += 1
                    if gain > best_s:
                        best_s = gain
                        best_e = x
            if best_e is None:
                break
            mask |= 1 << best_e
            elems.remove(best_e)
        return mask

    # ---------- Simulated annealing ----------
    def refine(mask, score, budget):
        best_mask = mask
        best_score = score
        T = 30.0
        T_min = 0.05
        cooling = 0.9992

        while T > T_min and time.time() < budget:
            if len(mask) > 0:
                # swap: remove one, try to add another
                if random.random() < 0.65:
                    # pick element to remove
                    elems_in = [i for i in range(1, N + 1) if (mask >> i) & 1]
                    if not elems_in:
                        break
                    rem = random.choice(elems_in)
                    new_mask = mask & ~(1 << rem)
                    # try to add a random element
                    candidates = [i for i in range(1, N + 1) if not ((new_mask >> i) & 1)]
                    if candidates:
                        add = random.choice(candidates)
                        cand_mask = new_mask | (1 << add)
                        if not has_4ap(cand_mask):
                            # compute score diff
                            # remove rem: lose 3-APs that had rem and were complete in mask
                            # add add: gain 3-APs that had add and are complete in cand_mask
                            lost = 0
                            for m in elem_3ap[rem]:
                                if (mask & m) == m:
                                    lost += 1
                            gained = 0
                            for m in elem_3ap[add]:
                                if (cand_mask & m) == m:
                                    gained += 1
                            # but 3-APs containing both rem and add? rem is removed, add added.
                            # A 3-AP containing both would be lost (rem gone) and not gained (add present but rem not).
                            # Actually if a 3-AP contains both rem and add, in mask it's complete only if all 3 present.
                            # After swap, rem gone so it's not complete. So it's in 'lost' if it was complete.
                            # In 'gained' it requires all 3 present including rem, which isn't, so not counted. Good.
                            delta = gained - lost
                            new_score = score + delta
                            if new_score >= score or random.random() < math.exp((new_score - score) / T):
                                mask = cand_mask
                                score = new_score
                                if score > best_score:
                                    best_score = score
                                    best_mask = mask
                else:
                    # add-only
                    candidates = [i for i in range(1, N + 1) if not ((mask >> i) & 1)]
                    if candidates:
                        add = random.choice(candidates)
                        cand_mask = mask | (1 << add)
                        if not has_4ap(cand_mask):
                            gained = 0
                            for m in elem_3ap[add]:
                                if (cand_mask & m) == m:
                                    gained += 1
                            new_score = score + gained
                            if new_score >= score or random.random() < math.exp((new_score - score) / T):
                                mask = cand_mask
                                score = new_score
                                if score > best_score:
                                    best_score = score
                                    best_mask = mask
            T *= cooling

        # Local 2-opt refinement: try all single-element swaps to improve
        for _ in range(200):
            improved = False
            if time.time() > budget:
                break
            elems_in = [i for i in range(1, N + 1) if (mask >> i) & 1]
            if not elems_in:
                break
            for rem in elems_in:
                if time.time() > budget:
                    break
                base = mask & ~(1 << rem)
                lost = 0
                for m in elem_3ap[rem]:
                    if (mask & m) == m:
                        lost += 1
                for add in range(1, N + 1):
                    if (base >> add) & 1:
                        continue
                    cand = base | (1 << add)
                    if has_4ap(cand):
                        continue
                    gained = 0
                    for m in elem_3ap[add]:
                        if (cand & m) == m:
                            gained += 1
                    if gained - lost > 0:
                        mask = cand
                        score += gained - lost
                        if score > best_score:
                            best_score = score
                            best_mask = mask
                        improved = True
                        break
            if not improved:
                break

        return best_mask, best_score

    # ---------- Multi-start ----------
    best_mask = 0
    best_score = 0

    # Try a few greedy seeds with different shuffles
    num_seeds = 3
    for s in range(num_seeds):
        if time.time() > deadline - 10.0:
            break
        mask = greedy_seed()
        score = count_3aps(mask)
        if score > best_score:
            best_score = score
            best_mask = mask
        # Refine this seed with a time budget
        budget = min(deadline - 5.0, time.time() + 15.0)
        mask, score = refine(mask, score, budget)
        if score > best_score:
            best_score = score
            best_mask = mask

    # Final deep refinement on the best
    if time.time() < deadline - 3.0:
        budget = deadline - 2.0
        mask, score = refine(best_mask, best_score, budget)
        if score > best_score:
            best_score = score
            best_mask = mask

    result = [i for i in range(1, N + 1) if (best_mask >> i) & 1]
    if not result:
        # Fallback: return a known small 4-AP-free set
        result = [1, 2, 4, 8, 16, 32, 64, 128]
    return sorted(result)
# EVOLVE-BLOCK-END
