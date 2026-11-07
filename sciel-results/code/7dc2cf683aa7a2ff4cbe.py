"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

P = {'n': 12, 'm': 40}

def construct():
    n = P['n']
    m = P['m']
    total_masks = 1 << n
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s limit

    # Precompute bit counts for all masks
    bit_counts = [0] * total_masks
    for mask in range(1, total_masks):
        bit_counts[mask] = bit_counts[mask & (mask - 1)] + 1

    # Precompute comparability: comp[a][b] = 1 if a and b are comparable.
    # Two masks a, b are comparable iff (a & b) == a or (a & b) == b.
    # Stored as a flat bytearray for O(1) lookup.
    comp = bytearray(total_masks * total_masks)
    for a in range(total_masks):
        base = a * total_masks
        for b in range(total_masks):
            if (a & b) == a or (a & b) == b:
                comp[base + b] = 1

    def calc_score(family):
        """Calculate number of comparable unordered pairs in family."""
        score = 0
        L = len(family)
        for i in range(L):
            fi = family[i]
            base = fi * total_masks
            for j in range(i + 1, L):
                if comp[base + family[j]]:
                    score += 1
        return score

    # Candidate pools by size
    by_size = [[] for _ in range(n + 1)]
    for mask in range(total_masks):
        by_size[bit_counts[mask]].append(mask)

    random.seed(42)

    # Build a good initial solution using greedy approach
    def build_greedy_initial(seed_val=42):
        """Build initial solution by greedily adding most comparable sets."""
        random.seed(seed_val)
        family = []
        used = set()

        # Start with a chain to ensure a good base (13 nested sets)
        chain_masks = [(1 << s) - 1 for s in range(n + 1)]
        for mask in chain_masks:
            family.append(mask)
            used.add(mask)

        # Now greedily add sets that maximize comparability with existing family
        remaining_slots = m - len(family)
        for _ in range(remaining_slots):
            best_mask = None
            best_comp_count = -1
            # Try masks from small and large sizes first (they nest well with the chain)
            for size in [0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 5, 6, 7]:
                candidates = [mask for mask in by_size[size] if mask not in used]
                random.shuffle(candidates)
                # Only check first few candidates for speed
                for mask in candidates[:80]:
                    base = mask * total_masks
                    comp_count = 0
                    for f in family:
                        if comp[base + f]:
                            comp_count += 1
                    if comp_count > best_comp_count:
                        best_comp_count = comp_count
                        best_mask = mask
                if best_mask is not None and best_comp_count >= len(family) * 0.75:
                    break

            if best_mask is not None:
                family.append(best_mask)
                used.add(best_mask)
            else:
                # Fallback: add any unused mask
                all_remaining = [mask for mask in range(total_masks) if mask not in used]
                if all_remaining:
                    random.shuffle(all_remaining)
                    family.append(all_remaining[0])
                    used.add(all_remaining[0])

        return family[:m]

    family = build_greedy_initial(42)
    current_score = calc_score(family)
    best_family = list(family)
    best_score = current_score

    # Local search with improved strategies
    no_improve_count = 0
    max_no_improve = 15000
    restarts = 0
    max_restarts = 15
    temperature = 2.0
    min_temperature = 0.01
    cooling = 0.9995

    family_set = set(family)

    # Precompute for each mask, the list of masks it is comparable to (for fast neighbor generation)
    # Actually, let's use a different approach: precompute comparability as a set of sets
    # For speed, keep the bytearray but also build a list-of-lists for "which masks are comparable to mask m"
    comp_neighbors = [None] * total_masks
    for a in range(total_masks):
        base = a * total_masks
        neighbors = []
        for b in range(total_masks):
            if comp[base + b]:
                neighbors.append(b)
        comp_neighbors[a] = neighbors

    while time.time() < deadline:
        if no_improve_count >= max_no_improve:
            # Try a restart with a different random seed
            restarts += 1
            if restarts > max_restarts:
                break

            seed_val = 42 + restarts * 7919
            family = build_greedy_initial(seed_val)
            family_set = set(family)
            current_score = calc_score(family)
            if current_score > best_score:
                best_score = current_score
                best_family = list(family)
            no_improve_count = 0
            temperature = 2.0
            continue

        # Try random swaps with bias toward promising sizes
        idx = random.randint(0, m - 1)
        old_mask = family[idx]

        # Pick a new mask with bias toward small and large sizes
        r = random.random()
        if r < 0.35:
            new_size = random.randint(0, 2)
        elif r < 0.65:
            new_size = random.randint(10, 12)
        elif r < 0.80:
            new_size = random.randint(3, 5)
        elif r < 0.90:
            new_size = random.randint(7, 9)
        else:
            new_size = random.randint(0, n)

        if not by_size[new_size]:
            continue

        # Bias selection: prefer masks comparable to many existing family members
        candidates = by_size[new_size]
        if len(candidates) > 20:
            # Score a subset of candidates by their comparability with current family
            sample_size = min(40, len(candidates))
            sampled = random.sample(candidates, sample_size)
            best_cand = None
            best_cand_score = -1
            for cand in sampled:
                if cand == old_mask or cand in family_set:
                    continue
                cand_base = cand * total_masks
                cs = 0
                for f in family:
                    if comp[cand_base + f]:
                        cs += 1
                if cs > best_cand_score:
                    best_cand_score = cs
                    best_cand = cand
            if best_cand is not None:
                new_mask = best_cand
            else:
                new_mask = random.choice(candidates)
                if new_mask == old_mask or new_mask in family_set:
                    continue
        else:
            valid_cands = [c for c in candidates if c != old_mask and c not in family_set]
            if not valid_cands:
                continue
            new_mask = random.choice(valid_cands)

        # Calculate delta score using precomputed comparability
        delta = 0
        old_base = old_mask * total_masks
        new_base = new_mask * total_masks
        for j in range(m):
            if j == idx:
                continue
            other = family[j]
            if comp[old_base + other]:
                delta -= 1
            if comp[new_base + other]:
                delta += 1

        # Accept with simulated annealing for negative deltas
        accept = False
        if delta > 0:
            accept = True
        elif delta == 0:
            accept = random.random() < 0.3
        else:
            # Probability of accepting worse move
            prob = math.exp(delta / temperature)
            accept = random.random() < prob

        if accept:
            family[idx] = new_mask
            family_set.remove(old_mask)
            family_set.add(new_mask)
            current_score += delta
            if delta > 0:
                no_improve_count = 0
            else:
                no_improve_count += 1
            if current_score > best_score:
                best_score = current_score
                best_family = list(family)
        else:
            no_improve_count += 1

        # Cool down
        temperature *= cooling
        if temperature < min_temperature:
            temperature = min_temperature

    # Exhaustive single-element replacement passes: for each position, try ALL masks
    # Multiple passes to converge
    for pass_num in range(4):
        if time.time() >= deadline - 3:
            break
        family_set = set(best_family)
        improved = False
        for idx in range(m):
            if time.time() >= deadline - 3:
                break
            old_mask = best_family[idx]
            old_base = old_mask * total_masks
            best_delta = 0
            best_new_mask = None
            for new_mask in range(total_masks):
                if new_mask in family_set or new_mask == old_mask:
                    continue
                new_base = new_mask * total_masks
                delta = 0
                for j in range(m):
                    if j == idx:
                        continue
                    other = best_family[j]
                    if comp[old_base + other]:
                        delta -= 1
                    if comp[new_base + other]:
                        delta += 1
                if delta > best_delta:
                    best_delta = delta
                    best_new_mask = new_mask
            if best_new_mask is not None and best_delta > 0:
                family_set.remove(old_mask)
                family_set.add(best_new_mask)
                best_family[idx] = best_new_mask
                best_score += best_delta
                improved = True
        if not improved:
            break

    # Final pair-swap polish: try swapping two elements simultaneously
    if time.time() < deadline - 2:
        family_set = set(best_family)
        for i in range(m):
            if time.time() >= deadline - 2:
                break
            for j in range(i + 1, m):
                if time.time() >= deadline - 2:
                    break
                mi = best_family[i]
                mj = best_family[j]
                # Try replacing both with new masks
                # For speed, only try a limited set of candidates
                for new_i in range(total_masks):
                    if new_i in family_set or new_i == mi:
                        continue
                    ni_base = new_i * total_masks
                    # Quick check: must be comparable to at least half the family minus mj
                    good_count = 0
                    for k in range(m):
                        if k == i or k == j:
                            continue
                        if comp[ni_base + best_family[k]]:
                            good_count += 1
                    if good_count < (m - 2) // 2:
                        continue
                    for new_j in range(total_masks):
                        if new_j in family_set or new_j == mj or new_j == new_i:
                            continue
                        nj_base = new_j * total_masks
                        # Calculate delta
                        mi_base = mi * total_masks
                        mj_base = mj * total_masks
                        delta = 0
                        for k in range(m):
                            if k == i or k == j:
                                other = best_family[k]
                                if comp[mi_base + other]:
                                    delta -= 1
                                if comp[ni_base + other]:
                                    delta += 1
                                if comp[mj_base + other]:
                                    delta -= 1
                                if comp[nj_base + other]:
                                    delta += 1
                        # Also check i-j pair
                        if comp[mi_base + mj]:
                            delta -= 1
                        if comp[ni_base + new_j]:
                            delta += 1
                        if delta > 0:
                            family_set.remove(mi)
                            family_set.remove(mj)
                            family_set.add(new_i)
                            family_set.add(new_j)
                            best_family[i] = new_i
                            best_family[j] = new_j
                            best_score += delta
                            break
                    if time.time() >= deadline - 2:
                        break

    # Convert best_family to lists of integers
    result = []
    for mask in best_family:
        subset = [b for b in range(n) if (mask >> b) & 1]
        result.append(subset)

    # Verify: exactly 40 distinct subsets
    assert len(result) == 40, f"Expected 40 subsets, got {len(result)}"
    assert len(set(tuple(s) for s in result)) == 40, "Subsets are not distinct"
    for s in result:
        assert isinstance(s, list), f"Subset is not a list: {type(s)}"
        assert len(s) == len(set(s)), f"Subset has duplicates: {s}"
        for x in s:
            assert isinstance(x, int) and not isinstance(x, bool), f"Non-int in subset: {x}"
            assert 0 <= x < 12, f"Vertex out of range: {x}"

    return result
# EVOLVE-BLOCK-END
