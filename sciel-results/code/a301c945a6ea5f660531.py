"""Improved: precomputed 4-AP conflicts, vectorized 3-AP deltas, simulated annealing, greedy polish."""
import time
import random
import math

N = 256
DEADLINE = 50.0  # safety margin under 60s kill


def construct():
    start = time.time()
    random.seed(42)

    # Precompute 4-AP conflict sets: conflict[x] = frozenset of y such that {x,y} participates in a 4-AP
    # More precisely: for each pair (x, y), if there exists d>0 such that x, x+d, x+2d, x+3d are all in [1,N]
    # and y is one of the other three terms, then x and y are in conflict.
    # Actually, we need: if we have set S and want to add x, check if S contains any y that with x forms a 4-AP.
    # Precompute: for each x, the set of all y such that {x,y} is a pair in some 4-AP.
    
    # For a 4-AP a, a+d, a+2d, a+3d: the pairs are (a,a+d), (a,a+2d), (a,a+3d), (a+d,a+2d), (a+d,a+3d), (a+2d,a+3d)
    # So conflict[a] includes a+d, a+2d, a+3d
    # conflict[a+d] includes a, a+2d, a+3d
    # conflict[a+2d] includes a, a+d, a+3d
    # conflict[a+3d] includes a, a+d, a+2d
    
    conflict = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            conflict[a].update([b, c, e])
            conflict[b].update([a, c, e])
            conflict[c].update([a, b, e])
            conflict[e].update([a, b, c])
    
    # Convert to frozensets for fast membership
    conflict = [frozenset(s) for s in conflict]

    # Precompute 3-AP partner lists for delta computation
    # For each x, precompute:
    # first_partners[x] = list of (y, z) where x, y, z form a 3-AP with x first: x, x+d, x+2d
    # mid_partners[x] = list of (y, z) where y, x, z form a 3-AP with x middle: x-d, x, x+d
    # last_partners[x] = list of (y, z) where y, z, x form a 3-AP with x last: x-2d, x-d, x
    
    first_partners = [[] for _ in range(N + 1)]
    mid_partners = [[] for _ in range(N + 1)]
    last_partners = [[] for _ in range(N + 1)]
    
    for x in range(1, N + 1):
        # x as first
        max_d = (N - x) // 2
        for d in range(1, max_d + 1):
            first_partners[x].append((x + d, x + 2 * d))
        # x as middle
        max_d = min(x - 1, N - x)
        for d in range(1, max_d + 1):
            mid_partners[x].append((x - d, x + d))
        # x as last
        max_d = (x - 1) // 2
        for d in range(1, max_d + 1):
            last_partners[x].append((x - 2 * d, x - d))

    def is_valid_add(mask, x):
        """Check if adding x to mask doesn't create a 4-AP. x not in mask."""
        # For each y in mask, check if y is in conflict[x]
        # i.e., check if (mask & conflict_mask[x]) != 0
        # Precompute conflict masks as bitmasks
        pass

    # Precompute conflict as bitmask for each x
    conflict_mask = [0] * (N + 1)
    for x in range(1, N + 1):
        m = 0
        for y in conflict[x]:
            m |= (1 << (y - 1))
        conflict_mask[x] = m

    def is_valid_add(mask, x):
        """Check if adding x to mask doesn't create a 4-AP."""
        return (mask & conflict_mask[x]) == 0

    def delta_add(mask, x):
        """Number of new 3-APs created by adding x to mask."""
        cnt = 0
        # x as first: need both partners in mask
        for y, z in first_partners[x]:
            if (mask >> (y - 1)) & 1 and (mask >> (z - 1)) & 1:
                cnt += 1
        # x as middle
        for y, z in mid_partners[x]:
            if (mask >> (y - 1)) & 1 and (mask >> (z - 1)) & 1:
                cnt += 1
        # x as last
        for y, z in last_partners[x]:
            if (mask >> (y - 1)) & 1 and (mask >> (z - 1)) & 1:
                cnt += 1
        return cnt

    def count_3aps(mask):
        """Total number of distinct 3-term APs in mask."""
        total = 0
        for x in range(1, N + 1):
            if (mask >> (x - 1)) & 1:
                m = mask & ~(1 << (x - 1))
                total += delta_add(m, x)
        return total

    def greedy_seed():
        """Greedy construction: add elements in order if no 4-AP is formed."""
        m = 0
        for x in range(1, N + 1):
            if is_valid_add(m, x):
                m |= (1 << (x - 1))
        return m

    def greedy_polish(mask, current_3ap):
        """Try to add any valid element that increases 3-AP count."""
        improved = True
        while improved:
            improved = False
            for x in range(1, N + 1):
                if not ((mask >> (x - 1)) & 1):
                    if is_valid_add(mask, x):
                        d = delta_add(mask, x)
                        if d > 0:
                            mask |= (1 << (x - 1))
                            current_3ap += d
                            improved = True
        return mask, current_3ap

    def simulated_annealing(mask, current_3ap, time_budget):
        """Run simulated annealing to improve 3-AP count while maintaining 4-AP-free constraint."""
        best_mask = mask
        best_3ap = current_3ap
        elems = [x for x in range(1, N + 1) if (mask >> (x - 1)) & 1]
        missing = [x for x in range(1, N + 1) if not ((mask >> (x - 1)) & 1)]
        
        T = 5.0  # initial temperature
        T_min = 0.01
        cooling = 0.9995
        
        while time.time() - start < time_budget:
            if T < T_min:
                T = T_min
            
            # Choose move type: swap (remove+add) or add or remove
            move_type = random.random()
            
            if move_type < 0.6 and elems and missing:
                # Swap: remove one, add another
                rem = random.choice(elems)
                add = random.choice(missing)
                
                new_mask = (mask & ~(1 << (rem - 1))) | (1 << (add - 1))
                
                # Check validity: add must not conflict with new_mask without add
                mask_without_add = new_mask & ~(1 << (add - 1))
                if not is_valid_add(mask_without_add, add):
                    continue
                
                # Compute delta
                rem_contrib = delta_add(mask & ~(1 << (rem - 1)), rem)
                add_contrib = delta_add(mask_without_add, add)
                delta = add_contrib - rem_contrib
                
                if delta >= 0 or random.random() < math.exp(delta / T):
                    mask = new_mask
                    current_3ap += delta
                    if current_3ap > best_3ap:
                        best_3ap = current_3ap
                        best_mask = mask
                    # Update elems and missing
                    elems.remove(rem)
                    elems.append(add)
                    missing.remove(add)
                    missing.append(rem)
                    
            elif move_type < 0.85 and missing:
                # Add
                add = random.choice(missing)
                if is_valid_add(mask, add):
                    d = delta_add(mask, add)
                    if d >= 0 or random.random() < math.exp(d / T):
                        mask |= (1 << (add - 1))
                        current_3ap += d
                        if current_3ap > best_3ap:
                            best_3ap = current_3ap
                            best_mask = mask
                        elems.append(add)
                        missing.remove(add)
                        
            elif elems:
                # Remove
                rem = random.choice(elems)
                rem_contrib = delta_add(mask & ~(1 << (rem - 1)), rem)
                delta = -rem_contrib
                if delta >= 0 or random.random() < math.exp(delta / T):
                    mask &= ~(1 << (rem - 1))
                    current_3ap += delta
                    if current_3ap > best_3ap:
                        best_3ap = current_3ap
                        best_mask = mask
                    elems.remove(rem)
                    missing.append(rem)
            
            T *= cooling
            
        return mask, current_3ap, best_mask, best_3ap

    def local_search(mask, current_3ap, best_mask, best_3ap, time_budget):
        """Run local search: try all swaps and additions for improvement."""
        while time.time() - start < time_budget:
            improved = False
            elems = [x for x in range(1, N + 1) if (mask >> (x - 1)) & 1]
            missing = [x for x in range(1, N + 1) if not ((mask >> (x - 1)) & 1)]

            if not elems or not missing:
                break

            random.shuffle(elems)
            random.shuffle(missing)

            # Try swaps
            for rem in elems:
                if time.time() - start > time_budget:
                    break
                rem_contrib = delta_add(mask & ~(1 << (rem - 1)), rem)
                for add in missing:
                    if time.time() - start > time_budget:
                        break
                    mask_without_rem = mask & ~(1 << (rem - 1))
                    if not is_valid_add(mask_without_rem, add):
                        continue
                    add_contrib = delta_add(mask_without_rem, add)
                    delta = add_contrib - rem_contrib
                    if delta > 0:
                        mask = mask_without_rem | (1 << (add - 1))
                        current_3ap += delta
                        if current_3ap > best_3ap:
                            best_3ap = current_3ap
                            best_mask = mask
                        improved = True
                        break
                if improved:
                    break

            if not improved:
                # Try single additions
                found = False
                for add in missing:
                    if time.time() - start > time_budget:
                        break
                    if not is_valid_add(mask, add):
                        continue
                    delta = delta_add(mask, add)
                    if delta > 0:
                        mask |= (1 << (add - 1))
                        current_3ap += delta
                        if current_3ap > best_3ap:
                            best_3ap = current_3ap
                            best_mask = mask
                        found = True
                        break
                if not found:
                    # Perturbation: remove a random element to escape local optimum
                    if elems:
                        rem = random.choice(elems)
                        rem_contrib = delta_add(mask & ~(1 << (rem - 1)), rem)
                        mask &= ~(1 << (rem - 1))
                        current_3ap -= rem_contrib

        return mask, current_3ap, best_mask, best_3ap

    # Multi-start with different strategies
    best_mask = 0
    best_3ap = 0

    # Start 1: greedy seed + polish
    mask = greedy_seed()
    current_3ap = count_3aps(mask)
    mask, current_3ap = greedy_polish(mask, current_3ap)
    if current_3ap > best_3ap:
        best_3ap = current_3ap
        best_mask = mask

    # Local search on greedy
    if time.time() - start < DEADLINE - 5:
        time_budget = min(time.time() - start + 10, DEADLINE - 10)
        mask, current_3ap, best_mask, best_3ap = local_search(
            mask, current_3ap, best_mask, best_3ap, time_budget
        )

    # Start 2: simulated annealing from greedy
    if time.time() - start < DEADLINE - 5:
        time_budget = min(time.time() - start + 10, DEADLINE - 5)
        mask = best_mask
        current_3ap = count_3aps(mask)
        mask, current_3ap, best_mask, best_3ap = simulated_annealing(
            mask, current_3ap, time_budget
        )

    # Start 3-5: random restarts
    for restart in range(3, 6):
        if time.time() - start > DEADLINE - 5:
            break
        time_budget = min(time.time() - start + 8, DEADLINE - 3)
        
        # Create perturbed version
        mask = best_mask
        elems = [x for x in range(1, N + 1) if (mask >> (x - 1)) & 1]
        num_remove = random.randint(5, 10)
        to_remove = random.sample(elems, min(num_remove, len(elems)))
        for x in to_remove:
            mask &= ~(1 << (x - 1))
        
        current_3ap = count_3aps(mask)
        # Polish
        mask, current_3ap = greedy_polish(mask, current_3ap)
        
        # Simulated annealing
        mask, current_3ap, best_mask, best_3ap = simulated_annealing(
            mask, current_3ap, time_budget
        )

    # Final polish
    if time.time() - start < DEADLINE - 1:
        current_3ap = count_3aps(best_mask)
        best_mask, best_3ap = greedy_polish(best_mask, current_3ap)
        
        # One more local search
        if time.time() - start < DEADLINE - 1:
            best_mask, best_3ap, best_mask, best_3ap = local_search(
                best_mask, best_3ap, best_mask, best_3ap, DEADLINE
            )

    # Convert best_mask to sorted list
    result = [x for x in range(1, N + 1) if (best_mask >> (x - 1)) & 1]
    return sorted(result)
