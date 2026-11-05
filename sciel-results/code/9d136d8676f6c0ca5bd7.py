import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # Safety margin before 60s eval limit

    N = 256
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1

    # Build adjacency as bitsets
    adj = [0] * N
    forbidden = [False] * N

    for i in range(N):
        v = i + 1
        if 2 * v in squares:
            forbidden[i] = True
        else:
            for sq in squares:
                u = sq - v
                if 1 <= u <= N:
                    j = u - 1
                    if not forbidden[j]:
                        adj[i] |= (1 << j)

    candidates = [i for i in range(N) if not forbidden[i]]
    num_candidates = len(candidates)

    cand_mask = 0
    for i in candidates:
        cand_mask |= (1 << i)

    def greedy_init(seed=None):
        """Greedy min-degree MIS with optional randomization."""
        if seed is not None:
            random.seed(seed)
        available_mask = cand_mask
        selected_mask = 0
        while available_mask:
            best_v = -1
            min_deg = N + 1
            m = available_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                m ^= lsb
                deg = (adj[v] & available_mask).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                    if deg == 0:
                        break
            if best_v == -1:
                break
            selected_mask |= (1 << best_v)
            available_mask &= ~((1 << best_v) | (adj[best_v] & available_mask))
        return selected_mask

    def greedy_random():
        """Greedy with random tie-breaking and noise."""
        available_mask = cand_mask
        selected_mask = 0
        while available_mask:
            best_v = -1
            min_deg = N + 1
            m = available_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                m ^= lsb
                deg = (adj[v] & available_mask).bit_count()
                deg += random.uniform(0, 1.0)
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
            if best_v == -1:
                break
            selected_mask |= (1 << best_v)
            available_mask &= ~((1 << best_v) | (adj[best_v] & available_mask))
        return selected_mask

    def local_improve(mask):
        """Add any vertex that doesn't conflict with current set."""
        improved = True
        while improved:
            improved = False
            not_in_mask = cand_mask & ~mask
            m = not_in_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                m ^= lsb
                if (adj[v] & mask) == 0:
                    mask |= lsb
                    improved = True
                    break
        return mask

    def try_exchange_11(mask):
        """Try 1-1 exchange: remove u, add v where v was blocked only by u."""
        m = mask
        while m:
            lsb_u = m & -m
            u = lsb_u.bit_length() - 1
            m ^= lsb_u
            # Find v in adj[u] that is not in mask and whose only conflict with mask is u
            potential = adj[u] & ~mask & cand_mask
            pm = potential
            while pm:
                lsb_v = pm & -pm
                v = lsb_v.bit_length() - 1
                pm ^= lsb_v
                # Check if v's only neighbor in mask is u
                if (adj[v] & mask) == lsb_u:
                    # Perform swap
                    return mask ^ lsb_u | lsb_v
        return mask

    def try_exchange_12(mask):
        """Try 1-2 exchange: remove u, add v1 and v2."""
        m = mask
        while m:
            lsb_u = m & -m
            u = lsb_u.bit_length() - 1
            m ^= lsb_u
            mask_wo_u = mask ^ lsb_u
            # Find pairs v1, v2 not in mask_wo_u, non-conflicting with mask_wo_u,
            # and non-conflicting with each other
            candidates_add = cand_mask & ~mask_wo_u
            # Filter: must not conflict with mask_wo_u
            valid_add = 0
            cm = candidates_add
            while cm:
                lsb_v = cm & -cm
                v = lsb_v.bit_length() - 1
                cm ^= lsb_v
                if (adj[v] & mask_wo_u) == 0:
                    valid_add |= lsb_v
            # Now find non-conflicting pairs in valid_add
            vm = valid_add
            while vm:
                lsb_v1 = vm & -vm
                v1 = lsb_v1.bit_length() - 1
                vm ^= lsb_v1
                # v2 must be in valid_add, not v1, and not conflict with v1
                remaining = valid_add & ~lsb_v1 & ~adj[v1]
                rm = remaining
                while rm:
                    lsb_v2 = rm & -rm
                    v2 = lsb_v2.bit_length() - 1
                    rm ^= lsb_v2
                    # Check v2 doesn't conflict with mask_wo_u (already ensured)
                    # and v1, v2 don't conflict with each other (ensured by ~adj[v1])
                    return mask_wo_u | lsb_v1 | lsb_v2
        return mask

    def local_search(mask, max_rounds=50):
        """Iterate local_improve and exchanges until no improvement."""
        for _ in range(max_rounds):
            if time.time() - start_time > DEADLINE:
                break
            old_mask = mask
            mask = local_improve(mask)
            # Try 1-1 exchanges
            for _ in range(10):
                new_mask = try_exchange_11(mask)
                if new_mask != mask:
                    mask = new_mask
                    break
            # Try 1-2 exchanges
            new_mask = try_exchange_12(mask)
            if new_mask != mask:
                mask = new_mask
            if mask == old_mask:
                break
        return mask

    def perturb(mask, strength=3):
        """Remove random vertices and optionally add random non-conflicting ones."""
        current = mask
        count = current.bit_count()
        remove_k = min(strength, count)
        for _ in range(remove_k):
            if not current:
                break
            # Pick random set bit
            bits = []
            m = current
            while m:
                lsb = m & -m
                bits.append(lsb)
                m ^= lsb
            if bits:
                current ^= random.choice(bits)
        # Optionally add some random non-conflicting vertices to explore new territory
        for _ in range(random.randint(0, 2)):
            not_in = cand_mask & ~current
            if not_in:
                m = not_in
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    m ^= lsb
                    if (adj[v] & current) == 0:
                        current |= lsb
                        break
        return current

    # Main: Iterated Local Search
    best_mask = 0
    best_size = 0

    # Multiple initializations
    for init_seed in [None, 42, 123, 777, 2024]:
        if time.time() - start_time > DEADLINE:
            break
        if init_seed is None:
            mask = greedy_init()
        else:
            mask = greedy_init(seed=init_seed)
        mask = local_search(mask)
        if mask.bit_count() > best_size:
            best_size = mask.bit_count()
            best_mask = mask

    # ILS loop
    no_improve_count = 0
    while time.time() - start_time < DEADLINE:
        # Perturb
        current = perturb(best_mask, strength=random.randint(2, 6))
        
        # Rebuild with greedy fill
        current = local_improve(current)
        
        # Local search
        current = local_search(current)
        
        if current.bit_count() > best_size:
            best_size = current.bit_count()
            best_mask = current
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Also try fresh random greedy periodically
        if no_improve_count % 10 == 0 or random.random() < 0.3:
            mask_r = greedy_random()
            mask_r = local_search(mask_r)
            if mask_r.bit_count() > best_size:
                best_size = mask_r.bit_count()
                best_mask = mask_r
                no_improve_count = 0

    # Final polish: try to add any remaining vertices
    best_mask = local_improve(best_mask)
    best_size = best_mask.bit_count()

    # Convert to list
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        result.append(v + 1)
        m ^= lsb

    return sorted(result)
