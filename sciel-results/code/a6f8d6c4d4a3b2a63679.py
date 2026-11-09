"""
Maximum independent set in the conflict graph on [1, 512].
Conflicts: lcm(a,b) > 512. Uses greedy MIS with degree ordering,
structural exploitation via prime-power dominance, and 2-swap local search.
"""
import time
import random
from math import gcd

N = 512
DEADLINE = 85.0


def _build_conflict():
    """
    Build conflict adjacency as bitmasks.
    conf[v] has bit u set iff lcm(v,u) > 512 (v and u conflict).
    """
    conf = [0] * (N + 1)
    for a in range(1, N + 1):
        mask = 0
        for b in range(1, N + 1):
            if a == b:
                continue
            g = gcd(a, b)
            if a * b // g > N:
                mask |= (1 << b)
        conf[a] = mask
    return conf


def _popcount(x):
    return bin(x).count('1')


def _greedy_mis(conf, start_mask, deadline):
    """
    Greedy maximum independent set on conflict graph.
    Repeatedly pick vertex with minimum conflict-degree within remaining candidates.
    """
    independent = []
    remaining = start_mask

    while remaining:
        if time.time() > deadline:
            break
        best_v = -1
        best_deg = N + 1
        m = remaining
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            deg = _popcount(conf[v] & remaining)
            if deg < best_deg:
                best_deg = deg
                best_v = v
            m ^= lsb

        if best_v < 0:
            break

        independent.append(best_v)
        # Remove best_v and all its conflicts from remaining
        remaining &= ~(1 << best_v)
        remaining &= ~conf[best_v]

    return independent


def _greedy_mis_max(conf, start_mask, deadline):
    """
    Greedy MIS picking vertex with maximum non-conflict degree (i.e., most compatible).
    """
    independent = []
    remaining = start_mask

    while remaining:
        if time.time() > deadline:
            break
        best_v = -1
        best_score = -1
        m = remaining
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            # Score: number of non-conflicting vertices in remaining
            non_conf = remaining & ~conf[v] & ~(1 << v)
            score = _popcount(non_conf)
            if score > best_score:
                best_score = score
                best_v = v
            m ^= lsb

        if best_v < 0:
            break

        independent.append(best_v)
        remaining &= ~(1 << best_v)
        remaining &= ~conf[best_v]

    return independent


def _local_search_2swap(conf, independent, deadline):
    """
    2-swap local search: try removing two vertices and adding two non-members.
    Also try 1-remove-1-add and 1-remove-2-add.
    """
    ind_set = set(independent)
    n_ind = len(independent)
    full_mask = (1 << (N + 1)) - 2  # bits 1..512

    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break

        # Try 1-remove-1-add
        for i in range(n_ind):
            if time.time() > deadline:
                break
            removed = independent[i]
            remaining = [v for v in independent if v != removed]
            rem_set = set(remaining)

            # Find vertices not in independent that don't conflict with any remaining
            addable = full_mask
            for u in remaining:
                addable &= ~conf[u]
            addable &= ~rem_set
            addable &= ~(1 << removed)  # don't re-add same vertex

            if addable == 0:
                continue

            # Pick best addable vertex (fewest conflicts with other addables)
            m = addable
            best_v = -1
            best_conf = N + 1
            while m:
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                m ^= lsb
                c = _popcount(conf[v] & addable)
                if c < best_conf:
                    best_conf = c
                    best_v = v

            if best_v > 0:
                new_ind = remaining + [best_v]
                if len(new_ind) > n_ind:
                    independent = new_ind
                    ind_set = set(independent)
                    n_ind = len(independent)
                    improved = True
                    break

        # Try 1-remove-2-add
        if not improved:
            for i in range(n_ind):
                if time.time() > deadline:
                    break
                removed = independent[i]
                remaining = [v for v in independent if v != removed]
                rem_set = set(remaining)

                addable = full_mask
                for u in remaining:
                    addable &= ~conf[u]
                addable &= ~rem_set
                addable &= ~(1 << removed)

                if _popcount(addable) < 2:
                    continue

                # Find pair in addable that don't conflict with each other
                verts = []
                m = addable
                while m:
                    lsb = m & (-m)
                    v = lsb.bit_length() - 1
                    verts.append(v)
                    m ^= lsb

                found_pair = False
                for a_idx in range(len(verts)):
                    if time.time() > deadline:
                        break
                    va = verts[a_idx]
                    for b_idx in range(a_idx + 1, len(verts)):
                        vb = verts[b_idx]
                        if not (conf[va] & (1 << vb)):
                            # No conflict between va and vb
                            new_ind = remaining + [va, vb]
                            if len(new_ind) > n_ind:
                                independent = new_ind
                                ind_set = set(independent)
                                n_ind = len(independent)
                                improved = True
                                found_pair = True
                                break
                    if found_pair:
                        break
                if found_pair:
                    break

    return independent


def _structural_init_conflict():
    """
    Initial independent set using structural knowledge.
    Powers of 2: {1, 2, 4, 8, ..., 512} - pairwise lcm is the max, so <= 512.
    Also add odd numbers that are coprime to powers of 2 up to their limit.
    """
    base = []
    v = 1
    while v <= N:
        base.append(v)
        v *= 2
    return base


def construct():
    start_time = time.time()

    # Build conflict graph
    conf = _build_conflict()

    # Full mask
    full_mask = (1 << (N + 1)) - 2

    # Initial solution from structural knowledge
    best_ind = _structural_init_conflict()

    # Greedy MIS from full graph
    g1 = _greedy_mis(conf, full_mask, DEADLINE - 50)
    if len(g1) > len(best_ind):
        best_ind = g1

    g2 = _greedy_mis_max(conf, full_mask, DEADLINE - 50)
    if len(g2) > len(best_ind):
        best_ind = g2

    # Local search on best
    best_ind = _local_search_2swap(conf, list(best_ind), DEADLINE - 30)

    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 25:
        return list(best_ind)

    # Multiple random restarts with different orderings
    rng = random.Random(123)
    for restart in range(200):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 20:
            break

        # Random permutation for greedy
        verts = list(range(1, N + 1))
        rng.shuffle(verts)
        # Build mask in shuffled order - but greedy_mis picks min-degree, so ordering matters less
        # Instead, use a biased mask
        bias = rng.randint(100, 400)
        sample = rng.sample(range(1, N + 1), bias)
        sample_mask = 0
        for v in sample:
            sample_mask |= (1 << v)

        # Try both greedy variants
        r1 = _greedy_mis(conf, sample_mask, DEADLINE - 20)
        if len(r1) > len(best_ind):
            best_ind = r1
            best_ind = _local_search_2swap(conf, list(best_ind), DEADLINE - 20)

        r2 = _greedy_mis_max(conf, sample_mask, DEADLINE - 20)
        if len(r2) > len(best_ind):
            best_ind = r2
            best_ind = _local_search_2swap(conf, list(best_ind), DEADLINE - 20)

    # Final local search
    best_ind = _local_search_2swap(conf, list(best_ind), DEADLINE - 2)

    return list(best_ind)
