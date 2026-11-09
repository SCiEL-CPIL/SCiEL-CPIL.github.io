"""Fresh design: randomized greedy + removal optimization for linear triple systems on 16 points."""
import random
import time

_START = time.time()
_LIMIT = 110.0  # safety margin well under the 120/150s kill

def _point_masks():
    return [1 << i for i in range(16)]

def _independence_number(triples, pm):
    """Exact independence number via branch-and-bound on the shadow graph.
    
    The shadow graph has an edge (u,v) if u and v co-occur in some triple.
    A set S is independent (contains no full triple) iff no two points in S
    share a triple, i.e., S is an independent set in the shadow graph.
    So alpha(hypergraph) = alpha(shadow graph).
    """
    # Build adjacency bitmasks for the shadow graph
    adj = [0] * 16
    for (a, b, c) in triples:
        adj[a] |= (1 << b) | (1 << c)
        adj[b] |= (1 << a) | (1 << c)
        adj[c] |= (1 << a) | (1 << b)
    
    # Branch and bound for Maximum Independent Set
    best = [0]
    
    def popcount(x):
        return bin(x).count('1')
    
    def bnb(cand, size):
        # cand: bitmask of candidate vertices that can still be added
        # size: current size of the independent set
        if size + popcount(cand) <= best[0]:
            return
        if cand == 0:
            if size > best[0]:
                best[0] = size
            return
        # Upper bound: size + popcount(cand)
        # Choose vertex with highest degree in induced subgraph for branching
        # Pick vertex v from cand
        # Use a heuristic: pick vertex with max degree within cand
        best_v = -1
        best_deg = -1
        c = cand
        while c:
            v_bit = c & (-c)
            v = v_bit.bit_length() - 1
            deg = popcount(adj[v] & cand)
            if deg > best_deg:
                best_deg = deg
                best_v = v
            c ^= v_bit
        
        v = best_v
        v_bit = 1 << v
        
        # Branch 1: exclude v
        bnb(cand & ~v_bit, size)
        
        # Branch 2: include v (then remove v and all its neighbors from cand)
        bnb(cand & ~v_bit & ~adj[v], size + 1)
    
    bnb((1 << 16) - 1, 0)
    return best[0]

def _is_valid(triples):
    # any two triples share at most one point
    for i in range(len(triples)):
        a, b, c = triples[i]
        for j in range(i + 1, len(triples)):
            x, y, z = triples[j]
            inter = 0
            if a == x or a == y or a == z: inter += 1
            if b == x or b == y or b == z: inter += 1
            if c == x or c == y or c == z: inter += 1
            if inter > 1:
                return False
    return True

def _greedy_build(rng, pm, order_mode):
    # Build a linear triple system greedily.
    # order_mode: how to pick which triple to add next.
    points = list(range(16))
    rng.shuffle(points)
    triples = []
    # We'll maintain, for each point, the count of triples it's in, and a set of used point-pairs.
    used_pairs = set()
    # Build candidate pool incrementally.
    # Represent each point's incidence as a bitmask of which triples it's in? Simpler: track used pairs.
    # Greedy: repeatedly pick a triple that doesn't conflict, preferring triples covering "underused" points.
    # To keep it fast, precompute all C(16,3)=560 triples.
    all_triples = []
    for i in range(16):
        for j in range(i + 1, 16):
            for k in range(j + 1, 16):
                all_triples.append((i, j, k))
    # Shuffle candidate order
    cand = all_triples[:]
    rng.shuffle(cand)
    # We'll add triples until no more can be added (or a cap).
    # Track point usage count to bias.
    usage = [0] * 16
    max_rounds = 400
    added = 0
    while added < max_rounds:
        progressed = False
        # Re-shuffle remaining candidates each round is expensive; instead scan and pick a random valid one.
        # Scan candidates, collect valid ones, pick one.
        valid_now = []
        for t in cand:
            a, b, c = t
            # check pair conflicts
            if (a, b) in used_pairs or (a, c) in used_pairs or (b, c) in used_pairs:
                continue
            valid_now.append(t)
        if not valid_now:
            break
        # pick one, biased toward covering underused points
        if order_mode == 0:
            t = valid_now[rng.randrange(len(valid_now))]
        else:
            # weight = sum of (16 - usage[p]) for p in t
            best = None
            best_w = -1
            for t in valid_now:
                w = (16 - usage[t[0]]) + (16 - usage[t[1]]) + (16 - usage[t[2]])
                if w > best_w:
                    best_w = w
                    best = t
            t = best
        a, b, c = t
        triples.append(t)
        used_pairs.add((a, b)); used_pairs.add((a, c)); used_pairs.add((b, c))
        usage[a] += 1; usage[b] += 1; usage[c] += 1
        added += 1
        progressed = True
        if not progressed:
            break
    return triples

def _remove_to_optimize(triples, pm, rng, time_budget):
    # Given a valid system, try removing triples to reduce independence number.
    # alpha is monotone non-decreasing as we remove triples.
    # We want to remove triples that most reduce alpha.
    # Local search: try removing each triple, keep the one that gives best alpha, repeat.
    cur = list(triples)
    cur_alpha = _independence_number(cur, pm)
    improved = True
    start = time.time()
    while improved and (time.time() - start) < time_budget:
        improved = False
        best_alpha = cur_alpha
        best_idx = -1
        n = len(cur)
        # Try removing each triple
        for i in range(n):
            trial = cur[:i] + cur[i+1:]
            a = _independence_number(trial, pm)
            if a < best_alpha:
                best_alpha = a
                best_idx = i
                improved = True
            # early stop if we hit a very good value
        if improved:
            cur = cur[:best_idx] + cur[best_idx+1:]
            cur_alpha = best_alpha
    return cur, cur_alpha

def _swap_move(triples, rng, pm, time_budget):
    """Try swapping one triple for another to reduce alpha."""
    cur = list(triples)
    cur_alpha = _independence_number(cur, pm)
    start = time.time()
    improved = True
    all_triples = []
    for i in range(16):
        for j in range(i + 1, 16):
            for k in range(j + 1, 16):
                all_triples.append((i, j, k))
    
    while improved and (time.time() - start) < time_budget:
        improved = False
        best_alpha = cur_alpha
        best_swap = None
        n = len(cur)
        for i in range(n):
            # Try replacing cur[i] with each candidate triple
            old = cur[i]
            # Build set of pairs used by all other triples
            other_pairs = set()
            for j in range(n):
                if j == i:
                    continue
                a, b, c = cur[j]
                other_pairs.add((a, b)); other_pairs.add((a, c)); other_pairs.add((b, c))
            # Find candidates that don't conflict with other_pairs
            for cand in all_triples:
                a, b, c = cand
                if (a, b) in other_pairs or (a, c) in other_pairs or (b, c) in other_pairs:
                    continue
                # Also check it doesn't conflict with itself (shouldn't happen for valid triples)
                trial = cur[:i] + [cand] + cur[i+1:]
                a_val = _independence_number(trial, pm)
                if a_val < best_alpha:
                    best_alpha = a_val
                    best_swap = (i, cand)
                    improved = True
        if improved:
            i, cand = best_swap
            cur[i] = cand
            cur_alpha = best_alpha
    return cur, cur_alpha

def construct():
    rng = random.Random(12345)
    pm = _point_masks()
    best_alpha = 16
    best_triples = None
    start = time.time()

    # A few fixed heuristic order modes and random restarts
    order_modes = [0, 1]
    restarts = 0
    while (time.time() - start) < (_LIMIT - 20):
        mode = order_modes[restarts % len(order_modes)]
        triples = _greedy_build(rng, pm, mode)
        if not triples:
            restarts += 1
            continue
        # Post-optimization: first remove, then swap
        remaining = _LIMIT - 25 - (time.time() - start)
        if remaining < 0.5:
            # Just evaluate as-is
            a = _independence_number(triples, pm)
            if a < best_alpha:
                best_alpha = a
                best_triples = triples
            restarts += 1
            continue
        
        # Phase 1: removal optimization
        rem_budget = min(4.0, remaining * 0.4)
        opt_triples, a = _remove_to_optimize(triples, pm, rng, rem_budget)
        
        # Phase 2: swap optimization on the result
        swap_budget = min(4.0, remaining * 0.4)
        if swap_budget > 0.3:
            opt_triples, a = _swap_move(opt_triples, rng, pm, swap_budget)
        
        if a < best_alpha:
            best_alpha = a
            best_triples = opt_triples
        restarts += 1
        if best_alpha <= 6:
            break

    if best_triples is None:
        # Fallback: return the seed-like system (still valid)
        best_triples = [[3*i, 3*i+1, 3*i+2] for i in range(5)]

    # Final safety: ensure validity and within time
    if not _is_valid(best_triples):
        best_triples = [[3*i, 3*i+1, 3*i+2] for i in range(5)]
    return best_triples
