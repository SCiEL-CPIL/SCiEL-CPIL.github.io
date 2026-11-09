"""HP lattice protein folding: Maximum contact set via branch-and-bound on conflict graph."""
# EVOLVE-BLOCK-START
import random
import time
import math
import heapq

_GLOBAL_START = time.time()
_GLOBAL_DEADLINE = _GLOBAL_START + 205.0

_DELTAS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _remaining_time():
    return max(0.0, _GLOBAL_DEADLINE - time.time())


def _energy(sequence, chain):
    """Compute energy: -1 for each H-H contact on non-adjacent lattice neighbours."""
    n = len(chain)
    if n < 2:
        return 0
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            x, y = p
            for dx, dy in _DELTAS:
                q = (x + dx, y + dy)
                j = where.get(q)
                if j is not None and abs(i - j) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts // 2


def _greedy_grow(sequence, rng, hh_weight=1.0, free_weight=0.1, noise=0.3):
    """Greedy growth with contact-aware bias."""
    nn = len(sequence)
    if nn < 2:
        return [(0, 0)] * nn
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, nn):
        options = []
        x, y = chain[-1]
        for site in _DELTAS:
            cand = (x + site[0], y + site[1])
            if cand in occupied:
                continue
            
            gain = 0
            if sequence[i] == 'H':
                cx, cy = cand
                for dx, dy in _DELTAS:
                    q = (cx + dx, cy + dy)
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
                        
            free = 0
            cx, cy = cand
            for dx, dy in _DELTAS:
                if (cx + dx, cy + dy) not in occupied:
                    free += 1
                        
            score = hh_weight * gain + free_weight * free + noise * rng.random()
            options.append((score, cand))
            
        if not options:
            return None
        
        options.sort(key=lambda x: x[0], reverse=True)
        if len(options) > 1 and rng.random() < 0.3:
            choice_idx = min(len(options) - 1, rng.randint(0, 2))
            best_score, best_site = options[choice_idx]
        else:
            best_score, best_site = options[0]
            
        occupied[best_site] = i
        chain.append(best_site)

    return chain


def _apply_reversal(chain, i, j):
    """Apply a reversal move, returning None if invalid."""
    nn = len(chain)
    if i >= j:
        return None
    if i > 0:
        px, py = chain[i - 1]
        ex, ey = chain[j]
        if abs(px - ex) + abs(py - ey) != 1:
            return None
    if j < nn - 1:
        sx, sy = chain[i]
        nx, ny = chain[j + 1]
        if abs(sx - nx) + abs(sy - ny) != 1:
            return None
    
    new_chain = chain[:i] + chain[i:j+1][::-1] + chain[j+1:]
    if len(set(new_chain)) != nn:
        return None
    return new_chain


def _apply_pivot(chain, i, angle):
    """Apply a pivot move, returning None if invalid."""
    nn = len(chain)
    if i <= 0 or i >= nn - 1:
        return None
    px, py = chain[i]
    tail = chain[i+1:]
    rotated = []
    for tx, ty in tail:
        rx, ry = tx - px, ty - py
        for _ in range(angle % 4):
            rx, ry = -ry, rx
        rotated.append((px + rx, py + ry))
    new_chain = chain[:i+1] + rotated
    if len(set(new_chain)) != nn:
        return None
    return new_chain


def _local_search(sequence, chain, time_limit, rng):
    """Iterated local search with simulated annealing."""
    nn = len(chain)
    current_energy = _energy(sequence, chain)
    best_energy = current_energy
    best_chain = chain[:]
    
    T = 1.5
    cooling = 0.9999
    min_t = 0.01
    
    deadline = time.time() + time_limit
    no_improve = 0
    max_no_improve = 2000
    
    h_indices = [i for i in range(nn) if sequence[i] == 'H']
    
    while time.time() < deadline:
        if _remaining_time() < 2.0:
            break

        move_type = rng.random()
        if move_type < 0.6:
            if h_indices and rng.random() < 0.5:
                hi = rng.choice(h_indices)
                span = rng.randint(2, min(15, nn - 1))
                i = max(0, hi - span // 2)
                j = min(nn - 1, i + span)
                if j <= i:
                    i = rng.randint(0, nn - 2)
                    j = rng.randint(i + 1, nn - 1)
            else:
                i = rng.randint(0, nn - 2)
                j = rng.randint(i + 1, nn - 1)
            new_chain = _apply_reversal(chain, i, j)
        else:
            if h_indices and rng.random() < 0.5:
                hi = rng.choice(h_indices)
                i = max(1, min(nn - 2, hi + rng.randint(-2, 2)))
            else:
                i = rng.randint(1, nn - 2)
            angle = rng.choice([1, 2, 3])
            new_chain = _apply_pivot(chain, i, angle)

        if new_chain is None:
            no_improve += 1
            if no_improve > max_no_improve:
                break
            continue

        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy

        accept = False
        if delta <= 0:
            accept = True
        elif T > min_t:
            exp_arg = -delta / T
            if exp_arg > -50:
                accept = rng.random() < math.exp(exp_arg)

        if accept:
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = chain[:]
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1

        if no_improve > max_no_improve:
            if _remaining_time() < 3.0:
                break
            chain = best_chain[:]
            current_energy = best_energy
            for _ in range(rng.randint(3, 8)):
                mt = rng.random()
                if mt < 0.6:
                    if h_indices and rng.random() < 0.5:
                        hi = rng.choice(h_indices)
                        span = rng.randint(2, min(15, nn - 1))
                        i = max(0, hi - span // 2)
                        j = min(nn - 1, i + span)
                        if j <= i:
                            i = rng.randint(0, nn - 2)
                            j = rng.randint(i + 1, nn - 1)
                    else:
                        i = rng.randint(0, nn - 2)
                        j = rng.randint(i + 1, nn - 1)
                    nc = _apply_reversal(chain, i, j)
                else:
                    if h_indices and rng.random() < 0.5:
                        hi = rng.choice(h_indices)
                        i = max(1, min(nn - 2, hi + rng.randint(-2, 2)))
                    else:
                        i = rng.randint(1, nn - 2)
                    angle = rng.choice([1, 2, 3])
                    nc = _apply_pivot(chain, i, angle)
                if nc is not None:
                    chain = nc
                    current_energy = _energy(sequence, chain)
                    break
            no_improve = 0
            T = max(T * 1.5, 0.5)
            max_no_improve = int(max_no_improve * 0.8)
            if max_no_improve < 100:
                break

        T *= cooling
        if T < min_t:
            break

    return best_chain, best_energy


def _branch_and_bound_contacts(sequence, chain, time_limit):
    """
    Branch-and-bound search to find maximum set of compatible H-H contacts.
    Nodes: potential contacts (pairs of H residues that are lattice neighbors but not chain-adjacent)
    Edges: conflicts (two contacts that cannot both be realized)
    """
    n = len(chain)
    h_positions = [i for i in range(n) if sequence[i] == 'H']
    if len(h_positions) < 2:
        return 0, []
    
    # Build contact map: which H-H pairs are currently in contact
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    
    current_contacts = set()
    for i in h_positions:
        x, y = chain[i]
        for dx, dy in _DELTAS:
            q = (x + dx, y + dy)
            j = where.get(q)
            if j is not None and j > i and abs(i - j) > 1 and sequence[j] == 'H':
                current_contacts.add((i, j))
    
    # For each potential contact, determine if it's currently realized
    # A contact (i,j) is "realizable" if residues i and j can be placed adjacent
    
    # Conflict graph: two contacts (i,j) and (k,l) conflict if they share a residue
    # or if the geometric constraints are incompatible
    # Simplified: conflicts only if they share a residue index
    
    contact_list = list(current_contacts)
    if not contact_list:
        return 0, []
    
    # Build adjacency: contact a conflicts with contact b if they share an index
    m = len(contact_list)
    conflict = [set() for _ in range(m)]
    for a in range(m):
        ia, ja = contact_list[a]
        for b in range(a+1, m):
            ib, jb = contact_list[b]
            if ia == ib or ia == jb or ja == ib or ja == jb:
                conflict[a].add(b)
                conflict[b].add(a)
    
    # Greedy coloring for upper bound
    color = [-1] * m
    max_color = 0
    for v in range(m):
        used_colors = set()
        for u in conflict[v]:
            if color[u] != -1:
                used_colors.add(color[u])
        c = 0
        while c in used_colors:
            c += 1
        color[v] = c
        max_color = max(max_color, c + 1)
    
    # Branch and bound
    best_count = 0
    best_set = []
    
    def bb(node, chosen, forbidden):
        nonlocal best_count, best_set
        if _remaining_time() < 1.0:
            return
        
        # Upper bound: greedy coloring bound
        remaining = [v for v in range(m) if v not in chosen and v not in forbidden]
        if not remaining:
            if len(chosen) > best_count:
                best_count = len(chosen)
                best_set = list(chosen)
            return
        
        # Greedy bound
        bound = len(chosen) + len(remaining)
        if bound <= best_count:
            return
        
        # Pick a vertex with highest degree in remaining
        v = max(remaining, key=lambda x: len(conflict[x] & set(remaining)))
        
        # Branch: include v
        new_chosen = set(chosen) | {v}
        new_forbidden = set(forbidden) | {v} | conflict[v]
        bb(node, new_chosen, new_forbidden)
        
        # Branch: exclude v
        new_forbidden2 = set(forbidden) | {v}
        bb(node, chosen, new_forbidden2)
    
    bb(0, set(), set())
    return best_count, [contact_list[i] for i in best_set]


def _backtrack_fold(sequence, time_limit, rng):
    """
    Fold sequence using diverse initialization and local refinement.
    """
    n = len(sequence)
    if n < 2:
        return [(0, 0)] * n
    
    is_h = [c == 'H' for c in sequence]
    h_indices = [i for i in range(n) if is_h[i]]
    
    best_chain = None
    best_energy = 0

    # Generate diverse initial solutions
    initial_chains = []
    configs = [
        (2.0, 0.0, 0.5),
        (1.0, 0.5, 0.5),
        (0.5, 1.0, 0.5),
        (1.5, 0.2, 0.3),
        (1.0, 0.3, 1.0),
        (3.0, 0.0, 0.2),
    ]
    
    growth_deadline = time.time() + min(2.0, time_limit * 0.1)
    for hh_w, free_w, noise in configs:
        if time.time() > growth_deadline:
            break
        for attempt in range(2):
            if time.time() > growth_deadline:
                break
            chain = _greedy_grow(sequence, rng, hh_w, free_w, noise)
            if chain is not None:
                initial_chains.append(chain)

    if not initial_chains:
        initial_chains.append([(i, 0) for i in range(n)])

    # Evaluate and sort initial chains
    scored = []
    for chain in initial_chains:
        e = _energy(sequence, chain)
        scored.append((e, chain))
    scored.sort(key=lambda x: x[0])
    candidates = [s[1] for s in scored[:min(4, len(scored))]]

    # Local refinement
    total_time = _remaining_time() - 3.0
    if total_time < 1.0:
        return candidates[0] if candidates else [(i, 0) for i in range(n)]

    weights = [0.5, 0.3, 0.15, 0.05]
    n_cands = len(candidates)
    if n_cands < len(weights):
        weights = weights[:n_cands]
        w_sum = sum(weights)
        weights = [w / w_sum for w in weights]
    
    for idx, chain in enumerate(candidates):
        if _remaining_time() < 2.0:
            break
        alloc = total_time * weights[idx]
        alloc = min(alloc, _remaining_time() - 1.0)
        if alloc < 0.5:
            continue
        
        refined_chain, refined_energy = _local_search(sequence, chain, alloc, rng)
        
        if best_chain is None or refined_energy < best_energy:
            best_energy = refined_energy
            best_chain = refined_chain

    if best_chain is None:
        best_chain = candidates[0] if candidates else [(i, 0) for i in range(n)]

    return best_chain


def fold(sequence, dimension):
    """
    Fold the sequence into a low-energy conformation.
    """
    rng = random.Random(42)
    n = len(sequence)

    if n < 2:
        return [[0, 0]] * n

    time_limit = max(0.5, _remaining_time() - 5.0)

    best_chain = _backtrack_fold(sequence, time_limit, rng)

    # Validate
    if best_chain is None or len(best_chain) != n:
        best_chain = [(i, 0) for i in range(n)]

    seen = set()
    valid = True
    for i, (x, y) in enumerate(best_chain):
        if (x, y) in seen:
            valid = False
            break
        seen.add((x, y))
        if i > 0:
            px, py = best_chain[i-1]
            if abs(px - x) + abs(py - y) != 1:
                valid = False
                break

    if not valid:
        best_chain = [(i, 0) for i in range(n)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
