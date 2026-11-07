"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

_N = P['n']
_EDGES = P['edges']
_MINV = P['minimum_vertices']

# Build adjacency bitsets (full graph) for fast induced-degree computation.
_ADJ = [0] * _N
for _a, _b in _EDGES:
    _ADJ[_a] |= (1 << _b)
    _ADJ[_b] |= (1 << _a)


def _score_mask(mask):
    """Number of distinct induced degrees for vertices in mask."""
    if bin(mask).count('1') < _MINV:
        return -1
    degs = set()
    m = mask
    while m:
        lsb = m & (-m)
        v = lsb.bit_length() - 1
        degs.add((_ADJ[v] & mask).bit_count())
        m ^= lsb
    return len(degs)


def _exhaustive_small_search(max_size=10, deadline=None):
    """Exhaustively search all subsets of size 8 to max_size for best distinct degree count."""
    best_mask = 0
    best_score = 0
    for size in range(_MINV, max_size + 1):
        for combo in combinations(range(_N), size):
            if deadline and time.time() > deadline:
                return best_mask, best_score
            mask = 0
            for v in combo:
                mask |= (1 << v)
            s = _score_mask(mask)
            if s > best_score:
                best_score = s
                best_mask = mask
    return best_mask, best_score


def _mask_to_list(mask):
    """Convert bitmask to sorted list of vertex indices."""
    result = []
    m = mask
    while m:
        lsb = m & (-m)
        v = lsb.bit_length() - 1
        result.append(v)
        m ^= lsb
    return result


def _random_mask(rng, minv, maxv=None):
    """Generate a random valid mask with size between minv and maxv."""
    if maxv is None:
        maxv = _N
    k = rng.randint(minv, maxv)
    verts = rng.sample(range(_N), k)
    mask = 0
    for v in verts:
        mask |= (1 << v)
    return mask


def _hill_climb_1(mask, score, rng, deadline, best_mask, best_score):
    """Greedy 1-move hill climbing."""
    while time.time() < deadline:
        improved = False
        for v in range(_N):
            bit = 1 << v
            new_mask = mask ^ bit
            if bin(new_mask).count('1') < _MINV:
                continue
            s = _score_mask(new_mask)
            if s > score:
                mask = new_mask
                score = s
                improved = True
                if score > best_score:
                    best_score = s
                    best_mask = mask
                break
        if not improved:
            break
    return mask, score, best_mask, best_score


def _hill_climb_2(mask, score, rng, deadline, best_mask, best_score):
    """Hill climbing with 1-move and 2-move neighborhoods."""
    while time.time() < deadline:
        improved = False
        
        # First try 1-moves
        for v in range(_N):
            bit = 1 << v
            new_mask = mask ^ bit
            if bin(new_mask).count('1') < _MINV:
                continue
            s = _score_mask(new_mask)
            if s > score:
                mask = new_mask
                score = s
                improved = True
                if score > best_score:
                    best_score = s
                    best_mask = mask
                break
        
        if improved:
            continue
        
        # Try 2-moves (toggle two vertices)
        found_2 = False
        for v1 in range(_N):
            if time.time() >= deadline:
                break
            for v2 in range(v1 + 1, _N):
                bit1 = 1 << v1
                bit2 = 1 << v2
                new_mask = mask ^ bit1 ^ bit2
                if bin(new_mask).count('1') < _MINV:
                    continue
                s = _score_mask(new_mask)
                if s > score:
                    mask = new_mask
                    score = s
                    improved = True
                    found_2 = True
                    if score > best_score:
                        best_score = s
                        best_mask = mask
                    break
            if found_2:
                break
        
        if not improved:
            break
    
    return mask, score, best_mask, best_score


def _simulated_annealing(mask, score, rng, deadline, best_mask, best_score):
    """Simulated annealing to escape local optima."""
    T = 2.0
    T_min = 0.01
    cooling = 0.98
    
    while time.time() < deadline and T > T_min:
        v = rng.randrange(_N)
        new_mask = mask ^ (1 << v)
        if bin(new_mask).count('1') < _MINV:
            continue
        
        new_score = _score_mask(new_mask)
        delta = new_score - score
        
        if delta > 0:
            mask = new_mask
            score = new_score
            if score > best_score:
                best_score = score
                best_mask = mask
        else:
            if rng.random() < math.exp(delta / T):
                mask = new_mask
                score = new_score
        
        T *= cooling
    
    return mask, score, best_mask, best_score


def _swap_annealing(vertices, score, rng, deadline, best_mask, best_score):
    """Simulated annealing with swap moves (keep size fixed, swap in/out)."""
    n = _N
    vset = set(vertices)
    T = 3.0
    T_min = 0.05
    cooling = 0.995
    current_vertices = list(vertices)
    current_score = score
    
    while time.time() < deadline and T > T_min:
        # Pick a vertex to remove and a vertex to add
        if not current_vertices:
            break
        vin = rng.choice(current_vertices)
        vout_candidates = [v for v in range(n) if v not in vset]
        if not vout_candidates:
            break
        vout = rng.choice(vout_candidates)
        
        new_vertices = [v for v in current_vertices if v != vin] + [vout]
        new_mask = 0
        for v in new_vertices:
            new_mask |= (1 << v)
        
        if bin(new_mask).count('1') < _MINV:
            T *= cooling
            continue
        
        new_score = _score_mask(new_mask)
        delta = new_score - current_score
        
        if delta > 0:
            current_vertices = new_vertices
            current_score = new_score
            vset = set(current_vertices)
            mask_val = 0
            for v in current_vertices:
                mask_val |= (1 << v)
            if current_score > best_score:
                best_score = current_score
                best_mask = mask_val
        else:
            if rng.random() < math.exp(delta / T):
                current_vertices = new_vertices
                current_score = new_score
                vset = set(current_vertices)
        
        T *= cooling
    
    return current_vertices, current_score, best_mask, best_score


def _greedy_construct(target_size, rng, deadline):
    """Greedy construction picking vertices to maximize degree diversity."""
    verts = []
    remaining = list(range(_N))
    rng.shuffle(remaining)
    
    for _ in range(target_size):
        if not remaining:
            break
        best_add = None
        best_add_score = -1
        for v in remaining:
            trial = verts + [v]
            mask = 0
            for x in trial:
                mask |= (1 << x)
            s = _score_mask(mask)
            if s > best_add_score:
                best_add_score = s
                best_add = v
        if best_add is not None:
            verts.append(best_add)
            remaining.remove(best_add)
    
    mask = 0
    for v in verts:
        mask |= (1 << v)
    return verts, _score_mask(mask)


def construct():
    deadline = time.time() + 80.0  # safety margin under 90s kill
    rng = random.Random(12345)

    best_mask = None
    best_score = 0

    def consider(mask):
        nonlocal best_mask, best_score
        s = _score_mask(mask)
        if s > best_score:
            best_score = s
            best_mask = mask

    # Phase 0: Exhaustive search over small subsets (size 8-10)
    # C(17,8)=24310, C(17,9)=24310, C(17,10)=19448 - feasible within time budget
    ex_mask, ex_score = _exhaustive_small_search(max_size=10, deadline=deadline)
    if ex_score > best_score:
        best_score = ex_score
        best_mask = ex_mask

    # Seed starts: multiple diverse random initializations
    seed_mask = 0
    for v in range(9):
        seed_mask |= (1 << v)
    consider(seed_mask)

    # Many random initial masks of varying sizes
    for _ in range(50):
        consider(_random_mask(rng, _MINV, _N))

    # Multi-phase local search with different neighborhood structures
    current_mask = best_mask
    current_score = best_score

    # Phase 1: 1-move hill climbing from best
    current_mask, current_score, best_mask, best_score = _hill_climb_1(
        current_mask, current_score, rng, deadline, best_mask, best_score
    )

    # Phase 2: 2-move hill climbing from best
    if time.time() < deadline:
        current_mask, current_score, best_mask, best_score = _hill_climb_2(
            best_mask, best_score, rng, deadline, best_mask, best_score
        )

    # Phase 3: Simulated annealing from best
    if time.time() < deadline:
        current_mask, current_score, best_mask, best_score = _simulated_annealing(
            best_mask, best_score, rng, deadline, best_mask, best_score
        )

    # Phase 4: More random restarts with 2-move climbing
    restarts = 0
    while time.time() < deadline and restarts < 20:
        current_mask = _random_mask(rng, _MINV, _N)
        current_score = _score_mask(current_mask)
        if current_score > best_score:
            best_score = current_score
            best_mask = current_mask
        
        current_mask, current_score, best_mask, best_score = _hill_climb_2(
            current_mask, current_score, rng, deadline, best_mask, best_score
        )
        restarts += 1

    # Phase 5: Swap-based annealing on the best solution
    if time.time() < deadline and best_mask is not None:
        best_verts = _mask_to_list(best_mask)
        _, _, best_mask, best_score = _swap_annealing(
            best_verts, best_score, rng, deadline, best_mask, best_score
        )

    # Phase 6: Greedy construction with multiple random orderings
    if time.time() < deadline:
        for target_size in range(_MINV, min(_MINV + 4, _N + 1)):
            for _ in range(30):
                if time.time() >= deadline:
                    break
                verts, s = _greedy_construct(target_size, rng, deadline)
                if verts:
                    mask = 0
                    for v in verts:
                        mask |= (1 << v)
                    if s > best_score:
                        best_score = s
                        best_mask = mask
                    # Polish with 2-move climbing
                    if time.time() < deadline:
                        _, _, best_mask, best_score = _hill_climb_2(
                            mask, s, rng, deadline, best_mask, best_score
                        )

    # Phase 7: Final polish with simulated annealing from best
    if time.time() < deadline and best_mask is not None:
        _simulated_annealing(best_mask, best_score, rng, deadline, best_mask, best_score)

    # Fallback: ensure we always have a valid result.
    if best_mask is None:
        best_mask = seed_mask
        best_score = _score_mask(best_mask)

    # Convert mask to a list of plain Python ints.
    result = _mask_to_list(best_mask)
    result.sort()
    return list(result)
# EVOLVE-BLOCK-END
