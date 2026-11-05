import random
import time
import numpy as np

N = 16
TOTAL = N * N
DEADLINE = time.time() + 280.0

# Precomputation: conflict[i][j] = bitmask of k such that {i,j,k} is isosceles
conflict = np.zeros((TOTAL, TOTAL), dtype=object)

def _precompute():
    pts = [(i // N, i % N) for i in range(TOTAL)]
    for i in range(TOTAL):
        for j in range(TOTAL):
            if i == j:
                continue
            pi = pts[i]
            pj = pts[j]
            d2_ij = (pi[0]-pj[0])**2 + (pi[1]-pj[1])**2
            mask = 0
            for k in range(TOTAL):
                if k == i or k == j:
                    continue
                pk = pts[k]
                d2_ik = (pi[0]-pk[0])**2 + (pi[1]-pk[1])**2
                d2_jk = (pj[0]-pk[0])**2 + (pj[1]-pk[1])**2
                if d2_ij == d2_ik or d2_ij == d2_jk or d2_ik == d2_jk:
                    mask |= (1 << k)
            conflict[i][j] = mask

_precompute()

def _point_idx(p):
    return p[0] * N + p[1]

def _idx_point(i):
    return (i // N, i % N)

def _creates_isosceles_idx(p_idx, S_bits):
    """Check if adding p_idx to S creates any isosceles triangle."""
    s = S_bits
    while s:
        lsb = s & -s
        a_idx = lsb.bit_length() - 1
        s ^= lsb
        if S_bits & conflict[p_idx][a_idx]:
            return True
    return False

def _is_maximal(S_bits):
    """Check if S is maximal (no point can be added without creating isosceles)."""
    for p_idx in range(TOTAL):
        if S_bits & (1 << p_idx):
            continue
        if not _creates_isosceles_idx(p_idx, S_bits):
            return False
    return True

def _greedy_order(order, S_bits=0):
    """Greedy: add points in given order if they don't create isosceles."""
    for p_idx in order:
        if not (S_bits & (1 << p_idx)):
            if not _creates_isosceles_idx(p_idx, S_bits):
                S_bits |= (1 << p_idx)
    return S_bits

def _greedy_maximal_from_seed(seed_indices, rng):
    """Start from seed, then greedily fill to maximal."""
    S_bits = 0
    for idx in seed_indices:
        if not _creates_isosceles_idx(idx, S_bits):
            S_bits |= (1 << idx)
    
    # Greedy fill with random order
    remaining = [i for i in range(TOTAL) if not (S_bits & (1 << i))]
    rng.shuffle(remaining)
    for p_idx in remaining:
        if not _creates_isosceles_idx(p_idx, S_bits):
            S_bits |= (1 << p_idx)
    return S_bits

def _remove_and_repair(S_bits, rng, max_repair_attempts=5):
    """
    Try to remove a random point from S, then repair to maximal with minimal additions.
    Returns (new_S_bits, new_count) or (None, None) if no improvement found.
    """
    # Get list of points in S
    s = S_bits
    point_list = []
    while s:
        lsb = s & -s
        idx = lsb.bit_length() - 1
        point_list.append(idx)
        s ^= lsb
    
    if len(point_list) < 2:
        return None, None
    
    # Try removing each point (random order)
    rng.shuffle(point_list)
    
    for removed_idx in point_list:
        candidate = S_bits & ~(1 << removed_idx)
        
        # Check if candidate is still maximal
        if _is_maximal(candidate):
            return candidate, len(point_list) - 1
        
        # Try to repair: add back minimal points to make it maximal
        # Use greedy with constraint propagation
        for _ in range(max_repair_attempts):
            repaired = candidate
            # Find points that can be added
            addable = []
            for p_idx in range(TOTAL):
                if repaired & (1 << p_idx):
                    continue
                if not _creates_isosceles_idx(p_idx, repaired):
                    addable.append(p_idx)
            
            if not addable:
                # Already maximal?
                if _is_maximal(repaired):
                    count = bin(repaired).count('1')
                    if count < len(point_list):
                        return repaired, count
                break
            
            # Pick the point that blocks the most other addable points
            best_p = None
            best_blocks = -1
            for p_idx in addable:
                new_repaired = repaired | (1 << p_idx)
                blocks = 0
                for q_idx in addable:
                    if q_idx == p_idx:
                        continue
                    if _creates_isosceles_idx(q_idx, new_repaired):
                        blocks += 1
                if blocks > best_blocks:
                    best_blocks = blocks
                    best_p = p_idx
            
            if best_p is None:
                break
            
            repaired |= (1 << best_p)
        
        # Check if repaired is maximal and smaller
        if _is_maximal(repaired):
            count = bin(repaired).count('1')
            if count < len(point_list):
                return repaired, count
    
    return None, None

def _simulated_annealing(initial_bits, rng, start_temp=10.0, end_temp=0.1, 
                          cooling=0.995, max_iters=5000):
    """
    Simulated annealing to find smaller maximal no-isosceles sets.
    Moves: remove a point, then repair to maximal.
    Accept if new size < old size, or with probability exp(-delta/T).
    """
    current_bits = initial_bits
    current_count = bin(current_bits).count('1')
    best_bits = initial_bits
    best_count = current_count
    
    temp = start_temp
    
    for it in range(max_iters):
        if time.time() > DEADLINE:
            break
        
        # Try remove-and-repair
        new_bits, new_count = _remove_and_repair(current_bits, rng, max_repair_attempts=3)
        
        if new_bits is None:
            # Try a different move: swap - remove one, add another
            s = current_bits
            point_list = []
            while s:
                lsb = s & -s
                idx = lsb.bit_length() - 1
                point_list.append(idx)
                s ^= lsb
            
            if len(point_list) < 2:
                break
                
            # Pick random point to remove
            remove_idx = rng.choice(point_list)
            candidate = current_bits & ~(1 << remove_idx)
            
            # Try to add a different point to make it maximal
            added = False
            for _ in range(20):  # Try a few random additions
                add_idx = rng.randint(0, TOTAL - 1)
                if candidate & (1 << add_idx):
                    continue
                if _creates_isosceles_idx(add_idx, candidate):
                    continue
                new_candidate = candidate | (1 << add_idx)
                if _is_maximal(new_candidate):
                    new_bits = new_candidate
                    new_count = bin(new_bits).count('1')
                    added = True
                    break
            
            if not added:
                temp *= cooling
                continue
        
        # Decide whether to accept
        delta = new_count - current_count
        if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-10)):
            current_bits = new_bits
            current_count = new_count
            if current_count < best_count:
                best_bits = current_bits
                best_count = current_count
        
        temp *= cooling
    
    return best_bits, best_count

def _bits_to_list(bits):
    res = []
    s = bits
    while s:
        lsb = s & -s
        idx = lsb.bit_length() - 1
        res.append(_idx_point(idx))
        s ^= lsb
    return res

def construct():
    rng = random.Random(42)
    best_bits = 0
    best_len = float('inf')
    
    all_indices = list(range(TOTAL))
    
    # Phase 1: Multi-start greedy with different strategies
    n_starts = 500
    for _ in range(n_starts):
        if time.time() > DEADLINE:
            break
        
        # Random seed of size 1-4
        seed_size = rng.randint(1, 4)
        seed_indices = rng.sample(all_indices, k=seed_size)
        res_bits = _greedy_maximal_from_seed(seed_indices, rng)
        
        if _is_maximal(res_bits):
            l = bin(res_bits).count('1')
            if l < best_len:
                best_len = l
                best_bits = res_bits
    
    # Phase 2: Simulated annealing from best solution
    if best_bits and time.time() < DEADLINE - 20:
        sa_bits, sa_count = _simulated_annealing(best_bits, rng, 
                                                  start_temp=5.0, 
                                                  end_temp=0.05,
                                                  cooling=0.998,
                                                  max_iters=3000)
        if sa_count < best_len:
            best_bits = sa_bits
            best_len = sa_count
    
    # Phase 3: More random starts with SA
    if time.time() < DEADLINE - 10:
        for _ in range(200):
            if time.time() > DEADLINE:
                break
            
            seed_size = rng.randint(2, 5)
            seed_indices = rng.sample(all_indices, k=seed_size)
            res_bits = _greedy_maximal_from_seed(seed_indices, rng)
            
            if _is_maximal(res_bits):
                l = bin(res_bits).count('1')
                if l < best_len:
                    best_len = l
                    best_bits = res_bits
                    # Quick SA
                    if time.time() < DEADLINE - 5:
                        sa_bits, sa_count = _simulated_annealing(best_bits, rng,
                                                                  start_temp=3.0,
                                                                  end_temp=0.05,
                                                                  cooling=0.999,
                                                                  max_iters=500)
                        if sa_count < best_len:
                            best_bits = sa_bits
                            best_len = sa_count
    
    return _bits_to_list(best_bits)
