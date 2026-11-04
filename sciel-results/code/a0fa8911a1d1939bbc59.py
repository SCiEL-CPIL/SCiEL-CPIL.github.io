import time
import random
import math
import numpy as np

def _score_set(U):
    """Compute the score for a set U (list of nonnegative ints containing 0)."""
    if not U or 0 not in U:
        return 0.0
    U_set = set(U)
    n = len(U_set)
    if n < 2:
        return 0.0
    max_u = max(U_set)
    # Compute sumset and diffset efficiently
    sums = set()
    diffs = set()
    diffs.add(0)
    U_list = sorted(U_set)
    for i in range(n):
        ui = U_list[i]
        for j in range(i, n):
            sums.add(ui + U_list[j])
        for j in range(i + 1, n):
            d = U_list[j] - ui
            diffs.add(d)
            diffs.add(-d)
    # Score: 1 + log(|U-U|/|U+U|)/log(2*max U + 1)
    if len(sums) == 0:
        return 0.0
    ratio = len(diffs) / len(sums)
    if ratio <= 1.0:
        return 1.0
    if 2 * max_u + 1 <= 1:
        return 1.0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 1.0
    return 1.0 + log_ratio / log_denom

def _generate_candidates(U, limit_val, random_small_count=15):
    """
    Generate candidate elements for addition based on sums and differences of existing elements.
    Also adds arithmetic progressions and scaled versions for diversity.
    """
    U_set = set(U)
    candidates = set()
    U_list = sorted(U_set)
    n = len(U_list)
    for i in range(n):
        a = U_list[i]
        for j in range(i, n):
            b = U_list[j]
            s = a + b
            d = b - a
            if 0 < s <= limit_val and s not in U_set:
                candidates.add(s)
            if 0 < d <= limit_val and d not in U_set:
                candidates.add(d)
            # Add scaled versions
            for mult in [2, 3]:
                sc = a * mult
                if 0 < sc <= limit_val and sc not in U_set:
                    candidates.add(sc)
    # Add arithmetic progressions from existing elements
    for i in range(min(n, 5)):
        base = U_list[i]
        for step in [1, 2, 3, 5, 8, 13]:
            for k in range(1, 5):
                val = base + k * step
                if 0 < val <= limit_val and val not in U_set:
                    candidates.add(val)
    # Add random small integers for diversity
    for _ in range(random_small_count):
        r = random.randint(1, min(100, limit_val))
        if r not in U_set:
            candidates.add(r)
    # Add some random larger values
    for _ in range(5):
        r = random.randint(1, limit_val)
        if r not in U_set:
            candidates.add(r)
    return [c for c in candidates if 0 < c <= limit_val]

def _local_search_dynamic(U, deadline, max_val_limit=1000, max_n_limit=40):
    """
    Local search with dynamic candidate generation and simulated annealing-style acceptance.
    """
    U = sorted(set(U))
    current_score = _score_set(U)
    best_local_score = current_score
    best_local_set = list(U)
    
    # Initial candidate pool
    candidates = _generate_candidates(U, max_val_limit)
    
    no_improvement_count = 0
    max_no_improvement = 50
    
    # Temperature for simulated annealing
    temperature = 0.05
    cooling_rate = 0.98
    
    while time.time() < deadline - 2:
        if no_improvement_count > max_no_improvement:
            break
            
        improved = False
        move_type = random.random()
        
        # Update candidate pool periodically or if empty
        if not candidates or random.random() < 0.15:
            current_max = max(U) if U else 0
            new_limit = max(max_val_limit, current_max * 2)
            candidates = _generate_candidates(U, new_limit)
        
        if move_type < 0.45:
            # Try to ADD an element
            if len(U) < max_n_limit and candidates:
                # Try a few random candidates
                shuffled = candidates[:]
                random.shuffle(shuffled)
                for c in shuffled[:15]:
                    if c in U:
                        continue
                    new_U = U + [c]
                    new_U.sort()
                    new_score = _score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        # Regenerate candidates immediately as structure changed significantly
                        candidates = _generate_candidates(U, max(max_val_limit, max(U)*2))
                        break
                    # Simulated annealing: accept worse moves with probability
                    elif random.random() < math.exp((new_score - current_score) / temperature):
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = _generate_candidates(U, max(max_val_limit, max(U)*2))
                        break
        elif move_type < 0.65:
            # Try to REMOVE an element
            if len(U) > 2:
                removable = [x for x in U if x != 0]
                if removable:
                    c = random.choice(removable)
                    new_U = [x for x in U if x != c]
                    new_score = _score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = _generate_candidates(U, max_val_limit)
                    elif random.random() < math.exp((new_score - current_score) / temperature):
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = _generate_candidates(U, max_val_limit)
        elif move_type < 0.85:
            # Try to REPLACE an element
            if len(U) > 1 and candidates:
                idx = random.randint(1, len(U) - 1)
                old_val = U[idx]
                shuffled = candidates[:]
                random.shuffle(shuffled)
                for new_val in shuffled[:15]:
                    if new_val == old_val or new_val in U:
                        continue
                    new_U = U.copy()
                    new_U[idx] = new_val
                    new_U.sort()
                    new_score = _score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = _generate_candidates(U, max(max_val_limit, max(U)*2))
                        break
                    elif random.random() < math.exp((new_score - current_score) / temperature):
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = _generate_candidates(U, max(max_val_limit, max(U)*2))
                        break
        else:
            # Try SWAP two elements
            if len(U) > 2 and candidates:
                idx1 = random.randint(1, len(U) - 1)
                idx2 = random.randint(1, len(U) - 1)
                if idx1 != idx2:
                    shuffled = candidates[:]
                    random.shuffle(shuffled)
                    for new_val in shuffled[:10]:
                        if new_val in U:
                            continue
                        new_U = U.copy()
                        new_U[idx1] = new_val
                        new_U.sort()
                        new_score = _score_set(new_U)
                        if new_score > current_score:
                            U = new_U
                            current_score = new_score
                            improved = True
                            candidates = _generate_candidates(U, max(max_val_limit, max(U)*2))
                            break
                        elif random.random() < math.exp((new_score - current_score) / temperature):
                            U = new_U
                            current_score = new_score
                            improved = True
                            candidates = _generate_candidates(U, max(max_val_limit, max(U)*2))
                            break
            
        if improved:
            no_improvement_count = 0
            if current_score > best_local_score:
                best_local_score = current_score
                best_local_set = list(U)
        else:
            no_improvement_count += 1
        
        # Cool down temperature
        temperature *= cooling_rate
        if temperature < 0.005:
            temperature = 0.005

    return best_local_score, best_local_set

def _score_geometric(b, m):
    """Score a geometric progression U = {floor(b^k) for k=0..m} with 0."""
    if b <= 1.0 or m < 1:
        return -1.0
    U = [0]
    val = 1.0
    for k in range(1, m + 1):
        U.append(int(val))
        val *= b
    U = sorted(set(U))
    if len(U) < 2:
        return -1.0
    return _score_set(U)

def _continuous_geometric_search(deadline):
    """Perform continuous parameter sweep over geometric progressions."""
    # Phase 1: Coarse grid search to identify promising regions
    best_score = -1.0
    best_params = None
    
    # Coarse grid over b and m
    b_grid = np.linspace(1.3, 6.0, 25)
    m_grid = range(3, 20)
    
    for m in m_grid:
        if time.time() > deadline - 15:
            break
        for b in b_grid:
            s = _score_geometric(b, m)
            if s > best_score:
                best_score = s
                best_params = (b, m)
    
    if best_params is None:
        return None, -1.0
    
    # Phase 2: Ternary search around the best coarse point
    b0, m0 = best_params
    b_lo = max(1.3, b0 - 0.8)
    b_hi = min(6.0, b0 + 0.8)
    
    lo = b_lo
    hi = b_hi
    for _ in range(50):  # 50 iterations for fine-grained search
        if time.time() > deadline - 15:
            break
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        s1 = _score_geometric(m1, m0)
        s2 = _score_geometric(m2, m0)
        if s1 < s2:
            lo = m1
        else:
            hi = m2
    
    b_opt = (lo + hi) / 2.0
    s_opt = _score_geometric(b_opt, m0)
    
    # Also try adjacent m values
    for m_adj in [m0 - 2, m0 - 1, m0 + 1, m0 + 2]:
        if m_adj < 2 or time.time() > deadline - 15:
            continue
        # Quick ternary search for this m
        lo2 = max(1.3, b_opt - 0.5)
        hi2 = min(6.0, b_opt + 0.5)
        for _ in range(30):
            if time.time() > deadline - 15:
                break
            m1 = lo2 + (hi2 - lo2) / 3.0
            m2 = hi2 - (hi2 - lo2) / 3.0
            s1 = _score_geometric(m1, m_adj)
            s2 = _score_geometric(m2, m_adj)
            if s1 < s2:
                lo2 = m1
            else:
                hi2 = m2
        b_adj = (lo2 + hi2) / 2.0
        s_adj = _score_geometric(b_adj, m_adj)
        if s_adj > s_opt:
            s_opt = s_adj
            b_opt = b_adj
            m0 = m_adj
    
    # Construct the final set
    U = [0]
    val = 1.0
    for k in range(1, m0 + 1):
        U.append(int(val))
        val *= b_opt
    U = sorted(set(U))
    
    return U, s_opt

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    # Initial seeds
    best_set = [0, 1, 3]
    best_score = _score_set(best_set)
    
    # Structured seeds
    structured_candidates = [
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81], # Squares
        [0, 1, 2, 4, 8, 16, 32, 64, 128], # Powers of 2
        [0, 1, 3, 7, 15, 31, 63, 127], # Mersenne-like
        [0, 1, 4, 10, 20, 35, 56, 84], # Triangular
        [0, 1, 3, 4, 9, 10, 12, 13], # Small Sidon-like
        [0, 1, 4, 10, 18, 28, 40, 54], # Quadratic-ish
        [0, 1, 2, 4, 8, 16, 32, 64, 128, 256], # Larger powers of 2
        [0, 1, 3, 9, 27, 81, 243], # Powers of 3
        [0, 1, 4, 6, 10, 15, 21, 28], # Triangular + 1?
        [0, 1, 3, 6, 10, 15, 21, 28], # Triangular
        [0, 1, 2, 5, 9, 14, 20, 27], # Quadratic sequence
        [0, 1, 3, 6, 10, 15, 21, 28, 36, 45], # More triangular
    ]
    
    for cand in structured_candidates:
        s = _score_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)
            
    # Geometric search (inspired by continuous optimization approach)
    if time.time() < deadline - 25:
        geo_set, geo_score = _continuous_geometric_search(deadline)
        if geo_score > best_score and geo_set:
            best_score = geo_score
            best_set = geo_set

    # Dynamic Local Search with multiple restarts
    seeds = [best_set]
    # Add a few other good structured seeds for diversity
    for cand in structured_candidates[:4]:
        if cand != best_set:
            seeds.append(cand)
    
    max_val_limit = 800
    max_n_limit = 35
    
    restarts = 0
    while time.time() < deadline - 15:
        restarts += 1
        # Cycle through seeds and random starts
        if restarts % 3 == 0:
            base = seeds[restarts % len(seeds)]
        else:
            # Random start
            n = random.randint(5, 18)
            if n - 1 > max_val_limit:
                n = max_val_limit
            elements = random.sample(range(1, max_val_limit + 1), n - 1)
            base = [0] + elements
            base.sort()
        
        U = list(base)
        
        # Run local search
        local_score, local_set = _local_search_dynamic(U, deadline, max_val_limit, max_n_limit)
        
        if local_score > best_score:
            best_score = local_score
            best_set = local_set
            # Add the new best to seeds for future restarts
            if len(seeds) < 6:
                seeds.append(local_set)
            else:
                seeds[-1] = local_set

    # Final Polish: High-precision local search on the best set
    # Use a tighter limit and more iterations if time permits
    if time.time() < deadline - 3:
        final_score, final_set = _local_search_dynamic(best_set, deadline, max_val_limit=3000, max_n_limit=50)
        if final_score > best_score:
            best_score = final_score
            best_set = final_set

    return best_set
