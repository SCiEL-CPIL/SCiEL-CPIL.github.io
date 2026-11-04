"""
Combined approach: Structural insight (dynamic candidate generation from arithmetic neighborhoods)
+ Refinement machinery (efficient scoring, ternary/geometric search, and local search with SA).
"""
import time
import random
import numpy as np
from math import log

# EVOLVE-BLOCK-START

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1.0
    n = len(U)
    if n < 2:
        return -1.0
        
    # Use numpy for vectorized sum/diff computation when n is moderate
    if n <= 200:
        arr = np.array(U, dtype=np.int64)
        # Compute all pairwise sums (upper triangle including diagonal)
        i_idx, j_idx = np.triu_indices(n)
        sums = arr[i_idx] + arr[j_idx]
        sum_count = len(np.unique(sums))
        
        # Compute all pairwise differences (strictly upper triangle)
        i_idx, j_idx = np.triu_indices(n, k=1)
        diffs = arr[j_idx] - arr[i_idx]
        # Add both positive and negative differences, plus 0
        if len(diffs) > 0:
            all_diffs = np.concatenate([diffs, -diffs, [0]])
        else:
            all_diffs = np.array([0])
        diff_count = len(np.unique(all_diffs))
    else:
        # Fallback to set-based for very large sets
        sums = set()
        diffs = set()
        for i in range(n):
            ui = U[i]
            for j in range(i, n):
                sums.add(ui + U[j])
            for j in range(i + 1, n):
                d = U[j] - ui
                diffs.add(d)
                diffs.add(-d)
        diffs.add(0)
        sum_count = len(sums)
        diff_count = len(diffs)
    
    if sum_count == 0:
        return -1.0
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1.0
        
    max_u = U[-1]
    D = 2 * max_u + 1
    if D <= 1:
        return -1.0
        
    return 1 + log(ratio) / log(D)

def generate_parametric_candidates(deadline):
    candidates = []
    
    # 1. Geometric Progressions
    for B in range(2, 15):
        for K in range(3, 12):
            U = [0]
            val = 1
            for i in range(1, K):
                U.append(val)
                val *= B
            if time.time() > deadline - 10: return candidates
            candidates.append(U)
            
    # 2. "Powers of B" with offset or scaling
    for B in range(2, 10):
        for K in range(3, 10):
            for offset in range(0, 3):
                U = [offset]
                val = offset + 1
                for i in range(1, K):
                    U.append(val)
                    val = val * B + offset
                if 0 not in U:
                    U = sorted(set(U + [0]))
                candidates.append(U)
                
    # 3. Fibonacci-like
    for K in range(3, 10):
        U = [0, 1]
        for i in range(2, K):
            U.append(U[-1] + U[-2])
        candidates.append(U)
        
    # 4. Randomized geometric
    random.seed(42)
    for _ in range(50):
        B = random.uniform(1.5, 5.0)
        K = random.randint(4, 12)
        U = [0]
        val = 1
        for i in range(1, K):
            U.append(int(val))
            val *= B
        U = sorted(set(U))
        candidates.append(U)
        
    return candidates

def score_geometric(b, m):
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
    return score_set(U)

def continuous_geometric_search(deadline):
    """Perform continuous parameter sweep over geometric progressions."""
    # Phase 1: Coarse grid search to identify promising regions
    best_score = -1.0
    best_params = None
    
    # Coarse grid over b and m
    b_grid = np.linspace(1.5, 5.0, 25)
    m_grid = range(3, 18)
    
    for m in m_grid:
        if time.time() > deadline - 10:
            break
        for b in b_grid:
            s = score_geometric(b, m)
            if s > best_score:
                best_score = s
                best_params = (b, m)
    
    if best_params is None:
        return None, -1.0
    
    # Phase 2: Ternary search around the best coarse point
    b0, m0 = best_params
    b_lo = max(1.5, b0 - 0.5)
    b_hi = min(5.0, b0 + 0.5)
    
    lo = b_lo
    hi = b_hi
    for _ in range(45):  # 45 iterations for fine-grained search
        if time.time() > deadline - 10:
            break
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        s1 = score_geometric(m1, m0)
        s2 = score_geometric(m2, m0)
        if s1 < s2:
            lo = m1
        else:
            hi = m2
    
    b_opt = (lo + hi) / 2.0
    s_opt = score_geometric(b_opt, m0)
    
    # Also try adjacent m values (wider range)
    for m_adj in [m0 - 2, m0 - 1, m0 + 1, m0 + 2]:
        if m_adj < 2 or time.time() > deadline - 10:
            continue
        # Quick ternary search for this m
        lo2 = max(1.5, b_opt - 0.4)
        hi2 = min(5.0, b_opt + 0.4)
        for _ in range(25):
            if time.time() > deadline - 10:
                break
            m1 = lo2 + (hi2 - lo2) / 3.0
            m2 = hi2 - (hi2 - lo2) / 3.0
            s1 = score_geometric(m1, m_adj)
            s2 = score_geometric(m2, m_adj)
            if s1 < s2:
                lo2 = m1
            else:
                hi2 = m2
        b_adj = (lo2 + hi2) / 2.0
        s_adj = score_geometric(b_adj, m_adj)
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

def generate_candidates(U, limit_val):
    """
    Generate candidate elements for addition based on sums and differences of existing elements.
    This is the structural improvement: exploring the arithmetic neighborhood.
    """
    U_set = set(U)
    candidates = set()
    
    # Sums and differences of existing elements
    for a in U:
        for b in U:
            s = a + b
            d = abs(a - b)
            if 0 < s <= limit_val and s not in U_set:
                candidates.add(s)
            if 0 < d <= limit_val and d not in U_set:
                candidates.add(d)
    
    # Add some random small integers to ensure diversity
    for _ in range(20):
        r = random.randint(1, min(50, limit_val))
        if r not in U_set:
            candidates.add(r)
            
    return list(candidates)

def local_search_dynamic(U, deadline, max_val_limit=1000, max_n_limit=40):
    """
    Local search with dynamic candidate generation.
    """
    U = sorted(set(U))
    current_score = score_set(U)
    best_local_score = current_score
    best_local_set = list(U)
    
    # Initial candidate pool
    candidates = generate_candidates(U, max_val_limit)
    
    no_improvement_count = 0
    max_no_improvement = 40
    
    while time.time() < deadline - 2:
        if no_improvement_count > max_no_improvement:
            break
            
        improved = False
        move_type = random.random()
        
        # Update candidate pool periodically or if empty
        if not candidates or random.random() < 0.15:
            # Expand limit if we are growing
            current_max = max(U)
            new_limit = max(max_val_limit, current_max * 2)
            candidates = generate_candidates(U, new_limit)
        
        if move_type < 0.5:
            # Try to ADD an element
            if len(U) < max_n_limit and candidates:
                # Try a few random candidates
                random.shuffle(candidates)
                for c in candidates[:15]:
                    if c in U:
                        continue
                    new_U = U + [c]
                    new_U.sort()
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        # Regenerate candidates immediately as structure changed significantly
                        candidates = generate_candidates(U, max(max_val_limit, max(U)*2))
                        break
        elif move_type < 0.7:
            # Try to REMOVE an element
            if len(U) > 2:
                removable = [x for x in U if x != 0]
                if removable:
                    c = random.choice(removable)
                    new_U = [x for x in U if x != c]
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = generate_candidates(U, max_val_limit)
        else:
            # Try to REPLACE an element
            if len(U) > 1 and candidates:
                idx = random.randint(1, len(U) - 1)
                old_val = U[idx]
                # Pick a random candidate
                random.shuffle(candidates)
                for new_val in candidates[:15]:
                    if new_val == old_val or new_val in U:
                        continue
                    new_U = U.copy()
                    new_U[idx] = new_val
                    new_U.sort()
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = generate_candidates(U, max(max_val_limit, max(U)*2))
                        break
            
        if improved:
            no_improvement_count = 0
            if current_score > best_local_score:
                best_local_score = current_score
                best_local_set = list(U)
        else:
            no_improvement_count += 1
                
    return best_local_score, best_local_set

def greedy_modular_search(p, V_init, deadline, max_k=15):
    """
    Greedy search for V subset such that U = {k*p : k in V} maximizes score.
    We work in the quotient space to eliminate low-order sum collisions.
    """
    V = sorted(set(V_init))
    if 0 not in V:
        V = sorted(V + [0])
    if len(V) < 2:
        V = [0, 1]
    
    current_score = score_set([k * p for k in V])
    best_score = current_score
    best_V = list(V)
    
    no_improvement = 0
    max_no_improvement = 30
    
    while time.time() < deadline - 5 and no_improvement < max_no_improvement:
        improved = False
        
        # Generate candidates for V
        candidates = set()
        for a in V:
            for b in V:
                s = a + b
                d = abs(a - b)
                if 0 < s <= max_k and s not in V:
                    candidates.add(s)
                if 0 < d <= max_k and d not in V:
                    candidates.add(d)
        # Add some random candidates
        for _ in range(10):
            r = random.randint(1, max_k)
            candidates.add(r)
        
        # Try adding elements
        if len(V) < max_k and candidates:
            cand_list = list(candidates)
            random.shuffle(cand_list)
            for c in cand_list[:10]:
                new_V = sorted(V + [c])
                new_score = score_set([k * p for k in new_V])
                if new_score > current_score:
                    V = new_V
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_V = list(V)
                    break
        
        # Try removing elements
        if len(V) > 2:
            removable = [x for x in V if x != 0]
            random.shuffle(removable)
            for c in removable[:5]:
                new_V = [x for x in V if x != c]
                new_score = score_set([k * p for k in new_V])
                if new_score > current_score:
                    V = new_V
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_V = list(V)
                    break
        
        # Try replacing elements
        if len(V) > 1 and candidates:
            idx = random.randint(1, len(V) - 1)
            old_val = V[idx]
            cand_list = list(candidates)
            random.shuffle(cand_list)
            for new_val in cand_list[:10]:
                if new_val == old_val or new_val in V:
                    continue
                new_V = V.copy()
                new_V[idx] = new_val
                new_V = sorted(new_V)
                new_score = score_set([k * p for k in new_V])
                if new_score > current_score:
                    V = new_V
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_V = list(V)
                    break
        
        if improved:
            no_improvement = 0
        else:
            no_improvement += 1
    
    return [k * p for k in best_V], best_score

def modular_arithmetic_search(deadline):
    """
    Search over prime moduli and use greedy optimization in quotient space.
    """
    best_score = -1.0
    best_set = None
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    for p in primes:
        if time.time() > deadline - 15:
            break
        
        # Try different initial V sets
        initial_Vs = [
            [0, 1],
            [0, 1, 2],
            [0, 1, 3],
            [0, 1, 2, 4],
            [0, 1, 3, 5],
            [0, 1, 2, 3, 5],
            [0, 1, 4, 9],  # Squares in quotient
            [0, 1, 3, 9],  # Powers of 3 in quotient
        ]
        
        for V_init in initial_Vs:
            if time.time() > deadline - 15:
                break
            U, s = greedy_modular_search(p, V_init, deadline)
            if U is not None and s > best_score:
                best_score = s
                best_set = U[:]
    
    return best_set, best_score

def construct():
    deadline = time.time() + 115  # Safety margin for 150s limit (using 150s budget)
    best_score = -1.0
    best_set = [0, 1, 4, 9, 11]
    
    # Initial best
    s = score_set(best_set)
    if s > best_score:
        best_score = s
        best_set = list(best_set)

    # 1. Try some structured candidates
    structured_candidates = [
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81], # Squares
        [0, 1, 2, 4, 8, 16, 32, 64, 128], # Powers of 2
        [0, 1, 3, 7, 15, 31, 63, 127], # Mersenne-like
        [0, 1, 4, 10, 20, 35, 56, 84], # Triangular
        [0, 1, 3, 4, 9, 10, 12, 13], # Small Sidon-like
        [0, 1, 4, 10, 18, 28, 40, 54], # Quadratic-ish
        [0, 1, 2, 4, 8, 16, 32, 64, 128, 256], # Larger powers of 2
        [0, 1, 3, 9, 27, 81, 243], # Powers of 3
        [0, 1, 5, 25, 125, 625], # Powers of 5
        [0, 1, 6, 36, 216], # Powers of 6
        [0, 2, 6, 18, 54, 162], # 2*3^k
    ]
    
    for cand in structured_candidates:
        s = score_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)
            
    # 2. Continuous geometric search
    if time.time() < deadline - 25:
        geo_set, geo_score = continuous_geometric_search(deadline)
        if geo_set is not None and geo_score > best_score:
            best_score = geo_score
            best_set = geo_set[:]
            structured_candidates.append(geo_set)
    
    # 3. Modular arithmetic search
    if time.time() < deadline - 30:
        mod_set, mod_score = modular_arithmetic_search(deadline)
        if mod_set is not None and mod_score > best_score:
            best_score = mod_score
            best_set = mod_set[:]
            structured_candidates.append(mod_set)
    
    # 4. Local Search with Dynamic Candidate Generation
    # We will run multiple restarts.
    max_val_limit = 600
    max_n_limit = 35
    
    restarts = 0
    while time.time() < deadline - 10:
        restarts += 1
        # Choose a starting set
        if restarts % 3 == 0:
            # Start from a structured set
            base = random.choice(structured_candidates)
            U = list(base)
        else:
            # Start from random
            n = random.randint(4, 18)
            if n - 1 > max_val_limit:
                n = max_val_limit
            elements = random.sample(range(1, max_val_limit + 1), n - 1)
            U = [0] + elements
            U.sort()
        
        # Run local search
        local_score, local_set = local_search_dynamic(U, deadline, max_val_limit, max_n_limit)
        
        if local_score > best_score:
            best_score = local_score
            best_set = local_set

    # Final polish: Try very small moves on the best set found
    if time.time() < deadline - 3:
        U = list(best_set)
        current_score = best_score
        U.sort()
        # We reuse the dynamic local search but with a very short deadline
        # and restricted to the current neighborhood
        final_score, final_set = local_search_dynamic(U, deadline, max_val_limit, max_n_limit)
        if final_score > best_score:
            best_score = final_score
            best_set = final_set
            
    return best_set

# EVOLVE-BLOCK-END
