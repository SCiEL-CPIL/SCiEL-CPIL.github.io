"""Combined program: Exhaustive small-set search + continuous geometric optimization + local search."""
import time
import random
from math import log
from itertools import combinations
import numpy as np

# EVOLVE-BLOCK-START

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
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
    ratio = len(diffs) / len(sums)
    if ratio <= 1:
        return -1
    max_u = U[-1]
    D = 2 * max_u + 1
    if D <= 1:
        return -1
    return 1 + log(ratio) / log(D)

def score_geometric(b, m):
    """Score a geometric progression U = {floor(b^k) for k=0..m} with 0."""
    if b <= 1.0 or m < 1:
        return -1
    U = [0]
    val = 1.0
    for k in range(1, m + 1):
        U.append(int(val))
        val *= b
    U = sorted(set(U))
    if len(U) < 2:
        return -1
    return score_set(U)

def exhaustive_small_search(deadline):
    """Exhaustively search small sets to find good structured candidates."""
    best_score = -1
    best_set = [0, 1]
    candidates_found = []
    
    # Search over sets of size 3 to 7 with values up to 15
    for size in range(3, 8):
        if time.time() > deadline - 30:
            break
        for combo in combinations(range(1, 16), size - 1):
            if time.time() > deadline - 30:
                break
            U = [0] + list(combo)
            U.sort()
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
            if s > 1.02:
                candidates_found.append((s, U[:]))
    
    # Also try slightly larger values for smaller sets
    for size in range(3, 6):
        if time.time() > deadline - 25:
            break
        for combo in combinations(range(1, 25), size - 1):
            if time.time() > deadline - 25:
                break
            U = [0] + list(combo)
            U.sort()
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
            if s > 1.02 and s > 1.015:
                candidates_found.append((s, U[:]))
    
    # Sort by score and return top candidates
    candidates_found.sort(key=lambda x: x[0], reverse=True)
    
    # Deduplicate
    seen = set()
    unique_candidates = []
    for s, U in candidates_found:
        key = tuple(U)
        if key not in seen:
            seen.add(key)
            unique_candidates.append((s, U))
    
    return best_set, best_score, unique_candidates[:10]

def continuous_geometric_search(deadline):
    """Perform continuous parameter sweep over geometric progressions."""
    best_score = -1
    best_set = None
    
    # Coarse grid to identify promising regions
    b_grid = np.linspace(1.3, 6.0, 30)
    m_range = range(2, 18)
    
    coarse_best_b = 2.0
    coarse_best_m = 5
    coarse_best_score = -1
    
    for m in m_range:
        if time.time() > deadline - 20:
            break
        for b in b_grid:
            s = score_geometric(b, m)
            if s > coarse_best_score:
                coarse_best_score = s
                coarse_best_b = b
                coarse_best_m = m
    
    # Fine ternary search around coarse best for each nearby m
    for m in range(max(2, coarse_best_m - 3), min(18, coarse_best_m + 4)):
        if time.time() > deadline - 15:
            break
        b_lo = max(1.2, coarse_best_b - 1.0)
        b_hi = min(6.5, coarse_best_b + 1.0)
        
        lo, hi = b_lo, b_hi
        for _ in range(60):
            if time.time() > deadline - 15:
                break
            m1 = lo + (hi - lo) / 3.0
            m2 = hi - (hi - lo) / 3.0
            s1 = score_geometric(m1, m)
            s2 = score_geometric(m2, m)
            if s1 < s2:
                lo = m1
            else:
                hi = m2
        
        b_opt = (lo + hi) / 2.0
        s_opt = score_geometric(b_opt, m)
        
        if s_opt > best_score:
            best_score = s_opt
            U = [0]
            val = 1.0
            for k in range(1, m + 1):
                U.append(int(val))
                val *= b_opt
            best_set = sorted(set(U))
    
    return best_set, best_score

def generate_parametric_candidates(deadline):
    """Generate diverse parametric candidate sets."""
    candidates = []
    
    # Geometric progressions with various bases
    for B in [1.5, 1.7, 2.0, 2.2, 2.5, 3.0, 3.5, 4.0, 5.0]:
        for K in range(3, 15):
            if time.time() > deadline - 10:
                return candidates
            U = [0]
            val = 1.0
            for i in range(1, K):
                U.append(int(val))
                val *= B
            U = sorted(set(U))
            if len(U) >= 2:
                candidates.append(U)
    
    # Fibonacci-like sequences
    for K in range(3, 13):
        U = [0, 1]
        for i in range(2, K):
            U.append(U[-1] + U[-2])
        candidates.append(U)
    
    # Randomized geometric with seed
    random.seed(123)
    for _ in range(30):
        B = random.uniform(1.4, 5.5)
        K = random.randint(4, 14)
        U = [0]
        val = 1.0
        for i in range(1, K):
            U.append(int(val))
            val *= B
        U = sorted(set(U))
        candidates.append(U)
    
    # Power-of-2 variants
    for K in range(3, 12):
        U = [0]
        val = 1
        for i in range(1, K):
            U.append(val)
            val *= 2
        candidates.append(U)
    
    # Mixed: geometric + small perturbations
    random.seed(456)
    for _ in range(20):
        B = random.uniform(1.8, 3.5)
        K = random.randint(4, 10)
        U = [0]
        val = 1.0
        for i in range(1, K):
            perturbed = int(val) + random.randint(-1, 1)
            if perturbed > 0:
                U.append(perturbed)
            val *= B
        U = sorted(set(U))
        candidates.append(U)
    
    return candidates

def local_search(start_set, deadline, max_iter=5000):
    """Enhanced local search with diverse move operators and adaptive step sizes."""
    current = sorted(start_set)
    current_score = score_set(current)
    if current_score < 0:
        return current, -1
    
    best_set = current[:]
    best_score = current_score
    
    def try_move(new_set):
        nonlocal current, current_score, best_set, best_score
        new_set = sorted(set(new_set))
        if 0 not in new_set:
            return False
        if len(new_set) < 2:
            return False
        new_score = score_set(new_set)
        if new_score > current_score + 1e-12:
            current = new_set
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_set = current[:]
            return True
        return False
    
    iterations = 0
    no_improve_count = 0
    max_no_improve = 150
    
    while iterations < max_iter and time.time() < deadline - 3:
        iterations += 1
        moved = False
        prev_score = current_score
        
        # 1. Move one element (small shifts)
        if len(current) > 1 and random.random() < 0.45:
            i = random.randint(1, len(current) - 1)
            old_val = current[i]
            # Adaptive step size based on element magnitude
            step = max(1, int(old_val * 0.05))
            deltas = list(range(-step, step + 1))
            random.shuffle(deltas)
            for delta in deltas[:5]:
                if delta == 0:
                    continue
                new_val = old_val + delta
                if new_val > 0:
                    new_set = current[:]
                    new_set[i] = new_val
                    if try_move(new_set):
                        moved = True
                        break
        
        # 2. Add an element
        if len(current) < 20 and random.random() < 0.20:
            max_val = current[-1]
            candidates = []
            for x in current:
                candidates.extend([x + 1, x - 1, x * 2, x + random.randint(1, max(2, x // 3))])
            candidates.append(max_val + random.randint(1, max(5, max_val // 5)))
            candidates.append(random.randint(1, max(5, max_val + 10)))
            
            random.shuffle(candidates)
            for new_val in candidates[:6]:
                if new_val > 0 and new_val not in current:
                    new_set = current + [new_val]
                    if try_move(new_set):
                        moved = True
                        break
        
        # 3. Remove an element
        if len(current) > 4 and random.random() < 0.12:
            i = random.randint(1, len(current) - 1)
            new_set = [x for j, x in enumerate(current) if j != i]
            if try_move(new_set):
                moved = True
        
        # 4. Swap two elements
        if len(current) > 3 and random.random() < 0.15:
            i, j = random.sample(range(1, len(current)), 2)
            vi, vj = current[i], current[j]
            step_i = max(1, int(vi * 0.05))
            step_j = max(1, int(vj * 0.05))
            di = random.randint(-step_i, step_i)
            dj = random.randint(-step_j, step_j)
            nvi, nvj = vi + di, vj + dj
            if nvi > 0 and nvj > 0:
                new_set = current[:]
                new_set[i] = nvi
                new_set[j] = nvj
                new_set = sorted(set(new_set))
                if 0 in new_set and len(new_set) >= 2:
                    new_score = score_set(new_set)
                    if new_score > current_score + 1e-12:
                        current = new_set
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_set = current[:]
                        moved = True
        
        # 5. Structural move: geometric extension
        if len(current) > 2 and random.random() < 0.08:
            i = random.randint(1, len(current) - 1)
            new_val = int(current[i] * random.uniform(1.5, 3.5))
            if new_val > 0 and new_val not in current:
                new_set = current[:]
                new_set[i] = new_val
                if try_move(new_set):
                    moved = True
        
        # Track no-improve streak for adaptive behavior
        if current_score > prev_score + 1e-9:
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # If stuck, try a bigger jump
        if no_improve_count > max_no_improve and random.random() < 0.4:
            i = random.randint(1, len(current) - 1)
            step = max(3, int(current[i] * 0.15))
            new_val = current[i] + random.randint(-step, step)
            if new_val > 0:
                new_set = current[:]
                new_set[i] = new_val
                new_set = sorted(set(new_set))
                if 0 in new_set and len(new_set) > 1:
                    new_score = score_set(new_set)
                    if new_score > current_score - 0.03:
                        current = new_set
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_set = current[:]
                        moved = True
                        no_improve_count = 0
    
    return best_set, best_score

def construct():
    deadline = time.time() + 110  # Safety margin for 120s limit
    
    # Phase 1: Exhaustive search over small sets to find good structured candidates
    small_best_set, small_best_score, small_candidates = exhaustive_small_search(deadline)
    
    best_set = small_best_set[:]
    best_score = small_best_score
    
    # Phase 2: Continuous geometric search
    geo_set, geo_score = continuous_geometric_search(deadline)
    
    if geo_set is not None and geo_score > best_score:
        best_score = geo_score
        best_set = geo_set[:]
    
    # Phase 3: Generate and evaluate parametric candidates
    candidates = generate_parametric_candidates(deadline)
    
    scored_candidates = []
    for c in candidates:
        if time.time() > deadline - 20:
            break
        s = score_set(c)
        if s > best_score:
            best_score = s
            best_set = c[:]
        if s > 0:
            scored_candidates.append((s, c))
    
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Phase 4: Multi-start local search
    seen_structures = set()
    top_starts = []
    
    # Add exhaustive search results
    for s, c in small_candidates:
        key = tuple(c)
        if key not in seen_structures:
            seen_structures.add(key)
            top_starts.append(c)
        if len(top_starts) >= 8:
            break
    
    # Add parametric candidates
    for s, c in scored_candidates:
        key = tuple(c)
        if key not in seen_structures:
            seen_structures.add(key)
            top_starts.append(c)
        if len(top_starts) >= 12:
            break
    
    # Include continuous geometric result and current best
    if geo_set is not None and tuple(geo_set) not in [tuple(x) for x in top_starts]:
        top_starts.append(geo_set)
    if tuple(best_set) not in [tuple(x) for x in top_starts]:
        top_starts.append(best_set)
    
    # Limit number of starts to manage time
    if len(top_starts) > 8:
        top_starts = top_starts[:8]
    
    # Allocate remaining time across local searches
    remaining_time = deadline - time.time() - 5
    if remaining_time > 10:
        time_per_search = max(8, remaining_time / max(1, len(top_starts)))
        
        for idx, start in enumerate(top_starts):
            if time.time() > deadline - 8:
                break
            search_deadline = min(deadline - 5, time.time() + time_per_search)
            local_best, local_score = local_search(start, search_deadline, max_iter=3000)
            if local_score > best_score:
                best_score = local_score
                best_set = local_best[:]
    
    # Final polish: one more local search on the best found
    if time.time() < deadline - 15:
        final_best, final_score = local_search(best_set, deadline - 3, max_iter=2000)
        if final_score > best_score:
            best_score = final_score
            best_set = final_best[:]
    
    return best_set

# EVOLVE-BLOCK-END
