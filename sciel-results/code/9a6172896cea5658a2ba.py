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
    # Use numpy for faster set operations
    arr = np.array(U, dtype=np.int64)
    # Sumset: all pairwise sums (i<=j for symmetry, but we need unique values)
    # Compute all pairwise sums
    sums = np.zeros(n * n, dtype=np.int64)
    for i in range(n):
        sums[i*n:(i+1)*n] = arr[i] + arr
    sums = np.unique(sums)
    # Difference set: all pairwise differences (including negatives and 0)
    diffs = np.zeros(n * n, dtype=np.int64)
    for i in range(n):
        diffs[i*n:(i+1)*n] = arr - arr[i]
    diffs = np.unique(diffs)
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
    
    # Search over sets of size 3 to 7 with values up to 12 (faster)
    for size in range(3, 8):
        if time.time() > deadline - 35:
            break
        for combo in combinations(range(1, 13), size - 1):
            if time.time() > deadline - 35:
                break
            U = [0] + list(combo)
            U.sort()
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
            if s > 1.015:
                candidates_found.append((s, U[:]))
    
    # Also try slightly larger values for smaller sets
    for size in range(3, 6):
        if time.time() > deadline - 30:
            break
        for combo in combinations(range(1, 20), size - 1):
            if time.time() > deadline - 30:
                break
            U = [0] + list(combo)
            U.sort()
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
            if s > 1.015:
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
    current = sorted(set(start_set))
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
    max_no_improve = 80
    # Track the scale of the set to adapt step sizes
    set_scale = max(1, current[-1])
    
    while iterations < max_iter and time.time() < deadline - 3:
        iterations += 1
        moved = False
        prev_score = current_score
        set_scale = max(1, current[-1])
        
        # 1. Move one element with adaptive step size
        if len(current) > 1 and random.random() < 0.40:
            i = random.randint(1, len(current) - 1)
            old_val = current[i]
            # Larger steps for larger elements, smaller for small ones
            if old_val < 10:
                step = max(1, random.randint(1, 3))
            elif old_val < 100:
                step = max(2, random.randint(2, 8))
            else:
                step = max(3, int(old_val * random.uniform(0.02, 0.15)))
            # Try multiple deltas
            for _ in range(8):
                delta = random.randint(-step, step)
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
        if len(current) < 25 and random.random() < 0.18:
            max_val = current[-1]
            candidates = []
            # Geometric extensions
            for x in current[1:]:
                candidates.append(int(x * random.uniform(1.5, 4.0)))
                candidates.append(int(x * random.uniform(0.5, 0.9)))
            # Near the max
            candidates.append(max_val + random.randint(1, max(5, int(max_val * 0.3))))
            candidates.append(random.randint(1, max(5, int(max_val * 1.5))))
            # Small values
            candidates.extend(random.randint(1, 20) for _ in range(5))
            
            random.shuffle(candidates)
            for new_val in candidates[:8]:
                if new_val > 0 and new_val not in current:
                    new_set = current + [new_val]
                    if try_move(new_set):
                        moved = True
                        break
        
        # 3. Remove an element
        if len(current) > 5 and random.random() < 0.10:
            i = random.randint(1, len(current) - 1)
            new_set = [x for j, x in enumerate(current) if j != i]
            if try_move(new_set):
                moved = True
        
        # 4. Swap two elements with larger perturbations
        if len(current) > 3 and random.random() < 0.12:
            i, j = random.sample(range(1, len(current)), 2)
            vi, vj = current[i], current[j]
            step_i = max(2, int(vi * random.uniform(0.03, 0.2)))
            step_j = max(2, int(vj * random.uniform(0.03, 0.2)))
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
        
        # 5. Structural move: replace with geometric-like value
        if len(current) > 2 and random.random() < 0.08:
            i = random.randint(1, len(current) - 1)
            # Choose a reference element and scale it
            ref = random.choice(current[1:])
            new_val = int(ref * random.uniform(1.3, 4.0))
            if new_val > 0 and new_val not in current:
                new_set = current[:]
                new_set[i] = new_val
                if try_move(new_set):
                    moved = True
        
        # 6. Global restart: randomly perturb multiple elements
        if no_improve_count > max_no_improve and random.random() < 0.5:
            new_set = current[:]
            num_perturb = random.randint(1, min(3, len(new_set) - 1))
            for _ in range(num_perturb):
                idx = random.randint(1, len(new_set) - 1)
                step = max(3, int(new_set[idx] * random.uniform(0.1, 0.3)))
                new_set[idx] = new_set[idx] + random.randint(-step, step)
            new_set = sorted(set(x for x in new_set if x > 0))
            if 0 not in new_set:
                new_set = [0] + new_set
            if len(new_set) >= 2:
                new_score = score_set(new_set)
                if new_score > current_score - 0.05:
                    current = new_set
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_set = current[:]
                    moved = True
                    no_improve_count = 0
        
        # Track no-improve streak
        if current_score > prev_score + 1e-9:
            no_improve_count = 0
        else:
            no_improve_count += 1
    
    return best_set, best_score

def construct():
    deadline = time.time() + 105  # Safety margin for 120s limit (15s buffer)
    
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
        if time.time() > deadline - 25:
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
        if len(top_starts) >= 6:
            break
    
    # Add parametric candidates
    for s, c in scored_candidates:
        key = tuple(c)
        if key not in seen_structures:
            seen_structures.add(key)
            top_starts.append(c)
        if len(top_starts) >= 10:
            break
    
    # Include continuous geometric result and current best
    if geo_set is not None and tuple(geo_set) not in [tuple(x) for x in top_starts]:
        top_starts.append(geo_set)
    if tuple(best_set) not in [tuple(x) for x in top_starts]:
        top_starts.append(best_set)
    
    # Limit number of starts to manage time
    if len(top_starts) > 6:
        top_starts = top_starts[:6]
    
    # Allocate remaining time across local searches
    remaining_time = deadline - time.time() - 8
    if remaining_time > 15:
        time_per_search = max(10, remaining_time / max(1, len(top_starts)))
        
        for idx, start in enumerate(top_starts):
            if time.time() > deadline - 10:
                break
            search_deadline = min(deadline - 6, time.time() + time_per_search)
            local_best, local_score = local_search(start, search_deadline, max_iter=4000)
            if local_score > best_score:
                best_score = local_score
                best_set = local_best[:]
    
    # Final polish: one more local search on the best found
    if time.time() < deadline - 12:
        final_best, final_score = local_search(best_set, deadline - 3, max_iter=3000)
        if final_score > best_score:
            best_score = final_score
            best_set = final_best[:]
    
    return best_set

# EVOLVE-BLOCK-END
