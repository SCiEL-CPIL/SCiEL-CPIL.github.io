"""Improved program using Model-and-Optimize with Local Search."""
import time
import random
import itertools
import numpy as np
from math import log

# EVOLVE-BLOCK-START

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    # Use numpy for faster computation when n is moderate
    if n <= 200:
        U_arr = np.array(U, dtype=np.int64)
        # Sumset: all pairs (i,j) with i <= j
        i_idx, j_idx = np.triu_indices(n)
        sums = U_arr[i_idx] + U_arr[j_idx]
        sums_set = set(sums.tolist())
        
        # Difference set: all pairs (i,j) with i < j, include both d and -d
        i_idx2, j_idx2 = np.triu_indices(n, k=1)
        diffs_pos = U_arr[j_idx2] - U_arr[i_idx2]
        diffs = set(diffs_pos.tolist())
        diffs.update([-d for d in diffs])
        diffs.add(0)
        
        ratio = len(diffs) / len(sums_set)
    else:
        # Fallback to pure Python for very large sets
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

def generate_parametric_candidates(deadline):
    candidates = []
    
    # 1. Pure geometric progressions (base-B powers)
    for B in range(2, 20):
        for K in range(3, 14):
            U = [0]
            val = 1
            for i in range(1, K):
                U.append(val)
                val *= B
            if time.time() > deadline - 15:
                return candidates
            candidates.append(U)
    
    # 2. "Powers of B" with offset
    for B in range(2, 12):
        for K in range(3, 12):
            for offset in range(0, 4):
                U = [offset]
                val = offset + 1
                for i in range(1, K):
                    U.append(val)
                    val = val * B + offset
                if 0 not in U:
                    U = sorted(set(U + [0]))
                candidates.append(U)
    
    # 3. Fibonacci-like
    for K in range(3, 12):
        U = [0, 1]
        for i in range(2, K):
            U.append(U[-1] + U[-2])
        candidates.append(U)
    
    # 4. Randomized geometric with various bases
    random.seed(42)
    for _ in range(80):
        B = random.uniform(1.5, 8.0)
        K = random.randint(4, 14)
        U = [0]
        val = 1
        for i in range(1, K):
            U.append(int(val))
            val *= B
        U = sorted(set(U))
        if len(U) >= 2:
            candidates.append(U)
    
    # 5. Sparse binary-like sets (inspired by Hamming codes)
    for K in range(3, 10):
        # Use positions that are powers of 2 or combinations
        U = [0]
        max_val = 1
        for i in range(1, K):
            # Add a value that's a power of 2 or sum of small powers
            candidate = random.choice([1 << i, (1 << i) - 1, (1 << i) + 1, (1 << (i-1)) + (1 << (i-2)) if i >= 2 else 3])
            if candidate > 0 and candidate not in U:
                U.append(candidate)
                max_val = max(max_val, candidate)
        U = sorted(set(U))
        if len(U) >= 2:
            candidates.append(U)
    
    # 6. Mixed base structures
    for B1 in range(2, 8):
        for B2 in range(B1+1, min(B1+5, 15)):
            for K in range(4, 10):
                U = [0]
                val = 1
                for i in range(1, K):
                    U.append(val)
                    # Alternate between bases
                    if i % 2 == 0:
                        val *= B1
                    else:
                        val *= B2
                candidates.append(U)
    
    return candidates

def local_search(start_set, deadline, max_iter=3000):
    current = sorted(start_set)
    current_score = score_set(current)
    if current_score < 0:
        return current, -1
    
    best_set = current[:]
    best_score = current_score
    no_improve_count = 0
    
    def try_move(new_set):
        nonlocal current, current_score, best_set, best_score, no_improve_count
        new_set = sorted(set(new_set))
        if 0 not in new_set:
            return False
        if len(new_set) < 2:
            return False
        new_score = score_set(new_set)
        if new_score > current_score + 1e-12:
            current = new_set
            current_score = new_score
            no_improve_count = 0
            if current_score > best_score:
                best_score = current_score
                best_set = current[:]
            return True
        return False

    # Precompute some useful values
    def get_candidate_values(current_set):
        """Generate candidate values for moves."""
        vals = set()
        max_val = current_set[-1]
        
        # Small shifts from existing elements
        for x in current_set:
            if x > 0:
                for delta in range(-3, 4):
                    v = x + delta
                    if v > 0:
                        vals.add(v)
        
        # Multiplicative moves
        for x in current_set:
            if x > 0:
                for mult in [2, 3]:
                    vals.add(x * mult)
        
        # Extend beyond max
        for delta in range(1, 15):
            vals.add(max_val + delta)
        
        # Random values
        for _ in range(10):
            vals.add(random.randint(1, max(2, max_val)))
        
        # Gaps: try to fill gaps in the set
        for i in range(len(current_set) - 1):
            gap_start = current_set[i] + 1
            gap_end = current_set[i+1] - 1
            if gap_end > gap_start:
                vals.add(random.randint(gap_start, gap_end))
        
        return list(vals)

    iterations = 0
    while iterations < max_iter and time.time() < deadline - 8:
        iterations += 1
        moved = False
        
        # Strategy 1: Move one element (most common)
        if len(current) > 1:
            i = random.randint(1, len(current) - 1)
            old_val = current[i]
            
            # Try multiple deltas
            deltas = random.sample([-5, -3, -2, -1, 1, 2, 3, 5, 7, 10], 4)
            for delta in deltas:
                new_val = old_val + delta
                if new_val > 0:
                    new_set = current[:]
                    new_set[i] = new_val
                    if try_move(new_set):
                        moved = True
                        break
        if moved:
            continue
        
        # Strategy 2: Add an element
        if len(current) < 25:
            max_val = current[-1]
            candidate_vals = get_candidate_values(current)
            random.shuffle(candidate_vals)
            
            for new_val in candidate_vals[:8]:
                if new_val > 0 and new_val not in current:
                    new_set = current + [new_val]
                    if try_move(new_set):
                        moved = True
                        break
        if moved:
            continue
        
        # Strategy 3: Remove an element (if set is large)
        if len(current) > 5:
            i = random.randint(1, len(current) - 1)
            new_set = [x for j, x in enumerate(current) if j != i]
            if try_move(new_set):
                moved = True
                continue
        
        # Strategy 4: Simulated annealing step (allow small worsening)
        if random.random() < 0.15:
            i = random.randint(1, len(current) - 1)
            new_val = current[i] + random.randint(-5, 5)
            if new_val > 0:
                new_set = current[:]
                new_set[i] = new_val
                new_set = sorted(set(new_set))
                if 0 in new_set and len(new_set) > 1:
                    new_score = score_set(new_set)
                    # Accept with probability based on score difference
                    delta_score = new_score - current_score
                    if delta_score > 0 or random.random() < np.exp(delta_score * 5):
                        if try_move(new_set):
                            moved = True
        
        # Track no-improvement streak
        if not moved:
            no_improve_count += 1
        
        # Restart if stuck for too long
        if no_improve_count > 100:
            # Perturb the set
            if len(current) > 2:
                i = random.randint(1, len(current) - 1)
                current[i] = current[i] + random.randint(-10, 10)
                current = sorted(set(x for x in current if x > 0))
                if 0 not in current:
                    current.insert(0, 0)
                current_score = score_set(current)
                no_improve_count = 0

    return best_set, best_score

def polish_search(start_set, deadline):
    """Final high-precision polish with tighter tolerance and more iterations."""
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
        if new_score > current_score + 1e-10:  # Tighter tolerance
            current = new_set
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_set = current[:]
            return True
        return False
    
    iterations = 0
    max_iter = 5000
    while iterations < max_iter and time.time() < deadline - 3:
        iterations += 1
        moved = False
        
        # Fine-grained moves
        if len(current) > 1:
            i = random.randint(1, len(current) - 1)
            old_val = current[i]
            
            # Very small shifts
            for delta in [-2, -1, 1, 2]:
                new_val = old_val + delta
                if new_val > 0:
                    new_set = current[:]
                    new_set[i] = new_val
                    if try_move(new_set):
                        moved = True
                        break
        if moved:
            continue
        
        # Add/remove with fine granularity
        if len(current) < 30:
            max_val = current[-1]
            # Try values near existing ones
            for x in random.sample(current, min(5, len(current))):
                for delta in [-1, 1]:
                    new_val = x + delta
                    if new_val > 0 and new_val not in current:
                        new_set = current + [new_val]
                        if try_move(new_set):
                            moved = True
                            break
                if moved:
                    break
        if moved:
            continue
        
        if len(current) > 4:
            i = random.randint(1, len(current) - 1)
            new_set = [x for j, x in enumerate(current) if j != i]
            if try_move(new_set):
                moved = True

    return best_set, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin (150s limit, keep 40s buffer)
    
    # 1. Generate and evaluate parametric candidates
    candidates = generate_parametric_candidates(deadline)
    
    best_set = [0, 1, 4, 9, 11] # Fallback
    best_score = score_set(best_set)
    
    # Sort candidates by initial score to prioritize good starts
    scored_candidates = []
    for c in candidates:
        s = score_set(c)
        if s > best_score:
            best_score = s
            best_set = c[:]
        if s > 0:
            scored_candidates.append((s, c))
    
    # Sort by score descending
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # 2. Run local search on top candidates
    seen_structures = set()
    top_starts = []
    for s, c in scored_candidates:
        key = tuple(c)
        if key not in seen_structures:
            seen_structures.add(key)
            top_starts.append(c)
        if len(top_starts) >= 8:  # More starting points
            break
    
    # Also include the current best if not in top starts
    if tuple(best_set) not in [tuple(x) for x in top_starts]:
        top_starts.append(best_set)
    
    # Run local search on each top start
    for start in top_starts:
        if time.time() > deadline - 30:  # Reserve time for polish
            break
        local_best, local_score = local_search(start, deadline, max_iter=800)
        if local_score > best_score:
            best_score = local_score
            best_set = local_best[:]
    
    # 3. Final polish pass on the best result
    if time.time() < deadline - 5:
        polished_set, polished_score = polish_search(best_set, deadline)
        if polished_score > best_score:
            best_score = polished_score
            best_set = polished_set
    
    return best_set

# EVOLVE-BLOCK-END
