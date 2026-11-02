<<<<<<< SEARCH
import time
import numpy as np
import random
import itertools
from functools import lru_cache

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)
    
    # Fallbacks for other parameters
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return [list(c) for c in itertools.product(range(q), repeat=n)]
=======
import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)
    
    # Fallbacks for other parameters
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return [list(c) for c in itertools.product(range(q), repeat=n)]
>>>>>>> REPLACE
<<<<<<< SEARCH
def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    
    # Precompute balls
    weights = [3 ** (n - 1 - i) for i in range(n)]
    balls = np.zeros((total, 1 + 2 * n), dtype=np.int32)
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                balls[w, idx] = new_w
                idx += 1
    
    # Precompute frozensets for fast intersection
    ball_sets = [frozenset(balls[w].tolist()) for w in range(total)]
    
    # Build bipartite graph adjacency: word -> set of covering codewords
    word_to_codewords = [frozenset(balls[w].tolist()) for w in range(total)]
    
    best_size = total
    best_set = None
    
    # Run multiple restarts
    iteration = 0
    while time.time() < deadline - 5.0:
        iteration += 1
        
        # 1. Randomized Greedy Construction with LP-guided bias
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_mask = np.ones(total, dtype=bool)
        
        # Compute initial dual potentials via simple LP relaxation approximation
        # Use subgradient method to approximate dual variables
        dual_potentials = _compute_dual_potentials(total, balls, word_to_codewords, deadline - 10.0)
        
        while np.any(uncovered_mask):
            if time.time() > deadline - 10.0:
                break
            # Count coverage for each candidate among uncovered words
            counts = uncovered_mask[balls].sum(axis=1)
            
            if np.max(counts) == 0:
                break
                
            max_count = np.max(counts)
            candidates = np.where(counts == max_count)[0]
            
            # Use dual potentials to bias selection: prefer codewords with lower potential
            # (lower potential = more "valuable" in dual sense)
            if len(candidates) > 1 and dual_potentials is not None:
                # Soft-min: weight by exp(-beta * potential) for candidates
                beta = 0.5
                potentials = dual_potentials[candidates]
                weights_vec = np.exp(-beta * potentials)
                weights_vec = weights_vec / np.sum(weights_vec)
                best_idx = np.random.choice(candidates, p=weights_vec)
            else:
                best_idx = candidates[np.random.randint(len(candidates))]
            
            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            uncovered_mask = (cover_count == 0)
            
        if np.any(uncovered_mask):
            continue # Invalid due to timeout
        
        # 2. Local Search with LP-guided moves
        # Phase 1: Remove redundant codewords (multiple passes)
        for _pass in range(3):
            changed = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
            if not changed:
                break
            
        # Phase 2: 1-Swap (Remove 1, Add 1) using LP-guided candidate selection
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    improved = True
                    continue
                
                critical_mask = (cover_count[balls[c]] == 1)
                critical_words = balls[c][critical_mask]
                
                if len(critical_words) == 0:
                    cover_count[balls[c]] -= 1
                    improved = True
                    continue

                # Use precomputed frozensets for fast intersection
                candidates = ball_sets[critical_words[0]]
                for cw in critical_words[1:]:
                    candidates = candidates & ball_sets[cw]
                    if not candidates:
                        break
                
                if c in candidates:
                    candidates = candidates - {c}
                
                if candidates:
                    # LP-guided selection: prefer candidates with lower dual potential
                    if dual_potentials is not None and len(candidates) > 1:
                        cand_list = list(candidates)
                        beta = 0.5
                        potentials = dual_potentials[cand_list]
                        weights_vec = np.exp(-beta * potentials)
                        weights_vec = weights_vec / np.sum(weights_vec)
                        c_prime = cand_list[np.random.choice(len(cand_list), p=weights_vec)]
                    else:
                        c_prime = random.choice(tuple(candidates))
                    cover_count[balls[c]] -= 1
                    cover_count[balls[c_prime]] += 1
                    new_selected.append(c_prime)
                    improved = True
                else:
                    new_selected.append(c)
            
            selected = new_selected
            
        # Phase 3: 2-Exchange (Remove 2, Add 1) with LP-guided sampling
        if len(selected) > 486 and time.time() < deadline - 8.0:
            improved_2ex = True
            while improved_2ex and time.time() < deadline - 5.0:
                improved_2ex = False
                num_trials = min(8000, len(selected) * (len(selected) - 1) // 2)
                
                for _ in range(num_trials):
                    if time.time() > deadline - 5.0:
                        break
                    
                    idx1 = random.randint(0, len(selected) - 1)
                    idx2 = random.randint(0, len(selected) - 1)
                    if idx1 == idx2:
                        continue
                        
                    c1 = selected[idx1]
                    c2 = selected[idx2]
                    
                    # Temporarily remove
                    cover_count[balls[c1]] -= 1
                    cover_count[balls[c2]] -= 1
                    
                    uncovered_after = (cover_count == 0)
                    
                    if np.any(uncovered_after):
                        uncovered_words = np.where(uncovered_after)[0]
                        
                        # Find c3 that covers all uncovered words using precomputed sets
                        candidates = ball_sets[uncovered_words[0]]
                        for uw in uncovered_words[1:]:
                            candidates = candidates & ball_sets[uw]
                            if not candidates:
                                break
                        
                        if c1 in candidates:
                            candidates = candidates - {c1}
                        if c2 in candidates:
                            candidates = candidates - {c2}
                        
                        if candidates:
                            # LP-guided selection for c3
                            if dual_potentials is not None and len(candidates) > 1:
                                cand_list = list(candidates)
                                beta = 0.5
                                potentials = dual_potentials[cand_list]
                                weights_vec = np.exp(-beta * potentials)
                                weights_vec = weights_vec / np.sum(weights_vec)
                                c3 = cand_list[np.random.choice(len(cand_list), p=weights_vec)]
                            else:
                                c3 = random.choice(tuple(candidates))
                            cover_count[balls[c3]] += 1
                            # Remove c1, c2 from selected
                            new_selected = []
                            for c in selected:
                                if c != c1 and c != c2:
                                    new_selected.append(c)
                            new_selected.append(c3)
                            selected = new_selected
                            improved_2ex = True
                            break
                        else:
                            # Revert
                            cover_count[balls[c1]] += 1
                            cover_count[balls[c2]] += 1
                    else:
                        # Both can be removed
                        new_selected = []
                        for c in selected:
                            if c != c1 and c != c2:
                                new_selected.append(c)
                        selected = new_selected
                        improved_2ex = True
                        break
                        
                    if not improved_2ex:
                        cover_count[balls[c1]] += 1
                        cover_count[balls[c2]] += 1
        
        # Phase 4: Simulated Annealing with penalty-based energy function
        # This helps escape local minima by allowing temporary increases in cost
        if len(selected) > 486 and time.time() < deadline - 10.0:
            # Energy function: E = len(selected) + penalty * (uncovered_count)
            # We want to minimize this; allow moves that increase E with probability exp(-dE/T)
            T = 50.0  # Initial temperature
            T_min = 5.0
            cooling_rate = 0.95
            
            for sa_iter in range(200):
                if time.time() > deadline - 8.0:
                    break
                if T < T_min:
                    break
                
                # Compute current energy
                current_E = float(len(selected))
                
                # Try a random move: swap one codeword with another
                if len(selected) > 1:
                    idx_swap = random.randint(0, len(selected) - 1)
                    c_old = selected[idx_swap]
                    
                    # Find a random alternative that covers at least some of c_old's critical words
                    critical_mask = (cover_count[balls[c_old]] == 1)
                    critical_words = balls[c_old][critical_mask]
                    
                    if len(critical_words) > 0:
                        # Pick a random critical word and find a random alternative covering it
                        cw = critical_words[random.randint(0, len(critical_words) - 1)]
                        alternatives = list(ball_sets[cw])
                        if c_old in alternatives:
                            alternatives.remove(c_old)
                        
                        if alternatives:
                            c_new = alternatives[random.randint(0, len(alternatives) - 1)]
                            
                            # Try the swap
                            cover_count[balls[c_old]] -= 1
                            cover_count[balls[c_new]] += 1
                            
                            # Check if still valid
                            uncovered_after = (cover_count == 0)
                            if not np.any(uncovered_after):
                                # Valid move: size stays same or decreases
                                new_E = float(len(selected))
                                dE = new_E - current_E
                                # Always accept if dE <= 0, else with probability exp(-dE/T)
                                if dE <= 0 or random.random() < np.exp(-dE / T):
                                    selected[idx_swap] = c_new
                                else:
                                    # Revert
                                    cover_count[balls[c_old]] += 1
                                    cover_count[balls[c_new]] -= 1
                            else:
                                # Invalid: revert
                                cover_count[balls[c_old]] += 1
                                cover_count[balls[c_new]] -= 1
                    
                    T *= cooling_rate
        
        # Update dual potentials periodically
        if iteration % 5 == 0 and time.time() < deadline - 15.0:
            dual_potentials = _compute_dual_potentials(total, balls, word_to_codewords, deadline - 15.0)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            
            if best_size <= 403:
                break
            if best_size <= 486:
                break

    # Convert best_set to output format
    if best_set is None:
        # Fallback
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_mask = np.ones(total, dtype=bool)
        while np.any(uncovered_mask):
            counts = uncovered_mask[balls].sum(axis=1)
            max_count = np.max(counts)
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[0]
            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            uncovered_mask = (cover_count == 0)
        best_set = selected

    result = []
    for w in best_set:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
        
    return result
=======
def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    
    # Precompute balls as numpy array for vectorized operations
    weights = [3 ** (n - 1 - i) for i in range(n)]
    balls = np.zeros((total, 1 + 2 * n), dtype=np.int32)
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                balls[w, idx] = new_w
                idx += 1
    
    # Precompute ball membership: for each word, which codewords cover it
    # Using inverse mapping
    word_to_codewords = [[] for _ in range(total)]
    for c in range(total):
        for w in balls[c]:
            word_to_codewords[w].append(c)
    
    best_size = total
    best_set = None
    
    # Run multiple restarts with different strategies
    iteration = 0
    while time.time() < deadline - 5.0:
        iteration += 1
        
        # Strategy selection: alternate between greedy, simulated annealing, and tabu
        strategy = iteration % 3
        
        if strategy == 0:
            # Greedy construction with randomization
            selected, cover_count = _greedy_construct(total, balls, word_to_codewords, deadline - 10.0)
        elif strategy == 1:
            # Start from greedy, then simulated annealing
            selected, cover_count = _greedy_construct(total, balls, word_to_codewords, deadline - 30.0)
            if time.time() < deadline - 15.0:
                selected, cover_count = _simulated_annealing(total, balls, word_to_codewords, selected, cover_count, deadline - 10.0)
        else:
            # Start from greedy, then tabu search
            selected, cover_count = _greedy_construct(total, balls, word_to_codewords, deadline - 30.0)
            if time.time() < deadline - 15.0:
                selected, cover_count = _tabu_search(total, balls, word_to_codewords, selected, cover_count, deadline - 10.0)
        
        # Local improvement: remove redundant codewords
        selected, cover_count = _remove_redundant(total, balls, selected, cover_count)
        
        # 1-swap improvement
        if time.time() < deadline - 8.0:
            selected, cover_count = _one_swap_improve(total, balls, word_to_codewords, selected, cover_count, deadline - 8.0)
        
        # 2-exchange improvement
        if len(selected) > 486 and time.time() < deadline - 5.0:
            selected, cover_count = _two_exchange_improve(total, balls, word_to_codewords, selected, cover_count, deadline - 5.0)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            
            if best_size <= 403:
                break
            if best_size <= 486:
                break

    # Convert best_set to output format
    if best_set is None:
        selected, _ = _greedy_construct(total, balls, word_to_codewords, deadline)
        best_set = selected

    result = []
    for w in best_set:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
        
    return result

def _greedy_construct(total, balls, word_to_codewords, deadline):
    """Randomized greedy set cover construction."""
    selected = []
    cover_count = np.zeros(total, dtype=np.int32)
    uncovered = np.ones(total, dtype=bool)
    
    while np.any(uncovered) and time.time() < deadline:
        # Count how many uncovered words each candidate covers
        counts = uncovered[balls].sum(axis=1)
        
        if np.max(counts) == 0:
            break
        
        max_count = np.max(counts)
        candidates = np.where(counts == max_count)[0]
        
        # Random selection among top candidates
        best_idx = candidates[np.random.randint(len(candidates))]
        
        selected.append(best_idx)
        cover_count[balls[best_idx]] += 1
        uncovered = (cover_count == 0)
    
    return selected, cover_count

def _remove_redundant(total, balls, selected, cover_count):
    """Remove codewords that are not uniquely needed."""
    changed = True
    while changed:
        changed = False
        new_selected = []
        for c in selected:
            if np.min(cover_count[balls[c]]) > 1:
                cover_count[balls[c]] -= 1
                changed = True
            else:
                new_selected.append(c)
        selected = new_selected
    return selected, cover_count

def _one_swap_improve(total, balls, word_to_codewords, selected, cover_count, deadline):
    """Try to replace each codeword with another that covers its critical words."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        new_selected = []
        for c in selected:
            if np.min(cover_count[balls[c]]) > 1:
                cover_count[balls[c]] -= 1
                improved = True
                continue
            
            critical_mask = (cover_count[balls[c]] == 1)
            critical_words = balls[c][critical_mask]
            
            if len(critical_words) == 0:
                cover_count[balls[c]] -= 1
                improved = True
                continue

            # Find candidates that cover all critical words
            candidates = set()
            for cw in critical_words:
                candidates = candidates.intersection(word_to_codewords[cw])
            
            if c in candidates:
                candidates.discard(c)
            
            if candidates:
                c_prime = random.choice(list(candidates))
                cover_count[balls[c]] -= 1
                cover_count[balls[c_prime]] += 1
                new_selected.append(c_prime)
                improved = True
            else:
                new_selected.append(c)
        
        selected = new_selected
    
    return selected, cover_count

def _two_exchange_improve(total, balls, word_to_codewords, selected, cover_count, deadline):
    """Try to remove 2 codewords and add 1."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        num_trials = min(5000, len(selected) * (len(selected) - 1) // 2)
        
        for _ in range(num_trials):
            if time.time() > deadline:
                break
            
            idx1 = random.randint(0, len(selected) - 1)
            idx2 = random.randint(0, len(selected) - 1)
            if idx1 == idx2:
                continue
                
            c1 = selected[idx1]
            c2 = selected[idx2]
            
            cover_count[balls[c1]] -= 1
            cover_count[balls[c2]] -= 1
            
            uncovered_after = (cover_count == 0)
            
            if np.any(uncovered_after):
                uncovered_words = np.where(uncovered_after)[0]
                
                # Find c3 that covers all uncovered words
                candidates = set()
                for uw in uncovered_words:
                    candidates = candidates.intersection(word_to_codewords[uw])
                
                if c1 in candidates:
                    candidates.discard(c1)
                if c2 in candidates:
                    candidates.discard(c2)
                
                if candidates:
                    c3 = random.choice(list(candidates))
                    cover_count[balls[c3]] += 1
                    new_selected = [c for c in selected if c != c1 and c != c2]
                    new_selected.append(c3)
                    selected = new_selected
                    improved = True
                    break
                else:
                    cover_count[balls[c1]] += 1
                    cover_count[balls[c2]] += 1
            else:
                new_selected = [c for c in selected if c != c1 and c != c2]
                selected = new_selected
                improved = True
                break
            
            if not improved:
                cover_count[balls[c1]] += 1
                cover_count[balls[c2]] += 1
    
    return selected, cover_count

def _simulated_annealing(total, balls, word_to_codewords, selected, cover_count, deadline):
    """Simulated annealing to escape local minima."""
    T = 30.0
    T_min = 5.0
    cooling_rate = 0.95
    
    for _ in range(100):
        if time.time() > deadline:
            break
        if T < T_min:
            break
        
        if len(selected) > 1:
            idx_swap = random.randint(0, len(selected) - 1)
            c_old = selected[idx_swap]
            
            critical_mask = (cover_count[balls[c_old]] == 1)
            critical_words = balls[c_old][critical_mask]
            
            if len(critical_words) > 0:
                cw = critical_words[random.randint(0, len(critical_words) - 1)]
                alternatives = word_to_codewords[cw]
                if c_old in alternatives:
                    alternatives = [a for a in alternatives if a != c_old]
                
                if alternatives:
                    c_new = random.choice(alternatives)
                    
                    cover_count[balls[c_old]] -= 1
                    cover_count[balls[c_new]] += 1
                    
                    uncovered_after = (cover_count == 0)
                    if not np.any(uncovered_after):
                        selected[idx_swap] = c_new
                    else:
                        cover_count[balls[c_old]] += 1
                        cover_count[balls[c_new]] -= 1
        
        T *= cooling_rate
    
    return selected, cover_count

def _tabu_search(total, balls, word_to_codewords, selected, cover_count, deadline):
    """Tabu search with short-term memory."""
    tabu_list = {}
    tabu_tenure = 10
    current_iter = 0
    
    for _ in range(50):
        if time.time() > deadline:
            break
        
        current_iter += 1
        
        # Clear expired tabu entries
        expired = [c for c, t in tabu_list.items() if current_iter - t >= tabu_tenure]
        for c in expired:
            del tabu_list[c]
        
        if len(selected) > 1:
            idx_swap = random.randint(0, len(selected) - 1)
            c_old = selected[idx_swap]
            
            critical_mask = (cover_count[balls[c_old]] == 1)
            critical_words = balls[c_old][critical_mask]
            
            if len(critical_words) > 0:
                cw = critical_words[random.randint(0, len(critical_words) - 1)]
                alternatives = [a for a in word_to_codewords[cw] if a != c_old and a not in tabu_list]
                
                if alternatives:
                    c_new = random.choice(alternatives)
                    
                    cover_count[balls[c_old]] -= 1
                    cover_count[balls[c_new]] += 1
                    
                    uncovered_after = (cover_count == 0)
                    if not np.any(uncovered_after):
                        selected[idx_swap] = c_new
                        tabu_list[c_old] = current_iter
                    else:
                        cover_count[balls[c_old]] += 1
                        cover_count[balls[c_new]] -= 1
    
    return selected, cover_count
>>>>>>> REPLACE
<<<<<<< SEARCH
def _compute_dual_potentials(total, balls, word_to_codewords, deadline):
    """
    Approximate dual potentials via subgradient method on the LP relaxation.
    The LP relaxation of the covering problem is:
      min sum(x_j) s.t. sum_{j: i in B_j} x_j >= 1 for all i, x_j >= 0
    Dual: max sum(y_i) s.t. sum_{i in B_j} y_i <= 1 for all j, y_i >= 0
    We approximate y_i via subgradient ascent.
    """
    y = np.ones(total, dtype=np.float64) * 0.1  # Initialize dual variables
    step_size = 0.01
    max_iters = 50
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        
        # Compute gradient: for each word i, gradient_i = 1 - min_{j: i in B_j} (sum_{k in B_j} y_k)
        # But this is expensive; use a simpler approximation
        # For each codeword j, compute slack_j = 1 - sum_{i in B_j} y_i
        # Words with high slack indicate they are "loose" constraints
        
        # Simplified: update y_i based on how "covered" word i is
        # If word i is covered by many codewords with high y, reduce y_i
        # If word i is covered by few codewords or low y, increase y_i
        
        # Use subgradient: for each word i, the constraint is sum_{j: i in B_j} x_j >= 1
        # Dual constraint: sum_{i in B_j} y_i <= 1
        # Gradient of dual objective w.r.t. y_i is 1 (for the primal feasibility)
        # But we need to project onto the dual feasible region
        
        # Simple heuristic: normalize y so that for each codeword j, sum_{i in B_j} y_i <= 1
        # Scale down y if any codeword violates
        
        max_violation = 0
        for j in range(total):
            s = np.sum(y[balls[j][1:]])  # Sum y over ball of j (excluding j itself, but balls[j][0]=j)
            # Actually balls[j] contains j and its neighbors; we want sum over all in ball
            s = np.sum(y[balls[j]])
            if s > 1.0:
                max_violation = max(max_violation, s - 1.0)
        
        if max_violation > 0:
            # Scale down y proportionally
            y = y / (1.0 + max_violation)
        else:
            # Can increase y slightly
            y = y * (1.0 + step_size)
            # But cap at reasonable values
            y = np.minimum(y, 0.5)
    
    return y
=======

>>>>>>> REPLACE
