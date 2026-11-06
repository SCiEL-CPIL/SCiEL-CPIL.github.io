import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 82  # safety margin before 90s kill
    
    n = 128
    TARGET_START = 128
    TARGET_END = 256
    TARGETS = list(range(TARGET_START, TARGET_END + 1))
    
    # Precompute target indices for faster access
    target_indices = np.arange(TARGET_START, TARGET_END + 1)
    
    def evaluate(A_set):
        """Return (max_multiplicity, covered) for the given set A."""
        v = np.zeros(129, dtype=np.int32)
        for x in A_set:
            v[x] = 1
        conv = np.convolve(v, v, mode='full')
        max_mult = 0
        covered = True
        for t in target_indices:
            ordered = conv[t]
            diag = 1 if (t % 2 == 0 and (t // 2) in A_set) else 0
            unordered = (ordered + diag) // 2
            if unordered == 0:
                covered = False
            if unordered > max_mult:
                max_mult = unordered
        return max_mult, covered
    
    def evaluate_fast(v_array, A_set):
        """Faster evaluation using precomputed convolution when possible."""
        conv = np.convolve(v_array, v_array, mode='full')
        max_mult = 0
        covered = True
        for t in target_indices:
            ordered = conv[t]
            diag = 1 if (t % 2 == 0 and (t // 2) in A_set) else 0
            unordered = (ordered + diag) // 2
            if unordered == 0:
                covered = False
            if unordered > max_mult:
                max_mult = unordered
        return max_mult, covered
    
    def make_initials():
        """Create diverse initial solutions."""
        initials = []
        # Full set
        initials.append(set(range(0, 129)))
        # Half: [64, 128]
        initials.append(set(range(64, 129)))
        # Even numbers
        initials.append(set(range(0, 129, 2)))
        # Odd numbers plus 0
        initials.append(set([0]) | set(range(1, 129, 2)))
        # Multiples of 3
        initials.append(set(range(0, 129, 3)))
        # Multiples of 4
        initials.append(set(range(0, 129, 4)))
        # Random sets of various sizes
        random.seed(123)
        for size in [45, 55, 65, 75, 85, 95, 105]:
            s = set(random.sample(range(0, 129), size))
            initials.append(s)
        # Structured: alternating blocks
        s = set()
        for i in range(0, 129, 8):
            s.update(range(i, i + 4))
        initials.append(s)
        # Another structured pattern
        s = set()
        for i in range(0, 129, 6):
            s.update(range(i, i + 3))
        initials.append(s)
        return initials
    
    def fix_coverage(A_set):
        """Greedily fix coverage by adding elements."""
        A = set(A_set)
        for t in TARGETS:
            found = False
            for a in sorted(A):
                b = t - a
                if b >= a and b in A:
                    found = True
                    break
            if not found:
                # Add elements to cover this sum
                mid = t // 2
                if 0 <= mid <= 128:
                    A.add(mid)
                if t % 2 == 1 and mid + 1 <= 128:
                    A.add(mid + 1)
                # Also try neighbors
                for offset in [-1, 1, -2, 2]:
                    x = mid + offset
                    if 0 <= x <= 128 and x not in A:
                        # Check if adding x helps
                        A.add(x)
                        break
        return A
    
    def local_search(A_set, time_budget, seed=None):
        """Simulated annealing local search with smarter moves."""
        if seed is not None:
            random.seed(seed)
        
        current_A = set(A_set)
        v = np.zeros(129, dtype=np.int32)
        for x in current_A:
            v[x] = 1
        
        current_score, current_covered = evaluate(current_A)
        
        if not current_covered:
            current_A = fix_coverage(current_A)
            v = np.zeros(129, dtype=np.int32)
            for x in current_A:
                v[x] = 1
            current_score, current_covered = evaluate(current_A)
        
        best_A = set(current_A)
        best_score = current_score if current_covered else float('inf')
        
        T = 20.0
        T_min = 0.3
        cooling = 0.999
        
        # Track which elements are in the set for fast membership
        in_set = [False] * 129
        for x in current_A:
            in_set[x] = True
        
        iter_count = 0
        while time.time() < time_budget and T > T_min:
            iter_count += 1
            
            # Generate a move: prefer removing elements that contribute to peak
            if len(current_A) > 10 and random.random() < 0.55:
                # Remove: bias toward elements in high-multiplicity sums
                x = random.choice(list(current_A))
                new_A = current_A - {x}
                new_v = v.copy()
                new_v[x] = 0
                new_score, new_covered = evaluate_fast(new_v, new_A)
            else:
                # Add: bias toward elements that help cover gaps or reduce peak
                # Pick from elements not in set, with some randomness
                candidates = [i for i in range(129) if not in_set[i]]
                if not candidates:
                    continue
                x = random.choice(candidates)
                new_A = current_A | {x}
                new_v = v.copy()
                new_v[x] = 1
                new_score, new_covered = evaluate_fast(new_v, new_A)
            
            # Score with heavy penalty for uncovered
            if new_covered:
                new_cost = new_score
            else:
                new_cost = new_score + 5000
            
            if current_covered:
                current_cost = current_score
            else:
                current_cost = current_score + 5000
            
            delta = new_cost - current_cost
            
            accept = False
            if delta < 0:
                accept = True
            elif T > 0:
                # Avoid overflow in exp
                if delta / T < 50:
                    accept = random.random() < np.exp(-delta / T)
            
            if accept:
                current_A = new_A
                current_score = new_score
                current_covered = new_covered
                v = new_v
                in_set[x] = (x in current_A)
                
                if current_covered and current_score < best_score:
                    best_A = set(current_A)
                    best_score = current_score
            
            T *= cooling
        
        return best_A, best_score
    
    # Multi-start local search
    best_A = None
    best_score = float('inf')
    
    initials = make_initials()
    
    # Shuffle for diversity
    random.seed(42)
    random.shuffle(initials)
    
    # Phase 1: Quick evaluation to rank starts
    quick_time = start_time + 12
    scored_initials = []
    for init in initials:
        if time.time() > quick_time:
            break
        s, c = evaluate(init)
        cost = s if c else s + 5000
        scored_initials.append((cost, init))
    
    scored_initials.sort(key=lambda x: x[0])
    
    # Phase 2: Local search from ranked starts
    for i, (cost, init) in enumerate(scored_initials):
        if time.time() >= deadline - 5:
            break
        remaining_starts = len(scored_initials) - i
        remaining_time = deadline - 5 - time.time()
        if remaining_time <= 0:
            break
        # Better starts get more time
        time_alloc = max(3, min(remaining_time / max(1, remaining_starts), 25))
        
        seed = 1000 + i
        ls_result, ls_score = local_search(init, time.time() + time_alloc, seed=seed)
        
        if ls_score < best_score:
            best_score = ls_score
            best_A = ls_result
    
    # Phase 3: Additional random restarts
    if time.time() < deadline - 15 and best_A is not None:
        num_extra = 4
        for j in range(num_extra):
            if time.time() >= deadline - 5:
                break
            random.seed(5000 + j)
            size = random.randint(50, 90)
            init = set(random.sample(range(0, 129), size))
            init = fix_coverage(init)
            
            time_alloc = max(3, (deadline - 5 - time.time()) / max(1, num_extra - j))
            ls_result, ls_score = local_search(init, time.time() + time_alloc, seed=7000 + j)
            
            if ls_score < best_score:
                best_score = ls_score
                best_A = ls_result
    
    # Fallback
    if best_A is None or best_score == float('inf'):
        best_A = set(range(0, 129))
        best_score, best_covered = evaluate(best_A)
        if not best_covered:
            best_A = set(range(64, 129))
    
    # Final verification
    final_score, final_covered = evaluate(best_A)
    if not final_covered:
        best_A = set(range(0, 129))
        final_score, final_covered = evaluate(best_A)
    
    return sorted(best_A)
