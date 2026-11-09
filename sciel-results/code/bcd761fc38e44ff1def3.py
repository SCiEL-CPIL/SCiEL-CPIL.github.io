import time
import random
import numpy as np

def construct():
    start_time = time.time()
    timeout = 145.0  # Safety margin before 150s limit
    
    def evaluate(S):
        """Count distinct d > 0 such that a, a+d, a+2d are all in S."""
        S_set = set(S)
        valid_diffs = set()
        S_list = sorted(S)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                d = b - a
                c = a + 2 * d
                if c in S_set:
                    valid_diffs.add(d)
        return len(valid_diffs)
    
    def get_covered_diffs(S):
        """Return the set of covered differences and for each diff, which triples cover it."""
        S_set = set(S)
        covered = {}
        S_list = sorted(S)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                d = b - a
                c = a + 2 * d
                if c in S_set:
                    if d not in covered:
                        covered[d] = []
                    covered[d].append((a, b, c))
        return covered
    
    def find_missing_diffs(S, max_check=200):
        """Find which small differences are NOT covered by S."""
        S_set = set(S)
        covered = get_covered_diffs(S)
        missing = []
        for d in range(1, max_check + 1):
            if d not in covered:
                missing.append(d)
        return missing, covered
    
    def try_add_element_for_diff(S, d):
        """Try to find an element x such that adding x creates a new progression with difference d."""
        S_set = set(S)
        S_list = sorted(S)
        
        # For diff d, we need a, a+d, a+2d all in S∪{x}
        # Cases:
        # 1. x is the missing 'a': need a+d and a+2d in S
        # 2. x is the missing 'a+d': need a and a+2d in S  
        # 3. x is the missing 'a+2d': need a and a+d in S
        
        candidates = []
        
        for a in S_list:
            # Case 1: x = a - d (if valid)
            x1 = a - d
            if 1 <= x1 <= 512 and x1 not in S_set:
                if a + d in S_set and a + 2*d in S_set:
                    candidates.append(x1)
            
            # Case 2: x = a + d (if valid)
            x2 = a + d
            if 1 <= x2 <= 512 and x2 not in S_set:
                if a + 2*d in S_set:
                    candidates.append(x2)
            
            # Case 3: x = a + 2*d (if valid)
            x3 = a + 2*d
            if 1 <= x3 <= 512 and x3 not in S_set:
                if a + d in S_set:
                    candidates.append(x3)
        
        return candidates
    
    def greedy_difference_coverage(S):
        """Greedy: repeatedly add the element that covers the most new differences."""
        S = set(S)
        for _ in range(100):
            if time.time() - start_time > timeout - 5.0:
                break
            if len(S) >= 32:
                break
            
            best_elem = None
            best_new_diffs = 0
            best_new_count = 0
            
            current_covered = set(get_covered_diffs(S).keys())
            
            # Try all possible elements
            for x in range(1, 513):
                if x in S:
                    continue
                
                new_S = S | {x}
                new_covered = set(get_covered_diffs(new_S).keys())
                new_diffs = new_covered - current_covered
                
                if len(new_diffs) > best_new_diffs:
                    best_new_diffs = len(new_diffs)
                    best_new_count = len(new_diffs)
                    best_elem = x
                elif len(new_diffs) == best_new_diffs and len(new_diffs) > 0:
                    # Tie-breaking: prefer elements that also keep score high
                    pass
            
            if best_elem is not None:
                S.add(best_elem)
            else:
                # No improvement, add random
                while True:
                    x = random.randint(1, 512)
                    if x not in S:
                        S.add(x)
                        break
        
        return S
    
    def directed_swap_search(S, max_iters=5000):
        """Directed search: prioritize swaps that cover more differences."""
        current = set(S)
        current_score = evaluate(current)
        best = set(current)
        best_sc = current_score
        
        T = 5.0
        T_min = 0.01
        cooling_rate = 0.9997
        
        for it in range(max_iters):
            if time.time() - start_time > timeout - 3.0:
                break
            
            # Directed move: try to cover a missing difference
            missing, covered = find_missing_diffs(current, max_check=100)
            
            if missing and random.random() < 0.6:
                # Pick a random missing diff and try to cover it
                target_d = random.choice(missing)
                candidates = try_add_element_for_diff(current, target_d)
                
                if candidates:
                    # Remove a random element and add one of the candidates
                    to_remove = random.choice(list(current))
                    new_S = set(current)
                    new_S.discard(to_remove)
                    # Pick candidate not equal to removed
                    valid_cands = [c for c in candidates if c != to_remove]
                    if valid_cands:
                        to_add = random.choice(valid_cands)
                        new_S.add(to_add)
                    else:
                        new_S.add(candidates[0])
                    
                    new_score = evaluate(new_S)
                    delta = new_score - current_score
                    
                    if delta > 0:
                        current = new_S
                        current_score = new_score
                        if new_score > best_sc:
                            best_sc = new_score
                            best = set(new_S)
                    else:
                        if random.random() < np.exp(delta / T):
                            current = new_S
                            current_score = new_score
                else:
                    # Fallback to random swap
                    new_S = set(current)
                    to_remove = random.choice(list(new_S))
                    new_S.discard(to_remove)
                    while True:
                        to_add = random.randint(1, 512)
                        if to_add not in new_S:
                            new_S.add(to_add)
                            break
                    new_score = evaluate(new_S)
                    delta = new_score - current_score
                    if delta > 0:
                        current = new_S
                        current_score = new_score
                        if new_score > best_sc:
                            best_sc = new_score
                            best = set(new_S)
                    else:
                        if random.random() < np.exp(delta / T):
                            current = new_S
                            current_score = new_score
            else:
                # Random swap (exploration)
                new_S = set(current)
                to_remove = random.choice(list(new_S))
                new_S.discard(to_remove)
                while True:
                    to_add = random.randint(1, 512)
                    if to_add not in new_S:
                        new_S.add(to_add)
                        break
                new_score = evaluate(new_S)
                delta = new_score - current_score
                if delta > 0:
                    current = new_S
                    current_score = new_score
                    if new_score > best_sc:
                        best_sc = new_score
                        best = set(new_S)
                else:
                    if random.random() < np.exp(delta / T):
                        current = new_S
                        current_score = new_score
            
            T *= cooling_rate
            if T < T_min:
                T = T_min
        
        return best, best_sc
    
    # Known good starting sets
    known_sets = [
        set(list(range(1, 27)) + [28, 29, 30, 31, 32, 48]),
        set(range(1, 33)),
        set(range(2, 34)),
        set(range(1, 20)) | set(range(30, 52)),  # Two blocks
        set(range(1, 16)) | set(range(30, 46)),   # Two closer blocks
        set(range(1, 11)) | set(range(20, 31)) | set(range(40, 51)),  # Three blocks
    ]
    
    # Add some structured sets
    def make_structured(block_sizes, spacings):
        s = set()
        pos = 1
        for bs, sp in zip(block_sizes, spacings):
            for k in range(bs):
                val = pos + k
                if 1 <= val <= 512:
                    s.add(val)
            pos += bs + sp
        while len(s) < 32:
            val = random.randint(1, 512)
            if val not in s:
                s.add(val)
        return s
    
    known_sets.append(make_structured([16, 16], [10]))
    known_sets.append(make_structured([8, 8, 8, 8], [5, 5, 5]))
    known_sets.append(make_structured([4, 4, 4, 4, 4, 4, 4, 4], [3, 3, 3, 3, 3, 3, 3]))
    
    # Evaluate all candidates
    scored = []
    for c in known_sets:
        if len(c) == 32:
            sc = evaluate(c)
            scored.append((sc, c))
    
    # Also try greedy difference coverage from a few starts
    for c in known_sets[:3]:
        if len(c) == 32:
            g = greedy_difference_coverage(c)
            if len(g) == 32:
                sc = evaluate(g)
                scored.append((sc, g))
    
    scored.sort(key=lambda x: -x[0])
    
    best_score = scored[0][0]
    best_set = scored[0][1].copy()
    
    # Multi-start directed swap search
    num_restarts = 20
    for restart in range(num_restarts):
        if time.time() - start_time > timeout - 5.0:
            break
        
        if restart == 0:
            init = best_set
        elif restart < 6:
            # Perturb best
            init = set(best_set)
            elems = list(init)
            n_mut = random.randint(2, 4)
            for _ in range(n_mut):
                to_remove = random.choice(list(init))
                init.discard(to_remove)
                while True:
                    to_add = random.randint(1, 512)
                    if to_add not in init:
                        init.add(to_add)
                        break
        else:
            # Use one of the known sets or random
            if random.random() < 0.5:
                init = random.choice(known_sets)
            else:
                init = set(random.sample(range(1, 513), 32))
        
        sa_best, sa_score = directed_swap_search(init, max_iters=4000)
        
        if sa_score > best_score:
            best_score = sa_score
            best_set = sa_best
    
    return list(best_set)
