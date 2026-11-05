# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    """
    Model-and-optimize approach:
    1. Fast validity check using precomputed 3-sums and forbidden residues.
    2. Greedy construction from multiple angles.
    3. Local search with add/remove/swap moves, guided by a scoring model.
    4. Simulated annealing-style exploration to escape local optima.
    """
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    def time_left():
        return deadline - time.time()
    
    N = 192
    ALL = list(range(1, N + 1))
    
    def compute_checkers(s):
        """
        Precompute structures to quickly check if x can be added to s.
        Returns:
        - sums3: set of all sums of 3 distinct elements from s
        - F: dict {a: set of forbidden residues mod a}
        """
        lst = sorted(s)
        n = len(lst)
        sums3 = set()
        if n >= 3:
            for i in range(n):
                for j in range(i + 1, n):
                    s2 = lst[i] + lst[j]
                    for k in range(j + 1, n):
                        sums3.add(s2 + lst[k])
        
        F = {}
        # Precompute pair sums for each exclusion
        for idx_a in range(n):
            a = lst[idx_a]
            forbidden = set()
            for i in range(n):
                if i == idx_a:
                    continue
                for j in range(i + 1, n):
                    if j == idx_a:
                        continue
                    residue = -(lst[i] + lst[j]) % a
                    forbidden.add(residue)
            F[a] = forbidden
        return sums3, F

    def can_add(x, s, checkers):
        """
        Check if x can be added to s using precomputed checkers.
        """
        sums3, F = checkers
        
        # Check 1: Does any sum of 3 from s equal a multiple of x?
        if sums3:
            min_sum = min(sums3) # Actually min is sum of 3 smallest
            # Optimize: just iterate multiples
            max_sum = max(sums3)
            # Multiples of x in [min_sum, max_sum]
            first_mult = ((min_sum + x - 1) // x) * x
            for m in range(first_mult, max_sum + 1, x):
                if m in sums3:
                    return False
        
        # Check 2: For each a in s, is x % a in F[a]?
        for a in s:
            if x % a in F[a]:
                return False
                
        return True

    def is_valid(s):
        """
        Check if set s is valid.
        For each element a in s, check that no sum of 3 distinct OTHER elements is divisible by a.
        """
        lst = sorted(s)
        n = len(lst)
        if n < 4:
            return True
        
        for idx_a in range(n):
            a = lst[idx_a]
            # Check all triples from s \ {a}
            for i in range(n):
                if i == idx_a:
                    continue
                for j in range(i + 1, n):
                    if j == idx_a:
                        continue
                    s2 = lst[i] + lst[j]
                    for k in range(j + 1, n):
                        if k == idx_a:
                            continue
                        if (s2 + lst[k]) % a == 0:
                            return False
        return True

    def greedy_from_high(rng):
        """Greedy: add elements from high to low if valid."""
        selected = set()
        checkers = (set(), {})
        for x in range(N, 0, -1):
            if can_add(x, selected, checkers):
                selected.add(x)
                checkers = compute_checkers(selected)
        return selected

    def greedy_from_low(rng):
        """Greedy: add elements from low to high if valid."""
        selected = set()
        checkers = (set(), {})
        for x in range(1, N + 1):
            if can_add(x, selected, checkers):
                selected.add(x)
                checkers = compute_checkers(selected)
        return selected

    def greedy_random(rng):
        """Greedy: random order."""
        order = ALL[:]
        rng.shuffle(order)
        selected = set()
        checkers = (set(), {})
        for x in order:
            if can_add(x, selected, checkers):
                selected.add(x)
                checkers = compute_checkers(selected)
        return selected

    def greedy_biased(rng):
        """Greedy: prefer larger numbers but with some randomness."""
        selected = set()
        candidates = ALL[:]
        rng.shuffle(candidates)
        candidates.sort(key=lambda x: -x + rng.uniform(0, 20))
        checkers = (set(), {})
        for x in candidates:
            if can_add(x, selected, checkers):
                selected.add(x)
                checkers = compute_checkers(selected)
        return selected

    def local_search(s, rng, max_iters=200):
        """
        Local search with add, remove, and swap moves.
        Uses fast can_add for additions.
        """
        s = set(s)
        cur_size = len(s)
        no_improve = 0
        checkers = compute_checkers(s)
        
        for _ in range(max_iters):
            if time_left() < 2:
                break
            
            improved = False
            moved = False
            
            # Try additions - try more candidates
            candidates = [x for x in ALL if x not in s]
            rng.shuffle(candidates)
            for x in candidates[:30]:
                if can_add(x, s, checkers):
                    s.add(x)
                    cur_size += 1
                    improved = True
                    moved = True
                    checkers = compute_checkers(s)
                    break
            
            if improved:
                no_improve = 0
                continue
            
            no_improve += 1
            
            # Try swaps - try more combinations
            elems = list(s)
            rng.shuffle(elems)
            for rem in elems[:8]:
                candidates = [x for x in ALL if x not in s]
                rng.shuffle(candidates)
                for add in candidates[:30]:
                    new_s = (s - {rem}) | {add}
                    if is_valid(new_s):
                        s = new_s
                        improved = True
                        moved = True
                        checkers = compute_checkers(s)
                        break
                if improved:
                    break
            
            if improved:
                no_improve = 0
                continue
            
            # Try double moves: remove two, add two
            if no_improve > 5:
                elems = list(s)
                rng.shuffle(elems)
                for i in range(min(6, len(elems))):
                    for j in range(i + 1, min(len(elems), i + 8)):
                        rem1, rem2 = elems[i], elems[j]
                        candidates = [x for x in ALL if x not in s]
                        rng.shuffle(candidates)
                        for add1 in candidates[:15]:
                            for add2 in candidates[:15]:
                                if add1 == add2:
                                    continue
                                new_s = (s - {rem1, rem2}) | {add1, add2}
                                if is_valid(new_s):
                                    s = new_s
                                    improved = True
                                    moved = True
                                    checkers = compute_checkers(s)
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    no_improve = 0
                    continue
            
            if not moved:
                break
        
        return s
    
    # Phase 1: Generate diverse starting solutions
    rng = random.Random(42)
    
    starts = []
    starts.append(greedy_from_high(rng))
    starts.append(greedy_from_low(rng))
    for _ in range(5):
        starts.append(greedy_random(rng))
    for _ in range(3):
        starts.append(greedy_biased(rng))
    
    # Also try some structured starts
    # Start with large numbers only
    s_large = set()
    checkers_large = (set(), {})
    for x in range(N, N // 2, -1):
        if can_add(x, s_large, checkers_large):
            s_large.add(x)
            checkers_large = compute_checkers(s_large)
    starts.append(s_large)
    
    # Start with numbers in certain ranges
    s_mid = set()
    checkers_mid = (set(), {})
    for x in range(N // 2, N // 4, -1):
        if can_add(x, s_mid, checkers_mid):
            s_mid.add(x)
            checkers_mid = compute_checkers(s_mid)
    starts.append(s_mid)
    
    best = max(starts, key=len)
    best_size = len(best)
    
    print(f"Initial best size: {best_size}", flush=True)
    
    # Phase 2: Local search from each start
    for i, start_s in enumerate(starts):
        if time_left() < 5:
            break
        
        result = local_search(start_s, rng, max_iters=150)
        if len(result) > best_size:
            best_size = len(result)
            best = result
            print(f"Improved to {best_size} at iteration {i}", flush=True)
    
    # Phase 3: More aggressive search with restarts
    for iteration in range(300):
        if time_left() < 10:
            break
        
        # Generate a new starting point with some randomness
        if iteration % 5 == 0:
            # Perturb the best solution
            s = set(best)
            n_remove = rng.randint(2, 5)
            to_remove = rng.sample(list(s), min(n_remove, len(s)))
            s -= set(to_remove)
            start_s = s
        elif iteration % 5 == 1:
            start_s = greedy_from_high(rng)
        elif iteration % 5 == 2:
            start_s = greedy_random(rng)
        elif iteration % 5 == 3:
            start_s = greedy_biased(rng)
        else:
            # Random subset
            k = rng.randint(30, 45)
            start_s = set(rng.sample(ALL, k))
            # Filter to valid subset
            valid_s = set()
            checkers_valid = (set(), {})
            for x in sorted(start_s, reverse=True):
                if can_add(x, valid_s, checkers_valid):
                    valid_s.add(x)
                    checkers_valid = compute_checkers(valid_s)
            start_s = valid_s
        
        result = local_search(start_s, rng, max_iters=100)
        if len(result) > best_size:
            best_size = len(result)
            best = result
            print(f"New best: {best_size} at iteration {iteration}", flush=True)
    
    # Phase 4: Final polish - more aggressive local search on best
    if time_left() > 15:
        print(f"Starting final polish, current best: {best_size}", flush=True)
        for _ in range(50):
            if time_left() < 5:
                break
            s = set(best)
            # Small perturbation
            n_remove = rng.randint(1, 3)
            to_remove = rng.sample(list(s), min(n_remove, len(s)))
            s -= set(to_remove)
            result = local_search(s, rng, max_iters=100)
            if len(result) > best_size:
                best_size = len(result)
                best = result
                print(f"Polish improved to {best_size}", flush=True)
    
    # Final check
    if not is_valid(best):
        # Fallback to greedy
        best = greedy_from_high(rng)
        while not is_valid(best):
            best = set(list(best)[:-1])
    
    return list(best)

# EVOLVE-BLOCK-END
