# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    """
    Maximum cardinality subset of [1,192] such that no element divides the sum of any 3 distinct others.
    Uses bitset-based conflict precomputation and simulated annealing / hill climbing.
    """
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    def time_left():
        return deadline - time.time()
    
    N = 192
    ALL = list(range(1, N + 1))
    
    # Precompute all triples and their sums
    # For each element x, we need to know: does x divide any sum of 3 distinct OTHER elements?
    # Strategy: Precompute for each pair (i,j) the sum, then for each k the total.
    # But for fast checking during search, we use a different approach.
    
    # Approach: For a given set S, element x in S is "bad" if there exist a,b,c in S\{x} with (a+b+c) % x == 0.
    # We can precompute for each x, the set of "forbidden residues" that would make it bad.
    # Actually, a better approach for speed:
    # Precompute all 3-sums: for each triple (i,j,k), sum = i+j+k.
    # For a candidate set S, x is bad if any triple from S\{x} has sum divisible by x.
    
    # Let's precompute for each x (1..192) and each pair (i,j) with i<j, the residue needed:
    # If x is in S, and i,j in S\{x}, then we need to check if there's k in S\{x,i,j} such that (i+j+k) % x == 0.
    # This is still O(n^3) per check.
    
    # Better: Precompute all triples (i,j,k) and their sums. There are C(192,3) ≈ 1.16M triples.
    # For each triple, store the sum. Then for a set S, x is bad if there's a triple from S\{x} with sum % x == 0.
    
    # Even better: For each x, precompute the set of all triples (i,j,k) such that (i+j+k) % x == 0.
    # Then x is bad in S if any such triple is entirely contained in S\{x}.
    
    # Let's use a different strategy for speed:
    # For each x, maintain a count of how many "conflicting triples" are present in S\{x}.
    # A conflicting triple for x is any triple (i,j,k) with i,j,k != x and (i+j+k) % x == 0.
    
    # Precomputation:
    # For each x in 1..192, find all triples (i,j,k) with i<j<k, none equal to x, and (i+j+k) % x == 0.
    # Store these as a list of triples for each x.
    
    # This might be too much memory. Let's think differently.
    
    # Alternative: For each x, and for each pair (i,j) with i,j != x, compute the required residue r = -(i+j) % x.
    # Then x is bad if there exists k in S\{x,i,j} with k % x == r.
    # We can maintain for each x a set of residues present in S\{x}.
    
    # Let's precompute for each x and each pair (i,j), i != j, i,j != x:
    # residue_needed[x][i][j] = -(i+j) % x
    # Then during search, for each x in S, check if there's a pair (i,j) in S\{x} such that
    # the residue -(i+j) % x is present in {k % x : k in S\{x,i,j}}.
    
    # This is still complex. Let's use a simpler but effective approach:
    
    # Precompute all triples and their sums. Store in a list: triples = [(i,j,k, sum) for ...]
    # For each x, precompute which triples have sum % x == 0.
    
    # Actually, let's just do a direct check but optimized with early termination.
    # For a set S of size n, checking validity is O(n^4) in the worst case.
    # But we can do better: for each x in S, iterate over pairs (i,j) in S\{x}, and check if there's k in S\{x,i,j} with (i+j+k) % x == 0.
    # We can precompute for each x and each residue r, the set of elements in [1,192] with element % x == r.
    
    # Precompute: for each x in 1..192, for each r in 0..x-1, the set of elements e in 1..192 with e % x == r.
    residue_sets = {}
    for x in range(1, N + 1):
        residue_sets[x] = {}
        for r in range(x):
            residue_sets[x][r] = set(e for e in range(1, N + 1) if e % x == r)
    
    def is_valid(s):
        """Check if set s is valid."""
        lst = sorted(s)
        n = len(lst)
        if n < 4:
            return True
        
        s_set = set(lst)
        
        for idx_a in range(n):
            a = lst[idx_a]
            others = [lst[i] for i in range(n) if i != idx_a]
            n_others = len(others)
            if n_others < 3:
                continue
            
            # For each pair (i,j) in others, check if there's k in others\{i,j} with (i+j+k) % a == 0
            for i in range(n_others):
                for j in range(i + 1, n_others):
                    s2 = others[i] + others[j]
                    r = (-s2) % a
                    # Check if there's k in others, k != others[i], k != others[j], with k % a == r
                    for k in range(n_others):
                        if k == i or k == j:
                            continue
                        if others[k] % a == r:
                            return False
        return True
    
    def can_add(x, s, s_set):
        """Check if x can be added to s."""
        # Two checks:
        # 1. For each a in s, does x create a bad triple with a?
        #    i.e., are there i,j in s\{x} such that (i+j+x) % a == 0?
        #    Equivalently, for each a in s, is there a pair (i,j) in s such that (i+j+x) % a == 0?
        #    Note: x is not in s, so all elements in s are "other" for x.
        #    Wait, the condition is: x must not divide the sum of any 3 distinct OTHER elements.
        #    So for x, we check: are there 3 distinct elements a,b,c in s such that (a+b+c) % x == 0?
        # 2. For each a in s, adding x might make a bad: are there i,j in s\{a} such that (i+j+x) % a == 0?
        #    Here x is one of the "other" elements for a.
        
        # Check 1: x divides sum of 3 from s
        lst_s = sorted(s_set)
        n_s = len(lst_s)
        if n_s >= 3:
            for i in range(n_s):
                for j in range(i + 1, n_s):
                    s2 = lst_s[i] + lst_s[j]
                    r = (-s2) % x
                    for k in range(j + 1, n_s):
                        if lst_s[k] % x == r:
                            return False
        
        # Check 2: For each a in s, does x create a bad triple for a?
        for a in s_set:
            # Need: exist i,j in s\{a} such that (i+j+x) % a == 0
            # i.e., (i+j) % a == (-x) % a
            r = (-x) % a
            # Check if there's a pair (i,j) in s\{a} with (i+j) % a == r
            others = [e for e in lst_s if e != a]
            n_others = len(others)
            if n_others < 2:
                continue
            for i in range(n_others):
                for j in range(i + 1, n_others):
                    if (others[i] + others[j]) % a == r:
                        return False
        
        return True
    
    def greedy_from_high():
        """Greedy: add elements from high to low if valid."""
        selected = set()
        for x in range(N, 0, -1):
            if can_add(x, selected, selected):
                selected.add(x)
        return selected
    
    def greedy_from_low():
        """Greedy: add elements from low to high if valid."""
        selected = set()
        for x in range(1, N + 1):
            if can_add(x, selected, selected):
                selected.add(x)
        return selected
    
    def greedy_random(rng):
        """Greedy: random order."""
        order = ALL[:]
        rng.shuffle(order)
        selected = set()
        for x in order:
            if can_add(x, selected, selected):
                selected.add(x)
        return selected
    
    def local_search(s, rng, max_iters=200):
        """Local search with add and swap moves."""
        s = set(s)
        checkers_cache = {}
        
        def get_checkers(s):
            key = frozenset(s)
            if key not in checkers_cache:
                checkers_cache[key] = (set(), {})
            return checkers_cache[key]
        
        for _ in range(max_iters):
            if time_left() < 2:
                break
            
            improved = False
            moved = False
            
            # Try additions
            candidates = [x for x in ALL if x not in s]
            rng.shuffle(candidates)
            for x in candidates[:20]:
                if can_add(x, s, s):
                    s.add(x)
                    improved = True
                    moved = True
                    break
            
            if improved:
                continue
            
            # Try swaps
            elems = list(s)
            rng.shuffle(elems)
            for rem in elems[:5]:
                candidates = [x for x in ALL if x not in s]
                rng.shuffle(candidates)
                for add in candidates[:20]:
                    new_s = (s - {rem}) | {add}
                    if is_valid(new_s):
                        s = new_s
                        improved = True
                        moved = True
                        break
                if improved:
                    break
            
            if not moved:
                break
        
        return s
    
    # Phase 1: Generate diverse starting solutions
    rng = random.Random(42)
    
    starts = []
    starts.append(greedy_from_high())
    starts.append(greedy_from_low())
    for _ in range(5):
        starts.append(greedy_random(rng))
    
    best = max(starts, key=len)
    best_size = len(best)
    
    print(f"Initial best size: {best_size}", flush=True)
    
    # Phase 2: Local search from each start
    for i, start_s in enumerate(starts):
        if time_left() < 5:
            break
        
        result = local_search(start_s, rng, max_iters=100)
        if len(result) > best_size:
            best_size = len(result)
            best = result
            print(f"Improved to {best_size} at iteration {i}", flush=True)
    
    # Phase 3: More aggressive search with restarts
    for iteration in range(200):
        if time_left() < 10:
            break
        
        # Generate a new starting point
        if iteration % 3 == 0:
            # Perturb the best solution
            s = set(best)
            n_remove = rng.randint(2, 5)
            to_remove = rng.sample(list(s), min(n_remove, len(s)))
            s -= set(to_remove)
            start_s = s
        elif iteration % 3 == 1:
            start_s = greedy_from_high()
        else:
            start_s = greedy_random(rng)
        
        result = local_search(start_s, rng, max_iters=80)
        if len(result) > best_size:
            best_size = len(result)
            best = result
            print(f"New best: {best_size} at iteration {iteration}", flush=True)
    
    # Final check
    if not is_valid(best):
        # Fallback
        best = greedy_from_high()
    
    return list(best)

# EVOLVE-BLOCK-END
