import time
import random

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def solve_one(values):
        n = len(values)
        if n == 0:
            return []
        
        best_subset = []
        best_size = 0
        
        # Heuristic 1: Odd numbers
        odd_subset = [x for x in values if x % 2 != 0]
        if len(odd_subset) > best_size:
            best_size = len(odd_subset)
            best_subset = odd_subset
            
        # Heuristic 2: Greedy with difference set tracking
        # We try multiple random orderings and specific orderings
        
        def greedy_solve(order):
            S = []
            S_set = set()
            F = set() # Set of differences s_i - s_j
            for x in order:
                # Check if x can be added
                # Violation if x = s_i - s_j for some s_i, s_j in S
                # i.e., x in F
                # Or if 2x in S
                if x in F or 2*x in S_set:
                    continue
                
                # Add x
                S.append(x)
                S_set.add(x)
                
                # Update F with new differences involving x
                # New pairs: (x, s) and (s, x) for all s in S (excluding x itself? No, x-x=0, but 0 not in S)
                # Differences: x - s and s - x
                for s in S[:-1]: # s is in S before x was added
                    F.add(x - s)
                    F.add(s - x)
                    
            return S

        # Order 1: Descending absolute value
        order1 = sorted(values, key=lambda x: abs(x), reverse=True)
        res1 = greedy_solve(order1)
        if len(res1) > best_size:
            best_size = len(res1)
            best_subset = res1
            
        # Order 2: Ascending absolute value
        order2 = sorted(values, key=lambda x: abs(x))
        res2 = greedy_solve(order2)
        if len(res2) > best_size:
            best_size = len(res2)
            best_subset = res2
            
        # Order 3: Random permutations
        # Estimate time per greedy run. N=240. Inner loop is O(N). Total O(N^2) ~ 57k ops.
        # Python can do ~10M ops/sec. So ~0.005s per run.
        # We can afford ~1000 runs in 5s.
        
        rng_local = random.Random(42)
        num_restarts = 200
        for i in range(num_restarts):
            if time.time() > deadline - 5.0:
                break
            order = values[:]
            rng_local.shuffle(order)
            res = greedy_solve(order)
            if len(res) > best_size:
                best_size = len(res)
                best_subset = res
                
        # Heuristic 3: Local Search (Simple Swap/Add)
        # Start with best_subset found so far
        current = set(best_subset)
        current_list = list(current)
        
        # Try to add elements from values not in current
        remaining = [x for x in values if x not in current]
        
        # Sort remaining by some metric? Maybe by "conflict count"
        # For a greedy add, we already did that.
        # Let's try a simple hill climbing: try to swap elements?
        # Or just re-run greedy with a different initialization?
        
        # Another powerful heuristic: Max Cut on the conflict graph?
        # Graph nodes are values. Edge (u, v) if u+v is in values? No, that's not right.
        # The condition is: no a,b,c in S such that a+b=c.
        # This is not a standard graph coloring.
        
        # Let's stick to the greedy with random restarts, it's robust.
        
        return best_subset

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > deadline:
            # Fallback to odd numbers if timeout
            vals = instance(size, seed)
            results.append([x for x in vals if x % 2 != 0])
            continue
            
        vals = instance(size, seed)
        res = solve_one(vals)
        results.append(res)
        
    return results
