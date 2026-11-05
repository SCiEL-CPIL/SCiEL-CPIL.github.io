"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import random
    import time

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_valid_subset(S):
        """Check if S is sum-free."""
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def greedy_init(elements, random_seed=0):
        """Greedy initialization with randomization."""
        rng = random.Random(random_seed)
        S = set()
        remaining = elements[:]
        rng.shuffle(remaining)
        
        # Sort by absolute value descending as a tie-breaker or primary key?
        # Let's use a mix: shuffle, then sort by abs value descending within the shuffled list?
        # No, just shuffle and iterate.
        # Or sort by "degree" (number of conflicts in full set) ascending.
        
        # Calculate static degree
        val_set = set(elements)
        degrees = {}
        for x in elements:
            deg = 0
            for y in elements:
                if x + y in val_set:
                    deg += 1
                if 2 * x == y:
                    deg += 1
                if 2 * y == x:
                    deg += 1
            degrees[x] = deg
            
        # Sort by degree ascending, then random
        remaining.sort(key=lambda x: (degrees[x], random.random()))
        
        for x in remaining:
            # Check if x can be added to S
            ok = True
            for y in S:
                if x + y in S or 2 * x == y or 2 * y == x:
                    ok = False
                    break
            if ok:
                S.add(x)
        return S

    def local_search(S, elements, max_iters=2000):
        """Local search to improve the set size."""
        S = set(S)
        elements_set = set(elements)
        current_size = len(S)
        no_improve_count = 0
        rng = random.Random(42)
        
        for _ in range(max_iters):
            improved = False
            
            # 1. Try to add an element
            candidates = [x for x in elements if x not in S]
            if candidates:
                # Try random candidates
                rng.shuffle(candidates)
                for x in candidates:
                    ok = True
                    for y in S:
                        if x + y in S or 2 * x == y or 2 * y == x:
                            ok = False
                            break
                    if ok:
                        S.add(x)
                        current_size += 1
                        improved = True
                        no_improve_count = 0
                        break # Restart loop with new S
            
            # 2. Try to swap
            if not improved:
                in_S = list(S)
                out_S = [x for x in elements if x not in S]
                if in_S and out_S:
                    # Try random swaps
                    for _ in range(10): # Limit swaps per iteration to avoid slowness
                        if not in_S or not out_S: break
                        x = rng.choice(in_S)
                        y = rng.choice(out_S)
                        
                        new_S = S - {x}
                        new_S.add(y)
                        
                        # Check validity
                        ok = True
                        for a in new_S:
                            for b in new_S:
                                if a + b in new_S:
                                    ok = False
                                    break
                            if not ok: break
                        
                        if ok:
                            S = new_S
                            improved = True
                            no_improve_count = 0
                            break

            if not improved:
                no_improve_count += 1
                if no_improve_count > 50:
                    break
        
        return S

    def solve_heuristic(elements):
        best_S = set()
        
        # Multiple random restarts for greedy init
        for i in range(20):
            S = greedy_init(elements, random_seed=i)
            if len(S) > len(best_S):
                best_S = S
        
        # Local search on the best greedy solution
        S = local_search(best_S, elements)
        if len(S) > len(best_S):
            best_S = S
            
        # Maybe run local search on a few other good greedy solutions
        for i in range(5):
            S = greedy_init(elements, random_seed=100 + i)
            if len(S) > len(best_S) - 2: # Close to best
                S = local_search(S, elements)
                if len(S) > len(best_S):
                    best_S = S

        return sorted(best_S)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        elems = instance(size, seed)
        res = solve_heuristic(elems)
        results.append(res)
        
    return results
# EVOLVE-BLOCK-END
