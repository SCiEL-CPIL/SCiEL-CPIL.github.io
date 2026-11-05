"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math

def construct():
    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def can_add(x, S):
        if 2 * x in S:
            return False
        for a in S:
            if x + a in S:
                return False
            if x - a in S:
                return False
        return True

    def solve_instance(elems, time_limit=30.0):
        import time
        start_time = time.time()
        n = len(elems)
        elem_set = set(elems)
        
        # Initial greedy solution
        S = set()
        # Try adding in random order? Or sorted?
        # Greedy by abs value descending often works well
        candidates = sorted(elems, key=abs, reverse=True)
        for x in candidates:
            if can_add(x, S):
                S.add(x)
        
        best_S = set(S)
        best_size = len(S)
        
        # Local Search with Random Restarts / Kicks
        # We will run several phases.
        # Phase 1: Hill Climbing (only accept improvements)
        # Phase 2: Simulated Annealing (accept worse moves)
        
        rng = random.Random(12345)
        
        # Simulated Annealing
        T = 1.0
        cooling = 0.99
        min_T = 0.01
        
        current_S = set(S)
        current_size = len(S)
        
        steps = 0
        max_steps = 10000 # Adjust based on n
        
        while T > min_T and (time.time() - start_time) < time_limit:
            steps += 1
            # Generate a move
            # 80% chance to try adding, 20% to try removing
            if rng.random() < 0.8:
                # Try adding a random element not in S
                not_in_S = [x for x in elems if x not in current_S]
                if not not_in_S:
                    continue
                x = rng.choice(not_in_S)
                if can_add(x, current_S):
                    new_S = current_S | {x}
                    new_size = current_size + 1
                    # Always accept improvement
                    current_S = new_S
                    current_size = new_size
                    if current_size > best_size:
                        best_size = current_size
                        best_S = set(current_S)
                else:
                    # Invalid add. Try to kick by removing a random element
                    if current_S:
                        y = rng.choice(list(current_S))
                        new_S = current_S - {y}
                        new_size = current_size - 1
                        # Accept with probability exp(1/T) ? 
                        # Delta = -1. Prob = exp(-Delta/T) = exp(1/T)
                        # Since T is small, this is close to 1 if T is not too small.
                        # Actually, standard SA: P = exp(-Delta E / T). 
                        # Here "Energy" is -size. Delta E = -1.
                        # P = exp(1/T).
                        # If T=1, P=2.7? No, prob must be <= 1.
                        # Let's use P = min(1, exp(1/T)).
                        # Or simpler: P = 1.0 if T > 1, else exp(1/T) is wrong direction?
                        # We want to accept worse moves (smaller size) to escape local optima.
                        # So P should be high when T is high.
                        # P = min(1.0, math.exp(1.0 / T))
                        if rng.random() < min(1.0, math.exp(1.0 / T)):
                            current_S = new_S
                            current_size = new_size
            else:
                # Try removing a random element
                if current_S:
                    y = rng.choice(list(current_S))
                    new_S = current_S - {y}
                    new_size = current_size - 1
                    if rng.random() < min(1.0, math.exp(1.0 / T)):
                        current_S = new_S
                        current_size = new_size
            
            T *= cooling
            
        return sorted(best_S)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        elems = instance(size, seed)
        # Allocate time based on size
        # 240 is bigger, give it more time?
        # Total 120s. Let's give 30s each.
        res = solve_instance(elems, time_limit=30.0)
        results.append(res)
        
    return results
# EVOLVE-BLOCK-END
