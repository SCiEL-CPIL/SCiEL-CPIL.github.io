import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 145  # Safety margin for 150s limit

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(subset):
        s = set(subset)
        for a in subset:
            for b in subset:
                if a + b in s:
                    return False
        return True

    def solve_instance(values, budget):
        n = len(values)
        if n <= 2:
            return list(values)
        
        v_set = set(values)
        
        # Precompute conflicts for faster evaluation? 
        # For n=240, n^3 is 13M, which is too slow for SA steps.
        # We need a fast penalty function.
        
        # Penalty: number of triples (a,b,c) in S such that a+b=c.
        # We can maintain a count of violations.
        
        def get_penalty(S):
            # S is a set of values
            # Count distinct triples (a,b,c) with a+b=c
            # Note: a,b,c distinct? No, a=b allowed.
            # But usually sum-free definitions allow a=b.
            # The task says: "no a + b = c with a, b, c in it (a = b allowed)"
            # So (x, x, 2x) is a violation.
            # (x, y, x+y) is a violation.
            
            # Efficient check:
            # For each pair (a, b) in S, check if a+b in S.
            # To avoid double counting, we can just count the number of such pairs.
            # Actually, we want the number of violating triples.
            # A simple count of pairs (a,b) such that a+b in S is a good proxy.
            count = 0
            s_list = list(S)
            for i in range(len(s_list)):
                a = s_list[i]
                for j in range(len(s_list)):
                    b = s_list[j]
                    if a + b in v_set: # Must be in the original set to be a valid triple from the instance? 
                        # Wait, S is a subset of values. So a+b must be in S.
                        if a + b in S:
                            count += 1
            return count

        # Simulated Annealing
        # State: set S
        # Objective: Maximize |S| subject to Penalty(S) == 0
        
        # Initialize with a good heuristic: Odds
        S_init = set(x for x in values if x % 2 != 0)
        # Or maybe mix of large positives and large negatives?
        # Let's try a few starts.
        
        best_S = set()
        best_score = -1
        
        # Start candidates
        starts = []
        starts.append(set(x for x in values if x % 2 != 0))
        starts.append(set(x for x in values if x > 5000))
        starts.append(set(x for x in values if x < -5000))
        # Random starts
        for _ in range(3):
            random_start = set(random.sample(values, n // 2))
            starts.append(random_start)
        
        current_time = time.time()
        total_budget = budget
        time_per_start = total_budget / len(starts)
        
        for S0 in starts:
            if time.time() - current_time > time_per_start:
                continue
                
            S = set(S0)
            T = 100.0
            T_min = 0.01
            cooling_rate = 0.995
            
            # Initial penalty
            penalty = get_penalty(S)
            
            # If initial is not sum-free, we need to fix it or SA will struggle.
            # Let's quickly remove elements causing conflicts if penalty > 0
            # Actually, SA handles this by accepting moves that reduce penalty.
            # But if we start with high penalty, we might get stuck.
            # Let's do a quick greedy repair first.
            if penalty > 0:
                # Remove elements involved in most conflicts
                # Simple: just remove random elements until sum-free?
                # Or use the greedy local search from the parent to clean up?
                # Let's just do a quick pass:
                changed = True
                while changed and time.time() - current_time < time_per_start * 0.5:
                    changed = False
                    for x in list(S):
                        if get_penalty(S) > 0:
                            # Check if removing x reduces penalty
                            S_temp = S - {x}
                            p_new = get_penalty(S_temp)
                            if p_new < penalty:
                                S = S_temp
                                penalty = p_new
                                changed = True
                                break
            
            # Now run SA
            while T > T_min and time.time() - current_time < time_per_start:
                # Move: flip one element
                # Choose random element from values
                x = random.choice(values)
                
                if x in S:
                    S_new = S - {x}
                    delta_size = -1
                else:
                    S_new = S | {x}
                    delta_size = 1
                
                penalty_new = get_penalty(S_new)
                delta_penalty = penalty_new - penalty
                
                # Objective: Maximize Size - Lambda * Penalty
                # We want Penalty 0. So we treat Penalty as a huge cost.
                # Score = Size - 100 * Penalty
                delta_score = delta_size - 100 * delta_penalty
                
                if delta_score >= 0:
                    accept = True
                else:
                    accept = random.random() < math.exp(delta_score / T)
                
                if accept:
                    S = S_new
                    penalty = penalty_new
                    if penalty == 0 and len(S) > best_score:
                        best_score = len(S)
                        best_S = set(S)
                
                T *= cooling_rate
        
        if best_score == -1:
            # Fallback
            return [x for x in values if x % 2 != 0]
            
        return list(best_S)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        remaining = DEADLINE - (time.time() - start_time)
        if remaining < 5:
            vals = instance(size, seed)
            results.append([x for x in vals if x % 2])
            continue

        # Allocate budget proportionally to problem size
        # n=60 is small, n=240 is large.
        # Penalty calculation is O(n^2).
        # SA steps need to be fast.
        # Let's give more time to larger instances.
        budget = min(remaining - 2, 40 * (size / 240))
        budget = max(budget, 5)

        vals = instance(size, seed)
        res = solve_instance(vals, budget)

        if not is_sum_free(res):
            # If SA failed to find a valid set, fallback to odds
            res = [x for x in vals if x % 2 != 0]

        results.append(res)

    return results
