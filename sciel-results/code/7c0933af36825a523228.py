import math
import time

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    def get_candidates():
        # 1 and primes < 512
        candidates = [1]
        for i in range(2, 512):
            if all(i % d != 0 for d in range(2, int(math.isqrt(i)) + 1)):
                candidates.append(i)
        return candidates

    def greedy(exclude_two=False):
        candidates = get_candidates()
        if exclude_two:
            candidates = [b for b in candidates if b != 2]
        
        chosen_b = []
        chosen_a = []
        
        # Sort by 1/b descending, i.e., b ascending
        for b in candidates:
            a = 512 - b
            valid = True
            for a_c in chosen_a:
                if math.gcd(a, a_c) != 1:
                    valid = False
                    break
            if valid:
                chosen_b.append(b)
                chosen_a.append(a)
                
        return chosen_a

    # Try both strategies
    sol1 = greedy(exclude_two=False)
    sol2 = greedy(exclude_two=True)
    
    def score(sol):
        return sum(1.0 / (512 - a) for a in sol)
    
    best = max([sol1, sol2], key=score)
    
    # Simple local search: try swapping one element with another
    # This is expensive, so only do it if time permits
    if time.time() < deadline - 10:
        current_score = score(best)
        best_set = set(best)
        all_candidates = get_candidates()
        # Try removing one and adding another
        # This is O(N^2) which is fine for N~500
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(len(best)):
                a_remove = best[i]
                temp_best = best[:i] + best[i+1:]
                temp_set = set(temp_best)
                for b_add in all_candidates:
                    if b_add in best_set: continue # b is in best, so a is in best
                    a_add = 512 - b_add
                    if a_add in temp_set: continue
                    valid = True
                    for a_c in temp_best:
                        if math.gcd(a_add, a_c) != 1:
                            valid = False
                            break
                    if valid:
                        new_score = score(temp_best) + 1.0 / (512 - a_add)
                        if new_score > current_score + 1e-9:
                            best = temp_best + [a_add]
                            best_set = set(best)
                            current_score = new_score
                            improved = True
                            break
                if improved:
                    break
        # Ensure return format is list
        if isinstance(best, set):
            best = list(best)
            
    return best
