import time
import random

def construct():
    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)
    
    def solve_instance(S, time_limit):
        n = len(S)
        S_set = set(S)
        
        # Precompute: for each element, what conflicts does it have?
        # conflict[a] = set of b such that a+b is in S_set or a-b is in S_set etc.
        # Actually, let's think differently.
        
        # For local search, I'll use a different approach:
        # 1. Start with a greedy solution
        # 2. Try to improve by removing one element and adding multiple others
        
        # Greedy initialization with multiple strategies
        def greedy_sorted(order):
            result = set()
            for x in order:
                # Check if x can be added
                ok = True
                for a in result:
                    if (x - a) in result or (a + x) in result:
                        ok = False
                        break
                if ok and (2*x) not in result:
                    result.add(x)
            return result
        
        # Multiple initializations
        inits = []
        # Sorted ascending
        inits.append(greedy_sorted(S))
        # Sorted descending
        inits.append(greedy_sorted(list(reversed(S))))
        # Odds only
        odds = [x for x in S if x % 2 != 0]
        inits.append(set(odds))
        # Suffix approach
        suffix = S
        for k in range(1, n+1):
            if 2 * S[n-k] > S[n-1]:
                suffix = S[n-k:]
            else:
                break
        inits.append(set(suffix))
        # Random orders
        for _ in range(5):
            shuffled = list(S)
            random.shuffle(shuffled)
            inits.append(greedy_sorted(shuffled))
        
        best = max(inits, key=len)
        
        # Local search: try removing one element and adding others
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            improved = False
            best_copy = set(best)
            
            # Try removing each element
            for x in list(best_copy):
                trial = set(best_copy)
                trial.discard(x)
                
                # Try to add elements from S \ trial
                added = 0
                for y in S:
                    if y in trial:
                        continue
                    # Check if y can be added to trial
                    ok = True
                    for a in trial:
                        if (y - a) in trial or (a + y) in trial:
                            ok = False
                            break
                    if ok and (2*y) not in trial:
                        trial.add(y)
                        added += 1
                
                if len(trial) > len(best):
                    best = trial
                    improved = True
                    break
            
            if not improved:
                # Try random restarts with perturbation
                # Remove 1-2 random elements and re-greedy
                for _ in range(10):
                    perturbed = set(best)
                    for _ in range(random.randint(1, 3)):
                        if perturbed:
                            perturbed.discard(random.choice(list(perturbed)))
                    # Re-greedy on remaining
                    remaining = [x for x in S if x not in perturbed]
                    random.shuffle(remaining)
                    for y in remaining:
                        ok = True
                        for a in perturbed:
                            if (y - a) in perturbed or (a + y) in perturbed:
                                ok = False
                                break
                        if ok and (2*y) not in perturbed:
                            perturbed.add(y)
                    if len(perturbed) > len(best):
                        best = perturbed
                        improved = True
                        break
            
            if not improved:
                # Try more aggressive: remove 2 elements, add many
                found = False
                elements = list(best)
                for i in range(len(elements)):
                    if time.time() - start_time > time_limit:
                        break
                    for j in range(i+1, len(elements)):
                        trial = set(best)
                        trial.discard(elements[i])
                        trial.discard(elements[j])
                        added = 0
                        for y in S:
                            if y in trial:
                                continue
                            ok = True
                            for a in trial:
                                if (y - a) in trial or (a + y) in trial:
                                    ok = False
                                    break
                            if ok and (2*y) not in trial:
                                trial.add(y)
                                added += 1
                        if len(trial) > len(best):
                            best = trial
                            found = True
                            break
                    if found:
                        break
                if not found:
                    break
        
        return list(best)
    
    instances = [
        instance(60, 801),
        instance(120, 802),
        instance(240, 803)
    ]
    
    results = []
    total_time = 140  # Safety margin
    start = time.time()
    
    for S in instances:
        elapsed = time.time() - start
        remaining_time = total_time - elapsed - 5
        
        if remaining_time <= 0:
            # Fallback
            results.append([x for x in S if x % 2 != 0])
            continue
        
        res = solve_instance(S, remaining_time)
        results.append(res)
    
    return results
