import time
import random
import sys

START_TIME = time.time()
DEADLINE = 85.0

def elapsed():
    return time.time() - START_TIME

def is_done():
    return elapsed() > DEADLINE

def validate(A):
    if not A:
        return True
    if len(set(A)) != len(A):
        return False
    for x in A:
        if not isinstance(x, int) or x < 1 or x > 256:
            return False
    n = len(A)
    S = [set() for _ in range(n + 1)]
    S[0].add(0)
    for i in range(n):
        x = A[i]
        for j in range(i, -1, -1):
            if not S[j]:
                continue
            for s in S[j]:
                S[j+1].add(s + x)
    for j in range(n + 1):
        for k in range(j + 1, n + 1):
            if S[j] & S[k]:
                return False
    return True

def build_sumsets(A):
    """Build cardinality-indexed subset sum sets for set A."""
    n = len(A)
    S = [set() for _ in range(n + 1)]
    S[0].add(0)
    for i in range(n):
        x = A[i]
        for j in range(i, -1, -1):
            if S[j]:
                for s in S[j]:
                    S[j+1].add(s + x)
    return S

def check_validity(S):
    """Check if cardinality-indexed sumsets have no cross-cardinality collisions."""
    n = len(S) - 1
    for j in range(n + 1):
        for k in range(j + 1, n + 1):
            if S[j] & S[k]:
                return False
    return True

def greedy_build(max_elements=256, time_budget=30.0):
    """Greedy construction trying to maximize set size."""
    start = time.time()
    chosen = []
    S = [set() for _ in range(61)]  # up to cardinality 60
    S[0].add(0)
    
    # Try adding elements in various orders
    candidates = list(range(1, 257))
    
    # Strategy: try to add elements that minimize future conflicts
    # Use a heuristic: prefer smaller elements first (they create more sum variety)
    # but also consider which elements are "safe" to add
    
    for attempt in range(100):
        if is_done() or (time.time() - start) > time_budget:
            break
        
        chosen = []
        S = [set() for _ in range(61)]
        S[0].add(0)
        used = set()
        
        # Shuffle candidates with bias toward smaller values
        order = list(range(1, 257))
        random.shuffle(order)
        # Sort by a heuristic: prefer values that are "spread out"
        # Actually, let's try multiple strategies
        
        for x in order:
            if len(chosen) >= 60:
                break
            if (time.time() - start) > time_budget:
                break
            
            # Check if x can be added
            current_k = len(chosen)
            # New sums at cardinality current_k+1: {s + x for s in S[current_k]}
            new_sums = set()
            for s in S[current_k]:
                new_sums.add(s + x)
            
            # Check for collisions with all existing cardinality sets
            valid = True
            for j in range(current_k + 1):
                if new_sums & S[j]:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Also need to check that adding x doesn't create collisions
            # when combined with other elements at different cardinalities
            # This is complex, so we'll do a full check after adding
            
            # Tentatively add x
            old_S = [set(s) for s in S]
            for j in range(current_k + 1):
                if S[j]:
                    for s in S[j]:
                        S[j+1].add(s + x)
            
            # Full validation
            if not check_validity(S):
                # Revert
                for j in range(len(S)):
                    S[j] = old_S[j]
                continue
            
            chosen.append(x)
            used.add(x)
        
        if len(chosen) > 0:
            return chosen
    
    return chosen

def local_search(initial, time_budget=45.0):
    """Local search: try to add elements or swap elements to increase size."""
    start = time.time()
    best = list(initial)
    best_size = len(best)
    best_S = build_sumsets(best)
    
    if not check_validity(best_S):
        return initial
    
    all_values = set(range(1, 257))
    current_set = set(best)
    
    while not is_done() and (time.time() - start) < time_budget:
        improved = False
        
        # Try adding elements
        for x in sorted(all_values - current_set, key=lambda v: random.random()):
            if len(best) >= 60:
                break
            if (time.time() - start) > time_budget:
                break
            
            new_set = list(best) + [x]
            new_S = build_sumsets(new_set)
            if check_validity(new_S):
                best = new_set
                best_size += 1
                best_S = new_S
                current_set.add(x)
                improved = True
        
        # Try swaps: remove one element, add two
        if not improved and len(best) < 60:
            for i in range(len(best)):
                if (time.time() - start) > time_budget:
                    break
                removed = best[i]
                remaining = [x for j, x in enumerate(best) if j != i]
                
                for x in sorted(all_values - current_set, key=lambda v: random.random()):
                    if len(remaining) + 1 >= 60:
                        break
                    new_set = remaining + [x]
                    new_S = build_sumsets(new_set)
                    if check_validity(new_S):
                        best = new_set
                        best_size = len(best)
                        best_S = new_S
                        current_set = set(best)
                        improved = True
                        break
                if improved:
                    break
        
        # Try double swaps: remove two, add two
        if not improved and len(best) < 60:
            for i in range(len(best)):
                for j in range(i + 1, len(best)):
                    if (time.time() - start) > time_budget:
                        break
                    remaining = [x for k, x in enumerate(best) if k != i and k != j]
                    
                    candidates = list(all_values - current_set)
                    random.shuffle(candidates)
                    for x in candidates:
                        for y in candidates:
                            if x == y:
                                continue
                            new_set = remaining + [x, y]
                            new_S = build_sumsets(new_set)
                            if check_validity(new_S):
                                best = new_set
                                best_size = len(best)
                                best_S = new_S
                                current_set = set(best)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if not improved:
            # Try random restarts with perturbation
            if len(best) > 0 and (time.time() - start) < time_budget - 5:
                perturbed = list(best)
                # Remove 1-2 random elements
                for _ in range(random.randint(1, 2)):
                    if perturbed:
                        idx = random.randint(0, len(perturbed) - 1)
                        perturbed.pop(idx)
                
                # Try to rebuild
                current_set = set(perturbed)
                for x in sorted(all_values - current_set, key=lambda v: random.random()):
                    if len(perturbed) >= 60:
                        break
                    if (time.time() - start) > time_budget:
                        break
                    new_set = perturbed + [x]
                    new_S = build_sumsets(new_set)
                    if check_validity(new_S):
                        perturbed = new_set
                        current_set.add(x)
                
                if len(perturbed) > best_size:
                    best = perturbed
                    best_size = len(best)
                    best_S = build_sumsets(best)
                    current_set = set(best)
                    improved = True
        
        if not improved:
            break
    
    return best

def construct():
    # Start with a good base
    base_solution = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    best = list(base_solution)
    
    # Try greedy with multiple random seeds
    time_left = DEADLINE - elapsed()
    if time_left > 10:
        greedy_time = min(time_left * 0.3, 15.0)
        greedy_result = greedy_build(time_budget=greedy_time)
        if len(greedy_result) > len(best):
            best = greedy_result
    
    # Local search
    time_left = DEADLINE - elapsed()
    if time_left > 10:
        ls_time = min(time_left * 0.6, 40.0)
        ls_result = local_search(best, time_budget=ls_time)
        if len(ls_result) > len(best):
            best = ls_result
    
    # Final validation
    if validate(best):
        return best
    
    return base_solution

if __name__ == "__main__":
    res = construct()
    print(f"Found set of size {len(res)}: {res}")
