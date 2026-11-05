import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 180
    # 1. Generate all forbidden triples
    triples = []
    for a in range(1, N + 1):
        for d in range(1, a):
            if (a * a) % d == 0:
                b = a + d
                c = (a * b) // d
                if c <= N and c > b:
                    triples.append((a, b, c))
                    
    # If no triples, return all
    if not triples:
        return list(range(1, N + 1))
    
    # Precompute adjacency: for each element, which triples contain it
    elem_to_triples = defaultdict(list)
    for idx, t in enumerate(triples):
        for x in t:
            elem_to_triples[x].append(idx)
    
    T = len(triples)
    
    # Greedy Hitting Set with random restarts
    # We want to minimize |R|
    
    def run_greedy_random():
        uncovered = [True] * T
        R = set()
        
        while True:
            if time.time() > deadline:
                return R
            
            # Check if all covered
            any_uncovered = False
            for i in range(T):
                if uncovered[i]:
                    any_uncovered = True
                    break
            if not any_uncovered:
                break
            
            # Find element with max coverage among uncovered
            candidates = set()
            for i in range(T):
                if uncovered[i]:
                    for x in triples[i]:
                        candidates.add(x)
                        
            counts = {x: 0 for x in candidates}
            for i in range(T):
                if uncovered[i]:
                    for x in triples[i]:
                        counts[x] += 1
                        
            max_count = max(counts.values())
            best_elems = [x for x, c in counts.items() if c == max_count]
            
            # Random pick among best
            best_elem = random.choice(best_elems)
            
            R.add(best_elem)
            
            # Remove covered triples
            for idx in elem_to_triples[best_elem]:
                uncovered[idx] = False
                
        return R

    # Local search: try to remove redundant elements from R
    def local_search_remove(R):
        R = set(R)
        # For each element in R, check if removing it still covers all triples
        changed = True
        while changed:
            changed = False
            for x in list(R):
                # Check if x is needed
                needed = False
                for idx in elem_to_triples[x]:
                    # Check if this triple is covered by other elements in R
                    covered = False
                    for y in triples[idx]:
                        if y != x and y in R:
                            covered = True
                            break
                    if not covered:
                        needed = True
                        break
                if not needed:
                    R.discard(x)
                    changed = True
        return R
    
    # Branch and bound for maximum independent set (minimize hitting set)
    def branch_and_bound(R_initial):
        """Try to find a smaller hitting set using B&B."""
        best_R = set(R_initial)
        best_size = len(best_R)
        
        # Sort elements by degree (number of triples they cover)
        all_elems = list(range(1, N + 1))
        degrees = {x: len(elem_to_triples[x]) for x in all_elems}
        
        # Start with the greedy solution as upper bound
        # We'll try to prove we can do better or find a better solution
        
        # Use a different approach: iterative improvement
        # Try removing elements one by one and see if we can compensate
        for _ in range(50):  # iterations
            if time.time() > deadline:
                break
            improved = False
            for x in list(best_R):
                # Try removing x
                new_R = best_R - {x}
                # Check if new_R is still a valid hitting set
                valid = True
                for idx in elem_to_triples[x]:
                    covered = False
                    for y in triples[idx]:
                        if y in new_R:
                            covered = True
                            break
                    if not covered:
                        valid = False
                        break
                if valid:
                    best_R = new_R
                    best_size = len(best_R)
                    improved = True
                    break
            if not improved:
                break
        
        return best_R
    
    # Run greedy multiple times
    best_R = None
    best_R_size = N
    
    for i in range(200):
        if time.time() > deadline - 10:
            break
        R = run_greedy_random()
        R = local_search_remove(R)
        if len(R) < best_R_size:
            best_R_size = len(R)
            best_R = R
    
    # Try to improve with local search
    if best_R is not None:
        best_R = branch_and_bound(best_R)
        best_R_size = len(best_R)
    
    # Final local search pass
    if best_R is not None:
        best_R = local_search_remove(best_R)
        best_R_size = len(best_R)
    
    S = list(set(range(1, N + 1)) - best_R)
    return S
