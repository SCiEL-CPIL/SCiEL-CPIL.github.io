import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    N = 180
    numbers = list(range(1, N + 1))
    numbers_set = set(numbers)
    
    # 1. Generate all forbidden triples (a, b, c) with distinct elements
    # 1/a = 1/b + 1/c => (b-a)(c-a) = a^2
    triples_set = set()
    for a in range(1, N + 1):
        a_sq = a * a
        for u in range(1, a + 1):
            if a_sq % u == 0:
                v = a_sq // u
                b = a + u
                c = a + v
                if b > N or c > N:
                    continue
                if u == v:
                    continue
                triples_set.add(tuple(sorted([a, b, c])))
                
    triples_list = list(triples_set)
    num_triples = len(triples_list)
    
    # Precompute: for each element, which triples contain it
    element_to_triples = {i: [] for i in range(1, N + 1)}
    for t in triples_list:
        for v in t:
            element_to_triples[v].append(t)
    
    def is_valid_addition(current_set, x_val):
        """Check if adding x_val to current_set creates any forbidden triple."""
        for t in element_to_triples[x_val]:
            others = [v for v in t if v != x_val]
            if others[0] in current_set and others[1] in current_set:
                return False
        return True
    
    def local_search_with_swaps(current_set, time_limit=10.0):
        """Local search using addition and swap moves to improve the set."""
        current_set = set(current_set)
        complement = numbers_set - current_set
        improved = True
        while improved and (time.time() - start_time) < deadline - 5:
            improved = False
            
            # Phase 1: Try to add elements
            for x_val in list(complement):
                if (time.time() - start_time) > deadline - 5:
                    break
                if is_valid_addition(current_set, x_val):
                    current_set.add(x_val)
                    complement.remove(x_val)
                    improved = True
            
            # Phase 2: Try swap moves - remove one element, add another
            for x_remove in list(current_set):
                if (time.time() - start_time) > deadline - 5:
                    break
                # Try removing x_remove and see if we can add more than one element
                trial_set = current_set - {x_remove}
                trial_complement = complement | {x_remove}
                added_count = 0
                for x_add in list(trial_complement):
                    if is_valid_addition(trial_set, x_add):
                        trial_set.add(x_add)
                        trial_complement.remove(x_add)
                        added_count += 1
                # If we can add more elements than we removed, it's an improvement
                if added_count > 1:
                    current_set = trial_set
                    complement = numbers_set - current_set
                    improved = True
                    break
        
        return current_set
    
    # 2. Formulate and solve MILP
    best_set = None
    
    try:
        c_obj = np.ones(N)
        
        # Build constraint matrix A
        A = np.zeros((num_triples, N), dtype=np.float64)
        for idx, t in enumerate(triples_list):
            a, b, c_val = t
            A[idx, a - 1] = 1.0
            A[idx, b - 1] = 1.0
            A[idx, c_val - 1] = 1.0
        
        integrality = np.ones(N)
        bounds = Bounds(lb=np.zeros(N), ub=np.ones(N))
        lin_cons = LinearConstraint(A, lb=-np.inf, ub=2.0)
        
        milp_result = milp(
            c=c_obj,
            constraints=[lin_cons],
            integrality=integrality,
            bounds=bounds
        )
        
        if milp_result.success and milp_result.x is not None:
            x = milp_result.x
            x_rounded = (np.round(x) > 0.5).astype(int)
            
            current_set = set()
            for i in range(N):
                if x_rounded[i] == 1:
                    current_set.add(i + 1)
            
            # Apply local search with swaps
            current_set = local_search_with_swaps(current_set)
            
            if len(current_set) > 0:
                best_set = current_set
        
    except Exception as e:
        pass
    
    # 3. If MILP failed or we have time, do greedy + local search
    if (time.time() - start_time) < deadline - 10:
        # Greedy construction: start empty, add elements in order of lowest degree
        degree = {i: len(element_to_triples[i]) for i in range(1, N + 1)}
        sorted_numbers = sorted(numbers, key=lambda x: degree[x])
        
        greedy_set = set()
        for x_val in sorted_numbers:
            if is_valid_addition(greedy_set, x_val):
                greedy_set.add(x_val)
        
        # Apply local search with swaps
        greedy_set = local_search_with_swaps(greedy_set)
        
        if best_set is None or len(greedy_set) > len(best_set):
            best_set = greedy_set
    
    # 4. Multiple random restarts with local search if time remains
    if (time.time() - start_time) < deadline - 15 and best_set is not None:
        random.seed(42)
        best_size = len(best_set)
        
        for restart in range(5):
            if (time.time() - start_time) > deadline - 10:
                break
            
            # Random initialization: include each element with probability 0.5
            random_set = set()
            for x_val in numbers:
                if random.random() < 0.5:
                    random_set.add(x_val)
            
            # Remove conflicts
            changed = True
            while changed:
                changed = False
                for t in triples_list:
                    if t[0] in random_set and t[1] in random_set and t[2] in random_set:
                        # Remove the element with highest degree among the three
                        max_deg_elem = max(t, key=lambda v: degree[v])
                        random_set.remove(max_deg_elem)
                        changed = True
            
            # Apply local search with swaps
            random_set = local_search_with_swaps(random_set)
            
            if len(random_set) > best_size:
                best_set = random_set
                best_size = len(best_set)
    
    # Fallback if nothing was found
    if best_set is None:
        best_set = {1}
    
    # Final verification - ensure no forbidden triples
    for t in triples_list:
        if t[0] in best_set and t[1] in best_set and t[2] in best_set:
            best_set.remove(t[0])
    
    if not best_set:
        best_set = {1}
        
    return list(best_set)
