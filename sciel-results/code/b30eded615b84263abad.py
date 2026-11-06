import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin (90s limit)
    
    N = 180
    
    # Precompute all forbidden triples (a, b, c) with a < b < c and 1/a = 1/b + 1/c
    # This means b*c = a*(b+c), i.e., (b-a)*(c-a) = a^2
    forbidden = set()
    for a in range(1, N + 1):
        a2 = a * a
        # Find all factor pairs (d, e) with d*e = a^2, d < e
        for d in range(1, int(math.isqrt(a2)) + 1):
            if a2 % d != 0:
                continue
            e = a2 // d
            if d >= e:
                continue
            b = a + d
            c = a + e
            if b <= N and c <= N and b < c:
                forbidden.add((a, b, c))
    
    # For each element, store the set of elements it conflicts with in some forbidden triple
    # Actually, we need: for each element x, which other elements y,z form a forbidden triple with x
    # Let's build a conflict structure: for each element, list of (other1, other2) pairs
    # that form a forbidden triple containing this element
    
    # Better approach: build a 3-uniform hypergraph where each forbidden triple is a hyperedge
    # We want a maximum independent set in this hypergraph
    
    # For branch and bound, we need efficient conflict checking
    # Let's precompute for each pair (x, y), the set of z such that {x,y,z} is forbidden
    # Actually, let's just store the forbidden triples and check membership
    
    # Convert to a more efficient structure
    # For each element, store which triples it participates in
    elem_triples = {i: [] for i in range(1, N + 1)}
    for (a, b, c) in forbidden:
        elem_triples[a].append((b, c))
        elem_triples[b].append((a, c))
        elem_triples[c].append((a, b))
    
    # Sort elements by degree (number of triples they participate in) for branching
    # Elements with more constraints should be branched on first
    
    # For a greedy lower bound, let's try a simple greedy approach
    def greedy_solution():
        remaining = set(range(1, N + 1))
        solution = set()
        # Process elements in order of increasing degree
        for x in sorted(range(1, N + 1), key=lambda x: len(elem_triples[x])):
            if x not in remaining:
                continue
            # Check if adding x creates a forbidden triple with elements already in solution
            can_add = True
            for (y, z) in elem_triples[x]:
                if y in solution and z in solution:
                    can_add = False
                    break
            if can_add:
                solution.add(x)
                remaining.discard(x)
                # Remove elements that would form a forbidden triple with x and any future element
                # Actually, we just need to track which elements are "blocked"
        return solution
    
    initial_solution = greedy_solution()
    best_solution = set(initial_solution)
    best_size = len(best_solution)
    
    # Now do branch and bound
    # We'll use a recursive approach with pruning
    # State: current set of included elements, remaining candidates
    # We process elements in a fixed order
    
    # Let's use a different approach: since N=180 is small, we can use a smarter method
    # The key insight: 1/a = 1/b + 1/c means a is the "harmonic mean" related value
    # For large a, there are few b,c that work. For small a, there are more.
    
    # Let's try a different strategy: start with all elements and remove conflicts
    # Or use local search / simulated annealing
    
    # Given time constraints, let's try a more sophisticated greedy with multiple passes
    
    def greedy_with_removal():
        # Start with all elements
        included = set(range(1, N + 1))
        # Repeatedly find and remove conflicts
        changed = True
        while changed:
            changed = False
            for (a, b, c) in forbidden:
                if a in included and b in included and c in included:
                    # Remove one of them - remove the one with highest degree
                    degrees = [len(elem_triples[a]), len(elem_triples[b]), len(elem_triples[c])]
                    max_idx = degrees.index(max(degrees))
                    to_remove = [a, b, c][max_idx]
                    included.discard(to_remove)
                    changed = True
        return included
    
    # Try multiple random orderings for the greedy
    import random
    random.seed(42)
    
    best_greedy = set()
    best_greedy_size = 0
    
    for trial in range(50):
        if time.time() > deadline - 5:
            break
        
        # Randomized greedy
        elements = list(range(1, N + 1))
        random.shuffle(elements)
        # Sort by degree as tiebreaker
        elements.sort(key=lambda x: (len(elem_triples[x]), random.random()))
        
        solution = set()
        for x in elements:
            can_add = True
            for (y, z) in elem_triples[x]:
                if y in solution and z in solution:
                    can_add = False
                    break
            if can_add:
                solution.add(x)
        
        if len(solution) > best_greedy_size:
            best_greedy_size = len(solution)
            best_greedy = set(solution)
    
    if best_greedy_size > best_size:
        best_solution = best_greedy
        best_size = best_greedy_size
    
    # Now try local search: swap elements to improve
    def is_valid_solution(sol):
        sol_set = set(sol)
        for (a, b, c) in forbidden:
            if a in sol_set and b in sol_set and c in sol_set:
                return False
        return True
    
    # Local search: try to add elements that were removed, or swap
    def local_search(start_sol, max_iter=1000):
        sol = set(start_sol)
        improved = True
        iterations = 0
        while improved and iterations < max_iter:
            improved = False
            iterations += 1
            if time.time() > deadline - 3:
                break
            
            # Try adding each excluded element
            excluded = [x for x in range(1, N + 1) if x not in sol]
            random.shuffle(excluded)
            
            for x in excluded:
                # Check if we can add x by removing some elements
                # Find all forbidden triples containing x and elements in sol
                conflicts = []
                for (y, z) in elem_triples[x]:
                    if y in sol and z in sol:
                        conflicts.append((y, z))
                
                if not conflicts:
                    sol.add(x)
                    improved = True
                    break
                else:
                    # Try removing one element from each conflict pair
                    # This is complex; just try removing one element at a time
                    for (y, z) in conflicts:
                        for to_remove in [y, z]:
                            if to_remove in sol:
                                new_sol = sol - {to_remove}
                                new_sol.add(x)
                                if len(new_sol) > len(sol):
                                    sol = new_sol
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
        return sol
    
    if time.time() < deadline - 5:
        improved_sol = local_search(best_solution, max_iter=200)
        if len(improved_sol) > best_size:
            best_solution = improved_sol
            best_size = len(improved_sol)
    
    # Verify the solution is valid
    sol_set = set(best_solution)
    for (a, b, c) in forbidden:
        if a in sol_set and b in sol_set and c in sol_set:
            # This shouldn't happen, but if it does, fix it
            # Remove the element with highest degree
            degrees = [len(elem_triples[a]), len(elem_triples[b]), len(elem_triples[c])]
            max_idx = degrees.index(max(degrees))
            to_remove = [a, b, c][max_idx]
            sol_set.discard(to_remove)
            best_solution = sol_set
            best_size = len(best_solution)
    
    # Ensure nonempty
    if not best_solution:
        best_solution = {180}
    
    return list(best_solution)
