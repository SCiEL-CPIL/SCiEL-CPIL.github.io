"""Rewritten program using Max-SAT / simulated annealing approach."""
import random
import time

def construct():
    # We want a nonempty subset S of [1, 180] with no three distinct elements a, b, c
    # such that 1/a = 1/b + 1/c. Maximize |S|.
    
    # First, enumerate all forbidden triples (a, b, c) with a, b, c distinct in [1, 180]
    # such that 1/a = 1/b + 1/c.
    # Note: 1/a = 1/b + 1/c  <=>  c = a*b/(a+b-a*c) ... wait, let's derive properly.
    # 1/a = 1/b + 1/c => 1/a = (b+c)/(bc) => bc = a(b+c) => bc - ab - ac = 0 => bc - ab - ac + a^2 = a^2 => (b-a)(c-a) = a^2
    # So for each a, we need (b-a)(c-a) = a^2 with b != c, b != a, c != a, all in [1, 180].
    # Let u = b - a, v = c - a. Then u*v = a^2, u != 0, v != 0, u != v (since b != c means u != v... wait, b != c means b-a != c-a so u != v).
    # Also b = a + u in [1, 180], c = a + v in [1, 180].
    
    # Build set of forbidden triples as frozensets of {a, b, c}
    forbidden = set()
    for a in range(1, 181):
        a2 = a * a
        # Find all divisors u of a^2 such that u != 0
        # For each divisor u, v = a^2 // u, and we need u * v == a^2
        # b = a + u, c = a + v
        # We need 1 <= b <= 180, 1 <= c <= 180, b != a, c != a, b != c
        # u can be negative or positive
        # Let's iterate over all possible u values where a^2 is divisible by u
        # Actually, let's iterate over possible b and c directly for efficiency
        for b in range(1, 181):
            if b == a:
                continue
            # 1/a = 1/b + 1/c => 1/c = 1/a - 1/b = (b-a)/(ab) => c = ab/(b-a)
            # c must be integer, positive, in [1,180], c != a, c != b
            if b - a == 0:
                continue
            num = a * b
            den = b - a
            if den == 0:
                continue
            if num % den != 0:
                continue
            c = num // den
            if c < 1 or c > 180:
                continue
            if c == a or c == b:
                continue
            # We have a valid triple (a, b, c)
            # Store as frozenset to avoid duplicates
            triple = frozenset((a, b, c))
            forbidden.add(triple)
    
    # Now we have the set of all forbidden triples.
    # We want to find a maximum-size subset S of [1, 180] that contains no forbidden triple.
    # This is equivalent to finding a maximum independent set in a 3-uniform hypergraph.
    # 
    # Approach: Simulated Annealing on the boolean vector x[1..180] where x[i]=1 if i is in S.
    # Objective: maximize sum(x[i]) - penalty * (number of forbidden triples fully contained in S)
    # We'll use a penalty that is large enough to ensure feasibility in the end, but we can also
    # just maximize |S| subject to no forbidden triples.
    # 
    # Simpler: Use simulated annealing where the state is a set S. 
    # Moves: add a random element not in S, remove a random element from S, or swap.
    # Energy: -|S| + lambda * (number of violated triples)
    # We want to minimize energy, so we want large |S| and few violated triples.
    # At the end, we ensure no violated triples.
    
    # Let's precompute for each element, the list of forbidden triples it participates in.
    n = 180
    elements = list(range(1, n+1))
    
    # Map element to list of triple indices
    elem_to_triples = {i: [] for i in range(1, n+1)}
    triple_list = list(forbidden)
    for idx, tri in enumerate(triple_list):
        for e in tri:
            elem_to_triples[e].append(idx)
    
    # Number of triples
    num_triples = len(triple_list)
    
    # State: set of included elements
    # Start with a greedy heuristic: start with all elements, then remove elements that cause conflicts
    # Greedy: start with S = all elements, repeatedly remove the element that is in the most violated triples
    S = set(elements)
    violated = set(range(num_triples))
    
    # Compute initial violated triples
    def is_violated(tri, S):
        return all(e in S for e in tri)
    
    # Initial: all violated
    violated_count = {}
    for idx, tri in enumerate(triple_list):
        if is_violated(tri, S):
            violated_count[idx] = 1
        else:
            violated_count[idx] = 0
    
    total_violated = num_triples
    
    # Greedy removal
    changed = True
    while changed:
        changed = False
        # For each element in S, count how many violated triples it's in
        elem_violated_count = {e: 0 for e in S}
        for idx in range(num_triples):
            if violated_count[idx] == 1:
                tri = triple_list[idx]
                for e in tri:
                    if e in S:
                        elem_violated_count[e] += 1
        
        # Find element with max violated count
        max_count = 0
        max_elem = None
        for e in S:
            if elem_violated_count[e] > max_count:
                max_count = elem_violated_count[e]
                max_elem = e
        
        if max_count == 0:
            break
        
        # Remove max_elem
        S.discard(max_elem)
        changed = True
        
        # Update violated counts for triples containing max_elem
        for idx in elem_to_triples[max_elem]:
            tri = triple_list[idx]
            was_violated = violated_count[idx]
            if was_violated == 1:
                # Now check if still violated
                if all(e in S for e in tri):
                    violated_count[idx] = 1
                else:
                    violated_count[idx] = 0
                    total_violated -= 1
            # If it wasn't violated before, it can't become violated by removing an element
    
    # Now S has no violated triples. Let's see its size.
    base_size = len(S)
    
    # Now try to improve with simulated annealing: try to add elements back
    # State: S (set of elements)
    # Energy: -|S| * large_weight + violated_count * weight
    # We want to minimize: violated_count * 1000 - |S|
    
    # Let's do simulated annealing
    best_S = set(S)
    best_energy = -len(best_S)  # since no violated triples
    
    T = 10.0
    T_min = 0.01
    cooling = 0.999
    max_iters = 50000
    start_time = time.time()
    
    current_S = set(S)
    current_violated = set()  # set of violated triple indices
    
    # Recompute violated for current_S
    for idx, tri in enumerate(triple_list):
        if all(e in current_S for e in tri):
            current_violated.add(idx)
    
    def energy(S_set, viol_set):
        return 1000 * len(viol_set) - len(S_set)
    
    current_energy = energy(current_S, current_violated)
    
    for it in range(max_iters):
        if time.time() - start_time > 55:
            break
        
        # Choose a move: add a random element not in S, or remove a random element from S
        if random.random() < 0.6:
            # Add
            candidates = [e for e in elements if e not in current_S]
            if not candidates:
                continue
            e = random.choice(candidates)
            new_S = set(current_S)
            new_S.add(e)
            # Compute new violated
            new_violated = set(current_violated)
            for idx in elem_to_triples[e]:
                tri = triple_list[idx]
                if idx in new_violated:
                    continue
                if all(x in new_S for x in tri):
                    new_violated.add(idx)
            # Also, adding e might not remove any violated triples
            new_energy = energy(new_S, new_violated)
            delta_e = new_energy - current_energy
            if delta_e < 0 or random.random() < (2.718281828 ** (-delta_e / T)):
                current_S = new_S
                current_violated = new_violated
                current_energy = new_energy
                if len(current_violated) == 0 and -current_energy > -best_energy:
                    best_S = set(current_S)
                    best_energy = current_energy
        else:
            # Remove
            if not current_S:
                continue
            e = random.choice(list(current_S))
            new_S = set(current_S)
            new_S.discard(e)
            # Compute new violated: removing e can only un-violate triples containing e
            new_violated = set(current_violated)
            for idx in elem_to_triples[e]:
                if idx in new_violated:
                    tri = triple_list[idx]
                    if not all(x in new_S for x in tri):
                        new_violated.discard(idx)
            new_energy = energy(new_S, new_violated)
            delta_e = new_energy - current_energy
            if delta_e < 0 or random.random() < (2.718281828 ** (-delta_e / T)):
                current_S = new_S
                current_violated = new_violated
                current_energy = new_energy
                if len(current_violated) == 0 and -current_energy > -best_energy:
                    best_S = set(current_S)
                    best_energy = current_energy
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    # Return the best feasible solution found
    # Verify best_S is feasible
    for tri in triple_list:
        if all(e in best_S for e in tri):
            # Should not happen, but just in case
            break
    
    # If best_S is empty, fall back to something nonempty
    if not best_S:
        best_S = {180}
    
    return list(best_S)
