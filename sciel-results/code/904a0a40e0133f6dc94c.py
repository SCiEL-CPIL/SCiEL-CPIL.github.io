"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import Counter
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    DEADLINE = 140  # Safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def find_violations(S):
        """Return list of (a, b, c) such that a+b=c, all in S."""
        S_set = set(S)
        viol = []
        n = len(S)
        for i in range(n):
            a = S[i]
            for j in range(i, n):
                b = S[j]
                c = a + b
                if c in S_set:
                    viol.append((a, b, c))
        return viol

    def can_add(S_set, x):
        """Check if x can be added to sum-free set S without violation."""
        for a in S_set:
            if a + x in S_set:
                return False
            if x - a in S_set:
                return False
        if 2 * x in S_set:
            return False
        return True

    def ilp_polish(S, V, time_limit=5.0):
        """Use MILP to find a larger sum-free superset of S within V.
        
        Formulation: maximize sum(y_i) subject to sum-free constraints.
        y_i = 1 if element i is in the solution.
        Constraint: for each triple (a,b,c) with a+b=c, y_a + y_b + y_c <= 2.
        Also: y_i >= 1 for i in S (must keep existing elements).
        """
        n = len(V)
        idx = {v: i for i, v in enumerate(V)}
        
        # Find all violation triples
        triples = []
        V_set = set(V)
        for i in range(n):
            a = V[i]
            for j in range(i, n):
                b = V[j]
                c = a + b
                if c in V_set:
                    ci = idx[c]
                    triples.append((i, j, ci))
        
        if not triples:
            return list(S)
        
        # Build constraint matrix
        m = len(triples)
        A = np.zeros((m, n))
        for k, (i, j, ci) in enumerate(triples):
            A[k, i] = 1
            A[k, j] = 1
            A[k, ci] = 1
        
        # Objective: maximize sum(y)
        c_obj = -np.ones(n)
        
        # Constraints: A @ y <= 2 for each triple
        # y >= 1 for elements in S
        # y >= 0, y <= 1, integer
        
        S_set = set(S)
        lower_bounds = np.zeros(n)
        upper_bounds = np.ones(n)
        
        for x in S:
            if x in idx:
                lower_bounds[idx[x]] = 1
        
        # LinearConstraint: A @ y <= 2
        constraint = LinearConstraint(A, ub=2)
        
        try:
            res = milp(
                c=c_obj,
                constraints=[constraint],
                integrality=np.ones(n),
                bounds=Bounds(lb=lower_bounds, ub=upper_bounds),
                options={"time_limit": time_limit, "mip_gap": 0.0}
            )
            
            if res.status == 0:  # Optimal
                y = res.x
                new_S = [V[i] for i in range(n) if y[i] > 0.5]
                return new_S
            elif res.status in (1, 2, 3):  # Feasible but not proven optimal
                y = res.x
                if y is not None:
                    new_S = [V[i] for i in range(n) if y[i] > 0.5]
                    # Verify sum-free
                    new_S_set = set(new_S)
                    valid = True
                    for x in new_S:
                        for a in new_S_set:
                            if a + x in new_S_set or (x - a in new_S_set and a != 0):
                                if a + x in new_S_set or (x - a in new_S_set):
                                    # Check properly
                                    pass
                        # Quick check
                        for a in new_S_set:
                            if a + x in new_S_set:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        return new_S
        except Exception:
            pass
        
        return list(S)

    def greedy_add(S, V, order=None):
        """Add elements from V to S in the given order (or sorted by magnitude descending)."""
        S_set = set(S)
        if order is None:
            order = sorted(V, key=lambda x: abs(x), reverse=True)
        for x in order:
            if x not in S_set:
                if can_add(S_set, x):
                    S_set.add(x)
        return list(S_set)

    def local_search(S, V, max_steps=200, rng=None):
        """Perform local search: try swaps and additions to improve size."""
        if rng is None:
            rng = random.Random()
        S_set = set(S)
        V_set = set(V)
        
        # First, do a greedy add pass
        S_set = set(greedy_add(list(S_set), V_set - S_set))
        
        improved = True
        steps = 0
        while improved and steps < max_steps:
            improved = False
            steps += 1
            
            # Try to add any element from V not in S
            candidates = [x for x in V_set if x not in S_set]
            for x in candidates:
                if can_add(S_set, x):
                    S_set.add(x)
                    improved = True
                    break
            
            # Try swaps: remove a, add b
            if not improved:
                # For each element in S, try removing it and adding something else
                S_list = list(S_set)
                rng.shuffle(S_list)
                for a in S_list[:30]:  # Limit to 30 elements to try
                    S_set.discard(a)
                    # Now try to add something
                    candidates = [x for x in V_set if x not in S_set]
                    rng.shuffle(candidates)
                    added = False
                    for x in candidates[:80]:  # Limit to 80 candidates
                        if can_add(S_set, x):
                            S_set.add(x)
                            added = True
                            improved = True
                            break
                    if not added:
                        S_set.add(a)  # Put it back
                    if improved:
                        break
        
        return list(S_set)

    def repair_and_expand(S, V, use_ilp=False, ilp_time=5.0):
        """Improve S by repairing violations and adding safe elements."""
        S_list = list(S)
        S_set = set(S_list)
        
        # Phase 1: Remove elements to fix violations
        while True:
            viol = find_violations(S_list)
            if not viol:
                break
            
            # Count how many times each element appears in violations
            counts = Counter()
            for a, b, c in viol:
                counts[a] += 1
                counts[b] += 1
                counts[c] += 1
            
            # Remove the element with the highest count
            if counts:
                worst = max(counts, key=counts.get)
                if worst in S_set:
                    S_list.remove(worst)
                    S_set.discard(worst)
                else:
                    break
            else:
                break

        # Phase 2: Add safe elements using greedy
        S_list = greedy_add(S_list, V)
        
        # Phase 3: Local search
        S_list = local_search(S_list, V, max_steps=150)
        
        # Phase 4: ILP polish if requested
        if use_ilp:
            S_list = ilp_polish(S_list, V, time_limit=ilp_time)
        
        return S_list

    def search_instance(elements, seed_offset=0, deadline=None):
        """Search for a large sum-free subset of elements."""
        rng = random.Random(seed_offset)
        best = []
        best_size = 0
        
        if deadline is None:
            deadline = time.time() + 40
        
        # Generate multiple initial candidates
        candidates = []
        
        # 1. Odds
        odd = [x for x in elements if x % 2 != 0]
        candidates.append(odd)
        
        # 2. Largest values
        sorted_desc = sorted(elements, reverse=True)
        for frac in [0.3, 0.4, 0.5]:
            k = int(len(elements) * frac)
            candidates.append(sorted_desc[:k])
        
        # 3. Random samples
        for _ in range(5):
            k = int(len(elements) * 0.5)
            candidates.append(rng.sample(elements, k))
            
        # 4. Positive numbers
        pos = [x for x in elements if x > 0]
        candidates.append(pos)
        
        # 5. Negative numbers
        neg = [x for x in elements if x < 0]
        candidates.append(neg)
        
        # 6. Large magnitude first
        sorted_mag = sorted(elements, key=lambda x: abs(x), reverse=True)
        for frac in [0.3, 0.4, 0.5]:
            k = int(len(elements) * frac)
            candidates.append(sorted_mag[:k])
        
        # Process each candidate
        for cand in candidates:
            if time.time() > deadline - 10:
                break
            S = repair_and_expand(cand, elements, use_ilp=False)
            if len(S) > best_size:
                best_size = len(S)
                best = S
        
        # Additional random restarts if time permits
        start_restarts = time.time()
        restart_count = 0
        while time.time() < deadline - 15 and restart_count < 40:
            k = int(len(elements) * rng.uniform(0.3, 0.8))
            cand = rng.sample(elements, k)
            S = repair_and_expand(cand, elements, use_ilp=False)
            if len(S) > best_size:
                best_size = len(S)
                best = S
            restart_count += 1
        
        # ILP polish on the best solution found
        if time.time() < deadline - 3 and best_size > 0:
            ilp_time = min(8.0, max(1.0, deadline - 3 - time.time()))
            polished = ilp_polish(best, elements, time_limit=ilp_time)
            if len(polished) > best_size:
                best = polished
                best_size = len(polished)
        
        return best

    # Generate instances
    instances = []
    for size, seed in [(60, 801), (120, 802), (240, 803)]:
        instances.append(instance(size, seed))
    
    results = []
    # Allocate time: ~40s per instance with safety margin
    per_instance_time = 40
    for i, elems in enumerate(instances):
        remaining = DEADLINE - (time.time() - start_time)
        if remaining < 5:
            # Fallback: odd numbers
            results.append([x for x in elems if x % 2 != 0])
            continue
        
        # Allocate time for this instance, leaving buffer for remaining
        instances_remaining = 3 - i
        time_budget = min(per_instance_time, remaining - 5 * instances_remaining)
        if time_budget < 5:
            time_budget = remaining - 5
        if time_budget < 3:
            results.append([x for x in elems if x % 2 != 0])
            continue
        
        instance_deadline = time.time() + time_budget
        result = search_instance(elems, seed_offset=801 + i, deadline=instance_deadline)
        results.append(result)
    
    return results
# EVOLVE-BLOCK-END
