import time
import random
import math
import numpy as np
from random import Random

def construct():
    deadline = time.time() + 145.0  # Safety margin

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        """Check if S is sum-free."""
        s = set(S)
        for a in s:
            for b in s:
                if a + b in s:
                    return False
        return True

    def solve_sa(elements, time_budget):
        """Solve using Simulated Annealing."""
        n = len(elements)
        if n == 0:
            return []
        
        val_to_idx = {v: i for i, v in enumerate(elements)}
        
        # Pre-compute conflicts for fast delta calculation
        # For each index i, store:
        # 1. List of (j, k) such that elements[j] + elements[k] == elements[i]
        # 2. List of (j, k) such that elements[i] + elements[j] == elements[k]
        # Note: We need to handle i=j cases carefully.
        
        # Let's define a function to get delta cost of adding/removing index i given current state S (bit array)
        # State: boolean array x of length n
        
        # Pre-compute:
        # sums_to[i] = list of (j, k) with j <= k such that v_j + v_k = v_i
        # sums_from[i] = list of (j, k) with j != i, k != i such that v_i + v_j = v_k
        # Actually, simpler:
        # For each i, we want to know which pairs (j, k) become valid/invalid if i is toggled.
        
        # Let's just use a set-based check for simplicity and speed given n <= 240.
        # O(n^2) check is 57600 ops. Very fast in Python if optimized or NumPy.
        # But we need delta for SA.
        
        # Let's precompute all conflict triples (i, j, k) such that v_i + v_j = v_k.
        # We store these as a list of triples of indices.
        # To speed up delta, for each index i, we store the list of triples involving i.
        
        triples = []
        triples_by_idx = [[] for _ in range(n)]
        
        for i in range(n):
            vi = elements[i]
            for j in range(i, n): # i <= j
                vj = elements[j]
                s = vi + vj
                if s in val_to_idx:
                    k = val_to_idx[s]
                    # We have a conflict triple (i, j, k)
                    # We need to add this to the neighbor lists of i, j, and k
                    triples.append((i, j, k))
                    triples_by_idx[i].append((j, k))
                    if j != i:
                        triples_by_idx[j].append((i, k))
                    triples_by_idx[k].append((i, j))
        
        # Initial solution: Greedy
        # Sort by absolute value descending? Or by number of conflicts?
        # Let's try sorting by number of conflicts ascending (least connected first)
        deg = [len(triples_by_idx[i]) for i in range(n)]
        order = sorted(range(n), key=lambda i: (deg[i], abs(elements[i])))
        
        x = np.zeros(n, dtype=bool)
        current_set = set()
        
        for i in order:
            # Check if i can be added
            ok = True
            for (j, k) in triples_by_idx[i]:
                # If j and k are both in set, we have a conflict
                # Note: triples_by_idx[i] contains pairs (j,k) such that {i,j,k} is a conflict triple.
                # If j==k, it means 2*elements[j] == elements[i] or similar.
                # We just check if both j and k are currently selected.
                if x[j] and x[k]:
                    ok = False
                    break
            if ok:
                x[i] = True
                current_set.add(i)
        
        best_x = x.copy()
        best_size = np.sum(x)
        
        # Simulated Annealing
        # Objective: Maximize size - Penalty * Violations
        # We want 0 violations.
        # Let's define Energy E = -size + Lambda * Violations
        # We want to minimize E.
        
        def get_violations(x):
            # Count number of triples (i,j,k) where x[i] & x[j] & x[k]
            # We can sum over all triples
            # For speed, we can precompute a matrix? No, n=240, triples ~ O(n^2)
            # Let's just count.
            count = 0
            for (i, j, k) in triples:
                if x[i] and x[j] and x[k]:
                    count += 1
            return count
        
        # Initial energy
        current_viol = get_violations(x)
        current_size = np.sum(x)
        current_energy = -current_size + 100.0 * current_viol
        
        best_energy = current_energy
        
        T = 10.0
        cooling = 0.995
        steps_per_sec = 100000 # Approximate target
        
        start_time = time.time()
        steps = 0
        
        while time.time() - start_time < time_budget:
            # Pick a random index to flip
            i = random.randint(0, n - 1)
            
            # Calculate delta energy
            # If we flip x[i], how does the violation count change?
            # A triple is violated if all 3 are 1.
            # Let V be the set of triples involving i.
            # If x[i] becomes 1:
            #   New violations = number of (j,k) in V where x[j] and x[k] are 1.
            # If x[i] becomes 0:
            #   Removed violations = number of (j,k) in V where x[j] and x[k] are 1.
            
            old_xi = x[i]
            new_viol = 0
            if old_xi:
                # Turning off: remove violations where i, j, k are all 1
                for (j, k) in triples_by_idx[i]:
                    if x[j] and x[k]:
                        new_viol -= 1 # This was a violation, now it's not
            else:
                # Turning on: add violations where i, j, k are all 1
                for (j, k) in triples_by_idx[i]:
                    if x[j] and x[k]:
                        new_viol += 1 # This becomes a violation
            
            delta_viol = new_viol
            delta_size = 1 if not old_xi else -1
            delta_energy = -delta_size + 100.0 * delta_viol
            
            # Metropolis acceptance
            if delta_energy < 0 or random.random() < math.exp(-delta_energy / T):
                x[i] = not x[i]
                current_viol += delta_viol
                current_size += delta_size
                current_energy += delta_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_x = x.copy()
                    best_size = current_size
            
            # Cooling
            T *= cooling
            if T < 0.01:
                T = 0.01
            
            steps += 1
            # Optional: restart if stuck? 
            # For now, just continue.
        
        # Final Polish: Ensure no violations
        # Greedily remove elements that are part of violations
        final_set = set(np.where(best_x)[0])
        changed = True
        while changed:
            changed = False
            for i in list(final_set):
                # Check if i is in any violation
                in_viol = False
                for (j, k) in triples_by_idx[i]:
                    if j in final_set and k in final_set:
                        in_viol = True
                        break
                if in_viol:
                    final_set.remove(i)
                    changed = True
        
        # Try to add back elements
        for i in range(n):
            if i not in final_set:
                ok = True
                for (j, k) in triples_by_idx[i]:
                    if j in final_set and k in final_set:
                        ok = False
                        break
                if ok:
                    final_set.add(i)
        
        return [elements[i] for i in final_set]

    def heuristic_fallback(elements):
        """Simple greedy fallback."""
        included_vals = set()
        for v in sorted(elements, key=abs, reverse=True):
            ok = True
            for a in included_vals:
                if (v - a) in included_vals or (v + a) in included_vals:
                    ok = False
                    break
                if 2 * v in included_vals:
                    ok = False
                    break
            if ok:
                included_vals.add(v)
        return list(included_vals)

    results = []
    instances = [(60, 801), (120, 802), (240, 803)]
    
    for size, seed in instances:
        elems = instance(size, seed)
        
        # Allocate time based on size
        # 60: 10s, 120: 20s, 240: 80s (Total 110s, leaving 5s for overhead)
        if size == 60:
            budget = 10.0
        elif size == 120:
            budget = 20.0
        else:
            budget = 80.0
            
        # Check if we have time
        if time.time() > deadline - budget - 2:
             # Not enough time, use fallback
             results.append(heuristic_fallback(elems))
             continue
             
        try:
            sol = solve_sa(elems, budget)
            if is_sum_free(sol):
                results.append(sol)
            else:
                results.append(heuristic_fallback(elems))
        except Exception:
            results.append(heuristic_fallback(elems))
            
        # Check time
        if time.time() > deadline - 5:
             break

    # Ensure we have 3 results
    while len(results) < 3:
        size, seed = instances[len(results)]
        elems = instance(size, seed)
        results.append(heuristic_fallback(elems))
        
    return results
