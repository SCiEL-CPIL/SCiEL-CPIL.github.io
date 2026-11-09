import time
import random
import math
import numpy as np
from random import Random
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    deadline = time.time() + 140.0  # Safety margin

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        s = set(S)
        for a in s:
            for b in s:
                if a + b in s:
                    return False
        return True

    def solve_ilp(elements):
        n = len(elements)
        if n == 0:
            return []
        
        v_set = set(elements)
        
        # Build conflict graph: edge (i,j) if elements[i] + elements[j] in v_set or 2*elements[i] == elements[j] or 2*elements[j] == elements[i]
        # For sum-free: no a+b=c with a,b,c in S (a=b allowed)
        # This is equivalent to: if we select both i and j, then elements[i]+elements[j] must NOT be in the selected set
        # The independent set formulation: x_i + x_j <= 1 for conflicting pairs
        
        edges = []
        for i in range(n):
            vi = elements[i]
            for j in range(i + 1, n):
                vj = elements[j]
                s = vi + vj
                if s in v_set:
                    edges.append((i, j))
        
        # Also need to handle the case where 2*vi = vj (i.e., vi+vi=vj)
        # This is already covered if we consider i=j, but in our edge list we only have i<j
        # So we need to add edges for (i, k) where 2*elements[i] == elements[k]
        for i in range(n):
            vi = elements[i]
            double = 2 * vi
            if double in v_set:
                # Find index of double
                # Since elements is sorted, we can use binary search or just check
                # But we already have v_set, so let's find the index
                for j in range(n):
                    if elements[j] == double and j != i:
                        # Check if this edge is already in our list
                        pair = (min(i, j), max(i, j))
                        if pair not in [(min(e[0], e[1]), max(e[0], e[1])) for e in edges]:
                            edges.append(pair)
                        break
        
        # Remove duplicate edges
        unique_edges = set()
        for i, j in edges:
            pair = (min(i, j), max(i, j))
            unique_edges.add(pair)
        edges = list(unique_edges)
        
        m = len(edges)
        
        if m == 0:
            return elements[:]
        
        # Objective: maximize sum(x) -> minimize -sum(x)
        c = -np.ones(n)
        
        # Constraints: x_i + x_j <= 1 for each edge
        A = np.zeros((m, n))
        for k, (i, j) in enumerate(edges):
            A[k, i] = 1.0
            A[k, j] = 1.0
        
        cons = LinearConstraint(A, lb=-np.inf, ub=1.0)
        bounds = Bounds(lb=0, ub=1)
        integrality = np.ones(n)
        
        try:
            res = milp(c=c, constraints=cons, integrality=integrality, bounds=bounds)
            if res.success:
                selected_indices = [i for i in range(n) if res.x[i] > 0.5]
                return [elements[i] for i in selected_indices]
            else:
                return None
        except Exception:
            return None

    def bitset_local_search(elements, time_budget):
        """Steepest-ascent local search using bitsets for fast conflict detection."""
        n = len(elements)
        v_to_idx = {v: i for i, v in enumerate(elements)}
        v_set = set(elements)
        
        # Build conflict bitsets: for each element i, bitset of elements that conflict with i
        # Two elements i and j conflict if elements[i] + elements[j] is in v_set
        # OR if 2*elements[i] == elements[j] (but this is a self-conflict, handled separately)
        
        # Actually, for the independent set formulation:
        # i and j (i != j) conflict if elements[i] + elements[j] in v_set
        # i and i conflict if 2*elements[i] in v_set (but we can't select i twice, so this is about i and k where 2*elements[i]==elements[k])
        
        # Let's define: conflict[i] is a bitmask of all j != i such that elements[i] + elements[j] in v_set
        # And also j such that 2*elements[j] == elements[i] (i.e., elements[j] + elements[j] = elements[i])
        
        conflict = [0] * n
        for i in range(n):
            vi = elements[i]
            mask = 0
            for j in range(n):
                if i == j:
                    continue
                vj = elements[j]
                s = vi + vj
                if s in v_set:
                    mask |= (1 << j)
            # Also check if 2*elements[j] == vi for any j (meaning j+j=vi, so j conflicts with vi if both selected)
            # This is already covered: if j+j=vi, then for element j, vi is in v_set, so j conflicts with itself? No.
            # Wait: the condition is "no a+b=c with a,b,c in S". 
            # If we select j and vi, and j+j=vi, then a=j, b=j, c=vi, all in S. Violation.
            # So j and vi conflict if 2*elements[j] == elements[vi].
            # In our loop above, when i=vi's index, j=j's index, we check vi+v_j. That's not 2*v_j.
            # We need to also check: for each pair (i,j), does 2*elements[i] == elements[j] OR 2*elements[j] == elements[i]?
            # Let's redo: conflict between i and j (i != j) exists if:
            # 1. elements[i] + elements[j] in v_set, OR
            # 2. 2*elements[i] == elements[j], OR  
            # 3. 2*elements[j] == elements[i]
            
            # Let me rebuild properly
            pass
        
        # Rebuild conflict bitsets correctly
        conflict = [0] * n
        for i in range(n):
            vi = elements[i]
            for j in range(i + 1, n):
                vj = elements[j]
                s = vi + vj
                if s in v_set or 2 * vi == vj or 2 * vj == vi:
                    conflict[i] |= (1 << j)
                    conflict[j] |= (1 << i)
        
        # Start with all elements selected
        selected_mask = (1 << n) - 1
        
        # Local search: iteratively remove the element whose removal resolves the most conflicts
        # Or: iteratively try to add back removed elements
        
        # Better: start with a greedy solution, then improve
        # Greedy: sort by degree, add if no conflict with already selected
        
        def greedy_init():
            # Sort by conflict degree (number of conflicts)
            degrees = [bin(conflict[i]).count('1') for i in range(n)]
            order = sorted(range(n), key=lambda i: degrees[i])
            
            sel = 0
            for idx in order:
                # Check if idx conflicts with any already selected
                if (conflict[idx] & sel) == 0:
                    sel |= (1 << idx)
            return sel
        
        best_sel = greedy_init()
        best_size = bin(best_sel).count('1')
        
        start = time.time()
        
        # Local search: try removing one element and adding another
        while time.time() - start < time_budget:
            improved = False
            # Try each possible removal
            for rem in range(n):
                if not (best_sel & (1 << rem)):
                    continue
                # Remove rem
                new_sel = best_sel & ~(1 << rem)
                # Try to add back any removed element
                for add in range(n):
                    if best_sel & (1 << add):
                        continue
                    # Check if add can be added to new_sel
                    if (conflict[add] & new_sel) == 0:
                        candidate = new_sel | (1 << add)
                        if bin(candidate).count('1') > best_size:
                            best_sel = candidate
                            best_size = bin(candidate).count('1')
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                break
        
        # Extract result
        result = [elements[i] for i in range(n) if best_sel & (1 << i)]
        return result

    def heuristic_fallback(elements):
        """Simple greedy fallback."""
        n = len(elements)
        v_set = set(elements)
        
        # Build conflict graph
        conflict = [set() for _ in range(n)]
        for i in range(n):
            vi = elements[i]
            for j in range(i + 1, n):
                vj = elements[j]
                if vi + vj in v_set or 2 * vi == vj or 2 * vj == vi:
                    conflict[i].add(j)
                    conflict[j].add(i)
        
        # Greedy: pick elements with fewest conflicts first
        order = sorted(range(n), key=lambda i: len(conflict[i]))
        selected = set()
        for idx in order:
            if not (conflict[idx] & selected):
                selected.add(idx)
        
        return [elements[i] for i in selected]

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        elems = instance(size, seed)
        
        # Try ILP first
        ilp_result = solve_ilp(elems)
        
        if ilp_result is not None and is_sum_free(ilp_result):
            results.append(ilp_result)
        else:
            # Try bitset local search with remaining time
            remaining = max(1.0, deadline - time.time() - 5.0)
            bs_result = bitset_local_search(elems, remaining)
            if is_sum_free(bs_result):
                results.append(bs_result)
            else:
                results.append(heuristic_fallback(elems))
        
        if time.time() > deadline - 10:
            break

    while len(results) < 3:
        size, seed = ((60, 801), (120, 802), (240, 803))[len(results)]
        elems = instance(size, seed)
        results.append(heuristic_fallback(elems))
        
    return results
