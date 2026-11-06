"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
sys.setrecursionlimit(100000)
from math import gcd

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 80.0
    N = 256
    n = N
    
    # Build conflict graph
    conflict = [0] * n
    for i in range(n):
        a = i + 1
        for j in range(i + 1, n):
            b = j + 1
            g = gcd(a, b)
            m = a // g
            nn = b // g
            if g % (m + nn) == 0:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
    
    full_mask = (1 << n) - 1
    compat = [full_mask & ~(conflict[i] | (1 << i)) for i in range(n)]
    
    # Greedy initial solution: pick vertices one by one
    def greedy_solution():
        chosen = set()
        for i in range(n):
            # Check if i is compatible with all chosen
            ok = True
            for j in chosen:
                if not (compat[i] & (1 << j)):
                    ok = False
                    break
            if ok:
                chosen.add(i)
        return chosen
    
    greedy = greedy_solution()
    best_clique = list(greedy)
    best_size = len(greedy)
    
    # Try to improve with local search
    def is_valid(clique):
        for i in range(len(clique)):
            for j in range(i+1, len(clique)):
                if not (compat[clique[i]] & (1 << clique[j])):
                    return False
        return True
    
    def local_search(clique):
        nonlocal best_clique, best_size
        clique_set = set(clique)
        improved = True
        while improved:
            improved = False
            # Try to add a vertex
            for v in range(n):
                if v in clique_set:
                    continue
                ok = True
                for u in clique_set:
                    if not (compat[v] & (1 << u)):
                        ok = False
                        break
                if ok:
                    clique_set.add(v)
                    improved = True
                    if len(clique_set) > best_size:
                        best_size = len(clique_set)
                        best_clique = list(clique_set)
                    break
            # Try swap: remove one, add two
            for u in list(clique_set):
                for v in range(n):
                    if v in clique_set:
                        continue
                    new_set = (clique_set - {u}) | {v}
                    if len(new_set) < len(clique_set):
                        continue
                    ok = True
                    for a in new_set:
                        for b in new_set:
                            if a < b and not (compat[a] & (1 << b)):
                                ok = False
                                break
                        if not ok:
                            break
                    if ok and len(new_set) > len(clique_set):
                        clique_set = new_set
                        improved = True
                        if len(clique_set) > best_size:
                            best_size = len(clique_set)
                            best_clique = list(clique_set)
                        break
                if improved:
                    break
        return clique_set
    
    if time.time() < deadline - 30:
        local_search(list(greedy))
    
    # Try exact search if time permits
    if time.time() < deadline - 50:
        def color_sort(candidates):
            if not candidates:
                return [], []
            color_of = {}
            order = []
            remaining = list(candidates)
            c = 0
            while remaining:
                c += 1
                new_remaining = []
                for v in remaining:
                    ok = True
                    for u in order:
                        if color_of[u] == c and (compat[v] & (1 << u)):
                            ok = False
                            break
                    if ok:
                        color_of[v] = c
                        order.append(v)
                    else:
                        new_remaining.append(v)
                remaining = new_remaining
            cb = [color_of[v] for v in order]
            return order, cb
        
        def expand(candidates, color_bound, current_clique):
            nonlocal best_clique, best_size
            if time.time() > deadline:
                return
            if not candidates:
                if len(current_clique) > best_size:
                    best_size = len(current_clique)
                    best_clique = list(current_clique)
                return
            i = len(candidates) - 1
            while i >= 0:
                if time.time() > deadline:
                    return
                if len(current_clique) + color_bound[i] <= best_size:
                    return
                v = candidates[i]
                current_clique.append(v)
                new_cands = [candidates[j] for j in range(i) if compat[v] & (1 << candidates[j])]
                if new_cands:
                    new_order, new_cb = color_sort(new_cands)
                    expand(new_order, new_cb, current_clique)
                else:
                    if len(current_clique) > best_size:
                        best_size = len(current_clique)
                        best_clique = list(current_clique)
                current_clique.pop()
                i -= 1
        
        init_order, init_cb = color_sort(list(range(n)))
        expand(init_order, init_cb, [])
    
    if best_clique:
        result = sorted([v + 1 for v in best_clique])
    else:
        result = [1]
    
    return result
# EVOLVE-BLOCK-END
