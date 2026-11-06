import time
from math import gcd
from itertools import combinations
import random

def construct():
    t0 = time.time()
    DEADLINE = 85.0
    N = 128
    UNIVERSE = list(range(1, N + 1))
    
    # Precompute gcd matrix
    gcd_mat = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = gcd(i, j)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g
    
    # Precompute forbidden triples: for each pair (a,b), the set of c that form a forbidden triple
    # forbidden[a][b] = set of c such that gcd(a,b) == gcd(a,c) == gcd(b,c)
    # Use bitmasks for fast operations
    # Map numbers 1..128 to indices 0..127
    idx = {x: x - 1 for x in UNIVERSE}
    num = [i + 1 for i in range(N)]
    
    # forbidden_mask[i][j] is a bitmask of k such that (num[i], num[j], num[k]) is forbidden
    forbidden_mask = [[0] * N for _ in range(N)]
    
    for i in range(N):
        a = num[i]
        for j in range(i + 1, N):
            b = num[j]
            gij = gcd_mat[a][b]
            mask = 0
            for k in range(N):
                if k == i or k == j:
                    continue
                c = num[k]
                if gcd_mat[a][c] == gij and gcd_mat[b][c] == gij:
                    mask |= (1 << k)
            forbidden_mask[i][j] = mask
            forbidden_mask[j][i] = mask
    
    def is_valid(S):
        S_set = set(S)
        L = len(S)
        for i in range(L):
            a = S[i]
            for j in range(i + 1, L):
                b = S[j]
                gij = gcd_mat[a][b]
                for k in range(j + 1, L):
                    c = S[k]
                    if gcd_mat[a][c] == gij and gcd_mat[b][c] == gij:
                        return False
        return True
    
    def can_add(S, x):
        # Check if adding x to S creates a forbidden triple
        # For each pair (a, b) in S, check if x is in forbidden_mask[a][b]
        S_set = set(S)
        for i in range(len(S)):
            a = S[i]
            gax = gcd_mat[a][x]
            for j in range(i + 1, len(S)):
                b = S[j]
                if gcd_mat[a][b] == gax and gcd_mat[b][x] == gax:
                    return False
        return True
    
    # Build conflict graph: edge (i,j) if there exists k such that (i,j,k) is forbidden
    # This is a relaxation: if no edge, i and j can't be in any forbidden triple together
    conflict = [[False] * N for _ in range(N)]
    degree = [0] * N
    for i in range(N):
        for j in range(i + 1, N):
            if forbidden_mask[i][j] != 0:
                conflict[i][j] = True
                conflict[j][i] = True
                degree[i] += 1
                degree[j] += 1
    
    # Bitmask-based local search with 1-remove-2-add
    def local_search_bitmask(selected_indices, max_iters=10000):
        # selected_indices is a list of indices (0..127)
        selected_set = set(selected_indices)
        selected_mask = 0
        for i in selected_indices:
            selected_mask |= (1 << i)
        
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < DEADLINE - 5:
            improved = False
            iters += 1
            
            # Try 1-add
            for x in range(N):
                if x in selected_set:
                    continue
                # Check if x can be added
                ok = True
                for i in selected_indices:
                    gix = gcd_mat[num[i]][num[x]]
                    for j in selected_indices:
                        if j <= i:
                            continue
                        if gcd_mat[num[i]][num[j]] == gix and gcd_mat[num[j]][num[x]] == gix:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    selected_indices.append(x)
                    selected_set.add(x)
                    selected_mask |= (1 << x)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try 1-remove-2-add: remove a, add b, c
            for i in range(len(selected_indices)):
                a = selected_indices[i]
                new_selected = [x for x in selected_indices if x != a]
                new_selected_set = set(new_selected)
                
                # Try to add two elements
                for b in range(N):
                    if b in new_selected_set:
                        continue
                    # Check if b can be added to new_selected
                    ok_b = True
                    for x in new_selected:
                        gxb = gcd_mat[num[x]][num[b]]
                        for y in new_selected:
                            if y <= x:
                                continue
                            if gcd_mat[num[x]][num[y]] == gxb and gcd_mat[num[y]][num[b]] == gxb:
                                ok_b = False
                                break
                        if not ok_b:
                            break
                    
                    if not ok_b:
                        continue
                    
                    # Now try to add c
                    for c in range(N):
                        if c in new_selected_set or c == b:
                            continue
                        # Check if c can be added to new_selected + [b]
                        ok_c = True
                        temp_selected = new_selected + [b]
                        for x in temp_selected:
                            gxc = gcd_mat[num[x]][num[c]]
                            for y in temp_selected:
                                if y <= x:
                                    continue
                                if gcd_mat[num[x]][num[y]] == gxc and gcd_mat[num[y]][num[c]] == gxc:
                                    ok_c = False
                                    break
                            if not ok_c:
                                break
                        
                        if ok_c:
                            selected_indices = new_selected + [b, c]
                            selected_set = set(selected_indices)
                            improved = True
                            break
                    
                    if improved:
                        break
                
                if improved:
                    break
        
        return selected_indices
    
    # Greedy initial solution
    def greedy_initial(order):
        selected = []
        for x in order:
            if can_add(selected, x):
                selected.append(x)
        return selected
    
    # Multiple orderings
    orderings = [
        sorted(range(N), key=lambda x: -degree[x]),  # high conflict first
        sorted(range(N), key=lambda x: degree[x]),    # low conflict first
        list(range(N)),
        list(reversed(range(N))),
        sorted(range(N), key=lambda x: -num[x]),
        sorted(range(N), key=lambda x: num[x]),
    ]
    
    best_solution = []
    best_len = 0
    for order in orderings:
        sol = greedy_initial(order)
        if len(sol) > best_len:
            best_len = len(sol)
            best_solution = sol[:]
    
    if time.time() >= DEADLINE - 10:
        return [num[i] for i in best_solution] if best_solution else [1]
    
    # Local search with bitmask
    best_solution = local_search_bitmask(best_solution, max_iters=20000)
    best_len = len(best_solution)
    
    if time.time() >= DEADLINE - 5:
        return [num[i] for i in best_solution] if best_solution else [1]
    
    # Multi-start local search
    random.seed(42)
    for _ in range(10):
        if time.time() >= DEADLINE - 5:
            break
        # Random perturbation
        perturbed = best_solution[:]
        remove_count = random.randint(1, 3)
        for _ in range(remove_count):
            if perturbed:
                idx_remove = random.randint(0, len(perturbed) - 1)
                perturbed.pop(idx_remove)
        
        # Greedily add back
        perturbed_set = set(perturbed)
        for x in sorted(range(N), key=lambda x: -degree[x]):
            if x not in perturbed_set and can_add([num[i] for i in perturbed], num[x]):
                perturbed.append(x)
                perturbed_set.add(x)
        
        # Local search
        perturbed = local_search_bitmask(perturbed, max_iters=5000)
        if len(perturbed) > best_len:
            best_len = len(perturbed)
            best_solution = perturbed[:]
    
    # Final validity check
    final_solution = [num[i] for i in best_solution]
    if not is_valid(final_solution):
        # Repair: remove elements involved in forbidden triples
        changed = True
        while changed:
            changed = False
            S_set = set(final_solution)
            for i in range(len(final_solution)):
                a = final_solution[i]
                for j in range(i + 1, len(final_solution)):
                    b = final_solution[j]
                    gij = gcd_mat[a][b]
                    for k in range(j + 1, len(final_solution)):
                        c = final_solution[k]
                        if gcd_mat[a][c] == gij and gcd_mat[b][c] == gij:
                            # Remove one of them
                            candidates = [(a, degree[idx[a]]), (b, degree[idx[b]]), (c, degree[idx[c]])]
                            candidates.sort(key=lambda t: -t[1])
                            rem = candidates[0][0]
                            final_solution.remove(rem)
                            changed = True
                            break
                    if changed:
                        break
                if changed:
                    break
        if not final_solution:
            final_solution = [1]
    
    return final_solution

# Wrap to ensure nonempty
_orig_construct = construct
def construct():
    result = _orig_construct()
    if not result:
        return [1]
    return result
