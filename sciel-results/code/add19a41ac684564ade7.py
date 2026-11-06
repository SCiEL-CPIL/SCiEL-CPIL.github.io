import time
from math import gcd
from itertools import combinations

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
        for i in range(len(S)):
            a = S[i]
            gax = gcd_mat[a][x]
            for j in range(i + 1, len(S)):
                b = S[j]
                if gcd_mat[a][b] == gax and gcd_mat[b][x] == gax:
                    return False
        return True
    
    # Build conflict graph: edge (x,y) if there exists z != x,y such that
    # gcd(x,y) == gcd(x,z) == gcd(y,z)
    # This is a relaxation: if no edge, then x,y can't be in any forbidden triple together
    # But having an edge doesn't mean x,y,z is forbidden for all z, just that some z exists
    # Use this as a guide for greedy ordering and a pairwise conflict filter
    
    conflict_graph = [[False] * (N + 1) for _ in range(N + 1)]
    for x in range(1, N + 1):
        for y in range(x + 1, N + 1):
            gxy = gcd_mat[x][y]
            for z in range(1, N + 1):
                if z == x or z == y:
                    continue
                if gcd_mat[x][z] == gxy and gcd_mat[y][z] == gxy:
                    conflict_graph[x][y] = True
                    conflict_graph[y][x] = True
                    break
    
    # Degree in conflict graph
    degree = [0] * (N + 1)
    for x in range(1, N + 1):
        for y in range(x + 1, N + 1):
            if conflict_graph[x][y]:
                degree[x] += 1
                degree[y] += 1
    
    # Structural greedy: process by decreasing "constraint tightness"
    # Use a multi-pass greedy with different orderings, then repair
    
    def greedy_with_order(order):
        selected = []
        for x in order:
            if can_add(selected, x):
                selected.append(x)
        return selected
    
    # Multiple orderings
    orderings = [
        sorted(UNIVERSE, key=lambda x: -degree[x]),  # high conflict first
        sorted(UNIVERSE, key=lambda x: degree[x]),    # low conflict first
        list(UNIVERSE),
        list(reversed(UNIVERSE)),
        sorted(UNIVERSE, key=lambda x: -x),
        sorted(UNIVERSE, key=lambda x: x),
    ]
    
    best_solution = []
    best_len = 0
    for order in orderings:
        sol = greedy_with_order(order)
        if len(sol) > best_len:
            best_len = len(sol)
            best_solution = sol[:]
    
    if time.time() >= DEADLINE - 10:
        return best_solution if best_solution else [1]
    
    # Local search: 1-swap and 1-add to improve
    import random
    random.seed(42)
    
    def try_improve(sol, max_iters=5000):
        sol = sol[:]
        S_set = set(sol)
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < DEADLINE - 5:
            improved = False
            iters += 1
            # Try 1-add
            for x in UNIVERSE:
                if x not in S_set:
                    if can_add(sol, x):
                        sol.append(x)
                        S_set.add(x)
                        improved = True
                        break
            if improved:
                continue
            # Try 1-swap: remove a, add b
            for i, a in enumerate(sol):
                for b in UNIVERSE:
                    if b in S_set:
                        continue
                    new_sol = sol[:i] + sol[i+1:]
                    if can_add(new_sol, b):
                        sol = new_sol + [b]
                        S_set = set(sol)
                        improved = True
                        break
                if improved:
                    break
        return sol
    
    best_solution = try_improve(best_solution)
    best_len = len(best_solution)
    
    if time.time() >= DEADLINE - 5:
        return best_solution if best_solution else [1]
    
    # Multi-start local search
    for _ in range(5):
        if time.time() >= DEADLINE - 5:
            break
        # Random perturbation
        perturbed = best_solution[:]
        # Remove 1-3 random elements
        remove_count = random.randint(1, 3)
        for _ in range(remove_count):
            if perturbed:
                idx = random.randint(0, len(perturbed) - 1)
                perturbed.pop(idx)
        # Greedily add back
        perturbed_set = set(perturbed)
        for x in sorted(UNIVERSE, key=lambda x: -degree[x]):
            if x not in perturbed_set and can_add(perturbed, x):
                perturbed.append(x)
                perturbed_set.add(x)
        # Local search
        perturbed = try_improve(perturbed, max_iters=2000)
        if len(perturbed) > best_len:
            best_len = len(perturbed)
            best_solution = perturbed[:]
    
    # Final validity check
    if not is_valid(best_solution):
        # Repair: remove elements involved in forbidden triples
        changed = True
        while changed:
            changed = False
            S_set = set(best_solution)
            for i in range(len(best_solution)):
                a = best_solution[i]
                for j in range(i + 1, len(best_solution)):
                    b = best_solution[j]
                    gij = gcd_mat[a][b]
                    for k in range(j + 1, len(best_solution)):
                        c = best_solution[k]
                        if gcd_mat[a][c] == gij and gcd_mat[b][c] == gij:
                            # Remove one of them (the one with highest degree)
                            candidates = [(a, degree[a]), (b, degree[b]), (c, degree[c])]
                            candidates.sort(key=lambda t: -t[1])
                            rem = candidates[0][0]
                            best_solution.remove(rem)
                            changed = True
                            break
                    if changed:
                        break
                if changed:
                    break
        if not best_solution:
            best_solution = [1]
    
    return best_solution


# Wrap to ensure nonempty
_orig_construct = construct
def construct():
    result = _orig_construct()
    if not result:
        return [1]
    return result
