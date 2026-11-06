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

    # Precompute forbidden triples
    # A triple (a, b, c) is forbidden if gcd(a,b) == gcd(a,c) == gcd(b,c)
    # We represent the hypergraph by mapping each pair (a, b) to the set of c's
    # that form a forbidden triple with a and b.
    # Actually, for branch and bound on independent set, we usually work with a graph.
    # Here we have a 3-uniform hypergraph.
    # A set S is independent if it contains no hyperedge.
    # This is equivalent to finding a maximum subset S such that for any a,b,c in S,
    # they do not form a forbidden triple.
    
    # Let's build the "conflict graph" for the upper bound.
    # Two vertices u, v are "conflicted" if there exists w such that {u, v, w} is a forbidden triple.
    # If u and v are not conflicted, they can never be in a forbidden triple together.
    # If u and v are conflicted, they *might* be in a forbidden triple, depending on w.
    
    # However, a simpler upper bound for the hypergraph independent set problem:
    # If we ignore the hyperstructure and just look at the pairwise conflict graph G_conflict,
    # any independent set in the hypergraph is an independent set in G_conflict?
    # No. An independent set in the hypergraph means NO triple is forbidden.
    # It does NOT mean no pair is "conflicted" in the pairwise sense.
    # Example: a, b are conflicted because of c. But if c is not in S, then {a, b} is fine.
    # So the pairwise conflict graph is NOT a supergraph in the sense that independent sets of H are independent sets of G.
    # In fact, {a, b} is allowed in H even if they are edges in G_conflict.
    
    # So we cannot directly use Max Independent Set on G_conflict as a valid solution.
    # But we can use the size of the Max Independent Set of G_conflict as an UPPER BOUND?
    # Let S be a valid solution (independent in H).
    # Does S have to be an independent set in G_conflict?
    # No. If a, b in S are "conflicted" by c, but c not in S, S is still valid.
    # So S is NOT necessarily an independent set in G_conflict.
    # Thus, MIS(G_conflict) is NOT an upper bound for MIS(H).
    # Actually, MIS(H) could be larger than MIS(G_conflict).
    
    # Let's rethink.
    # We want to maximize |S| such that no (a,b,c) in S is forbidden.
    # This is exactly the Maximum Independent Set in the 3-uniform hypergraph H.
    
    # Branch and Bound strategy:
    # State: (P, X) where P is the set of candidates to add, X is the set of elements excluded.
    # Current solution size k.
    # Upper bound: k + |P| (trivial).
    # Better upper bound:
    # If we pick a vertex v in P, any valid extension of the current solution that includes v
    # must not contain any pair (u, w) from P \ {v} such that (v, u, w) is a forbidden triple.
    # This is complex.
    
    # Alternative: Use a greedy coloring bound on the "conflict graph" derived from the hypergraph?
    # Or simply use a recursive search with strong pruning.
    
    # Let's define the forbidden triples explicitly for fast lookup.
    # forbidden[a][b] = set of c such that (a, b, c) is forbidden.
    forbidden = [[set() for _ in range(N + 1)] for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            gab = gcd_mat[a][b]
            for c in range(1, N + 1):
                if c == a or c == b:
                    continue
                if gcd_mat[a][c] == gab and gcd_mat[b][c] == gab:
                    forbidden[a][b].add(c)
                    forbidden[b][a].add(c)
    
    # For efficient branch and bound, we maintain:
    # S: current selected set
    # P: pool of available candidates
    # We need to check if adding x to S is valid.
    # x is valid if for all pairs (a, b) in S, x is not in forbidden[a][b].
    # Equivalently, for all a in S, x is not in forbidden[a][x]? No.
    # The condition is: no triple in S ∪ {x} is forbidden.
    # New triples involving x are (x, a, b) for a, b in S.
    # So x is valid iff for all a, b in S, x not in forbidden[a][b].
    
    def can_add(S, x):
        for i in range(len(S)):
            a = S[i]
            for j in range(i + 1, len(S)):
                b = S[j]
                if x in forbidden[a][b]:
                    return False
        return True

    # Branch and Bound
    # We will use a recursive function.
    # To speed up, we can precompute a "compatibility" mask or list.
    # Since N=128, recursion depth can be up to 128.
    # We need a good ordering of candidates in P.
    
    # Let's implement a simple branch and bound with a greedy upper bound.
    # Upper bound: |S| + greedy_independent_set_size(P)
    # Where greedy_independent_set_size(P) finds a large independent set in the hypergraph induced by P?
    # That's as hard as the original problem.
    
    # Simpler Upper Bound:
    # Consider the graph G_P where edges connect pairs that *cannot* coexist in any valid set?
    # No, pairs can coexist.
    
    # Let's try a different approach:
    # The problem is small enough (N=128) that a well-optimized branch and bound might work.
    # We need a tight bound.
    
    # Observation:
    # If we sort P by some heuristic, we can branch on the first element.
    # Branch 1: Include v. Remove from P all elements that conflict with v?
    # "Conflict" here means: if we include v, we cannot include any pair (u, w) such that (v, u, w) is forbidden.
    # This doesn't remove single elements from P, it removes pairs.
    # This makes the state space huge (sets of pairs).
    
    # Alternative formulation:
    # This is a 3-SAT like problem? Or a Max-CSP?
    
    # Let's look at the structure of forbidden triples.
    # gcd(a,b) = gcd(a,c) = gcd(b,c) = g.
    # This implies a=g*a', b=g*b', c=g*c' with gcd(a',b')=gcd(a',c')=gcd(b',c')=1?
    # No. gcd(a,b)=g means a=g*a', b=g*b', gcd(a',b')=1.
    # gcd(a,c)=g means c=g*c', gcd(a',c')=1.
    # gcd(b,c)=g means gcd(b',c')=1.
    # So a', b', c' are pairwise coprime.
    # So forbidden triples correspond to 3 pairwise coprime integers scaled by g.
    
    # This structure might help, but let's stick to the general B&B.
    
    # To make B&B work, we need a good upper bound.
    # Let's use the size of the maximum independent set in the *pairwise conflict graph* of P as a lower bound for the bound? No.
    
    # Let's try a heuristic:
    # 1. Greedy construction to get a good initial solution S0.
    # 2. B&B to prove optimality or find better.
    
    # Since we only need to maximize, and the time limit is 90s, we can run B&B with a time limit.
    
    # Let's implement the B&B carefully.
    # State: (S, P)
    # If P is empty, update best.
    # Upper bound: len(S) + len(P). If <= best, prune.
    # Choose v in P.
    # Branch Include v:
    #   New S = S ∪ {v}
    #   New P = P \ {v}
    #   We must ensure no forbidden triple in New S.
    #   If valid, recurse.
    #   Note: We don't remove elements from P based on v immediately, because v only conflicts with *pairs* in P.
    #   However, if we include v, we must ensure that for any future u, w in P, (v, u, w) is not forbidden.
    #   This constraint is on the *future* selections.
    
    # This is tricky. Standard B&B for hypergraphs is hard.
    
    # Let's try a different strategy:
    # Use the pairwise conflict graph G_conflict.
    # If S is a valid solution, is it an independent set in G_conflict?
    # No, as established.
    
    # However, if we define G_conflict differently:
    # Edge (u, v) if u and v *always* form a forbidden triple with any w? No.
    
    # Let's go back to the inspiration: "branch-and-bound maximum independent set solver on the 3-uniform hypergraph... using tight upper bound derived from the maximum clique size in the conflict graph".
    # Wait, Maximum Clique in conflict graph?
    # If u, v, w form a forbidden triple, they form a triangle in G_conflict.
    # A valid solution S contains NO triangle from H.
    # It may contain triangles in G_conflict that are not in H.
    # So S is a "triangle-free" subset with respect to H, but not necessarily G_conflict.
    
    # Maybe the bound is:
    # Any valid solution S is an independent set in the graph H' where edges are pairs that are in *some* forbidden triple?
    # No, that's G_conflict.
    
    # Let's assume the inspiration meant:
    # Use the size of the Maximum Independent Set of G_conflict as an upper bound?
    # If S is valid, S is NOT necessarily independent in G_conflict.
    # So MIS(G_conflict) is NOT an upper bound.
    # In fact, MIS(H) >= MIS(G_conflict).
    
    # So we can't use it as an upper bound.
    
    # What if we use the *Clique* number?
    # No.
    
    # Let's try a simpler, more robust approach:
    # 1. Generate a good initial solution using a greedy algorithm that accounts for future constraints (e.g., minimize degree in G_conflict).
    # 2. Use a local search with "pair swaps" and "triple swaps" to improve.
    # 3. Use a time limit.
    
    # The current program does this but scores 1.25. The inspiration scored 1.625 with B&B.
    # I will implement a B&B that branches on the *most constrained* element.
    
    # To make B&B efficient:
    # 1. Sort P by degree in G_conflict (descending).
    # 2. Branch on the first element v.
    # 3. Include v:
    #    - Check validity with current S.
    #    - If valid, new S = S ∪ {v}.
    #    - New P = P \ {v}.
    #    - Prune P: Remove any element u from P if (v, u, w) is forbidden for some w in S? No.
    #    - We must ensure that no future pair (u, w) from P completes a forbidden triple with v.
    #    - This means: For any u in P, if there exists w in P such that (v, u, w) is forbidden, we can't just remove u.
    #    - This constraint is global.
    
    # Alternative:
    # Branch on a *pair* (u, v) from P?
    # If we include both u and v, we must exclude all w such that (u, v, w) is forbidden.
    # This reduces P significantly.
    
    # Strategy:
    # 1. Greedy to get S0.
    # 2. B&B:
    #    - State: (S, P)
    #    - If P is empty, update best.
    #    - Upper bound: len(S) + len(P).
    #    - If upper bound <= best, prune.
    #    - Pick a pair (u, v) from P that has the maximum number of conflicts (i.e., max |forbidden[u][v] ∩ P|).
    #    - Branch 1: Include u and v.
    #      - Check if (u, v) is compatible with S.
    #      - If yes, New S = S ∪ {u, v}.
    #      - New P = P \ {u, v} \ (forbidden[u][v] ∩ P).
    #      - Recurse.
    #    - Branch 2: Exclude u.
    #      - New P = P \ {u}.
    #      - Recurse.
    
    # This branching on pairs might be effective because it removes many candidates.
    
    # Let's implement this.
    
    # Precompute degrees for ordering
    degree = [0] * (N + 1)
    for u in range(1, N + 1):
        for v in range(u + 1, N + 1):
            if forbidden[u][v]:
                degree[u] += 1
                degree[v] += 1
    
    # Initial greedy solution
    def greedy_initial():
        S = []
        P = set(UNIVERSE)
        while P:
            # Pick element with min degree in remaining P?
            # Or max?
            # Let's try min degree in G_conflict induced by P.
            best_v = None
            min_deg = float('inf')
            for v in P:
                deg = sum(1 for u in P if u != v and forbidden[u][v])
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
            if can_add(S, best_v):
                S.append(best_v)
            P.remove(best_v)
        return S
    
    S0 = greedy_initial()
    best_solution = S0[:]
    best_len = len(S0)
    
    # B&B
    import sys
    sys.setrecursionlimit(2000)
    
    def branch_and_bound(S, P):
        nonlocal best_solution, best_len
        if time.time() > DEADLINE:
            return
        
        P = set(P)
        if not P:
            if len(S) > best_len:
                best_len = len(S)
                best_solution = S[:]
            return
        
        # Upper bound
        if len(S) + len(P) <= best_len:
            return
        
        # Find most constrained pair in P
        # Pair (u, v) maximizes |forbidden[u][v] ∩ P|
        # If no pair has conflicts, we can just add all of P?
        # No, we still need to check triples within P.
        
        # Optimization: If P is small, just brute force or greedy fill.
        if len(P) < 10:
            # Try to add all elements in P that are compatible
            # Sort P by degree
            P_list = sorted(P, key=lambda x: -degree[x])
            temp_S = S[:]
            for v in P_list:
                if can_add(temp_S, v):
                    temp_S.append(v)
            if len(temp_S) > best_len:
                best_len = len(temp_S)
                best_solution = temp_S[:]
            return
        
        max_conflicts = -1
        best_pair = None
        # Sample pairs to find a good one? Or scan all?
        # N=128, P can be large. Scanning all pairs is O(|P|^2).
        # Let's scan a subset or the whole thing if P is small enough.
        P_list = list(P)
        if len(P_list) < 50:
            for i in range(len(P_list)):
                u = P_list[i]
                for j in range(i + 1, len(P_list)):
                    v = P_list[j]
                    conf = len(forbidden[u][v] & P)
                    if conf > max_conflicts:
                        max_conflicts = conf
                        best_pair = (u, v)
        else:
            # Heuristic: pick random pairs or highest degree vertices
            # Pick two highest degree vertices in P
            P_sorted = sorted(P_list, key=lambda x: -degree[x])
            if len(P_sorted) >= 2:
                best_pair = (P_sorted[0], P_sorted[1])
                max_conflicts = len(forbidden[best_pair[0]][best_pair[1]] & P)
        
        if not best_pair:
            # No conflicts? Then all pairs are "safe" in terms of immediate removal?
            # But triples might still exist.
            # Just add the most compatible element
            best_v = None
            min_conf = float('inf')
            for v in P:
                # Count how many pairs in P are forbidden with v?
                # Actually, just pick any
                pass
            # Fallback: greedy add
            temp_S = S[:]
            for v in P:
                if can_add(temp_S, v):
                    temp_S.append(v)
            if len(temp_S) > best_len:
                best_len = len(temp_S)
                best_solution = temp_S[:]
            return
        
        u, v = best_pair
        
        # Branch 1: Include u and v
        # Check if u, v are compatible with S
        if can_add(S, u) and can_add(S + [u], v):
            new_S = S + [u, v]
            # Remove u, v and all w in P such that (u, v, w) is forbidden
            new_P = P - {u, v}
            new_P -= forbidden[u][v]
            branch_and_bound(new_S, new_P)
        
        # Branch 2: Exclude u
        new_P = P - {u}
        branch_and_bound(S, new_P)
        
    # Start B&B
    # To avoid deep recursion on large P, we can start with a smaller P?
    # No, we want to find the global max.
    # But B&B on 128 elements might be too slow.
    # Let's limit the depth or use a heuristic to reduce P first.
    
    # Strategy:
    # 1. Get a good initial solution S0.
    # 2. Identify elements that are "unlikely" to be in the optimal solution?
    # 3. Or just run B&B with a time limit.
    
    # Let's try running B&B on the full set.
    # If it's too slow, we fall back to local search.
    
    # To speed up B&B, we can use a "conflict graph" based pruning.
    # If we include u, we don't remove v unless (u, v, w) is forbidden for some w in S?
    # No, the pair branching handles this.
    
    # Let's add a time check inside the loop.
    
    # Run B&B
    branch_and_bound([], set(UNIVERSE))
    
    # If B&B didn't finish or didn't improve much, do local search on the best found.
    if time.time() < DEADLINE - 5:
        # Local search on best_solution
        S = best_solution[:]
        S_set = set(S)
        improved = True
        while improved and time.time() < DEADLINE - 2:
            improved = False
            # Try 1-add
            for x in UNIVERSE:
                if x not in S_set:
                    if can_add(S, x):
                        S.append(x)
                        S_set.add(x)
                        improved = True
                        break
            if improved:
                continue
            # Try 1-swap
            for i, a in enumerate(S):
                for b in UNIVERSE:
                    if b in S_set:
                        continue
                    new_S = S[:i] + S[i+1:]
                    if can_add(new_S, b):
                        S = new_S + [b]
                        S_set = set(S)
                        improved = True
                        break
                if improved:
                    break
        if len(S) > best_len:
            best_solution = S
            best_len = len(S)
            
    return best_solution if best_solution else [1]

# Wrap to ensure nonempty
_orig_construct = construct
def construct():
    result = _orig_construct()
    if not result:
        return [1]
    return result
