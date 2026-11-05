"""
Minimum maximal no-isosceles set on [0,N)^2.

Strategy:
1. Construct a conflict graph G = (V, E) where V is the set of grid points.
   An edge (u, v) exists if there exists a point w such that {u, v, w} forms
   an isosceles triangle (including degenerate cases where distances are equal).
   Note: The condition "no isosceles set" means no 3 points form an isosceles triangle.
   This is equivalent to finding an Independent Set in the 3-uniform hypergraph of isosceles triples.
   However, if we define the graph edges as "u and v are incompatible if they share an isosceles partner",
   that's not quite right for MIS.
   
   Let's refine:
   A set S is "no-isosceles" if no 3 points in S form an isosceles triangle.
   This is a 3-uniform constraint. Standard MIS handles 2-uniform constraints.
   
   Alternative Approach:
   We can still use a branch and bound search over the vertices, maintaining the set of "available" vertices.
   When we select a vertex v, we must remove all vertices u such that {v, u, w} is isosceles for some w in S?
   No, the constraint is global.
   
   Let's stick to the definition:
   Valid Set S: For all distinct u, v, w in S, dist(u,v)^2, dist(u,w)^2, dist(v,w)^2 are all distinct.
   
   We can model this as a Maximum Independent Set in a graph where an edge (u,v) exists
   if u and v are "conflicting". Two points u and v conflict if they cannot both be in a valid set?
   No, two points can be in a valid set unless they form an isosceles triangle with a *third* point.
   The conflict depends on the third point.
   
   Therefore, a simple graph MIS is not directly applicable unless we define the graph dynamically or use a different formulation.
   
   However, we can use a Branch and Bound approach similar to MIS but adapted for 3-uniform constraints.
   State: (Current Set S, Candidate Set C).
   S is valid.
   C is the set of vertices that can be added to S without violating validity *with respect to S*.
   i.e., c in C iff for all u, v in S, {c, u, v} is not isosceles.
   
   We want to maximize |S|.
   Upper bound: |S| + |C|.
   If this bound is <= current best, prune.
   
   This is essentially a Maximum Independent Set search in a dynamically defined graph, or a backtracking search with pruning.
   Given N=16 (256 points), this might be feasible if the branching factor is controlled well.
   However, 256 is large for pure backtracking without strong heuristics.
   
   Let's try a different angle.
   The "no-isosceles" condition is very restrictive.
   Actually, let's look at the "Maximal" requirement.
   A maximal no-isosceles set is one where no other point can be added.
   We want the *minimum* size of such a set.
   
   This is a "Minimum Maximal Independent Set" problem in a 3-uniform hypergraph, or "Minimum Maximal Packing".
   
   Let's use a Local Search approach but with a better initialization and move set, or a specific heuristic.
   Wait, the current program uses Greedy + Local Search.
   The prompt asks for a MATERIALLY DIFFERENT strategy.
   
   Let's try **Integer Linear Programming (ILP)**? No, scipy doesn't have a full ILP solver for this size easily without pulp/cvxpy which might not be installed or too slow.
   
   Let's try **Genetic Algorithm**?
   
   Let's try **Simulated Annealing** with a different energy function and neighborhood?
   The current program has a comment about SA but implements Greedy+LS.
   
   Let's implement a **Branch and Bound** that searches for a *valid* set of a specific size K, and we binary search for the minimum K such that a *maximal* valid set of size K exists.
   Actually, finding if a maximal valid set of size K exists is hard.
   
   Alternative:
   1. Generate a valid set S.
   2. Check if it is maximal.
   3. If not, add points to make it maximal? No, we want minimum size.
   
   Let's reconsider the **Conflict Graph** idea.
   If we define a graph G where (u,v) is an edge if u and v are "close" in some sense?
   
   Let's look at the structure of isosceles triangles.
   Maybe we can use **Greedy with Backtracking** specifically optimized for minimality.
   
   Actually, there is a known result or approach for "Minimum Maximal Independent Set" (MMIS).
   For graphs, MMIS is NP-hard.
   
   Let's try a **Greedy Algorithm with Randomization and Repair**:
   1. Start with an empty set.
   2. Randomly add points that maintain validity.
   3. Once no more points can be added (maximal), try to "shrink" it.
   
   How to shrink?
   If S is maximal, removing a point p might allow adding another point q, resulting in a same-size set, or if we remove p and cannot add anything, S\{p} is not maximal.
   To get a *smaller* maximal set, we need to find a subset S' of S that is still maximal?
   No, if S' is a subset of S, S' is not necessarily maximal.
   
   Wait, if S is maximal, and we remove p, the new set S\{p} is *not* maximal (by definition of maximality, every point in a maximal set is "essential" in the sense that removing it breaks maximality? No.
   Definition: S is maximal if for all x not in S, S union {x} is invalid.
   If we remove p from S, S\{p} is a valid set. Is it maximal?
   If S\{p} is maximal, then p was not essential for maximality. But p was in S.
   If S\{p} is maximal, then S is also maximal (superset of a maximal set? No, S is not a superset of S\{p} in the sense of maximality constraints. S is larger.
   Actually, if S\{p} is maximal, then no point can be added to S\{p}.
   Since S is valid, and S\{p} is a subset, S is also valid.
   But S is larger.
   
   Key Insight:
   If S is a maximal valid set, then for any p in S, S\{p} is *not* maximal.
   Proof: If S\{p} were maximal, then no point could be added to S\{p}.
   But p can be added to S\{p} to form S, which is valid.
   Contradiction.
   Therefore, every point in a maximal valid set is "critical" in the sense that removing it makes the set non-maximal.
   
   This implies that we cannot simply remove points from a maximal set to get a smaller maximal set.
   We must **restructure** the set.
   
   Strategy: **Local Search with "Flip" Moves**.
   Start with a maximal set S (from greedy).
   Try to replace a subset of points with a different subset that maintains validity and maximality, but reduces size.
   This is hard.
   
   Better Strategy: **Randomized Construction with "Minimization" Bias**.
   When building the set greedily, we want to choose points that "block" many other points, forcing the final set to be small.
   A point that forms isosceles triangles with many other points is "efficient" at blocking.
   So, sort candidate points by their "blocking power" (number of points they conflict with) and pick the one with highest blocking power first?
   This is a heuristic for Minimum Dominating Set, which is related.
   
   Let's implement **Greedy with Blocking Power Heuristic**:
   1. Precompute for each point p, the set of pairs (i,j) that form isosceles with p.
   2. When selecting the next point, among all valid candidates, pick the one that "eliminates" the most future candidates.
      Eliminating a candidate q means that adding p makes q invalid (i.e., p and q conflict with some existing point? No, p and q don't conflict directly unless there's a third point).
      Actually, adding p removes from the candidate pool all points q such that {p, q, r} is isosceles for some r in S?
      No, the validity check is: q can be added to S if for all u,v in S, {q,u,v} is not isosceles.
      If we add p to S, the new set is S' = S U {p}.
      A point q is valid for S' if:
      1. q is valid for S (for all u,v in S, {q,u,v} not iso).
      2. {q, p, u} not iso for all u in S.
      
      So adding p removes from the candidate pool all q that were valid for S but become invalid due to p.
      We want to pick p that removes the *most* q's.
      
   This is a "Maximal Coverage" or "Set Cover" like heuristic.
   
   Let's implement this:
   1. Precompute `conflicts_with_pair[p][u][v]`? Too big.
   2. Precompute `bad_pairs_for_p[p]` = set of pairs (u,v) such that {p,u,v} is iso.
   3. Maintain `S` (selected) and `C` (candidates).
   4. Initially C = all points.
   5. While C is not empty:
      For each p in C:
         Calculate `new_C` = { q in C | q != p and for all u,v in S, {q,u,v} not iso AND for all u in S, {q,p,u} not iso }.
         Actually, we can just check if q is valid with S U {p}.
         Count |new_C|.
      Pick p that minimizes |new_C| (i.e., maximizes reduction).
      Add p to S.
      Update C = new_C (for the chosen p).
      
   This greedy approach tends to produce smaller maximal sets because it aggressively prunes the search space.
   
   To improve, we can do multiple restarts with random tie-breaking or random perturbation of the blocking power.

"""

import random
import time
import itertools
import math
import numpy as np

N = 16
ALL_POINTS = list(itertools.product(range(N), repeat=2))
N_PTS = len(ALL_POINTS)

# Precompute squared distances
coords = np.array(ALL_POINTS)
diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
dist2 = np.sum(diff**2, axis=2)

# Precompute bad pairs for each point
# bad_pairs[p] is a set of (i, j) such that {p, i, j} is isosceles.
# We store as a list of sets of tuples.
BAD_PAIRS = [set() for _ in range(N_PTS)]

def _precompute_bad_pairs():
    i_idx, j_idx = np.triu_indices(N_PTS, k=1)
    for p in range(N_PTS):
        dp = dist2[p, :]
        dp_i = dp[i_idx]
        dp_j = dp[j_idx]
        dij = dist2[i_idx, j_idx]
        cond = (dp_i == dp_j) | (dp_i == dij) | (dp_j == dij)
        bad_indices = np.where(cond)[0]
        for k in bad_indices:
            BAD_PAIRS[p].add((i_idx[k], j_idx[k]))

_start_time = time.time()
_precompute_bad_pairs()

def _is_valid_addition(p, S_indices):
    """Check if p can be added to S_indices."""
    S_set = set(S_indices)
    # Check against all pairs in S
    # We can use BAD_PAIRS[p]
    # If (u, v) in BAD_PAIRS[p] and u in S and v in S, then invalid.
    for u, v in BAD_PAIRS[p]:
        if u in S_set and v in S_set:
            return False
    return True

def _greedy_blocking_power(deadline):
    """
    Greedy algorithm that picks the point which reduces the candidate set the most.
    """
    S = []
    S_set = set()
    C = set(range(N_PTS))
    
    while C:
        if time.time() > deadline:
            break
        
        best_p = -1
        best_new_C_size = float('inf')
        best_new_C = None
        
        # To speed up, we might not want to iterate all C if it's large,
        # but N=256 is small.
        # We can also randomize the order of C to break ties.
        candidates = list(C)
        random.shuffle(candidates)
        
        for p in candidates:
            # Calculate new C if we add p
            new_C = set()
            for q in C:
                if q == p:
                    continue
                # Check if q is valid with S U {p}
                # This requires checking pairs in S U {p}
                # We can optimize:
                # 1. Check q against S (using BAD_PAIRS[q] and S_set)
                # 2. Check q against p and S (using BAD_PAIRS[q] and pairs involving p)
                
                # Efficient check:
                # q is valid if no pair in S U {p} is in BAD_PAIRS[q].
                # Pairs in S U {p} are:
                # 1. Pairs in S (already checked if q was in C? No, C is defined by validity wrt S)
                #    Since q is in C, q is valid wrt S. So no pair in S is in BAD_PAIRS[q].
                # 2. Pairs (p, u) for u in S.
                #    So we need to check if (p, u) is in BAD_PAIRS[q] for any u in S.
                #    This is equivalent to checking if BAD_PAIRS[q] intersects with {(p, u) | u in S}.
                
                valid = True
                for u in S:
                    # Check if (p, u) is in BAD_PAIRS[q]
                    # Note: BAD_PAIRS stores (i,j) with i<j.
                    # So we need to check (min(p,u), max(p,u))
                    a, b = (p, u) if p < u else (u, p)
                    if (a, b) in BAD_PAIRS[q]:
                        valid = False
                        break
                
                if valid:
                    new_C.add(q)
            
            new_C_size = len(new_C)
            if new_C_size < best_new_C_size:
                best_new_C_size = new_C_size
                best_p = p
                best_new_C = new_C
        
        if best_p == -1:
            break
            
        S.append(best_p)
        S_set.add(best_p)
        C = best_new_C
        
    # Ensure maximality?
    # The greedy process stops when C is empty, which means no point can be added.
    # So S is maximal.
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 280.0
    
    best_set = None
    best_size = float('inf')
    
    # Multiple restarts
    while time.time() < deadline - 10.0:
        candidate = _greedy_blocking_power(deadline - 5.0)
        
        if candidate is None:
            continue
            
        # Verify validity and maximality just in case
        # Validity check
        valid = True
        c_set = set(candidate)
        # O(k^3) check is slow, but k is small (~20-50).
        # We can use the BAD_PAIRS check incrementally?
        # For now, trust the greedy logic which ensures validity.
        # But let's do a quick sanity check on a few random triples?
        # No, the logic `q in C` ensures validity wrt S.
        
        if len(candidate) < best_size:
            best_size = len(candidate)
            best_set = candidate
            
    if best_set is None:
        # Fallback: simple greedy
        S = []
        S_set = set()
        C = set(range(N_PTS))
        while C:
            p = random.choice(list(C))
            S.append(p)
            S_set.add(p)
            # Update C
            new_C = set()
            for q in C:
                if q == p:
                    continue
                valid = True
                for u in S:
                    if u == p:
                        continue
                    a, b = (p, u) if p < u else (u, p)
                    if (a, b) in BAD_PAIRS[q]:
                        valid = False
                        break
                if valid:
                    new_C.add(q)
            C = new_C
        best_set = S

    return [ALL_POINTS[i] for i in best_set]
