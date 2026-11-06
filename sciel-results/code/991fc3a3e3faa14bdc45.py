"""
Strategy: Constraint Satisfaction with Backtracking and Propagation.

This approach models the problem as a CSP where:
- Variables: Grid points P
- Domains: {IN, OUT}
- Constraints:
  1. No-Isosceles: For any triplet (a,b,c), it is not the case that all three are IN and form an isosceles triangle.
  2. Maximality: For any point p, if p is OUT, then there must exist a pair (a,b) IN such that (p,a,b) is isosceles.
     Equivalently: If p is OUT, p must be "covered" by S.
     Equivalently: If p is NOT covered by S, p MUST be IN.

We use a backtracking search with constraint propagation to find a minimal valid assignment.
To minimize the set size, we prioritize assigning points to OUT (if possible) or use a heuristic that prefers OUT when feasible, but since we want MINIMAL size, we actually want to find a valid set and then minimize its size. 
Wait, the standard CSP formulation for "Minimal Maximal Set" is tricky because "Maximal" is a global property.
However, we can reframe:
A set S is maximal if and only if S is a maximal independent set in the graph G where edges connect points that form an isosceles triangle with a third point? No, that's not right.
The "no-isosceles" constraint is on triplets.

Let's use a different CSP formulation:
We decide the status of each point.
If we decide p is IN, we must ensure no isosceles triangle is formed with other IN points.
If we decide p is OUT, we must ensure it is covered by IN points.

A powerful heuristic for minimal maximal independent sets (analogous) is:
1. Start with an empty set of IN points.
2. Identify points that are "forced IN" because they cannot be covered by any potential IN points?
   Actually, let's use the property: If p is not in S, it MUST be covered.
   If we choose a set of IN points S, it is valid if:
   - No isosceles triangle in S.
   - Every point not in S is covered by S.

We can search for S by adding points.
To get a MINIMAL S, we want to add as few points as possible.
This is a Set Covering problem with conflicts.
"Select a subset S of points such that:
 1. S is an independent set in the 3-uniform hypergraph of isosceles triangles.
 2. S covers all points (where 'covers' means for every p not in S, there is a pair in S forming an isosceles with p).
 3. Minimize |S|."

This is a Minimum Independent Set Covering problem.

We will use a Backtracking Search with the following logic:
- State: Current set S (IN points), Current set C (OUT points, already covered).
- We iterate through points.
- Heuristic: Pick a point p.
- Option 1: Put p in S.
  - Check if S U {p} is independent. If yes, recurse.
  - Update coverage: Points covered by pairs involving p are now "safe" to be OUT.
- Option 2: Put p in C (OUT).
  - This is only valid if p is already covered by S.
  - If p is NOT covered by S, we CANNOT put it in C yet. We MUST put it in S?
    No, it could be covered by a point added later.
    However, if we process points in a specific order, and we know that no future point can cover p without conflicting with S, then p MUST be in S.
    
To make this efficient, we will use **Forward Checking**:
- Maintain `S` (IN) and `U` (Uncovered points not in S).
- Initially S = {}, U = All Points.
- When we add p to S:
  - Check independence.
  - Remove p from U.
  - For every point q in U that is covered by a pair in S (including new pairs with p), remove q from U? No, q is covered, so it can be OUT. But we haven't decided it's OUT yet.
  - Actually, if q is covered by S, it is SAFE to be OUT.
  - If q is NOT covered by S, it MUST be IN (eventually).
  
So, the propagation rule is:
If a point q is in U (not in S) and is NOT covered by S, then q MUST be added to S.
This is a deterministic propagation!

Algorithm:
1. Start with S = {}, Covered = {} (set of points covered by S).
   Note: A point is covered if there exists a pair (a,b) in S such that (p,a,b) is isosceles.
   Initially, no points are covered. So ALL points must be in S?
   No, that would mean S = All Points, which is not independent.
   
   Wait, the logic "If q is not in S and not covered by S, q MUST be in S" implies that we must build S such that every point not in S is covered.
   If we start with empty S, no point is covered. So ALL points MUST be in S.
   But if all points are in S, it's not independent.
   So this simple propagation leads to a contradiction immediately unless we allow "future" points to cover "current" uncovered points.
   
   The flaw: "Not covered by S" does not mean "Cannot be covered by future S".
   It means "Not covered by CURRENT S".
   
   However, if we use **Backtracking with Lookahead**:
   We try to construct S.
   At any step, we have a partial S.
   We look for a point p not in S.
   If p is covered by S, we can choose to leave it out (it's a candidate for OUT).
   If p is NOT covered by S, we have two choices:
     a) Add p to S.
     b) Leave p out, BUT we must ensure that some future point added to S will cover p.
    
   To simplify, we can use a **Greedy Set Cover with Conflict Resolution**:
   1. Start with S = {}.
   2. While there are uncovered points (points not in S and not covered by S):
      - Pick an uncovered point p.
      - We MUST add SOME point to S to cover p.
      - Candidates for addition:
        - p itself.
        - Any point q such that adding q to S would cover p?
          Adding q covers p if there exists r in S such that (p,q,r) is isosceles?
          Or if q is p?
          Actually, p is covered by S if there exist a,b in S such that (p,a,b) is isosceles.
          If S is empty, p is not covered.
          If we add q to S, does q cover p?
          No, a single point doesn't cover another. A PAIR in S covers a point.
          So if S is empty, adding q to S makes S={q}. No pairs. So no point is covered.
          This suggests we need at least 2 points in S to cover anything.
          
   This implies the "Uncovered" logic is hard to propagate incrementally with small S.
   
   Alternative Strategy: **Local Search on Independent Sets**
   1. Generate a random maximal independent set S (greedy).
   2. Check if it is maximal in the "covering" sense (i.e., every point not in S is covered by S).
      If not, it's not a valid solution for the task.
      The task requires "Maximal No-Isosceles Set".
      Definition of Maximal: No further grid point can be added.
      This is equivalent to: For every p not in S, p conflicts with S (i.e., p cannot be added).
      p cannot be added if adding p creates an isosceles triangle with two points in S.
      So, p is "blocked" if there exist a,b in S such that (p,a,b) is isosceles.
      This is EXACTLY the "covered" definition I used above!
      So, "Maximal No-Isosceles" is equivalent to "Every point not in S is covered by S".
      
   So the problem is: Find a set S such that:
   1. S is independent (no isosceles triangles in S).
   2. S is a dominating set in the "isosceles hypergraph" (every point not in S is covered by S).
   3. Minimize |S|.
   
   This is a **Minimum Independent Dominating Set** problem on a specific hypergraph.
   
   We can solve this using **Iterated Local Search**:
   1. Start with a random valid set (independent and dominating).
      How to get a valid start?
      - Greedy Independent Set: Add points if they don't conflict.
      - This yields an independent set, but it might not be dominating (maximal).
      - To make it maximal, we can try to add points? No, if it's not dominating, there are points p not in S that are NOT covered.
        This means p CAN be added to S without creating an isosceles triangle.
        So we can simply ADD all such p to S?
        If we add p, it might conflict with other "addable" points.
        So we can just run the Greedy Independent Set algorithm again, but starting from the current S?
        Actually, if S is independent, and p is not covered by S, then p is independent of S (no isosceles triangle with two points in S).
        So S U {p} is independent.
        So we can just add ALL uncovered points to S?
        If we add p1 and p2, and p1, p2, q form an isosceles, then S U {p1, p2} is NOT independent.
        So we must be careful.
        
      Better Initial Construction:
      1. Start with S = {}.
      2. While True:
         - Find all points p not in S that are "safe" to add (i.e., adding p keeps S independent).
           Actually, "safe" means p does not form an isosceles with any two points in S.
           If S is small, most points are safe.
         - If no points are safe, break? No, we need to ensure maximality.
         
      Let's use a standard **Maximal Independent Set** generator (Greedy).
      This ensures S is independent and maximal (in the graph sense, where edges are conflicts).
      Wait, the conflict is 3-uniform.
      A set S is maximal independent if for every p not in S, p conflicts with S.
      Conflict: p forms an isosceles with two points in S.
      So YES, a Greedy Maximal Independent Set is exactly a valid solution for the task!
      
      So:
      1. Generate a random Greedy Maximal Independent Set S.
      2. This S is valid (independent and maximal).
      3. Now, minimize |S| using Local Search.
      
      Local Search Moves:
      - **Remove**: Try to remove p from S.
        If S \ {p} is still independent (it is, since it's a subset) and still maximal (dominating)?
        If p is removed, p becomes a candidate for OUT.
        For S \ {p} to be maximal, p must be covered by S \ {p}.
        So, if p is covered by S \ {p}, we can remove p. This reduces size by 1.
        If p is NOT covered by S \ {p}, we cannot simply remove it, because then p would be addable, violating maximality.
        However, we might be able to **Swap**: Remove p and add q?
        If we remove p, we lose coverage of some points.
        If we add q, we might regain coverage.
        
      So the local search should focus on **Swaps** and **Removals**.
      1. **Removal**: If p is covered by S \ {p}, remove p. (Strict improvement in size).
      2. **Swap**: Remove p, add q.
         - Check if S \ {p} U {q} is independent.
         - Check if S \ {p} U {q} is maximal (all points not in new S are covered by new S).
         - If valid, accept if it reduces size? No, swap keeps size same.
         - We want to find a sequence of swaps that leads to a state where a removal is possible?
         - Or just accept valid swaps to explore the space of valid sets, hoping to find one where a point is redundant.
         
      Since we want to MINIMIZE size, we should prioritize Removals.
      If no removal is possible, we are at a local minimum.
      To escape, we use Swaps.
      We can use Simulated Annealing to accept non-improving swaps (or even temporarily invalid sets? No, let's stick to valid sets to ensure we always return a valid answer).
      
      So:
      - State: Valid set S (independent and maximal).
      - Moves:
        - Try to remove any p in S such that p is covered by S \ {p}. If found, do it. (Greedy reduction).
        - If no removal found, perform a Swap:
          - Pick p in S, q not in S.
          - Check if S' = S \ {p} U {q} is valid.
          - If valid, accept with some probability (to explore).
          - If invalid, reject.
          
      This is a random walk on the space of valid minimal maximal sets (locally minimal).
      To get a smaller set, we need to find a global minimum.
      
      Implementation details:
      - Precompute `IsoWithPoint[p]`: Set of pairs (a,b) such that (p,a,b) is isosceles.
      - Precompute `CoveredBy[p]`: Not directly, but we can check coverage dynamically.
        `is_covered(p, S)`: Returns True if there exist a,b in S such that (p,a,b) is isosceles.
        This check is O(|S|^2) if we iterate pairs, or O(|IsoWithPoint[p]|) if we check if any pair in IsoWithPoint[p] is subset of S.
        Since |S| is small (~20-50), and |IsoWithPoint[p]| might be large, iterating pairs in S might be faster if |S|^2 < |IsoWithPoint[p]|.
        Actually, for a fixed p, we just need to know if ANY pair in S is in IsoWithPoint[p].
        We can precompute for each p, the list of pairs.
        To check if S covers p:
        `any(pair <= S for pair in IsoPairs[p])`
        This is fast if we represent S as a set and pairs as frozensets.
        
      - Independence check for S' = S \ {p} U {q}:
        - S \ {p} is independent.
        - We need to check if q conflicts with S \ {p}.
        - q conflicts with S \ {p} if there exist a,b in S \ {p} such that (q,a,b) is isosceles.
        - This is equivalent to: `any(pair <= (S - {p}) for pair in IsoPairs[q])`.
        
      - Maximality check for S':
        - For every r not in S', is r covered by S'?
        - We only need to check r that might have LOST coverage when p was removed, or GAINED coverage when q was added?
        - Actually, if S was maximal, all r not in S were covered by S.
        - In S', the points not in S' are: (POINTS \ S) U {p} \ {q} (if q was not in S).
        - Points that were in S \ {p} are still in S'.
        - p is now out. Is p covered by S'?
          p is covered by S' if there exist a,b in S' such that (p,a,b) is isosceles.
          Note: S' = (S \ {p}) U {q}.
          So we check if p is covered by (S \ {p}) U {q}.
        - q is now in S'. So q doesn't need to be covered.
        - Other points r not in S:
          They were covered by S.
          Are they still covered by S'?
          S' has q instead of p.
          If r was covered ONLY by pairs involving p, it might lose coverage.
          If r is covered by pairs involving q, it gains coverage (but it was already covered, so no change in status, just different coverers).
          So we only need to check points r that were covered by S using pairs involving p, AND are NOT covered by S' using pairs involving q (or other pairs).
          Actually, simpler: Just check all r not in S'.
          Since |POINTS| = 256, and |S'| is small, checking all 256 points for coverage is O(256 * |S'|^2) or O(256 * |IsoPairs[r]|).
          This is fast enough for local search steps.
          
      Let's refine the Local Search:
      1. Generate initial S (Greedy Maximal Independent Set).
      2. Loop until time limit:
         a. Try to remove any point p in S that is covered by S \ {p}.
            If found, remove it. Continue (restart loop to find more removals).
         b. If no removal found:
            - Pick random p in S, random q not in S.
            - Check if S' = S \ {p} U {q} is valid.
            - If valid, accept S'.
            - (Optional: Use SA to accept invalid swaps? No, let's stick to valid to guarantee correctness).
            - To escape local minima, we might need to accept swaps that don't change size but move us to a different region.
            
      To improve on size, we can also try **2-Swaps** or **K-Swaps** if 1-Swap is stuck.
      But 1-Swap with random restarts might be sufficient.
      
      Let's implement this.
"""

import time
import random
import itertools
import sys

N = 16
POINTS = list(itertools.product(range(N), repeat=2))
POINT_SET = set(POINTS)
DEADLINE = 320.0

def _dist2(a, b):
    return (a[0] - b[0])**2 + (a[1] - b[1])**2

print("Precomputing...", file=sys.stderr)
start_time = time.time()

# IsoPairs[p] = set of frozensets({a,b}) such that (p,a,b) is isosceles
IsoPairs = {p: set() for p in POINTS}

# For fast independence check, we might also want to know for a pair (a,b), which points p are isosceles with them.
# IsoWithPair[pair] = set of points p.
IsoWithPair = {}

for a in POINTS:
    for b in POINTS:
        if a == b:
            continue
        pair = frozenset((a,b))
        d_ab = _dist2(a,b)
        for p in POINTS:
            if p == a or p == b:
                continue
            d_pa = _dist2(p,a)
            d_pb = _dist2(p,b)
            if d_pa == d_ab or d_pb == d_ab or d_pa == d_pb:
                IsoPairs[p].add(pair)
                if pair not in IsoWithPair:
                    IsoWithPair[pair] = set()
                IsoWithPair[pair].add(p)

print(f"Precomputation took {time.time() - start_time:.2f}s", file=sys.stderr)

def is_independent(S):
    """Check if S contains no isosceles triangle."""
    S_list = list(S)
    for i in range(len(S_list)):
        for j in range(i+1, len(S_list)):
            a, b = S_list[i], S_list[j]
            pair = frozenset((a,b))
            # If any point in S is in IsoWithPair[pair], then that point with a,b forms isosceles.
            # We need to check if there exists c in S such that c in IsoWithPair[pair].
            # Since a,b are in S, we just check if IsoWithPair[pair] intersects S \ {a,b}.
            # Actually, IsoWithPair[pair] contains points c such that (c,a,b) is isosceles.
            # If c is in S, then (a,b,c) is an isosceles triangle in S.
            if pair in IsoWithPair:
                for c in IsoWithPair[pair]:
                    if c in S:
                        return False
    return True

def is_maximal(S):
    """Check if every point not in S is covered by S."""
    for p in POINTS:
        if p in S:
            continue
        # Check if p is covered by S
        # p is covered if there exist a,b in S such that (p,a,b) is isosceles.
        # This is equivalent to: any(pair <= S for pair in IsoPairs[p])
        covered = False
        for pair in IsoPairs[p]:
            if pair <= S:
                covered = True
                break
        if not covered:
            return False
    return True

def is_valid(S):
    return is_independent(S) and is_maximal(S)

def greedy_maximal_independent_set(random_order):
    """
    Construct a maximal independent set using greedy addition.
    We process points in `random_order`.
    We add p to S if S U {p} is independent.
    At the end, S is independent.
    Is it maximal?
    If we process all points, any point not in S was rejected because it conflicted with some points in S.
    Conflict means: Adding p creates an isosceles triangle with two points in S.
    So p is covered by S.
    Thus, S is maximal.
    """
    S = set()
    for p in random_order:
        if p in S:
            continue
        # Check if adding p keeps S independent
        # p conflicts with S if there exist a,b in S such that (p,a,b) is isosceles.
        # This is equivalent to: any(pair <= S for pair in IsoPairs[p])
        # Wait, if (p,a,b) is isosceles, then p is covered by {a,b}.
        # If p is covered by S, then p CANNOT be added to S?
        # No. The condition for S U {p} to be independent is that NO isosceles triangle is formed.
        # If (p,a,b) is isosceles and a,b in S, then S U {p} is NOT independent.
        # So we can only add p if p is NOT covered by S.
        
        # So the greedy step is:
        # If p is not covered by S, add p to S.
        
        if not any(pair <= S for pair in IsoPairs[p]):
            S.add(p)
            
    return S

def construct():
    start_time = time.time()
    
    best_S = None
    best_size = float('inf')
    
    # Run multiple restarts to find a smaller set
    while time.time() < DEADLINE:
        random.seed(int(time.time() * 1000) % 100000) # Change seed for variety
        pts = list(POINTS)
        random.shuffle(pts)
        
        S = greedy_maximal_independent_set(pts)
        
        if not S:
            continue
            
        # Local Search: Try to remove points
        # We repeatedly try to remove any point that is redundant (covered by rest of S)
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            S_list = list(S)
            random.shuffle(S_list) # Random order for removal attempts
            for p in S_list:
                if time.time() > DEADLINE:
                    break
                S_without_p = S - {p}
                # Check if p is covered by S_without_p
                # If yes, we can remove p.
                if any(pair <= S_without_p for pair in IsoPairs[p]):
                    S = S_without_p
                    improved = True
                    break # Restart loop to check for more removals
            
        # If we found a smaller set, update best
        if len(S) < best_size:
            best_size = len(S)
            best_S = set(S)
            print(f"New best size: {best_size}", file=sys.stderr)
            
        # Optional: Local Search with Swaps to escape local minima
        # If we are stuck, try swaps
        # For now, just rely on restarts with different random orders.
        # The greedy order significantly impacts the resulting set size.
        
        # Check time
        if time.time() > DEADLINE - 5:
            break
            
    if best_S is None:
        # Fallback: Return a valid set (should not happen)
        return list(POINTS[:10])
        
    return list(best_S)

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")

