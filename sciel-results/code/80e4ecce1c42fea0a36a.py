"""
Strategy: Backtracking Search with Conflict Graph and MRV Heuristic.

The problem is to find a set S of points in [0, N)^2 such that:
1. S is "no-isosceles": No three points in S form an isosceles triangle.
2. S is "maximal": For every point p not in S, S U {p} is NOT no-isosceles.
   This means for every p not in S, there exist a, b in S such that (p, a, b) is isosceles.

Reframing:
Let U be the set of all points.
For a point p, let Conf(p) be the set of points q such that {p, q} is part of an isosceles triangle with some other point? No, that's not quite right.
The condition "S U {p} is not no-isosceles" means there exist a, b in S such that (p, a, b) is isosceles.
This implies that p is "covered" by the set S if there exist a, b in S such that p forms an isosceles triangle with a and b.

So we want a set S such that:
1. S is an independent set in the "Isosceles Graph" G_iso, where edges connect pairs that form an isosceles triangle with some third point? No.
   The graph G_iso should have vertices U. An edge (u, v) exists if... wait.
   The condition "S is no-isosceles" means no triplet (a,b,c) in S is isosceles.
   This is a hypergraph constraint.
   
   However, we can define a graph G_conf where two points u, v are connected if they CANNOT be in the same no-isosceles set?
   No, two points can always be in a no-isosceles set (a set of size 2 is never isosceles).
   The constraint is on triplets.
   
   Let's look at the maximality condition again.
   For any p not in S, there must be a, b in S such that (p, a, b) is isosceles.
   This means that the set S must "hit" the set of isosceles triangles containing p.
   
   Let's define for each point p, the set of pairs P_p = { {a,b} | (p,a,b) is isosceles }.
   Then S is maximal iff for all p not in S, S contains at least one pair from P_p.
   
   This looks like a hitting set problem or a covering problem, but with the additional constraint that S itself must be isosceles-free.
   
   Algorithm:
   1. Precompute for each point p, the list of pairs (a,b) that form an isosceles triangle with p.
      Actually, it's easier to precompute for each pair (a,b), which points p form an isosceles triangle with them.
      Let IsoTriples be a list of sets of 3 points.
      Or better: For each point p, let Blocks[p] be the set of pairs {a,b} such that {p,a,b} is isosceles.
      If S contains a pair {a,b} in Blocks[p], then p is "blocked" (cannot be added).
      
   2. We want to find a set S such that:
      - S is isosceles-free (no triplet in S is isosceles).
      - For all p not in S, there exists {a,b} subset S such that {a,b} in Blocks[p].
      
   This is a hard combinatorial optimization problem.
   We can use a backtracking search.
   
   State:
   - S: current set of selected points.
   - Available: set of points that can still be added to S without violating isosceles-freeness.
     Note: If we add p to S, we must ensure no existing pair in S forms an isosceles with p.
     Also, adding p might remove other points from "Available" if they form isosceles with p and some other point in S?
     No, the isosceles constraint is global. If S is isosceles-free, and we add p, we just need to check if {p} U S is isosceles-free.
     This requires checking if p forms an isosceles triangle with any pair in S.
     
   Heuristic:
   - We want to minimize |S|.
   - We should prioritize adding points that "block" many other points.
   - A point p blocks another point q if there exists a third point r such that (p,q,r) is isosceles?
     No, p blocks q (in the sense that q cannot be added if p is in S) only if there is some r in S such that (p,q,r) is isosceles.
     This depends on the rest of S.
     
   Alternative Heuristic:
   Let's try to construct S such that every point not in S is "covered".
   A point p is covered if there exist a,b in S such that (p,a,b) is isosceles.
   
   Let's define a "coverage" score for a candidate point c.
   If we add c to S, how many points p not in S become covered?
   p becomes covered if there exists a in S such that (p,c,a) is isosceles.
   So the "gain" of adding c is the number of points p such that (p,c,a) is isosceles for some a in S.
   
   We can use a greedy algorithm with randomization (Simulated Annealing or Randomized Greedy) to build S, then try to remove points.
   But the previous program did local search on removals.
   
   Let's try a different approach: **Integer Linear Programming (ILP)**?
   Standard library doesn't have an ILP solver. SciPy has `milp`? No, `scipy.optimize.milp` is available in newer versions, but let's stick to robust methods.
   
   Let's refine the Backtracking Search.
   
   Variables: x_p in {0,1} for each point p.
   Constraints:
   1. For every isosceles triangle (a,b,c): x_a + x_b + x_c <= 2.
   2. For every point p: 
      If x_p = 0, then there must exist a pair (a,b) such that x_a=1, x_b=1 and (p,a,b) is isosceles.
      This is a disjunctive constraint: (1-x_p) <= sum_{(a,b) in P_p} (x_a * x_b).
      This is non-linear.
      
   We can linearize x_a * x_b by introducing y_ab = x_a * x_b.
   y_ab <= x_a, y_ab <= x_b, y_ab >= x_a + x_b - 1.
   Constraint: (1-x_p) <= sum_{(a,b) in P_p} y_ab.
   
   Objective: Minimize sum x_p.
   
   Since we don't have a solver, we can implement a simple Branch and Bound or use a heuristic.
   
   Given the small size (N=16, 256 points), a well-tuned backtracking might work.
   
   Let's try a **Greedy Set Cover** style approach.
   1. Start with S = empty.
   2. While there exists a point p not "covered" (i.e., no pair in S forms isosceles with p):
      - Choose a point c to add to S.
      - Criteria: c must be compatible with S (S U {c} is isosceles-free).
      - Among compatible c, choose the one that covers the most currently uncovered points.
      - "Covers" p if there exists a in S such that (p,c,a) is isosceles.
      - Note: If S is empty, no point is covered. We must pick a first point.
        If S={c}, a point p is covered if there exists a=c such that (p,c,c) is isosceles? No, (p,c,c) is degenerate.
        Wait, (p,a,b) is isosceles. If a=b, distance is 0.
        Usually, triangles are non-degenerate. The problem says "no-isosceles set".
        Does (p, a, a) count? Distances: d(p,a), 0, d(a,p). 0 == d(p,a) only if p=a.
        If p!=a, distances are d, 0, d. Two sides equal. So it is isosceles.
        So if S={c}, then any p such that d(p,c) == 0 (impossible) or ...
        Wait, d(p,c) == d(c,c)=0 implies p=c.
        So S={c} covers NO points?
        Let's check: Is (p, c, c) isosceles?
        Sides: p-c, c-c, c-p. Lengths: L, 0, L.
        Yes, L==L. So it is isosceles.
        So if S={c}, then for any p, the set {p,c,c} is isosceles?
        No, the set is {p,c}. The triangle is formed by points p, c, and... wait.
        The condition for S U {p} to be invalid is that there exist a,b in S such that (p,a,b) is isosceles.
        If S={c}, then a=c, b=c.
        Triangle (p,c,c). Sides: dist(p,c), 0, dist(c,p).
        Since dist(p,c) == dist(c,p), we have side1 == side3.
        So yes, (p,c,c) is always isosceles for p != c.
        So S={c} covers ALL other points?
        If so, S={c} is maximal?
        Let's check if S={c} is no-isosceles.
        A set of 1 point has no triangles. So it is no-isosceles.
        Is it maximal?
        Can we add p?
        S U {p} = {c, p}.
        Does {c, p} contain an isosceles triangle?
        A triangle requires 3 distinct points.
        The definition of "isosceles triangle" usually implies 3 distinct vertices.
        If the problem allows degenerate triangles (2 points), then any set of size >1 is invalid?
        "No-isosceles set" usually means no 3 distinct points form an isosceles triangle.
        If distinctness is required, then (p,c,c) is NOT a triangle.
        Let's assume distinct points.
        So (p,a,b) requires p,a,b distinct.
        If S={c}, and we add p, S'={c,p}. No 3 distinct points. So S' is no-isosceles.
        So S={c} is NOT maximal.
        
        So my "coverage" logic needs to account for distinctness.
        p is covered by S if there exist distinct a,b in S such that p,a,b are distinct and (p,a,b) is isosceles.
        
        If |S| < 2, S covers no points (unless we consider degenerate cases, which we don't).
        So we must build S until |S| >= 2.
        
        Greedy Step:
        1. Pick initial pair (a,b) such that they are "good" (cover many points).
        2. Add points one by one.
        
        Let's implement a Randomized Greedy with Restarts.
        1. Precompute `IsoPairs[p]`: List of pairs (a,b) such that (p,a,b) is isosceles and p,a,b distinct.
           Actually, we need to check if adding p to S creates an isosceles triangle.
           This happens if there exist a,b in S such that (p,a,b) is isosceles.
           
        2. Greedy Construction:
           S = []
           While not all points covered:
               Find all candidates c not in S such that S U {c} is no-isosceles.
               For each candidate c, calculate `gain(c)`:
                   Number of points p not in S U {c} that are covered by S U {c} but not by S.
                   p is covered by S' if there exist a,b in S' distinct from p such that (p,a,b) is isosceles.
                   Since S is no-isosceles, the only new isosceles triangles involving p and S' must involve c.
                   So p is newly covered if there exist a in S (or a=c) such that (p,c,a) is isosceles.
                   Specifically, if a=c, then (p,c,c) is degenerate, so ignored.
                   So we need a in S, a != c, a != p, such that (p,c,a) is isosceles.
                   Also, if S has only c (size 1), gain is 0.
                   
               Pick c with max gain. If tie, pick randomly.
               If max gain is 0, we might be stuck or need to pick a "strategic" point.
               Actually, if gain is 0, adding c doesn't help cover any new points immediately.
               But it might help later.
               In that case, pick a random candidate.
               
           After S is built (all points covered), S is maximal.
           Then, try to remove points to minimize size.
           Local search:
           For each p in S:
               If S \ {p} is still maximal, remove p.
               
           This is similar to the previous program, but the construction phase is different (greedy covering vs random greedy validity).
           The previous program used `_greedy_maximal` which just added points if valid. It didn't prioritize covering.
           Prioritizing covering might lead to smaller sets?
           Actually, adding points that cover many points might allow us to stop earlier?
           No, we must cover ALL points.
           But maybe a "dense" covering set is smaller?
           
           Let's try this.

"""

import time
import random
import itertools
import sys

N = 16
POINTS = list(itertools.product(range(N), repeat=2))
POINT_SET = set(POINTS)
DEADLINE = 320.0  # Safety margin

# Precomputation
# dist2[(x1,y1),(x2,y2)]
# isosceles[(p,a,b)] -> bool

def _dist2(a, b):
    return (a[0] - b[0])**2 + (a[1] - b[1])**2

# Precompute all isosceles triangles
# Triangles are sets of 3 distinct points {a,b,c}
# We need to know for a given p and a set S, if there is an isosceles triangle involving p and S.
# This is equivalent to: does there exist a,b in S such that (p,a,b) is isosceles?

# Let's precompute for each point p, the set of pairs (a,b) that form an isosceles triangle with p.
# Pairs are unordered.
# IsoPairs[p] = set of frozenset({a,b})

IsoPairs = {p: set() for p in POINTS}

# Iterate over all triplets
# O(N^6) which is 16^6 = 16M. Might be slow in Python.
# Optimization: 
# For each pair (a,b), find all p such that (p,a,b) is isosceles.
# Then add (a,b) to IsoPairs[p].

# Number of pairs is ~ 256*255/2 = 32640.
# For each pair, finding p is O(N^2) = 256.
# Total ~ 8M ops. Acceptable.

print("Precomputing isosceles pairs...", file=sys.stderr)
start_time = time.time()

# Map from pair (frozenset) to list of p's
# Actually, we just need IsoPairs[p].

for a in POINTS:
    for b in POINTS:
        if a == b:
            continue
        pair = frozenset((a,b))
        # Find all p such that (p,a,b) is isosceles
        d_ab = _dist2(a,b)
        for p in POINTS:
            if p == a or p == b:
                continue
            d_pa = _dist2(p,a)
            d_pb = _dist2(p,b)
            if d_pa == d_ab or d_pb == d_ab or d_pa == d_pb:
                IsoPairs[p].add(pair)

print(f"Precomputation took {time.time() - start_time:.2f}s", file=sys.stderr)

def _is_valid_set(S):
    """Check if set S contains no isosceles triangles."""
    S_list = list(S)
    for i in range(len(S_list)):
        for j in range(i+1, len(S_list)):
            for k in range(j+1, len(S_list)):
                a, b, c = S_list[i], S_list[j], S_list[k]
                # Check if (a,b,c) is isosceles
                # We can use IsoPairs?
                # If (a,b,c) is isosceles, then pair {b,c} is in IsoPairs[a].
                if frozenset((b,c)) in IsoPairs[a]:
                    return False
    return True

def _is_maximal(S):
    """Check if set S is maximal."""
    S_set = set(S)
    for p in POINTS:
        if p in S_set:
            continue
        # Check if p is covered by S
        # p is covered if there exists pair {a,b} in S such that {a,b} is in IsoPairs[p]
        # Note: a,b must be distinct and distinct from p.
        # Since p not in S, a,b are distinct from p automatically.
        # a,b must be in S.
        covered = False
        # Iterate through pairs in S? Or iterate through IsoPairs[p] and check if both are in S?
        # IsoPairs[p] can be large.
        # Let's check pairs in S.
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i+1, len(S_list)):
                a, b = S_list[i], S_list[j]
                if frozenset((a,b)) in IsoPairs[p]:
                    covered = True
                    break
            if covered:
                break
        if not covered:
            return False
    return True

def _greedy_covering():
    """Construct a maximal set using greedy covering."""
    S = set()
    # We need to ensure S is no-isosceles.
    # We also need to cover all points.
    
    # Initial state: S is empty.
    # We can't cover anything with |S| < 2.
    
    # Let's pick the first point randomly? Or the one that has the most potential?
    # Potential of p: number of pairs in IsoPairs[p].
    # If we pick p, it doesn't cover anything yet.
    
    # Better: Pick a pair (a,b) first.
    # Which pair is best?
    # A pair (a,b) covers a point p if {a,b} in IsoPairs[p].
    # So we want a pair that covers many points.
    
    # Calculate coverage of each pair.
    pair_coverage = {}
    for a in POINTS:
        for b in POINTS:
            if a == b:
                continue
            pair = frozenset((a,b))
            if pair in pair_coverage:
                continue
            # Count how many p have this pair in IsoPairs[p]
            count = 0
            for p in POINTS:
                if p == a or p == b:
                    continue
                if pair in IsoPairs[p]:
                    count += 1
            pair_coverage[pair] = count
            
    # Sort pairs by coverage
    sorted_pairs = sorted(pair_coverage.items(), key=lambda x: x[1], reverse=True)
    
    # Randomize the top pairs to avoid local minima
    top_pairs = sorted_pairs[:20]
    random.shuffle(top_pairs)
    
    for pair, _ in top_pairs:
        a, b = list(pair)
        # Start with S = {a, b}
        # Check if {a,b} is no-isosceles (trivially yes)
        S = {a, b}
        
        # Check if S is maximal
        if _is_maximal(S):
            return S
            
        # Greedy expansion
        # While not maximal:
        while not _is_maximal(S):
            # Find candidates c not in S such that S U {c} is no-isosceles
            candidates = []
            for c in POINTS:
                if c in S:
                    continue
                # Check if c can be added
                valid = True
                for a in S:
                    for b in S:
                        if a == b:
                            continue
                        if frozenset((a,b)) in IsoPairs[c]:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    candidates.append(c)
            
            if not candidates:
                # Stuck, break and try next initial pair
                break
                
            # Calculate gain for each candidate
            # Gain(c) = number of points p not in S U {c} that become covered
            # p is covered by S' if there is a pair in S' in IsoPairs[p].
            # Since S is not maximal, there are uncovered points.
            # Let U = set of points not in S that are not covered by S.
            # For c in candidates:
            #   S' = S U {c}
            #   p is newly covered if p was not covered by S, but is covered by S'.
            #   This happens if there is a pair {x,y} in S' such that {x,y} in IsoPairs[p].
            #   Since S was not covering p, no pair in S is in IsoPairs[p].
            #   So the new pair must involve c.
            #   So {c,x} in IsoPairs[p] for some x in S.
            
            # Precompute uncovered points for S
            uncovered = []
            for p in POINTS:
                if p in S:
                    continue
                # Check if p is covered by S
                covered = False
                S_list = list(S)
                for i in range(len(S_list)):
                    for j in range(i+1, len(S_list)):
                        if frozenset((S_list[i], S_list[j])) in IsoPairs[p]:
                            covered = True
                            break
                    if covered:
                        break
                if not covered:
                    uncovered.append(p)
            
            if not uncovered:
                # Should be maximal, but _is_maximal said no?
                # Maybe _is_maximal is slower or different?
                break
                
            best_c = None
            best_gain = -1
            best_cands = []
            
            for c in candidates:
                gain = 0
                for p in uncovered:
                    # Check if c helps cover p
                    # Need x in S such that {c,x} in IsoPairs[p]
                    for x in S:
                        if frozenset((c,x)) in IsoPairs[p]:
                            gain += 1
                            break
                if gain > best_gain:
                    best_gain = gain
                    best_cands = [c]
                elif gain == best_gain:
                    best_cands.append(c)
            
            if best_cands:
                c = random.choice(best_cands)
                S.add(c)
            else:
                # No gain, pick random candidate
                c = random.choice(candidates)
                S.add(c)
                
        # After loop, check if S is maximal
        if _is_maximal(S):
            return S
            
    return None

def construct():
    start_time = time.time()
    best_set = None
    best_size = float('inf')
    
    # Try multiple random initializations
    for _ in range(50):
        if time.time() > DEADLINE:
            break
            
        S = _greedy_covering()
        if S is None:
            continue
            
        if len(S) < best_size:
            best_size = len(S)
            best_set = set(S)
            
        # Local search: try to remove points
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            S_list = list(best_set)
            random.shuffle(S_list)
            for p in S_list:
                if time.time() > DEADLINE:
                    break
                S_temp = best_set - {p}
                if _is_maximal(S_temp):
                    best_set = S_temp
                    best_size -= 1
                    improved = True
                    break
                    
    if best_set is None:
        # Fallback
        # Just return a valid maximal set
        # Greedy random
        random.shuffle(POINTS)
        S = []
        S_set = set()
        for p in POINTS:
            if p in S_set:
                continue
            valid = True
            for a in S:
                for b in S:
                    if a == b:
                        continue
                    if frozenset((a,b)) in IsoPairs[p]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                S.append(p)
                S_set.add(p)
        return S
        
    return list(best_set)

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
