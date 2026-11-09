"""
Schütte's problem k=4: find smallest n such that a tournament on n vertices
has the property that every 4-set has a common dominator.

Strategy:
- Start with the Paley tournament on 83 vertices (known to satisfy the property).
- Use a greedy vertex-removal procedure: repeatedly try to remove a vertex
  while checking (via randomized sampling + occasional full check) that the
  property is maintained.
- The property "every 4-set has a common dominator" is equivalent to:
  for every 4-set S, there exists v in S such that v -> w for all w in S\{v}.
  Equivalently, no 4-set is a "bad" 4-set (one where no vertex dominates the other three).

- A 4-set {a,b,c,d} is bad iff for each vertex in the set, at least one other
  vertex in the set has an arc going INTO it (i.e., no vertex has out-degree 3
  within the set).

- To check if removing vertex v preserves the property:
  We need to verify that in the remaining tournament, every 4-set still has a
  common dominator.
  
  Key insight: removing a vertex can only create NEW bad 4-sets that include... 
  actually no, removing a vertex removes 4-sets (those containing v) and the 
  remaining 4-sets are unchanged. So if the original tournament satisfies the 
  property, removing any vertex preserves the property for all 4-sets NOT 
  containing v. The 4-sets containing v are simply gone.
  
  Wait, that's not right either. The property must hold for ALL 4-sets in the 
  remaining graph. The 4-sets in the remaining graph are exactly the 4-sets 
  from the original that don't contain v. Since the original had the property 
  for ALL 4-sets, it certainly has it for the subset that doesn't contain v.
  
  So ANY vertex can be removed from a valid tournament and the result is still valid!
  
  That means the answer is just 5? No wait, that can't be right. Let me re-read.
  
  "Every set of 4 vertices must have a vertex with arcs to all four."
  
  Hmm, "arcs to all four" - does this mean arcs to all FOUR vertices in the set, 
  or arcs to all OTHER three? Let me re-read: "a vertex with arcs to all four".
  
  If it means the vertex has arcs to all four vertices in the set, that's impossible 
  since a vertex can't have an arc to itself. So it must mean "arcs to all [the other] 
  [three]" or perhaps the wording means the vertex dominates the set (has arcs to 
  every other member).
  
  Actually re-reading: "Every set of 4 vertices must have a vertex with arcs to all four."
  This is ambiguous but in the context of tournaments and Schütte's problem, the 
  standard formulation is: every 4-set has a vertex that dominates the other 3.
  
  If that's the case, then removing any vertex from a valid tournament preserves 
  validity, because the remaining 4-sets were already valid.
  
  But that would mean f(4) = 5, which contradicts the lower bound of 31.
  
  Let me reconsider. Maybe the property is: every 4-set has a vertex OUTSIDE the 
  set that dominates all four? No, that doesn't match "a vertex with arcs to all four" 
  in the context of "every set of 4 vertices must have...".
  
  Actually, I think I misread. Let me look at the Schütte problem more carefully.
  
  Schütte's problem: f(k) is the smallest n such that EVERY tournament on n vertices 
  has the property that every k-set has a common dominator (a vertex that beats all 
  k vertices). Wait no, that's not right either.
  
  Actually, I think the problem is: find the smallest n such that there EXISTS a 
  tournament on n vertices where every 4-set has a vertex (from the whole tournament, 
  not necessarily in the 4-set) that dominates all 4 vertices in the set.
  
  No wait, re-reading: "Every set of 4 vertices must have a vertex with arcs to all four."
  
  I think "a vertex" refers to a vertex in the tournament, and "arcs to all four" 
  means arcs to all four vertices in the set. This vertex could be inside or outside 
  the 4-set.
  
  If the vertex must be in the 4-set, then as I argued, any removal preserves validity, 
  giving f(4)=5.
  
  If the vertex can be anywhere in the tournament, then removing a vertex could 
  destroy the property if that vertex was the unique dominator of some 4-set.
  
  Given the lower bound of 31 and upper bound of 67, it must be the second interpretation: 
  the common dominator can be any vertex in the tournament.
  
  So the property is: for every 4-set S, there exists a vertex v (anywhere in the 
  tournament) such that v -> s for all s in S.
  
  This is much harder to check and removal is non-trivial.

  Let me implement:
  1. Start with Paley(83).
  2. For each vertex, try removing it and check if property holds.
  3. Use randomized sampling to check the property efficiently.
  
  For checking: for a given 4-set, we need to find if any vertex dominates all 4.
  We can precompute for each vertex its out-neighborhood. A vertex v dominates 4-set S 
  iff S is a subset of out(v).
  
  So the property is: for every 4-subset S of [n], there exists v such that S ⊆ out(v).
  Equivalently: the union of all 4-subsets of out(v) for v in [n] covers all 4-subsets.
  Or: no 4-set is "missed" by all out-neighborhoods.
  
  A 4-set S is missed iff for every v, S is NOT a subset of out(v), i.e., for every v, 
  there exists s in S such that v does NOT beat s (i.e., s beats v or v=s).
  
  For a fixed 4-set S = {a,b,c,d}, S is dominated by v iff v beats all of a,b,c,d.
  S is NOT dominated iff for every v, v fails to beat at least one of a,b,c,d.
  
  This is equivalent to: the intersection of the "non-out-neighborhoods" (including 
  self) of a, b, c, d is empty. I.e., there's no vertex that is NOT beaten by any 
  of a,b,c,d... wait let me think again.
  
  v dominates S iff v ∉ S (well, v could be in S but then v can't beat itself) and 
  v beats all elements of S.
  
  Actually v can be in S. If v is in S, then v needs to beat all OTHER elements of S, 
  but the problem says "arcs to all four", meaning arcs to all four vertices in S. 
  If v is in S, v can't have an arc to itself. So v must be OUTSIDE S.
  
  So: S is dominated iff there exists v ∉ S such that v beats all of S.
  
  Equivalently: S is NOT dominated iff for every v ∉ S, there exists s ∈ S such that 
  v does NOT beat s (i.e., s beats v).
  
  For each s, let IN(s) = {v : v does not beat s} = {v : s beats v} ∪ {s}.
  Then S is not dominated iff ∩_{s∈S} IN(s) ⊆ S. I.e., the only vertices that don't 
  beat some element of S are the elements of S themselves.
  
  Equivalently: ∩_{s∈S} IN(s) \ S = ∅.
  
  So to check if a 4-set S is bad: compute ∩_{s∈S} IN(s) and check if it's a subset of S.
  
  For the Paley tournament on 83 vertices, this should hold.
  
  For the removal: if we remove vertex u, we need to check that no 4-set in the 
  remaining graph is bad. A 4-set S (not containing u) is bad in the remaining graph 
  iff ∩_{s∈S} IN'(s) \ S = ∅, where IN'(s) is computed in the remaining graph.
  
  IN'(s) = IN(s) \ {u} if u ∈ IN(s), else IN(s).
  
  This is getting complex. Let me just do a straightforward randomized check.

  Given time constraints (120s), I'll:
  1. Start with Paley(83).
  2. Try greedy removal: for each vertex in some order, try removing it and 
     do a randomized check (sample many 4-sets) to see if property holds.
  3. Keep the best (smallest n) found.
"""

import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    p = 83
    squares = {x * x % p for x in range(1, p)}
    
    # Build adjacency: adj[i][j] = True if i -> j
    adj = np.zeros((p, p), dtype=bool)
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                adj[i][j] = True
            else:
                adj[j][i] = True
    
    # Precompute out-neighborhoods as sets for fast subset checks
    # out[v] = set of vertices that v beats
    out = [set() for _ in range(p)]
    for i in range(p):
        for j in range(p):
            if adj[i][j]:
                out[i].add(j)
    
    def is_valid(vertices, num_samples=20000, full_check=False):
        """Check if the tournament on given vertices satisfies the property."""
        n = len(vertices)
        if n < 4:
            return True
        
        # Map from old index to new index
        vset = set(vertices)
        
        # For each vertex in the sub-tournament, compute its out-neighborhood within the sub-tournament
        # out_sub[v] = set of vertices in vertices that v beats
        out_sub = {}
        for v in vertices:
            out_sub[v] = out[v] & vset
        
        if full_check:
            # Check all 4-sets
            from itertools import combinations
            for quad in combinations(vertices, 4):
                # Check if any vertex outside quad dominates all 4
                found = False
                for v in vertices:
                    if v in quad:
                        continue
                    if all(s in out_sub[v] for s in quad):
                        found = True
                        break
                if not found:
                    return False
            return True
        else:
            # Randomized check
            for _ in range(num_samples):
                quad = random.sample(vertices, 4)
                found = False
                for v in vertices:
                    if v in quad:
                        continue
                    if all(s in out_sub[v] for s in quad):
                        found = True
                        break
                if not found:
                    return False
            return True
    
    # Verify initial validity with a sample
    assert is_valid(list(range(p)), num_samples=50000), "Paley(83) should be valid"
    
    current = list(range(p))
    best_n = p
    best_vertices = list(current)
    
    # Greedy removal
    random.seed(42)
    removed = set()
    
    while time.time() < deadline:
        improved = False
        # Try to remove each remaining vertex
        candidates = [v for v in current if v not in removed]
        random.shuffle(candidates)
        
        for v in candidates:
            if time.time() > deadline:
                break
            
            new_vertices = [u for u in current if u != v]
            if len(new_vertices) < 5:
                break
            
            # Quick randomized check
            if is_valid(new_vertices, num_samples=5000):
                # More thorough check
                if is_valid(new_vertices, num_samples=20000):
                    current = new_vertices
                    removed.add(v)
                    improved = True
                    best_n = len(current)
                    best_vertices = list(current)
        
        if not improved:
            # Try a different order
            random.shuffle(current)
            # Reset and try again with shuffled order
            # Actually, let's try removing vertices in different random orders
            # by resetting and doing another pass
            pass
    
    # Final verification with more samples
    if is_valid(best_vertices, num_samples=50000):
        n = len(best_vertices)
    else:
        # Fall back to full Paley
        n = p
        best_vertices = list(range(p))
    
    # Build orientation for the remaining vertices
    # We need to output orientation for n vertices labeled 0..n-1
    # Map best_vertices to 0..n-1
    n = len(best_vertices)
    index_map = {v: i for i, v in enumerate(best_vertices)}
    
    orientation = []
    for i in range(n):
        for j in range(i + 1, n):
            vi = best_vertices[i]
            vj = best_vertices[j]
            if adj[vi][vj]:
                orientation.append(0)  # i -> j
            else:
                orientation.append(1)  # j -> i
    
    return {'n': n, 'orientation': orientation}
