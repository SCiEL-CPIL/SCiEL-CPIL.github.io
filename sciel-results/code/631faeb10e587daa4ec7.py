import time

def construct():
       start_time = time.time()
       deadline = start_time + 55 # 5s margin
       
       n = 256
       # Precompute conflicts
       # conflict[u] = set of v such that (u+v) | uv
       conflicts = [set() for _ in range(n+1)]
       for u in range(1, n+1):
           for v in range(u+1, n+1):
               if (u*v) % (u+v) == 0:
                   conflicts[u].add(v)
                   conflicts[v].add(u)
                   
       # Initial set: Odds
       S = set(range(1, n+1, 2))
       
       # Greedy add evens that don't conflict with S
       for e in range(2, n+1, 2):
           if not conflicts[e].intersection(S):
               S.add(e)
               
       # Local Search
       # Strategy: Try to add a random even, removing necessary odds if beneficial?
       # Or just try to add any node if free.
       # Since we added all free evens, no free nodes remain.
       # We need to swap.
       
       # Let's try to find a set of evens E to add, and odds O to remove.
       # Simplified: Pick a random even e not in S.
       # Let C = conflicts[e] & S.
       # If we remove C, can we add e? Yes.
       # Is it worth it? Only if we can add MORE than |C| nodes.
       # This is complex.
       
       # Alternative: Randomized Local Search
       # Move: Remove a random node u from S, then greedily add as many nodes as possible?
       # Or: Pick random u in S, remove it. Then try to add random v not in S if free.
       
       # Better: "2-opt" style.
       # Pick u in S, v not in S.
       # If we swap u for v, is it valid?
       # v must not conflict with S \ {u}.
       # i.e., conflicts[v] & (S - {u}) is empty.
       # This keeps size same. We want to INCREASE size.
       
       # To increase size, we need to add k nodes and remove < k nodes.
       # Since no node can be added for free (greedy exhausted), we must remove at least 1 to add 1.
       # So we need to add 2 and remove 1, or add 1 and remove 0 (impossible).
       
       # So look for a pair of evens {e1, e2} and a set of odds O such that:
       # 1. e1, e2 not in S.
       # 2. e1, e2 don't conflict with each other.
       # 3. Conflicts of e1, e2 with S are subset of O.
       # 4. |O| < 2.
       # If |O|=0, greedy would have added them.
       # If |O|=1, say {o}, then removing o allows adding e1, e2. Net gain +1.
       
       # Let's search for such triples (e1, e2, o).
       
       best_S = S
       
       while time.time() < deadline:
           found_improvement = False
           # Randomly sample evens
           evens_not_in_S = [e for e in range(2, n+1, 2) if e not in S]
           if not evens_not_in_S:
               break
               
           # Try pairs of evens
           # To be fast, limit pair checks
           import random
           random.seed(42) # Fixed seed for reproducibility? No, random is fine.
           
           # Maybe just iterate through all pairs of evens not in S?
           # Count of evens not in S is ~120. Pairs ~7000.
           # For each pair, check conflicts.
           
           for i in range(len(evens_not_in_S)):
               e1 = evens_not_in_S[i]
               if time.time() > deadline: break
               for j in range(i+1, len(evens_not_in_S)):
                   e2 = evens_not_in_S[j]
                   
                   # Check e1, e2 conflict
                   if e2 in conflicts[e1]:
                       continue
                       
                   # Check conflicts with S
                   c1 = conflicts[e1] & S
                   c2 = conflicts[e2] & S
                   common_conflicts = c1 | c2
                   
                   # We want to remove a small set O such that common_conflicts is covered.
                   # Ideally |common_conflicts| == 1.
                   if len(common_conflicts) == 1:
                       o = next(iter(common_conflicts))
                       # Check if removing o allows adding e1, e2
                       # i.e., c1 - {o} is empty and c2 - {o} is empty
                       # This is true by definition of common_conflicts.
                       
                       # Check if e1, e2 conflict with any other in S? No, c1,c2 are all conflicts.
                       
                       # Perform swap
                       S.remove(o)
                       S.add(e1)
                       S.add(e2)
                       found_improvement = True
                       break
                   elif len(common_conflicts) == 0:
                       # Should have been added by greedy, but maybe not if S changed?
                       S.add(e1)
                       S.add(e2)
                       found_improvement = True
                       break
                   
               if found_improvement:
                   break
                   
           if not found_improvement:
               # Try single additions? No, none free.
               # Try removing 2 odds to add 3 evens?
               # Or just break if no improvement found in this pass?
               # Let's try a random restart or different heuristic if stuck.
               # For now, if no pair improvement, break.
               break
               
       return list(S)
