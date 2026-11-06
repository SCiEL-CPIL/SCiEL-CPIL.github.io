import time
from math import gcd
import random

def construct():
    """
    Return a nonempty subset of [1, 255] containing 255,
    with gcd(a,b) > 1 for all distinct pairs. Maximize cardinality.
    """
    start_time = time.time()
    deadline = 50.0  # Keep 40s margin for safety

    # Candidate pool: numbers in [1, 255] that share a factor with 255
    # 255 = 3 * 5 * 17
    # x must be divisible by 3, 5, or 17
    candidates = []
    for i in range(1, 256):
        if i % 3 == 0 or i % 5 == 0 or i % 17 == 0:
            candidates.append(i)
    
    # Ensure 255 is in candidates
    if 255 not in candidates:
        candidates.append(255)
        
    n = len(candidates)
    idx_map = {val: i for i, val in enumerate(candidates)}
    
    # Precompute adjacency as sets of indices
    # neighbors[i] contains indices j such that gcd(candidates[i], candidates[j]) > 1
    neighbors = [set() for _ in range(n)]
    for i in range(n):
        vi = candidates[i]
        for j in range(i + 1, n):
            vj = candidates[j]
            if gcd(vi, vj) > 1:
                neighbors[i].add(j)
                neighbors[j].add(i)
    
    # Bron-Kerbosch algorithm with pivot selection
    # We want the maximum clique.
    # Since 255 is a universal vertex in this graph (connected to all candidates),
    # the maximum clique will always contain 255.
    
    best_clique = set()
    
    def bron_kerbosch(R, P, X):
        nonlocal best_clique
        if time.time() > deadline:
            return
        
        if not P and not X:
            if len(R) > len(best_clique):
                best_clique = set(R)
            return
        
        # Pivot selection: pick u in P U X with max |P intersect N(u)|
        pivot = None
        max_count = -1
        # Iterate over P U X
        # For efficiency, we can just iterate over X then P, or just pick first one?
        # Standard pivot: u in P U X maximizing |P ∩ N(u)|
        # If P is empty, X is not empty, we return (handled above)
        
        # Optimization: If X is empty, we can just pick any u in P?
        # No, pivot is to reduce branching.
        
        candidates_for_pivot = list(P) + list(X)
        # If candidates_for_pivot is large, this is expensive.
        # But N is small.
        
        for u in candidates_for_pivot:
            # Count neighbors in P
            count = len(P.intersection(neighbors[u]))
            if count > max_count:
                max_count = count
                pivot = u
        
        # If P is empty, we would have returned.
        # If pivot is None (should not happen if P not empty), pick arbitrary
        if pivot is None:
            pivot = next(iter(P))
            
        # Branch on P \ N(pivot)
        # For each v in P \ N(pivot):
        #   R = R U {v}
        #   P' = P ∩ N(v)
        #   X' = X ∩ N(v)
        #   BK(R, P', X')
        #   P = P \ {v}
        #   X = X U {v}
        
        # To optimize, iterate over P \ N(pivot)
        # We need to be careful with modifying P and X in place or creating copies.
        # Creating copies is safer but slower.
        
        # Get the set of vertices to branch on
        # P_not_N_pivot = P - neighbors[pivot]
        
        # We can iterate over a list to avoid modification issues during iteration
        # But we must update P and X after recursion.
        
        # Efficient implementation:
        # We iterate over a copy of P \ N(pivot)
        # But we also need to remove v from P and add to X after each step.
        
        # Let's use a list of vertices to visit
        # Actually, standard BK modifies P and X.
        
        # Let's try a simple recursive implementation first, relying on Python's speed for N~120.
        
        # To speed up pivot selection, we can precompute? No, P changes.
        
        # Iterate over vertices in P that are NOT neighbors of pivot
        # We need to collect them first because we modify P
        # Or we can just iterate over P and skip if in N(pivot)
        
        # Let's use a list to avoid "set changed size during iteration"
        # We will build a list of candidates
        # candidates_to_branch = [v for v in P if v not in neighbors[pivot]]
        
        # However, we also need to update P and X.
        
        # Let's use the standard algorithm structure.
        
        # Optimization: If P is small, just iterate.
        
        # We'll use a helper to get the list
        # Note: P and X are sets.
        
        # We iterate over a snapshot of P \ N(pivot)
        # But we must remove v from P and add to X after recursion.
        
        # Let's do this:
        # 1. Determine the set of vertices to branch on: V_branch = P - neighbors[pivot]
        # 2. Iterate over V_branch
        # 3. For each v:
        #    R.add(v)
        #    P_new = P & neighbors[v]
        #    X_new = X & neighbors[v]
        #    bron_kerbosch(R, P_new, X_new)
        #    R.remove(v)
        #    P.remove(v)
        #    X.add(v)
            
        # This is O(|V_branch| * (cost of set ops)).
        
        # To make it faster, we can order vertices?
        # Maybe not needed.
        
        # Let's implement this.
        
        # One issue: Recursion depth. Max clique size ~100. Depth 100 is fine.
        
        # Let's try to optimize the pivot selection.
        # If we pick pivot, we branch on P \ N(pivot).
        
        # Implementation detail:
        # We need to pass P and X by value (copies) or modify and revert.
        # Modifying and reverting is faster.
        
        # But P_new and X_new are subsets.
        
        # Let's write the recursive function.
        
        # We'll use a global time check.
        
        # To avoid deep recursion overhead, we might use iterative? No, BK is naturally recursive.
        
        # Let's add a heuristic: if len(P) == 0, return.
        
        # We'll implement the branching.
        
        # Get list of vertices to branch
        # We must be careful: P is modified.
        
        # We'll create a list of vertices to process
        # But we need to remove them from P one by one.
        
        # Let's just iterate over a list of P, and check if in N(pivot).
        # If not in N(pivot), process it.
        # But if we remove it from P, the list is stale?
        # We can just collect the vertices first.
        
        # However, the set of vertices to branch on depends on pivot.
        
        # Let's collect:
        # branch_vertices = [v for v in P if v not in neighbors[pivot]]
        
        # But P might be large.
        
        # Let's try.
        
        # One more optimization:
        # If X is not empty, we can use it to prune?
        # BK handles this.
        
        # Let's implement.
        
        # We'll use a nonlocal flag for timeout
        # Already have time check at start.
        
        # To speed up, we can use bitsets?
        # Python sets are okay for N=120.
        
        # Let's try to improve pivot selection.
        # Instead of checking all P U X, we can just check P?
        # Theorem: Pivot can be in P U X.
        # If we pick u in P, we branch on P \ N(u).
        # If we pick u in X, we branch on P \ N(u).
        # Usually picking u in P U X with max degree in P is best.
        
        # Let's stick to the standard.
        
        # We'll use a simple list comprehension for branch vertices.
        
        # To avoid creating huge lists, we can iterate over P and filter.
        
        # But we need to remove v from P.
        
        # Let's use a while loop?
        
        # Actually, the standard implementation:
        # for v in P \ N(pivot):
        #    ...
        #    P.remove(v)
        #    X.add(v)
        
        # In Python, you can't modify a set while iterating.
        # So we must create a list.
        
        # Let's do that.
        
        # Optimization: If len(P) is small, it's fast.
        
        # Let's go.
        
        # We'll use a global variable to track if we should stop
        # Already done via time.time()
        
        # Let's implement the inner logic.
        
        # We'll use a helper function to check time
        # Already inline.
        
        # Let's write the code.
        
        # We'll use a list of vertices to branch on
        # But we need to be careful: the set of vertices to branch on is P \ N(pivot)
        # This set is fixed at the start of the recursion step.
        
        # So:
        # branch_list = list(P.difference(neighbors[pivot]))
        
        # Then iterate.
        
        # But wait, if P is large, this is expensive.
        
        # Alternative:
        # Iterate over P, and if v not in N(pivot), process.
        # But we need to remove v from P.
        
        # We can just collect all such v in a list first.
        
        # Let's do that.
        
        # Also, we can order branch_list to find large cliques faster?
        # Maybe sort by degree in P?
        # Higher degree vertices are more likely to be in large cliques.
        # Let's try sorting by len(P & neighbors[v]) descending.
        
        # This might help find the maximum clique earlier, allowing pruning?
        # BK doesn't prune based on size unless we add a bound.
        # We can add a bound: if len(R) + len(P) <= len(best_clique), return.
        
        # This is a crucial optimization!
        
        # Add this bound.
        
        # Revised BK:
        # if len(R) + len(P) <= len(best_clique): return
        # if not P and not X: update best
        # ...
        
        # This will prune a lot.
        
        # Let's implement this version.

        # Pivot selection
        # We can also optimize pivot selection by only looking at P?
        # If P is not empty, picking u in P is usually good.
        
        # Let's just use the full P U X.
        
        # Code:
        
        # 1. Bound check
        if len(R) + len(P) <= len(best_clique):
            return
            
        # 2. Base case
        if not P and not X:
            if len(R) > len(best_clique):
                best_clique = set(R)
            return
            
        # 3. Pivot
        pivot = None
        max_count = -1
        # Iterate over P U X
        # To speed up, we can just pick a random one? No, pivot matters.
        # Let's iterate over P first, then X?
        # Or just list(P) + list(X)
        
        # If P is empty, X is not, we return (handled above)
        
        # Let's iterate over P U X
        # We can use a set union? No, just iterate.
        
        # Optimization: If P is large, this is slow.
        # But N is small.
        
        for u in P:
            count = len(P.intersection(neighbors[u]))
            if count > max_count:
                max_count = count
                pivot = u
                
        # If P is empty, pivot is None. But we handled not P and not X.
        # If P is empty and X is not, we return.
        # So if we are here, P is not empty?
        # Wait, if P is empty and X is not empty, we return.
        # So P is not empty.
        # So pivot will be found in P.
        
        # We don't need to check X for pivot if P is not empty?
        # Theorem says pivot in P U X.
        # But picking from P is sufficient and often better?
        # Actually, picking from X can be better if it has high degree in P.
        # But let's stick to P for now, or check both.
        
        # Let's check X as well to be safe.
        for u in X:
            count = len(P.intersection(neighbors[u]))
            if count > max_count:
                max_count = count
                pivot = u
                
        if pivot is None:
            pivot = next(iter(P))
            
        # 4. Branch
        # Branch on P \ N(pivot)
        
        # Get list of vertices
        # We need to remove them from P and add to X.
        
        # We'll create a list of vertices to branch on
        # But we need to be careful: we are modifying P.
        
        # Let's collect the vertices first.
        # branch_vertices = [v for v in P if v not in neighbors[pivot]]
        
        # To improve, sort by degree in P descending?
        # Let's try.
        
        # But computing degree for all is expensive.
        # Let's just use the natural order or random?
        # Random might help avoid bad ordering.
        
        # Let's use random.shuffle on the list?
        # No, deterministic is better for reproducibility, but we want max clique.
        
        # Let's just iterate.
        
        # To avoid "set changed size", we iterate over a list.
        
        # We'll create a list of candidates
        # But we need to remove them from P one by one.
        
        # Let's do this:
        # We'll iterate over a copy of P, but only process those not in N(pivot).
        # And remove from P.
        
        # Actually, we can just do:
        # for v in list(P):
        #    if v in neighbors[pivot]: continue
        #    ... process v ...
        #    P.remove(v)
        #    X.add(v)
            
        # This works.
        
        for v in list(P):
            if v in neighbors[pivot]:
                continue
                
            # Recurse
            R.add(v)
            P_new = P.intersection(neighbors[v])
            X_new = X.intersection(neighbors[v])
            
            bron_kerbosch(R, P_new, X_new)
            
            R.remove(v)
            P.remove(v)
            X.add(v)
            
            # Check time
            if time.time() > deadline:
                return

    # Initialize
    # R = {255}?
    # No, we established that Max Clique in V contains 255.
    # So we can just find Max Clique in V.
    # R = set()
    # P = set(range(n))
    # X = set()
    
    # But wait, if we start with R=empty, we might find a clique that doesn't contain 255?
    # No, because 255 is a universal vertex.
    # Any maximal clique must contain 255?
    # Yes. If C is a clique, and 255 not in C, then C U {255} is a larger clique.
    # So the maximum clique MUST contain 255.
    
    # So we can just run BK on the whole graph.
    
    R = set()
    P = set(range(n))
    X = set()
    
    bron_kerbosch(R, P, X)
    
    # If timeout, best_clique might be empty or partial.
    # We need a fallback.
    # The multiples of 3 is a good fallback.
    
    if not best_clique:
        # Fallback to multiples of 3
        init_set_idx = set()
        for i, v in enumerate(candidates):
            if v % 3 == 0:
                init_set_idx.add(i)
        best_clique = init_set_idx
        
    # Ensure 255 is in the result (it should be, but just in case)
    idx_255 = idx_map[255]
    if idx_255 not in best_clique:
        # If not, add it?
        # But it must be a clique.
        # If 255 is not in best_clique, it means best_clique is not maximal?
        # Or BK failed.
        # Since 255 is universal, we can just add it if it's a clique?
        # If best_clique is a clique, then best_clique U {255} is a clique.
        # So we can always add 255.
        best_clique.add(idx_255)
        
    result_vals = [candidates[i] for i in best_clique]
    
    # Final verification
    for i in range(len(result_vals)):
        for j in range(i+1, len(result_vals)):
            if gcd(result_vals[i], result_vals[j]) == 1:
                # Fallback
                init_set_idx = set()
                for i, v in enumerate(candidates):
                    if v % 3 == 0:
                        init_set_idx.add(i)
                result_vals = [candidates[i] for i in init_set_idx]
                break
                
    return result_vals
