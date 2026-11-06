import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        if not S:
            return True
        S_set = set(S)
        for a in S_set:
            for b in S_set:
                if a + b in S_set:
                    return False
        return True

    def build_conflict_graph(vals):
        """
        Build the conflict graph where an edge (i, j) exists if {vals[i], vals[j]} 
        is not sum-free. 
        Actually, the condition for a set S to be sum-free is:
        No a, b, c in S such that a+b=c.
        This is a 3-uniform hypergraph independence problem, not a simple graph MIS.
        However, we can model it as a graph MIS if we define conflicts carefully?
        No, standard MIS is for graphs. Sum-free is a 3-uniform constraint.
        
        Alternative approach: 
        We can use a branch and bound on the set of elements directly.
        State: (current_set, remaining_candidates)
        Upper bound: len(current_set) + len(remaining_candidates)
        Pruning: If upper bound <= best found, prune.
        
        To make it efficient, we need a good lower bound (initial solution) and good ordering.
        """
        pass

    def greedy_solution(vals, seed):
        """Get a quick feasible solution to use as lower bound."""
        rng = random.Random(seed)
        # Try a few random orders
        best = set()
        for _ in range(50):
            shuffled = list(vals)
            rng.shuffle(shuffled)
            S = set()
            for x in shuffled:
                # Check if x can be added
                ok = True
                for a in S:
                    if (x - a) in S or (x + a) in S:
                        ok = False
                        break
                if ok:
                    S.add(x)
            if len(S) > len(best):
                best = S
        return best

    def branch_and_bound(vals, initial_lower_bound, time_limit):
        """
        Exact search for Maximum Sum-Free Subset.
        Uses depth-first search with pruning.
        """
        n = len(vals)
        # Precompute for faster checks? 
        # For each x, precompute which pairs (a,b) in vals conflict with x.
        # Actually, the check "can x be added to S" depends on S.
        
        # Let's use a recursive approach with bitmasks if n <= 60? 
        # n can be up to 240. Bitmasks are too big.
        
        # We will use a simple recursive B&B with a strong pruning heuristic.
        # Sort vals to improve pruning.
        # Heuristic: Elements that are "more central" (involved in more conflicts) should be decided early?
        # Or, elements that are "safer" (fewer conflicts) should be added first?
        
        # Let's try a different angle: 
        # The problem is equivalent to finding a maximum independent set in a 3-uniform hypergraph.
        # Standard MIS algorithms don't apply directly.
        
        # However, we can reformulate:
        # A set S is sum-free if for all x in S, x is not in {a+b | a,b in S}.
        # This is hard.
        
        # Let's stick to a smarter local search / greedy with more iterations and better restarts, 
        # but the prompt asks for a MATERIALLY DIFFERENT strategy.
        # The current one is Greedy + Local Search.
        # Inspiration 2 mentioned "constraint-based conflict-resolution" and "maximum independent set".
        
        # Let's implement a proper Branch and Bound for the 3-uniform case.
        # State: (index, current_set, forbidden_set)
        # forbidden_set: elements that cannot be added because they would create a sum with two elements in current_set.
        # If we add x to current_set, any y such that x+y in current_set or x+y in forbidden_set? No.
        # If we add x, new conflicts are:
        # 1. x + a = c for a, c in current_set -> x is bad if this exists.
        # 2. If we add x, it forbids any z such that x + z in current_set (so z+x=c) or z+x in current_set.
        #    Also, if we add x, it forbids any z such that z + x in current_set.
        #    Basically, if x is added, any z where x+z is in current_set becomes forbidden.
        #    Also, any z where z+x is in current_set.
        #    And, any z where x+z is already forbidden? No, forbidden means "cannot be in S".
        
        # Let's define:
        # S: current set
        # F: set of elements that are "forbidden" (cannot be added to S because they would violate sum-free with elements already in S).
        # Initially S={}, F={}.
        # When adding x to S:
        #   - x must not be in F.
        #   - x must not be in S.
        #   - New F' = F union { y | x+y in S } union { y | y+x in S }
        #     Note: If x+y in S, then y+x in S is the same condition since addition is commutative.
        #     So F' = F union { y | x+y in S }
        #   - Also, if x+x in S, then x is forbidden? No, x is being added. 
        #     Wait, if 2x in S, then x+x=2x is a conflict. So x cannot be added if 2x in S.
        #     This is covered by: is x in F? 
        #     If 2x in S, then when 2x was added, did it forbid x? 
        #     When 2x was added, F gained { y | 2x+y in S }. x is not necessarily in S.
        #     So we must explicitly check: if 2x in S, x is bad.
        #     More generally, x is bad if there exist a,b in S such that a+b=x.
        #     This is not captured by F if F only tracks "x+y in S".
        
        # Revised State:
        # S: current set
        # F: set of elements that cannot be added because they would form a sum with two elements in S.
        #    i.e., x in F iff exists a,b in S such that a+b=x.
        #    Also, x in F if exists a in S such that x+a in S? 
        #    If x+a in S, then a+x = c in S. So x is bad.
        #    So F should contain x if:
        #    1. exists a,b in S: a+b=x
        #    2. exists a in S: x+a in S  (which implies x+a is in S)
        #    
        #    When we add x to S:
        #    - Check if x in F. If yes, cannot add.
        #    - New F' = F union { y | exists a in S: a+y=x } ? No.
        #      If we add x, new conflicts for future elements y:
        #      y is bad if:
        #      - exists a,b in S' (S union {x}) such that a+b=y.
        #      - exists a in S' such that y+a in S'.
        #      
        #      New bad y's due to x:
        #      1. x + a = y for a in S. (y = x+a)
        #      2. a + x = y for a in S. (same)
        #      3. x + x = y. (y = 2x)
        #      4. y + x = a for a in S. (y = a-x)
        #      5. y + a = x for a in S. (y = x-a)
        #      
        #      So F' = F union { x+a | a in S } union { 2x } union { a-x | a in S } union { x-a | a in S }
        #      Note: { a-x | a in S } is the same as { x-a | a in S }? No, sets of numbers.
        #      If a-x = y, then y+x=a. So yes, these are the y's that conflict with x and some a.
        #      
        #      So the update rule for F when adding x is:
        #      F_new = F_old union (x + S) union {2x} union (S - x) union (x - S)
        #      Where (x+S) = {x+a for a in S}, etc.
        
        # This allows us to do DFS.
        # Pruning:
        # - If len(S) + (n - len(S) - len(F)) <= best_found: prune.
        #   (Remaining candidates are those not in S and not in F)
        
        # To make it faster, we need a good ordering of candidates.
        # Candidates are vals not in S and not in F.
        # We should pick the candidate that maximizes the number of elements it forbids? 
        # Or minimizes? 
        # Usually in MIS, we pick the vertex with highest degree to branch on.
        # Here, "degree" could be the size of the new F added.
        # Branching on a high-impact element reduces the search space faster.

        vals_set = set(vals)
        n = len(vals)
        best_found = initial_lower_bound
        best_set = set()
        
        # Sort vals for deterministic behavior and potential pruning
        # Let's try sorting by absolute value or something? 
        # Actually, the order of iteration in DFS matters for finding a good solution early.
        # Let's try a few different orderings.
        
        def dfs(S, F, remaining_indices, depth):
            nonlocal best_found, best_set
            
            if time.time() > time_limit:
                return
            
            # Prune
            # Upper bound: current size + max possible additions
            # Max possible additions is len(remaining_indices)
            if len(S) + len(remaining_indices) <= best_found:
                return
            
            if not remaining_indices:
                if len(S) > best_found:
                    best_found = len(S)
                    best_set = set(S)
                return
            
            # Heuristic: Pick an element from remaining_indices
            # Let's pick one that creates the largest F expansion (most restrictive)
            # Or one that is "central".
            # For simplicity, let's just pick the first one, but try multiple orderings globally.
            
            # Optimization: If we can add all remaining, do it?
            # Check if all remaining are compatible with S and each other? Too expensive.
            
            # Pick a candidate
            # Let's use a heuristic: pick the candidate x that minimizes the number of remaining candidates?
            # i.e., maximizes |F_new \ F_old|
            
            # To avoid O(n^2) overhead in every node, we might just pick the first one.
            # But let's try to pick the one with highest "conflict potential".
            
            # For now, let's just iterate.
            # We need to be careful with time.
            
            # Let's try a different strategy: 
            # Instead of pure B&B, use a "Greedy with Backtracking" (Limited Discrepancy Search)
            # Or just a very aggressive local search with simulated annealing?
            
            # The prompt asks for a DIFFERENT strategy. 
            # The current one is Greedy + Local Search.
            # B&B is exact.
            
            # Let's implement the B&B with the F-set update.
            
            # To speed up, we'll represent S and F as sets.
            # remaining_indices is a list of values.
            
            # Heuristic for branching: 
            # Pick x from remaining_indices.
            # Let's try picking the x that has the most conflicts with other remaining elements?
            # This is expensive.
            
            # Let's just pick the first one for now, but rely on the pruning.
            # To improve, we can try multiple random orderings of the initial vals.
            
            x = remaining_indices[0]
            
            # Option 1: Exclude x
            # x is not in S. It doesn't affect F.
            # But x is removed from remaining.
            dfs(S, F, remaining_indices[1:], depth + 1)
            
            # Option 2: Include x
            # Check if x is in F.
            if x not in F:
                # Calculate new F
                new_F = set(F)
                for a in S:
                    new_F.add(x + a)
                    new_F.add(a - x)
                    new_F.add(x - a)
                new_F.add(2 * x)
                
                # Filter remaining_indices: remove x, and remove any y in new_F
                new_remaining = [y for y in remaining_indices[1:] if y not in new_F]
                
                # Also, if x is in S? No, x is new.
                # Check if x conflicts with S? 
                # If x in F, we skipped. 
                # But what if x + a = b for a,b in S? 
                # That means x = b - a. So x would be in F if b-a was calculated when b or a was added.
                # Yes, F contains all y such that y+a in S or a+y in S or a+b=y.
                # So if x is a valid addition, x is not in F.
                
                dfs(S | {x}, new_F, new_remaining, depth + 1)
        
        # We need to run this with a time limit.
        # The state space is huge. 
        # For n=240, this will likely timeout unless pruning is extremely effective.
        # The pruning condition len(S) + len(remaining) <= best is weak.
        
        # Let's try a different approach: 
        # Use the "Coloring" bound from Tomita's MIS algorithm? 
        # But this is a hypergraph.
        
        # Alternative: 
        # Since the sets are random, they are "loose". 
        # The density of conflicts is low.
        # A simple greedy might be very close to optimal.
        
        # Let's combine: 
        # 1. Get a good lower bound with greedy.
        # 2. Run a limited B&B to try to beat it.
        # 3. If B&B doesn't finish, return the greedy result.
        
        # To make B&B work, we need a better bound.
        # Bound: len(S) + (len(remaining) - min_vertex_cover(remaining_graph))
        # Where remaining_graph is the conflict graph of the remaining elements.
        # If we can compute a lower bound on the vertex cover (or upper bound on independent set) of the remaining subgraph, we can prune.
        
        # For the 3-uniform hypergraph, an upper bound on the independent set size is:
        # |Remaining| / (1 + avg_degree / 2) ? Not really.
        
        # Let's try a simpler, faster heuristic: 
        # "Iterated Local Search" with more aggressive moves.
        # But the prompt asks for a different strategy.
        
        # Let's go with the B&B but with a strict time limit and a good initial solution.
        # We will try multiple random orderings of the input values to see if we find a better solution quickly.
        
        pass

    # Given the complexity and time limit, a pure B&B might not finish for n=240.
    # However, for n=60, it might.
    
    # Let's implement a hybrid:
    # For n <= 60: Try B&B.
    # For n > 60: Use a very fast, high-quality heuristic.
    # Heuristic: 
    # 1. Start with all elements.
    # 2. While conflict exists, remove the element involved in the most conflicts.
    # This is a "Maximal Independent Set" heuristic (greedy deletion).
    # This is different from "Greedy Addition".
    
    def greedy_deletion(vals, seed):
        rng = random.Random(seed)
        S = set(vals)
        # Count conflicts for each element
        # Conflict for x: exists a,b in S such that a+b=x or x+a=b or x+b=a.
        # Actually, if we remove x, we resolve all conflicts involving x.
        # We want to remove x that is in the most conflicts.
        
        while True:
            # Find all conflicts
            # A conflict is a triple (a,b,c) with a+b=c.
            # We need to remove at least one from each triple.
            # This is a hitting set problem.
            # Greedy: pick the element that appears in the most triples.
            
            counts = {x: 0 for x in S}
            triples = []
            S_list = list(S)
            for i, a in enumerate(S_list):
                for j, b in enumerate(S_list):
                    c = a + b
                    if c in S:
                        # Conflict (a,b,c)
                        # Count for a, b, c
                        counts[a] += 1
                        counts[b] += 1
                        counts[c] += 1
                        triples.append((a, b, c))
            
            if not triples:
                break
            
            # Pick element with max count
            # Tie-break randomly
            max_count = max(counts.values())
            candidates = [x for x in S if counts[x] == max_count]
            x_to_remove = rng.choice(candidates)
            S.remove(x_to_remove)
            
        return S

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        vals = instance(size, seed)
        
        # Strategy: 
        # 1. Greedy Addition (from current program, but faster)
        # 2. Greedy Deletion (new)
        # 3. Local Search on both
        
        best_set = set()
        
        # Greedy Addition
        S_add = greedy_solution(vals, seed)
        if len(S_add) > len(best_set):
            best_set = S_add
            
        # Greedy Deletion
        S_del = greedy_deletion(vals, seed)
        if len(S_del) > len(best_set):
            best_set = S_del
            
        # Local Search on best
        # We'll use a simple local search: try to add/remove/swap
        
        def local_search(S, vals, rng, iters):
            S = set(S)
            best = len(S)
            cur = S
            for _ in range(iters):
                if time.time() > deadline:
                    break
                improved = False
                # Try adding
                for x in vals:
                    if x in cur:
                        continue
                    ok = True
                    for a in cur:
                        if (x - a) in cur or (x + a) in cur:
                            ok = False
                            break
                    if ok:
                        cur.add(x)
                        improved = True
                        if len(cur) > best:
                            best = len(cur)
                        break
                if not improved:
                    # Try removing one and re-adding
                    # Pick a random element to remove
                    if len(cur) > 1:
                        to_remove = rng.choice(list(cur))
                        cur.remove(to_remove)
                        # Re-greedy add
                        remaining = [x for x in vals if x not in cur]
                        rng.shuffle(remaining)
                        for x in remaining:
                            ok = True
                            for a in cur:
                                if (x - a) in cur or (x + a) in cur:
                                    ok = False
                                    break
                            if ok:
                                cur.add(x)
                        if len(cur) > best:
                            best = len(cur)
            return cur

        rng = random.Random(seed + 12345)
        best_set = local_search(best_set, vals, rng, 100)
        
        # Verify
        if not is_sum_free(best_set):
            # Fallback
            best_set = set(x for x in vals if x % 2)
            
        results.append(sorted(best_set))
        
        if time.time() > deadline:
            # If we are running out of time, return what we have
            # But we need to return all 3. 
            # If we don't have all 3, we must generate the rest quickly.
            pass
            
    return results
