"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
from collections import defaultdict

N = 256

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    # Precompute conflict graph
    # conflict[u] is a set of v such that {u, v} is NOT a Sidon set pair
    # A pair (u, v) is a conflict if:
    # 1. 2u == 2v (impossible for distinct u,v)
    # 2. 2u == v + w for some w in set? No, this is pairwise.
    # In a Sidon set, all sums a+b (a<=b) must be distinct.
    # If we add u and v, the sums are 2u, 2v, u+v.
    # These must be distinct from each other (always true if u!=v) and from existing sums.
    # For the graph formulation of "cannot coexist":
    # Two numbers u and v cannot be in the same Sidon set if they create a sum collision with EACH OTHER?
    # No, u and v can coexist if 2u != 2v, 2u != u+v, 2v != u+v.
    # 2u = u+v => u=v.
    # So any pair (u, v) with u != v is "locally compatible" in terms of their own sums.
    # The conflict arises when adding u and v to a SET S creates a collision with sums in S.
    # This is a hypergraph problem, not a simple graph MIS.
    
    # However, the prompt suggests: "Model the problem as a Maximum Independent Set (MIS) on a conflict graph 
    # where vertices are numbers 1–256 and edges connect elements that cannot coexist in a Sidon set"
    
    # Wait, if any pair can coexist locally, the graph has no edges?
    # That's not right. The Sidon condition is global.
    # Let's re-read the inspiration: "edges connect elements that cannot coexist in a Sidon set".
    # If we just pick 2 numbers, they are always a Sidon set (unless u=v).
    # If we pick 3 numbers u, v, w:
    # Sums: 2u, 2v, 2w, u+v, u+w, v+w.
    # Collision if any two are equal.
    # e.g., 2u = v+w. Then u, v, w cannot all be in the set.
    # This is a 3-uniform hypergraph of conflicts.
    
    # The MIS approach on a graph usually applies to problems like "no two adjacent vertices".
    # Here, the constraint is "no sum collision".
    # We can approximate this or use a different graph structure.
    # Actually, a simpler interpretation for "conflict graph" in this context might be:
    # If we fix a set S, an element x conflicts with S if it creates a sum collision.
    
    # Let's stick to the prompt's specific directive: 
    # "greedy vertex elimination heuristic that iteratively removes the vertex with the highest degree 
    # (most conflicts) to force the remaining set into a small, inclusion-maximal independent set."
    
    # This sounds like we start with ALL vertices and remove them?
    # Or start with empty and add?
    # "removes the vertex with the highest degree... to force the REMAINING set"
    # This implies we start with the full set [1..N] and remove vertices until the remaining set is a valid Sidon set.
    # But [1..N] is definitely not Sidon.
    # If we remove vertices to make it Sidon, we will likely end up with a LARGE Sidon set (maximal cardinality), 
    # because we are removing the "worst" offenders. We want MINIMAL cardinality.
    
    # To get MINIMAL cardinality, we should be ADDING vertices carefully, or REMOVING vertices from a large valid set?
    # No, if we start with a valid large Sidon set and remove vertices, we lose maximality easily.
    
    # Let's reconsider the "Conflict Graph" for Sidon sets.
    # Maybe the graph is defined differently?
    # If we define the graph such that an edge exists between u and v if they CANNOT be in a minimal maximal set?
    
    # Let's try a different strategy that fits the "MIS" vibe but targets minimality:
    # We want a small set. 
    # Greedy construction:
    # 1. Start with empty set.
    # 2. Pick a vertex v.
    # 3. Add v to set.
    # 4. Remove all vertices that conflict with v (cannot be added later).
    # 5. Repeat on remaining vertices.
    # This creates a maximal independent set in the conflict hypergraph.
    # To minimize size, we want to pick vertices that "block" as many other candidates as possible (High Degree).
    # This is a standard heuristic for Minimum Dominating Set or similar, but here it's for Independent Set.
    # Wait, if I pick a vertex that blocks many others, I am left with fewer candidates.
    # If I pick a vertex that blocks few others, I have more candidates left, but I might need more vertices to cover the "space"?
    # Actually, for a Sidon set, picking "spread out" numbers is good.
    
    # Let's implement the "Greedy Vertex Elimination" as described:
    # 1. Start with candidate pool P = [1..N].
    # 2. While P is not empty:
    #    a. For each v in P, calculate "conflict degree" D(v) = number of u in P \ {v} such that {v, u} creates a sum collision with some third element?
    #    No, that's complex.
    
    # Let's look at the specific directive again: "removes the vertex with the highest degree (most conflicts) to force the remaining set into a small, inclusion-maximal independent set."
    # This phrasing is slightly ambiguous for a minimization problem.
    # If we remove high-degree vertices from the FULL set, we are left with a sparse set.
    # Is that sparse set Sidon? Not necessarily.
    # Is it maximal? Not necessarily.
    
    # Alternative interpretation:
    # We construct the set incrementally.
    # At each step, we look at the current "available" vertices (those that can be added to the current set).
    # We pick the one that "conflicts" the most with the remaining available vertices?
    # If we pick a vertex that conflicts with many others, we reduce the pool of available vertices more aggressively.
    # This might lead to a smaller final set because we "use up" the search space faster?
    
    # Let's try this:
    # 1. Current Set S = []
    # 2. Current Sums = set()
    # 3. Available = [1..N]
    # 4. While Available:
    #    a. For each v in Available, check if it can be added to S (valid Sidon).
    #    b. Let Valid = [v in Available | v can be added to S]
    #    c. If Valid is empty, break.
    #    d. For each v in Valid, calculate "blocking power": how many u in Available \ {v} would become INVALID if we added v?
    #       (i.e., adding v creates sum collisions that prevent u from being added later)
    #    e. Pick v with MAXIMUM blocking power.
    #    f. Add v to S. Update Sums. Remove v and all newly invalidated u from Available.
    # 5. Check if S is maximal. If not, we might need to repair?
    #    But by construction, if we stop when no valid v exists, it IS maximal.
    #    So this algorithm ALWAYS produces a maximal Sidon set.
    #    The goal is to minimize |S|.
    #    Picking the vertex that blocks the most others should theoretically lead to a smaller set?
    #    Let's test this logic.
    #    If I pick a vertex that blocks 100 others, I remove 100 candidates.
    #    If I pick a vertex that blocks 1 other, I remove 1 candidate.
    #    The first choice reduces the pool faster. It might allow me to "cover" the interval with fewer picks?
    #    Yes, this is analogous to Set Cover or Dominating Set heuristics where you pick the element covering the most uncovered items.
    
    # This strategy is:
    # 1. Greedy construction.
    # 2. Selection criterion: Maximize "conflict degree" (number of other available elements that become incompatible).
    # 3. This is a specific "Greedy Vertex Elimination" on the availability graph.
    
    # Complexity:
    # N=256.
    # Loop runs |S| times (~15-20).
    # Inside loop:
    #   Check validity for all available (O(N^2) sum checks).
    #   Calculate blocking for all valid (O(N^3) in worst case if we check pairs).
    #   20 * 256^3 is too slow (300M ops).
    #   Need optimization.
    
    # Optimization:
    # We can precompute sum conflicts.
    # When we add v, the new sums are {2v, v+u for u in S}.
    # A candidate w is blocked if any of its potential sums {2w, w+x for x in S} intersects with the NEW sums?
    # No, w is blocked if adding w to S \cup {v} causes a collision.
    # Collision if:
    # 1. 2w is in Sums_new
    # 2. w+x is in Sums_new for some x in S
    # 3. w+v is in Sums_old? No, w+v is a new sum.
    #    Wait, if we add v, Sums_new = Sums_old \cup {2v} \cup {v+s for s in S}.
    #    w is invalid if:
    #    - 2w in Sums_new
    #    - w+s in Sums_new for any s in S
    #    - w+v in Sums_old? No, w+v is not in Sums_old.
    #    - Is there a collision between w's sums and v's sums?
    #      2w = 2v => w=v (no)
    #      2w = v+s => w = (v+s)/2
    #      w+s = 2v => s = 2v-w
    #      w+s = v+s' => w = v + s' - s
    #    
    #    Actually, simpler:
    #    w is valid in S \cup {v} iff:
    #    1. w is valid in S (it was in Available)
    #    2. The set of sums {2w} U {w+s for s in S} U {w+v} has no intersection with Sums_old
    #    3. The new sums are distinct among themselves (always true)
    #    
    #    So w is blocked by v if:
    #    - 2w in Sums_old
    #    - w+s in Sums_old for some s in S
    #    - 2w = 2v (no)
    #    - 2w = v+s (i.e., 2w - v is in S)
    #    - w+s = 2v (i.e., 2v-w is in S)
    #    - w+s = v+s' (i.e., w - v = s' - s)
    #    
    #    This is getting complicated to calculate on the fly for all pairs.
    
    # Let's simplify. We just want a good heuristic.
    # Instead of exact blocking power, we can use a proxy:
    # "Density" of the number.
    # Or, we can just use the current SA approach but with a better initialization or move set?
    # The prompt asks for a DIFFERENT strategy.
    
    # Let's try the "Max Blocking" greedy but with a simplified metric.
    # Metric: For each valid candidate v, count how many other valid candidates u would become invalid if v is added.
    # We can approximate this by checking a random sample of u's?
    # Or, we can precompute a "conflict matrix" for the current state?
    
    # Given the time limit, let's implement the Greedy Max-Blocking with efficient checks.
    
    def get_sums(selected):
        sums = set()
        for i in range(len(selected)):
            a = selected[i]
            sums.add(2 * a)
            for j in range(i + 1, len(selected)):
                sums.add(a + selected[j])
        return sums

    def is_valid_addition(v, selected, sums):
        # Check if v can be added to selected
        # New sums: 2v, v+x for x in selected
        # Must not intersect sums
        if 2 * v in sums:
            return False
        for x in selected:
            if (v + x) in sums:
                return False
        # Internal collisions of new sums:
        # 2v == v+x => v==x (false)
        # v+x1 == v+x2 => x1==x2 (false)
        return True

    def construct_greedy_max_blocking():
        selected = []
        sums = set()
        available = set(range(1, N + 1))
        
        while available:
            # Find all valid candidates
            valid_candidates = []
            for v in available:
                if is_valid_addition(v, selected, sums):
                    valid_candidates.append(v)
            
            if not valid_candidates:
                break
            
            # If only one, pick it
            if len(valid_candidates) == 1:
                v = valid_candidates[0]
            else:
                # Pick the one that blocks the most other valid candidates
                # blocking_power(v) = count of u in valid_candidates \ {v} such that u is NOT valid after adding v
                best_v = None
                max_block = -1
                
                # Optimization: Only check a subset if time is tight?
                # For now, check all.
                for v in valid_candidates:
                    # Simulate adding v
                    new_sums = set(sums)
                    new_sums.add(2 * v)
                    for x in selected:
                        new_sums.add(v + x)
                    
                    block_count = 0
                    for u in valid_candidates:
                        if u == v:
                            continue
                        # Check if u is valid with new_sums and selected+[v]
                        # Note: u was valid with selected, but might not be with selected+[v]
                        # We need to check if u can be added to selected U {v}
                        # This is is_valid_addition(u, selected + [v], new_sums)
                        # But is_valid_addition checks against the sums of the set.
                        # The sums of selected U {v} are new_sums.
                        # So we just check if u is valid against new_sums.
                        if is_valid_addition(u, selected + [v], new_sums):
                            # u is still valid, so v did NOT block u
                            pass
                        else:
                            block_count += 1
                    
                    if block_count > max_block:
                        max_block = block_count
                        best_v = v
                        
                v = best_v
            
            # Add v to selected
            selected.append(v)
            sums.add(2 * v)
            for x in selected:
                if x != v:
                    sums.add(v + x)
            
            # Remove v and all newly blocked elements from available
            # We need to re-evaluate which elements are still valid?
            # No, we just remove v. The loop will re-evaluate validity for the next step.
            # But to speed up, we can remove elements that are definitely blocked.
            # However, "blocked" depends on the current set.
            # So we just remove v from available.
            available.remove(v)
            
            # Time check
            if time.time() > deadline:
                break
        
        return selected

    # Run multiple restarts with different randomizations?
    # The greedy is deterministic. To get different results, we can randomize the tie-breaking
    # or shuffle the initial available set?
    # The "Max Blocking" is deterministic.
    # We can add a random component: if multiple v have the same max_block, pick one randomly.
    
    best_set = None
    best_size = float('inf')
    
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Randomize the order of iteration in valid_candidates to break ties randomly
        # Or just run the deterministic one?
        # Let's try a hybrid: Greedy with random tie-breaking.
        
        selected = []
        sums = set()
        available = set(range(1, N + 1))
        
        while available:
            valid_candidates = []
            for v in available:
                if is_valid_addition(v, selected, sums):
                    valid_candidates.append(v)
            
            if not valid_candidates:
                break
            
            if len(valid_candidates) == 1:
                v = valid_candidates[0]
            else:
                # Calculate blocking power
                # To save time, we might not want to check ALL pairs in every step.
                # But N=256 is small.
                powers = []
                for v in valid_candidates:
                    new_sums = set(sums)
                    new_sums.add(2 * v)
                    for x in selected:
                        new_sums.add(v + x)
                    
                    block_count = 0
                    # Sample a few u's if list is large?
                    # For now, check all.
                    for u in valid_candidates:
                        if u == v:
                            continue
                        if not is_valid_addition(u, selected + [v], new_sums):
                            block_count += 1
                    powers.append((block_count, v))
                
                # Find max block_count
                max_block = max(p[0] for p in powers)
                # Get all v with max_block
                candidates_max = [v for b, v in powers if b == max_block]
                v = random.choice(candidates_max)
            
            selected.append(v)
            sums.add(2 * v)
            for x in selected:
                if x != v:
                    sums.add(v + x)
            
            available.remove(v)
            
            if time.time() > deadline:
                break
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = list(selected)
            
        # Early stop if we hit theoretical lower bound?
        # sqrt(256) = 16.
        if best_size <= 16:
            break

    if best_set is None:
        # Fallback
        selected, sums = [], set()
        for x in range(1, N + 1):
            if is_valid_addition(x, selected, sums):
                selected.append(x)
                sums.add(2 * x)
                for y in selected:
                    if y != x:
                        sums.add(x + y)
        return selected
        
    return best_set
# EVOLVE-BLOCK-END
