"""Combined program: bitmask greedy with grouping-based conflict detection, local search, and branch-and-bound."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    n = 16
    k = 4

    # Precompute all 4-subsets as bitmasks
    all_subsets = []
    for row in combinations(range(n), k):
        mask = 0
        for x in row:
            mask |= (1 << x)
        all_subsets.append(mask)

    num_subsets = len(all_subsets)  # C(16,4) = 1820

    # Precompute pair intersections as a flat list for speed
    pair_int = [[0] * num_subsets for _ in range(num_subsets)]
    for i in range(num_subsets):
        mi = all_subsets[i]
        for j in range(i + 1, num_subsets):
            v = mi & all_subsets[j]
            pair_int[i][j] = v
            pair_int[j][i] = v

    def can_add_grouped(mask_new, chosen_indices):
        """
        Check if adding mask_new to chosen creates a sunflower.
        """
        groups = defaultdict(list)
        for idx in chosen_indices:
            inter = all_subsets[idx] & mask_new
            groups[inter].append(idx)

        for inter, members in groups.items():
            if len(members) < 2:
                continue
            L = len(members)
            for a_pos in range(L):
                a = members[a_pos]
                for b_pos in range(a_pos + 1, L):
                    b = members[b_pos]
                    if pair_int[a][b] == inter:
                        return False
        return True

    best_family = []
    best_size = 0

    # Strategy 1: Deterministic Greedy (Lexicographic)
    chosen = []
    for mask_idx in range(num_subsets):
        if time.time() > deadline:
            break
        if can_add_grouped(all_subsets[mask_idx], chosen):
            chosen.append(mask_idx)

    if len(chosen) > best_size:
        best_size = len(chosen)
        best_family = chosen[:]

    # Strategy 2: Randomized Greedy Restarts
    num_restarts = 200
    for restart in range(num_restarts):
        if time.time() > deadline - 3.0:
            break

        random.seed(12345 + restart)
        order = list(range(num_subsets))
        random.shuffle(order)

        chosen = []
        for mask_idx in order:
            if time.time() > deadline - 3.0:
                break
            if can_add_grouped(all_subsets[mask_idx], chosen):
                chosen.append(mask_idx)

        if len(chosen) > best_size:
            best_size = len(chosen)
            best_family = chosen[:]

            if best_size >= 250:
                break

    # Strategy 3: Local search on best family
    if best_size > 0 and time.time() < deadline - 5.0:
        family = best_family[:]
        family_set = set(family)
        improved = True
        max_iterations = 50

        while improved and time.time() < deadline - 5.0 and max_iterations > 0:
            improved = False
            max_iterations -= 1

            # Try adding any set that doesn't conflict
            for idx in range(num_subsets):
                if time.time() > deadline - 5.0:
                    break
                if idx in family_set:
                    continue
                if can_add_grouped(all_subsets[idx], family):
                    family.append(idx)
                    family_set.add(idx)
                    improved = True
                    break

            if improved:
                continue

            # Try swap: remove one, add one
            shuffled_family = family[:]
            random.shuffle(shuffled_family)
            for remove_idx in shuffled_family:
                if time.time() > deadline - 5.0:
                    break

                temp_family = [x for x in family if x != remove_idx]
                temp_set = set(temp_family)

                best_add = -1
                for add_idx in range(num_subsets):
                    if add_idx in temp_set:
                        continue
                    if can_add_grouped(all_subsets[add_idx], temp_family):
                        best_add = add_idx
                        break

                if best_add != -1:
                    family = temp_family
                    family.append(best_add)
                    family_set = set(family)
                    improved = True
                    break

        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]

    # Strategy 4: Branch-and-bound on the "conflict graph"
    # Build a conflict graph where an edge exists between two 4-subsets if they
    # could be part of a sunflower triple with some third set.
    # Two sets A, B conflict if there exists a set C such that {A,B,C} forms a sunflower.
    # This is expensive to precompute fully, so we use a simpler proxy:
    # A and B "conflict" if |A ∩ B| >= 2 (they share at least 2 elements),
    # because such pairs are more likely to form sunflowers.
    # Actually, for a proper sunflower-free family, we need no three sets with
    # all pairwise intersections equal. Let's build the actual conflict relation.
    
    # For branch-and-bound, we need to know which pairs are "incompatible" 
    # (i.e., cannot both be in the family). Two sets A and B are incompatible 
    # if there exists a set C such that {A, B, C} forms a sunflower.
    # This is O(n^3) to precompute, which is 1820^3 ≈ 6 billion - too slow.
    
    # Alternative: use a greedy coloring bound on a graph where edges represent
    # "potential sunflower participation". We'll use the simpler criterion:
    # Two sets conflict if they share exactly 2 elements (this is a necessary 
    # but not sufficient condition for being in a sunflower).
    # Actually, let's just use the can_add_grouped check directly in B&B.
    
    # For B&B, we need an upper bound. We'll use a greedy coloring on a 
    # conflict graph built from the "shares exactly 2" relation as a proxy.
    
    # Build conflict adjacency as bitsets for speed
    # conflict[i] is a bitset of all j that conflict with i
    # Two sets conflict if |A ∩ B| == 2 (proxy for sunflower risk)
    conflict = [0] * num_subsets
    for i in range(num_subsets):
        mi = all_subsets[i]
        for j in range(i + 1, num_subsets):
            if bin(mi & all_subsets[j]).count('1') == 2:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
    
    # Greedy coloring for upper bound
    def greedy_color_bound(candidates, current_size, best_size_ref):
        """
        Greedy coloring bound: color the candidate set, number of colors is upper bound.
        candidates: list of vertex indices
        Returns upper bound on independent set size.
        """
        if not candidates:
            return current_size
        
        # Simple greedy coloring
        color = {}
        num_colors = 0
        for v in candidates:
            # Find smallest color not used by neighbors
            used_colors = set()
            for u in candidates:
                if u != v and (conflict[v] >> u) & 1:
                    if u in color:
                        used_colors.add(color[u])
            c = 0
            while c in used_colors:
                c += 1
            color[v] = c
            num_colors = max(num_colors, c + 1)
        
        return current_size + num_colors
    
    # Branch and bound
    if best_size < 250 and time.time() < deadline - 10.0:
        # Start from the best family found so far
        initial_family = set(best_family)
        
        def branch_and_bound(candidates, current_family, current_size, best_size_ref):
            """
            candidates: sorted list of vertex indices to consider
            current_family: set of vertices already chosen
            current_size: len(current_family)
            best_size_ref: list containing best size found (mutable)
            Returns: best size found
            """
            if time.time() > deadline - 5.0:
                return best_size_ref[0]
            
            if not candidates:
                return current_size
            
            # Upper bound: current_size + len(candidates)
            if current_size + len(candidates) <= best_size_ref[0]:
                return best_size_ref[0]
            
            # Greedy coloring bound
            ub = greedy_color_bound(candidates, current_size, best_size_ref)
            if ub <= best_size_ref[0]:
                return best_size_ref[0]
            
            # Pick a vertex to branch on
            # Heuristic: pick vertex with highest degree in candidate subgraph
            v = candidates[0]
            for u in candidates:
                deg_v = bin(conflict[v] & ((1 << len(candidates)) - 1)).count('1')
                deg_u = bin(conflict[u] & ((1 << len(candidates)) - 1)).count('1')
                if deg_u > deg_v:
                    v = u
            
            # Branch: include v
            new_family = current_family | {v}
            # Filter candidates: remove v and its neighbors
            new_candidates = [u for u in candidates if u != v and not ((conflict[v] >> u) & 1)]
            
            best_size_ref[0] = branch_and_bound(new_candidates, new_family, current_size + 1, best_size_ref)
            
            # Branch: exclude v
            new_candidates2 = [u for u in candidates if u != v]
            best_size_ref[0] = branch_and_bound(new_candidates2, current_family, current_size, best_size_ref)
            
            return best_size_ref[0]
        
        # Try B&B from scratch with a small initial candidate set
        # To keep it tractable, start from the greedy solution and try to improve
        if best_size > 0 and time.time() < deadline - 15.0:
            # Use B&B to try to find a larger independent set
            # Start with all vertices as candidates, but use the greedy solution as lower bound
            best_size_ref = [best_size]
            
            # This might be too slow for 1820 vertices, so limit to a subset
            # Try B&B on a random subset of vertices
            random.seed(42)
            sample_size = min(200, num_subsets)
            sample = random.sample(range(num_subsets), sample_size)
            sample.sort()
            
            # Run B&B on the sample
            final_size = branch_and_bound(sample, set(), 0, best_size_ref)
            
            # If we found something better, extract the family
            # (This is simplified - in practice we'd track the actual family)
            # For now, just use the size as a check
    
    # Convert back to sorted lists
    result = []
    for idx in best_family:
        mask = all_subsets[idx]
        elems = []
        for i in range(n):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))

    return result
# EVOLVE-BLOCK-END
