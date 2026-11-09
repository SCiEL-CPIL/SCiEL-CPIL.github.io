"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import defaultdict

def _is_sunflower_triple(a, b, c):
    """Check if three sets form a sunflower (all pairwise intersections equal)."""
    return (a & b) == (a & c) == (b & c)

def _build_conflict_graph(all_sets):
    """
    Build a conflict structure: for each pair of sets, track which third sets
    would form a sunflower with them.
    
    Returns:
        conflicts: dict mapping (i, j) -> set of k such that (i,j,k) is a sunflower
    """
    n = len(all_sets)
    # For each pair (i,j), find all k > j that form a sunflower with i and j
    pair_to_thirds = defaultdict(set)
    
    # Group by intersection pattern for efficiency
    # A sunflower triple (A,B,C) has A∩B = A∩C = B∩C = core
    # So for a fixed core S (subset of [1,16]), all sets containing S
    # can form sunflowers with each other if they all contain S and
    # A∩B = S for each pair.
    
    # Alternative: enumerate all triples and check
    # C(1820, 3) ≈ 1 billion - too slow
    # Better: for each pair, check if adding a third creates sunflower
    
    # For pair (i, j), the common intersection is core = all_sets[i] & all_sets[j]
    # A third set k forms a sunflower with i and j iff:
    #   all_sets[i] & all_sets[k] == core AND all_sets[j] & all_sets[k] == core
    # This means all_sets[k] must contain core, and
    #   (all_sets[i] - core) ∩ all_sets[k] == empty AND
    #   (all_sets[j] - core) ∩ all_sets[k] == empty
    # i.e., all_sets[k] must avoid (all_sets[i] - core) and (all_sets[j] - core)
    # and contain core.
    
    # Precompute for each set its frozenset for fast lookup
    set_to_indices = defaultdict(list)
    for idx, s in enumerate(all_sets):
        set_to_indices[frozenset(s)].append(idx)
    
    # For each pair (i, j), determine constraints on k
    # k must: contain core, avoid (Si - core), avoid (Sj - core)
    # Let forbidden = (Si - core) ∪ (Sj - core)
    # k must be a 4-set containing core and disjoint from forbidden
    
    # This is still expensive. Let's use a different approach:
    # Enumerate all possible cores (subsets of size 0,1,2,3)
    # For each core, find all 4-sets containing it
    # Within that group, two sets A,B form a "partial sunflower" with core
    # if A∩B = core. Then any third set C in the group with A∩C=core and B∩C=core
    # completes a sunflower.
    
    # Actually, let's just build the hyperedges directly but smartly.
    
    # For each core size 0,1,2,3:
    #   Find all 4-sets containing core
    #   Among those, build a graph where edge (A,B) exists if A∩B = core
    #   Then for each triangle in that graph, we have a sunflower triple
    #   But we need ALL sunflower triples, not just those with a specific core.
    #   Each sunflower triple has exactly one core (the common intersection).
    
    # Let's enumerate by core:
    universe = set(range(1, 17))
    set_index = {frozenset(s): i for i, s in enumerate(all_sets)}
    
    for core_size in range(0, 4):
        for core in combinations(range(1, 17), core_size):
            core_set = set(core)
            # Find all 4-sets containing core
            remaining = universe - core_set
            need_more = 4 - core_size
            if need_more > len(remaining):
                continue
            members = []
            for extra in combinations(remaining, need_more):
                s = core_set | set(extra)
                if frozenset(s) in set_index:
                    members.append(set_index[frozenset(s)])
            
            if len(members) < 3:
                continue
            
            # Among members, find triples (i,j,k) where all pairwise intersections = core
            # i.e., for pair (i,j): all_sets[i] & all_sets[j] == core
            # This means (all_sets[i] - core) ∩ (all_sets[j] - core) == empty
            
            # Build adjacency: adj[i] = set of j in members where Si∩Sj = core
            m = len(members)
            adj = [set() for _ in range(m)]
            for a in range(m):
                for b in range(a+1, m):
                    sa = all_sets[members[a]]
                    sb = all_sets[members[b]]
                    if (sa & sb) == core_set:
                        adj[a].add(b)
                        adj[b].add(a)
            
            # Find all triangles in this graph
            # For each node a, for each pair (b,c) in adj[a] where b<c and c in adj[b]:
            for a in range(m):
                neighbors = adj[a]
                for b in neighbors:
                    if b <= a:
                        continue
                    # c must be in adj[a] ∩ adj[b] and c > b
                    common = neighbors & adj[b]
                    for c in common:
                        if c > b:
                            # Triangle (a, b, c) in the core graph
                            # This is a sunflower triple
                            i, j, k = members[a], members[b], members[c]
                            # Store in a global structure
                            # We'll collect all forbidden triples
                            pass  # We'll handle this differently below
    
    # The above is getting complex. Let's use a simpler approach:
    # Build a "conflict" representation where for each set i, we store
    # which other sets j conflict with i in the sense that there exists
    # a k such that (i,j,k) is a sunflower.
    # But that's not quite right for MIS.
    
    # Actually, for the independent set formulation:
    # We want the largest subset S of {0,...,n-1} such that no three elements
    # in S form a sunflower triple.
    # This is a 3-uniform hypergraph independent set problem.
    # Hyperedge = sunflower triple.
    
    # Let's just enumerate all sunflower triples directly.
    # n = C(16,4) = 1820
    # Number of triples = C(1820,3) ≈ 1.02 billion - too many to enumerate naively.
    
    # Better approach: for each pair (i,j), compute core = Si∩Sj
    # Then k must satisfy: Sk contains core, and Sk ∩ (Si-core) = empty, Sk ∩ (Sj-core) = empty
    # So Sk must be a 4-set containing core and disjoint from (Si-core)∪(Sj-core)
    # Let forbidden = (Si - core) ∪ (Sj - core)
    # |forbidden| can be up to 4+4-2*|core|... let's think
    # Si has 4 elements, core ⊆ Si, so Si-core has 4-|core| elements
    # Similarly Sj-core has 4-|core| elements
    # forbidden = (Si-core) ∪ (Sj-core), size at most 2*(4-|core|)
    # Sk must be a 4-set containing core (|core| elements) and avoiding forbidden
    # So Sk must choose (4-|core|) elements from universe - core - forbidden
    # |universe - core - forbidden| = 16 - |core| - |forbidden|
    
    # For each pair, we can enumerate possible k's.
    # Total pairs = C(1820,2) ≈ 1.65 million
    # For each pair, number of valid k's depends on the situation.
    
    # Let's build the hyperedges:
    hyperedges = []  # list of (i, j, k) with i < j < k
    pair_to_thirds = defaultdict(set)
    
    for i in range(len(all_sets)):
        si = all_sets[i]
        for j in range(i+1, len(all_sets)):
            sj = all_sets[j]
            core = si & sj
            core_frozen = frozenset(core)
            forbidden = (si - core) | (sj - core)
            # k must be a 4-set containing core and disjoint from forbidden
            available = universe - core - forbidden
            need = 4 - len(core)
            if len(available) < need:
                continue
            for extra in combinations(available, need):
                sk = core | set(extra)
                sk_frozen = frozenset(sk)
                if sk_frozen in set_index:
                    k = set_index[sk_frozen]
                    if k > j:  # ensure i < j < k
                        hyperedges.append((i, j, k))
                        pair_to_thirds[(i,j)].add(k)
    
    return hyperedges, pair_to_thirds

def _greedy_construct(order, deadline):
    """Greedy construction given an ordering of all 4-subsets."""
    chosen = []
    chosen_list = []
    for s in order:
        if time.time() > deadline:
            break
        # Check against all pairs in chosen
        conflict = False
        for i in range(len(chosen_list)):
            for j in range(i+1, len(chosen_list)):
                if _is_sunflower_triple(chosen_list[i], chosen_list[j], s):
                    conflict = True
                    break
            if conflict:
                break
        if not conflict:
            chosen_list.append(s)
            chosen.append(s)
    return chosen_list

def _branch_and_bound(all_sets, hyperedges, pair_to_thirds, deadline, initial_solution):
    """
    Branch and bound for maximum independent set in 3-uniform hypergraph.
    """
    n = len(all_sets)
    # Build adjacency for each node: which other nodes are in a hyperedge with this node
    node_neighbors = [set() for _ in range(n)]
    for (i, j, k) in hyperedges:
        node_neighbors[i].add(j)
        node_neighbors[i].add(k)
        node_neighbors[j].add(i)
        node_neighbors[j].add(k)
        node_neighbors[k].add(i)
        node_neighbors[k].add(j)
    
    best_solution = list(initial_solution)
    best_size = len(best_solution)
    
    # Greedy upper bound: greedy coloring of the "conflict graph"
    # where two nodes are connected if they appear in some hyperedge together
    # This gives an upper bound on the independent set size
    
    def greedy_color_upper_bound(remaining_nodes):
        """
        Compute a greedy coloring upper bound for the induced subgraph on remaining_nodes.
        """
        if not remaining_nodes:
            return 0
        
        # Build subgraph
        rem_set = set(remaining_nodes)
        # Color greedily
        colors = {}
        color_classes = []
        for node in sorted(rem_set, key=lambda x: -len(node_neighbors[x] & rem_set)):
            # Try to assign smallest available color
            used_colors = set()
            for neighbor in node_neighbors[node] & rem_set:
                if neighbor in colors:
                    used_colors.add(colors[neighbor])
            c = 0
            while c in used_colors:
                c += 1
            colors[node] = c
            if c >= len(color_classes):
                color_classes.append(set())
            color_classes[c].add(node)
        
        return len(color_classes)
    
    # Branch and bound
    # State: (current_independent_set, candidates, forbidden_pairs)
    # A candidate can be added if it doesn't form a sunflower with any two in current_set
    # and doesn't form a sunflower with any two other candidates that are also added
    
    # Simpler: recursive search
    # At each step, pick a candidate to include or exclude
    
    # For efficiency, we'll use a different approach:
    # Maintain a set of "active" candidates
    # When we include a node v, we remove all nodes that conflict with v
    # (i.e., nodes u such that there exists w in active where (v,u,w) is a hyperedge,
    #  or nodes u such that (v,u,w) is a hyperedge for some w already included)
    
    # Actually, the conflict is: if we include v, then for any pair (u,w) in the remaining
    # candidates, if (v,u,w) is a hyperedge, we cannot include both u and w.
    # This is complex. Let's use a simpler constraint propagation.
    
    # Let's use a different strategy: local search / simulated annealing
    # Start from the greedy solution and try to improve by swapping elements
    
    if time.time() > deadline:
        return best_solution
    
    # Local improvement: try to add elements
    active_set = set(best_solution)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            if i in active_set:
                continue
            # Check if i can be added
            can_add = True
            for j in active_set:
                for k in active_set:
                    if j >= k:
                        continue
                    if (i, j, k) in pair_to_thirds or (i, k, j) in pair_to_thirds or \
                       (j, i, k) in pair_to_thirds or (j, k, i) in pair_to_thirds or \
                       (k, i, j) in pair_to_thirds or (k, j, i) in pair_to_thirds:
                        # Check if (i,j,k) or any permutation is a hyperedge
                        # More efficiently: check if i is in pair_to_thirds[(j,k)] or similar
                        pass
            # This is getting messy. Let's use a cleaner approach.
    
    # Let me rewrite with a cleaner local search
    return best_solution

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    # Precompute all 4-element subsets
    all_sets = [set(row) for row in combinations(range(1, 17), 4)]
    n_sets = len(all_sets)
    
    # Build hyperedges (sunflower triples)
    hyperedges, pair_to_thirds = _build_conflict_graph(all_sets)
    
    # For efficient checking, build a set of frozensets of hyperedges
    hyperedge_set = set()
    for (i, j, k) in hyperedges:
        hyperedge_set.add(frozenset((i, j, k)))
    
    def _can_add(candidate_idx, current_set):
        """Check if candidate_idx can be added to current_set without forming a sunflower."""
        # For every pair (a, b) in current_set, check if (candidate, a, b) is a hyperedge
        for a in current_set:
            for b in current_set:
                if a >= b:
                    continue
                if frozenset((candidate_idx, a, b)) in hyperedge_set:
                    return False
        return True
    
    best_family = []
    
    # First: lexicographic greedy
    if time.time() < deadline:
        chosen = []
        for i in range(n_sets):
            if time.time() > deadline:
                break
            if _can_add(i, chosen):
                chosen.append(i)
        if len(chosen) > len(best_family):
            best_family = chosen
    
    # Try multiple random orderings
    rng = random.Random(42)
    for attempt in range(50):
        if time.time() > deadline:
            break
        order = list(range(n_sets))
        rng.shuffle(order)
        chosen = []
        for i in order:
            if time.time() > deadline:
                break
            if _can_add(i, chosen):
                chosen.append(i)
        if len(chosen) > len(best_family):
            best_family = chosen
    
    # Local improvement: try to add more elements
    if time.time() < deadline and best_family:
        active_set = set(best_family)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n_sets):
                if time.time() > deadline:
                    break
                if i in active_set:
                    continue
                if _can_add(i, active_set):
                    active_set.add(i)
                    improved = True
        
        if len(active_set) > len(best_family):
            best_family = list(active_set)
    
    # Try to swap elements to get a larger set
    if time.time() < deadline and best_family:
        active_set = set(best_family)
        for _ in range(20):  # limited iterations
            if time.time() > deadline:
                break
            improved = False
            # Try removing one element and adding multiple others
            for remove_idx in list(active_set):
                if time.time() > deadline:
                    break
                temp_set = active_set - {remove_idx}
                added = 0
                for i in range(n_sets):
                    if time.time() > deadline:
                        break
                    if i in temp_set:
                        continue
                    if _can_add(i, temp_set):
                        temp_set.add(i)
                        added += 1
                if len(temp_set) > len(active_set):
                    active_set = temp_set
                    improved = True
                    break
            if not improved:
                break
    
    # Convert indices back to sets
    result = [sorted(all_sets[i]) for i in best_family]
    
    # Verify no sunflower triples
    # (skip verification for speed, trust the algorithm)
    
    return result
# EVOLVE-BLOCK-END

