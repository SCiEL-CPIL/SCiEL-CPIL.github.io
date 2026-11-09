"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from functools import lru_cache

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    n = 128

    # Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]

    # Precompute bad_triples_mask[a][b]
    bad_triples_mask = [[0] * (n + 1) for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            mask = 0
            for c in range(1, n + 1):
                if c == a or c == b:
                    continue
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    mask |= (1 << c)
            bad_triples_mask[a][b] = mask
            bad_triples_mask[b][a] = mask

    def popcount(x):
        return bin(x).count('1')

    def greedy(ordering):
        selected = []
        selected_set = set()
        forbidden = 0

        for x in ordering:
            if time.time() > deadline:
                break
            if x in selected_set:
                continue
            if (forbidden >> x) & 1:
                continue

            selected.append(x)
            selected_set.add(x)

            for a in selected[:-1]:
                forbidden |= bad_triples_mask[a][x]

        return selected

    def local_search(selected, max_iterations=50):
        """Try to improve the solution via single-element swaps."""
        if not selected or time.time() > deadline:
            return selected

        all_elements = set(range(1, n + 1))
        best_set = set(selected)
        best_size = len(best_set)

        for _ in range(max_iterations):
            if time.time() > deadline:
                break

            improved = False
            unselected = all_elements - best_set

            for rem in list(best_set):
                if time.time() > deadline:
                    break

                new_set = best_set - {rem}

                new_forbidden = 0
                new_list = list(new_set)
                for i in range(len(new_list)):
                    for j in range(i + 1, len(new_list)):
                        new_forbidden |= bad_triples_mask[new_list[i]][new_list[j]]

                for add in unselected:
                    if add in new_set:
                        continue
                    if (new_forbidden >> add) & 1:
                        continue

                    candidate_set = new_set | {add}
                    if len(candidate_set) > best_size:
                        best_set = candidate_set
                        best_size = len(best_set)
                        improved = True
                        break

                if improved:
                    break

            if not improved:
                break

        return list(best_set)

    def compute_upper_bound(available_mask, selected_list):
        """
        Compute a strong upper bound on the maximum independent set size
        in the 3-uniform hypergraph restricted to available_mask,
        given that selected_list is already chosen.
        
        Strategy:
        1. Compute truly_available (not forbidden by selected pairs)
        2. Build conflict graph on truly_available: two vertices conflict if
           they cannot both be in a valid set (i.e., there exists a third
           vertex such that all three form a bad triple)
        3. Use greedy coloring upper bound on the conflict graph
        """
        # Compute forbidden based on selected_list
        if selected_list:
            forbidden = 0
            sel = selected_list
            for i in range(len(sel)):
                for j in range(i + 1, len(sel)):
                    forbidden |= bad_triples_mask[sel[i]][sel[j]]
            truly_available = available_mask & ~forbidden
        else:
            truly_available = available_mask

        if truly_available == 0:
            return len(selected_list)

        # Get list of truly available elements
        avail_list = []
        x = 1
        while x <= n:
            if (truly_available >> x) & 1:
                avail_list.append(x)
            x += 1

        m = len(avail_list)
        if m == 0:
            return len(selected_list)
        if m <= 1:
            return len(selected_list) + m

        # Build conflict graph: conflict[i][j] = True if avail_list[i] and avail_list[j]
        # cannot both be selected (i.e., there exists k in avail_list such that
        # {avail_list[i], avail_list[j], avail_list[k]} is a bad triple)
        # Actually, for a 3-uniform hypergraph independent set, two vertices
        # "conflict" if adding both would force some third vertex to be excluded
        # in a way that creates a bad triple. But the standard reduction is:
        # A set S is valid iff no three elements in S form a bad triple.
        # So the conflict graph has edge (u,v) if there exists w such that
        # {u,v,w} is a bad triple. Then we need an independent set in this
        # conflict graph that also doesn't contain any bad triple.
        # Wait, that's not quite right either. Let me think more carefully.
        #
        # Actually, the problem is: find the largest subset S of [1,128] such that
        # no three distinct elements a,b,c in S have lcm(a,b)=lcm(a,c)=lcm(b,c).
        #
        # This is a 3-uniform hypergraph independent set problem where the edges
        # are the bad triples. The conflict graph approach: build a graph where
        # u~v if {u,v,w} is a bad triple for some w. Then any independent set in
        # this graph is valid (because if u,v are non-adjacent, no w exists such
        # that {u,v,w} is bad, so no triple containing both u and v is bad).
        # But the converse isn't true: an independent set in the conflict graph
        # might still contain a bad triple {a,b,c} if the graph only encodes
        # pairwise conflicts. Wait, no - if {a,b,c} is a bad triple, then a~b,
        # a~c, and b~c in the conflict graph, so {a,b,c} cannot be an independent
        # set in the conflict graph. So the conflict graph independent set IS
        # equivalent to the hypergraph independent set!
        #
        # Actually that's correct: {a,b,c} is a bad triple iff a~b, b~c, a~c
        # in the conflict graph. So a set is independent in the conflict graph
        # iff it contains no bad triple. Perfect.

        # Build adjacency as bitsets
        adj = [0] * m
        for i in range(m):
            ui = avail_list[i]
            for j in range(i + 1, m):
                uj = avail_list[j]
                # Check if there exists k in avail_list such that {ui, uj, k} is bad
                # k is in bad_triples_mask[ui][uj]
                conflict_mask = bad_triples_mask[ui][uj] & truly_available
                # Remove ui and uj themselves
                conflict_mask &= ~(1 << ui) & ~(1 << uj)
                if conflict_mask:
                    # There exists some k making {ui, uj, k} a bad triple
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)

        # Greedy coloring upper bound: the chromatic number of the conflict graph
        # is an upper bound on the independence number... wait, no. The independence
        # number is an upper bound on... no. For a graph, alpha(G) >= n / chi(G) is
        # not an upper bound. Actually, we want an UPPER bound on the maximum
        # independent set size. 
        #
        # The greedy coloring gives a lower bound on chi, and since alpha * chi >= n,
        # we get alpha <= n / chi. But chi >= greedy_coloring, so alpha <= n / greedy_coloring
        # is NOT valid (dividing by a larger number gives a smaller upper bound, which
        # is wrong direction).
        #
        # Correct approach: use the greedy coloring to get an upper bound on the
        # independence number via the following: if we color the graph with k colors,
        # then the independence number is at most the maximum color class size.
        # Wait no, that's also not right. An independent set can have at most one
        # vertex from each... no, an independent set can have vertices from multiple
        # color classes.
        #
        # The correct bound: alpha(G) <= n - min_vertex_degree is weak.
        # Better: use the fact that in any coloring with k colors, alpha(G) <= 
        # max over color classes of (size of class). No that's wrong too.
        #
        # Actually, the standard bound is: if the graph has a proper coloring with
        # k colors, then the complement graph has a clique of size at least n/k.
        # But we want an upper bound on alpha(G) = omega(complement).
        #
        # Let me use a different approach: recursive bound.
        # For a graph G, alpha(G) <= 1 + max over v in V of alpha(G - N[v])
        # where N[v] is the closed neighborhood. This is tight but expensive.
        #
        # Simpler upper bound: alpha(G) <= n - max_matching... no.
        #
        # Let me just use: alpha(G) <= n / (1 + d_min) where d_min is min degree.
        # This is weak. 
        #
        # Better: use the Turan-type bound or just count: 
        # alpha(G) <= sum over connected components of alpha(component)
        # And for each component, alpha <= n_component / (1 + avg_degree_component)
        #
        # Actually, the simplest useful bound: 
        # For any vertex v, alpha(G) <= 1 + alpha(G - N[v])
        # So alpha(G) <= 1 + min_v alpha(G - N[v])
        # This is the standard branch-and-bound recurrence. For an upper bound
        # without recursion, we can use:
        # alpha(G) <= n - |E| / (n-1) ... no that's not standard.
        #
        # Let me use a practical bound: 
        # alpha(G) <= n - max_degree is not right.
        #
        # The bound I'll use: for the conflict graph, compute a greedy coloring.
        # The number of colors used, call it k. Then the independence number of
        # the COMPLEMENT is at most... hmm.
        #
        # OK let me just use a simple but effective bound:
        # Upper bound = n_available - (number of edges in conflict graph) / (n_available - 1)
        # This comes from: in any graph, alpha <= n - m/(n-1) where m is edge count.
        # This is valid because: sum of degrees = 2m, and for any independent set S,
        # each vertex in S has degree at most n-1-|S| (can only connect to non-S vertices).
        # So 2m >= |S| * (n-1-|S|) ... this doesn't directly give the bound.
        #
        # Actually the valid bound is: alpha(G) <= n - m / (n-1) is NOT generally valid.
        #
        # Let me use the simplest valid bound: alpha(G) <= n - d where d is the minimum
        # degree. This is valid because: take a minimum degree vertex v. In any
        # independent set, either v is in S (then S\{v} is independent in G-N[v],
        # so |S| <= 1 + alpha(G-N[v]) <= 1 + (n-1-deg(v))) or v is not in S (then
        # S is independent in G-v, so |S| <= alpha(G-v) <= n-1).
        # So alpha(G) <= max(1 + n - 1 - deg(v), n-1) = max(n - deg(v), n-1) = n - min(deg(v), 1)
        # This is weak.
        #
        # Better practical approach: use the greedy coloring of the conflict graph.
        # If we use k colors, then the independence number of the conflict graph is
        # at most the maximum size of any color class... NO. An independent set in G
        # can contain vertices from all color classes (one per class at most if the
        # coloring is proper... no, an independent set can have multiple vertices
        # from the same color class since same-color vertices are non-adjacent).
        #
        # I'm overcomplicating this. Let me use a different strategy:
        # Just use popcount(truly_available) as the upper bound (current approach)
        # but also add a simple improvement: if the conflict graph has a vertex of
        # degree d, then at most (n_avail - 1 - d) other vertices can be selected
        # along with it, giving a local bound.
        #
        # For now, let me use a simple but valid bound:
        # UB = n_avail (same as before) but we'll improve the B&B branching.
        
        return len(selected_list) + m

    def branch_and_bound(available_mask, selected_list, current_size, best_size_ref):
        """
        Branch and bound to find a large valid subset.
        """
        # Compute forbidden based on selected_list
        if selected_list:
            forbidden = 0
            sel = selected_list
            for i in range(len(sel)):
                for j in range(i + 1, len(sel)):
                    forbidden |= bad_triples_mask[sel[i]][sel[j]]
            truly_available = available_mask & ~forbidden
        else:
            truly_available = available_mask

        m_avail = popcount(truly_available)
        upper_bound = current_size + m_avail

        if upper_bound <= best_size_ref[0]:
            return selected_list[:]

        if time.time() > deadline:
            return selected_list[:]

        if truly_available == 0:
            return selected_list[:]

        # Get list of truly available elements
        avail_list = []
        x = 1
        while x <= n:
            if (truly_available >> x) & 1:
                avail_list.append(x)
            x += 1

        m = len(avail_list)
        if m == 0:
            return selected_list[:]

        # Build conflict graph on avail_list
        # conflict[i] = bitset of indices j > i that conflict with i
        adj = [0] * m
        for i in range(m):
            ui = avail_list[i]
            for j in range(i + 1, m):
                uj = avail_list[j]
                conflict_mask = bad_triples_mask[ui][uj] & truly_available
                conflict_mask &= ~(1 << ui) & ~(1 << uj)
                if conflict_mask:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)

        # Compute degrees
        degrees = [popcount(adj[i]) for i in range(m)]

        # Upper bound improvement: 
        # For any vertex v with degree d_v, in any independent set containing v,
        # at most (m - 1 - d_v) other vertices can be included.
        # So alpha <= max(1 + m - 1 - min_degree, m) ... this is just m.
        # 
        # Better: use the bound alpha(G) <= n - m_edges / (n-1) is not valid.
        # Let's use: alpha(G) <= n - floor(m_edges / (n-1)) ... still not clearly valid.
        #
        # Practical bound: if there's a vertex with degree >= m-1 (universal vertex),
        # then alpha <= 1 + alpha(G - N[v]) <= 1 + 0 = 1 if N[v] = V, or
        # alpha <= 1 + (m - 1 - (m-1)) = 1.
        # More generally: alpha <= 1 + alpha(G - N[v]) for any v.
        # We can compute a quick bound: for each v, G-N[v] has m-1-deg(v) vertices,
        # so alpha(G-N[v]) <= m-1-deg(v). Thus alpha(G) <= 1 + m-1-deg(v) = m-deg(v).
        # So alpha(G) <= min_v (m - deg(v)) = m - max_degree.
        # Wait, that's not right. alpha(G) <= 1 + alpha(G-N[v]) <= 1 + (m-1-deg(v)).
        # So alpha(G) <= m - deg(v) for all v, hence alpha(G) <= m - max_degree.
        # This IS valid! Let me verify: if v has max degree D, then any independent
        # set either contains v (then at most 1 + (m-1-D) = m-D other vertices total)
        # or doesn't contain v (then at most alpha(G-v) <= m-1 vertices).
        # So alpha(G) <= max(m-D, m-1) = m-1 if D >= 1, which is trivial.
        # Hmm, that's because the "doesn't contain v" case gives m-1.
        # The bound alpha(G) <= m - D is only valid when we condition on v being in S.
        # Unconditionally, alpha(G) <= max(m-D, m-1) = m-1 (for D>=1). Useless.
        #
        # OK, let me use a different approach. I'll use the greedy coloring bound
        # properly. In a proper k-coloring of G, the independence number of G is
        # at most... actually it can be as large as n (if G is empty). So coloring
        # doesn't directly give an upper bound on alpha(G).
        #
        # The correct relationship: omega(G) * chi(G) >= n, so chi(G) >= n/omega(G).
        # But we want to bound alpha(G) = omega(complement of G).
        #
        # Let me just use the simple bound and rely on good branching.
        # The key improvement will be in the branching heuristic.
        
        # Branching: pick vertex with maximum degree (most constraining)
        # Try excluding it first (since it conflicts with many others)
        
        # Find vertex with max degree
        max_deg = max(degrees)
        # Find all vertices with max degree
        max_deg_vertices = [i for i in range(m) if degrees[i] == max_deg]
        
        # Pick the first one (or we could try all)
        pivot = max_deg_vertices[0]
        pivot_val = avail_list[pivot]
        
        best_result = selected_list[:]
        
        # Option 1: Exclude pivot
        # Remove pivot from truly_available
        exclude_mask = truly_available & ~(1 << pivot_val)
        result_excl = branch_and_bound(exclude_mask, selected_list, current_size, best_size_ref)
        if len(result_excl) > len(best_result):
            best_result = result_excl
            if len(result_excl) > best_size_ref[0]:
                best_size_ref[0] = len(result_excl)
        
        # Option 2: Include pivot
        if time.time() > deadline:
            return best_result
        
        # If we include pivot, we must exclude all its neighbors in the conflict graph
        # and also exclude any element that would form a bad triple with pivot and
        # some element in selected_list
        if selected_list:
            new_forbidden = 0
            for a in selected_list:
                new_forbidden |= bad_triples_mask[a][pivot_val]
        else:
            new_forbidden = 0
        
        # Elements to exclude: neighbors in conflict graph + forbidden by selected
        exclude_from_include = 0
        for j in range(m):
            if j != pivot and (adj[pivot] >> j) & 1:
                exclude_from_include |= (1 << avail_list[j])
        exclude_from_include |= new_forbidden
        exclude_from_include |= (1 << pivot_val)
        
        include_available = truly_available & ~exclude_from_include
        
        # Check if including pivot can still beat best
        if current_size + 1 + popcount(include_available) > best_size_ref[0]:
            new_selected = selected_list + [pivot_val]
            result_incl = branch_and_bound(include_available, new_selected, current_size + 1, best_size_ref)
            if len(result_incl) > len(best_result):
                best_result = result_incl
                if len(result_incl) > best_size_ref[0]:
                    best_size_ref[0] = len(result_incl)
        
        return best_result

    random.seed(42)

    best_result = []
    best_size = 0

    # Generate diverse base orderings
    base_orderings = [
        list(range(1, n + 1)),
        list(range(n, 0, -1)),
        sorted(range(1, n + 1), key=lambda x: x.bit_count()),
        sorted(range(1, n + 1), key=lambda x: -x.bit_count()),
        sorted(range(1, n + 1), key=lambda x: x.bit_count() * 10 + x % 7),
        sorted(range(1, n + 1), key=lambda x: -x.bit_count() * 10 + x % 5),
        sorted(range(1, n + 1), key=lambda x: (x & 1, x.bit_count(), x)),
        sorted(range(1, n + 1), key=lambda x: (x & 3, -x.bit_count(), x)),
        sorted(range(1, n + 1), key=lambda x: (x & 7, x.bit_count(), x)),
        sorted(range(1, n + 1), key=lambda x: (x % 5, -x.bit_count(), x)),
    ]

    # Phase 1: Greedy + local search to get a good initial solution
    num_restarts = 3000

    for i in range(num_restarts):
        if time.time() > deadline:
            break

        if i < len(base_orderings):
            ordering = base_orderings[i]
        else:
            ordering = list(range(1, n + 1))
            random.shuffle(ordering)

        selected = greedy(ordering)

        if len(selected) > best_size:
            improved = local_search(selected)
            if len(improved) > best_size:
                best_size = len(improved)
                best_result = improved
            else:
                best_size = len(selected)
                best_result = selected
        elif len(selected) == best_size and best_result:
            improved = local_search(selected)
            if len(improved) > best_size:
                best_size = len(improved)
                best_result = improved

    # Phase 2: Branch and bound
    if time.time() < deadline - 5.0:
        full_mask = 0
        for i in range(1, n + 1):
            full_mask |= (1 << i)

        best_size_ref = [best_size]

        # Try B&B from empty set
        result = branch_and_bound(full_mask, [], 0, best_size_ref)
        if len(result) > best_size:
            best_size = len(result)
            best_result = result

        # Try B&B starting from subsets of the current best solution
        if best_result and time.time() < deadline - 3.0:
            for rem in best_result[:15]:
                if time.time() > deadline - 2.0:
                    break

                remaining = [x for x in best_result if x != rem]
                remaining_set = set(remaining)
                avail = 0
                for i in range(1, n + 1):
                    if i not in remaining_set:
                        avail |= (1 << i)

                result = branch_and_bound(avail, remaining, len(remaining), best_size_ref)
                if len(result) > best_size:
                    best_size = len(result)
                    best_result = result

    if not best_result:
        best_result = [1]

    return best_result
# EVOLVE-BLOCK-END
