import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # safety margin before 90s kill

    N = 192
    MAX_K = 48
    candidates = list(range(1, N + 1))
    n = len(candidates)

    # Map value -> index
    idx_of = {v: i for i, v in enumerate(candidates)}

    # Precompute all triples (i,j,k) of distinct indices, and their sums.
    # We'll build, for each 'a', the set of indices x that conflict with a.
    # x conflicts with a if there exist distinct i,j,k all != a, all != x,
    # with i,j,k distinct, and (candidates[i]+candidates[j]+candidates[k]) % a == 0.
    #
    # Efficient approach: For each a, we want the union over all triples (i,j,k)
    # with a not in {i,j,k} and sum%a==0 of {i,j,k}.
    #
    # To speed up, precompute for each a and each pair (i,j) with i,j != a,
    # the set of k != a, k != i, k != j such that (c[a] ... ) -- no, sum is c[i]+c[j]+c[k].
    #
    # Simpler: iterate triples once, and for each triple (i,j,k) with sum s,
    # for each a in [1,192] not in {c[i],c[j],c[k]} such that s % a == 0,
    # mark i,j,k as conflicting with a.
    # Number of triples: C(192,3) ~ 1.16M. For each, divisors of s up to 192.
    # s <= 576. Number of divisors <= 192 of s is small.
    # This is feasible.

    # conflicts[a_idx] = bitset (int) of candidate indices that conflict with a.
    # We'll build a list of ints, one per candidate index.
    conflicts = [0] * n

    # Precompute divisors up to 192 for each possible sum 3..576.
    max_sum = 3 * N
    divisors = [[] for _ in range(max_sum + 1)]
    for s in range(3, max_sum + 1):
        dlist = []
        # a must be in [1,192] and a != any of the triple values, but we'll filter later.
        for a in range(1, N + 1):
            if s % a == 0:
                dlist.append(a)
        divisors[s] = dlist

    # Iterate over all triples of distinct candidate indices.
    # For each, compute sum, then for each divisor a of sum, if a not equal to any of the three values,
    # mark the three indices in conflicts[idx_of[a]].
    # To keep this fast, we precompute value->index.
    vals = candidates  # 1..192

    # Use a set of (a, x) to avoid duplicate work? We just OR bits, so duplicates are fine.
    # But iterating 1.16M triples * ~10 divisors = ~11.6M operations, each doing a few dict lookups and bit ORs.
    # That's okay in Python if optimized. Let's precompute index arrays.
    # idx_of is a dict; use list indexed by value for speed: idx_by_val[v] = index.
    idx_by_val = [0] * (N + 1)
    for i, v in enumerate(vals):
        idx_by_val[v] = i

    # Precompute for each triple we need the three values and indices.
    # We'll iterate i<j<k.
    # To reduce overhead, precompute for each i, the list of (j,k) pairs? That's memory heavy.
    # Just do nested loops.

    # For speed, we can precompute, for each pair (i,j) with i<j, the list of k>j.
    # And the sum c[i]+c[j]. Then for each k, sum = that + c[k].
    # Let's precompute pair_sums[i][j] = c[i]+c[j] for i<j. And store as 2D list.
    # n=192, pairs ~ 18k, fine.

    pair_sum = [[0] * n for _ in range(n)]
    for i in range(n):
        ci = vals[i]
        for j in range(i + 1, n):
            pair_sum[i][j] = ci + vals[j]

    # Now iterate.
    for i in range(n):
        for j in range(i + 1, n):
            ps = pair_sum[i][j]
            vi = vals[i]
            vj = vals[j]
            for k in range(j + 1, n):
                s = ps + vals[k]
                vk = vals[k]
                for a in divisors[s]:
                    # a must not equal vi, vj, vk
                    if a == vi or a == vj or a == vk:
                        continue
                    ai = idx_by_val[a]
                    # mark i, j, k as conflicting with ai
                    bit_i = 1 << i
                    bit_j = 1 << j
                    bit_k = 1 << k
                    conflicts[ai] |= (bit_i | bit_j | bit_k)

    # Now conflicts[ai] is the set of indices that conflict with candidate ai.
    # Note: a candidate does NOT conflict with itself (we excluded a == value).
    # Also, conflict is symmetric: if x conflicts with a, then a conflicts with x?
    # The condition "a divides sum of 3 others" is asymmetric in formulation but the
    # *independence* condition is: we cannot have both a and x if x is in a triple summing to mult of a,
    # OR if a is in a triple (of others including x) summing to mult of x.
    # Actually the constraint is: for the chosen set S, for every a in S, and every triple T of 3 distinct
    # elements of S\{a}, a does not divide sum(T).
    # So the "conflict" between a and x exists if EITHER:
    #   (i) there exist two other elements u,v in S (distinct from a,x) such that a | (x+u+v), OR
    #   (ii) there exist two other elements u,v in S (distinct from a,x) such that x | (a+u+v).
    # My triple-based construction captures: x conflicts with a if x appears in some triple (x,u,v)
    # with a | (x+u+v)  [case (i)], OR a appears in some triple (a,u,v) with x | (a+u+v) [case (ii)].
    # But my construction only marks, for a given a, the elements that appear in triples whose sum is mult of a.
    # That captures case (i) for the pair (a, x): x is marked in conflicts[a] if x is in a triple (x,u,v)
    # with a | sum. Good. But case (ii) would require a to be marked in conflicts[x] (a in triple (a,u,v) with x|sum).
    # Since I iterate over ALL a and ALL triples, when a' = x (the other element), I mark a in conflicts[x]
    # if a is in a triple (a,u,v) with x | sum. So conflicts[x] contains a. 
    # So the union conflicts[a] over the construction already includes both directions, because I run the
    # marking for every a. So the final conflict relation is: a and x are incompatible iff
    # (x in conflicts[a]) OR (a in conflicts[x]).
    # To build a proper symmetric conflict graph for MIS, define edge(a,x) iff x in conflicts[a] or a in conflicts[x].
    # Let's build adj bitsets: adj[i] = bitset of j != i such that edge exists.

    adj = [0] * n
    for i in range(n):
        ci = conflicts[i]
        # edge if j in ci or i in conflicts[j]
        # We'll build by: adj[i] = ci  (j where i divides some triple containing j)
        # plus for each j, if i in conflicts[j], then edge. That's symmetric capture.
        # Simpler: adj[i] = ci | (bitset of j such that i in conflicts[j])
        # The second part: for each j, if (conflicts[j] >> i) & 1, then j conflicts with i.
        # Let's compute the reverse.
        rev = 0
        cj = conflicts
        for j in range(n):
            if j == i:
                continue
            if (cj[j] >> i) & 1:
                rev |= (1 << j)
        adj[i] = (ci | rev) & ~(1 << i)

    # Now adj[i] is the set of candidates incompatible with i.
    # We want a maximum independent set of size up to 48 in this graph.
    # Independent set: no two elements have an edge between them.

    # Greedy lower bound: build a large independent set.
    def greedy_iset(order):
        chosen = []
        chosen_bits = 0
        for x in order:
            if (adj[x] & chosen_bits) == 0:
                chosen.append(x)
                chosen_bits |= (1 << x)
        return chosen

    # Try several greedy orders to get a strong initial bound.
    best_greedy = []
    orders = [
        list(range(n)),
        list(range(n - 1, -1, -1)),
        # biased large-first
        [x for x in range(n) if vals[x] > N // 2] + [x for x in range(n) if vals[x] <= N // 2],
    ]
    for _ in range(30):
        o = list(range(n))
        random.shuffle(o)
        orders.append(o)
    for o in orders:
        g = greedy_iset(o)
        if len(g) > len(best_greedy):
            best_greedy = g

    best_size = len(best_greedy)
    best_set = [vals[i] for i in best_greedy]

    # Branch and bound for maximum independent set.
    # Represent candidate pool as a bitset of available vertices.
    # We'll do recursive search: pick a vertex, branch include/exclude, with bound.
    # Bound: greedy upper bound on independent set size in the induced subgraph on available.
    # Use a simple bound: |available|, but tighter via coloring or greedy packing.
    # A common fast bound: repeatedly pick a vertex, remove it and its neighbors, count picks (this is a
    # greedy independent set lower bound, not upper). For an UPPER bound we can use a greedy *clique cover* of the
    # complement, or a simple fractional bound. Simpler upper bound: |available| (trivial) is too loose.
    # Use: upper bound = size of a maximal independent set found greedily on available (this is a LOWER bound on the
    # true max, not upper). Hmm.
    # For an upper bound, we can use the fact that in the complement graph, independent set = clique. Upper bound on
    # clique size can be bounded by a greedy coloring of the complement (color classes are independent in complement =
    # cliques in original? no). Let's use a simpler but effective approach: branch-and-bound with the bound being
    # |available| minus a greedy packing lower bound is not an upper bound.
    #
    # Let's use the standard approach: bound = number of vertices in available (trivial) won't prune much.
    # Better: use a greedy coloring of the *complement* graph restricted to available to get an upper bound on the
    # maximum clique in the complement = maximum independent set in original.
    # Actually, a greedy coloring of a graph G gives an upper bound on the clique number of G (since a clique needs
    # distinct colors). So color the complement graph on available; the number of colors is an upper bound on the
    # max clique in the complement = max independent set in original. 
    #
    # Let's implement greedy coloring upper bound on the complement.

    # Complement adjacency: comp_adj[i] = available vertices not adjacent to i in original (and != i).
    # We'll compute on the fly within bound function given available bitset.

    def coloring_upper_bound(available):
        # Greedy color the complement graph induced on available.
        # In complement, edge between u,v iff NOT edge in original (u!=v).
        # We color greedily: assign smallest color not used by neighbors in complement.
        # Number of colors = upper bound on clique in complement = upper bound on independent set in original.
        # Work with list of vertices in available.
        verts = []
        a = available
        while a:
            lsb = a & (-a)
            v = lsb.bit_length() - 1
            verts.append(v)
            a ^= lsb
        if not verts:
            return 0
        # For each vertex, its complement-neighbors within available are: available & ~adj[v] & ~(1<<v)
        # Greedy coloring in order of verts.
        color_of = {}
        next_color = 0
        for v in verts:
            comp_neigh = available & ~adj[v] & ~(1 << v)
            # colors used by comp neighbors
            used = set()
            cn = comp_neigh
            while cn:
                lsb = cn & (-cn)
                u = lsb.bit_length() - 1
                if u in color_of:
                    used.add(color_of[u])
                cn ^= lsb
            c = 0
            while c in used:
                c += 1
            color_of[v] = c
            if c + 1 > next_color:
                next_color = c + 1
        return next_color

    # Branch and bound.
    # We'll search for an independent set of size > best_size.
    # State: available bitset, current chosen list, current size.
    # Pick a vertex v from available. Branch: include v (remove v and its neighbors from available), or exclude v.
    # To improve, choose v with high degree in the induced subgraph (to prune).
    # Bound: if current_size + coloring_upper_bound(available) <= best_size, prune.

    # To keep within time, limit recursion and use iterative deepening on target size.
    # Actually, we just want to beat best_size. Let's do a DFS that tries to find any independent set
    # of size best_size+1, best_size+2, ... up to 48. Use branch and bound.

    # Precompute degree within full graph for vertex ordering heuristic.
    full_deg = [bin(adj[i]).count('1') for i in range(n)]

    # We'll do iterative deepening: target = best_size+1, then +2, ...
    # For each target, run a bounded search that stops when it finds one set of that size.
    # Use a global found flag.

    # To make search fast, order vertices by descending full degree when picking.
    # Precompute a fixed ordering of vertices by descending degree for tie-breaking.
    vertex_order = sorted(range(n), key=lambda i: -full_deg[i])

    # We'll implement search(available, chosen_bits, chosen_list, current_size, target).
    # Returns True if found an independent set of size >= target.

    # Use a recursion with a node budget and time checks.
    node_budget = [0]
    time_budget = deadline - 8.0

    def find_independent_set(available, chosen_bits, chosen_list, current_size, target):
        if current_size >= target:
            return True
        # Bound
        if current_size + coloring_upper_bound(available) <= best_size:
            return False
        # If available empty
        if available == 0:
            return False
        # Time / node check
        if time.time() > time_budget:
            return False
        # Pick a vertex: choose one in available with highest degree within available.
        # Compute degrees within available for candidates in available, pick max.
        # To limit cost, just pick the first in vertex_order that is in available? 
        # Better: pick the vertex in available with max (adj[i] & available) popcount.
        # This is O(|available|) per node; acceptable.
        a = available
        best_v = -1
        best_d = -1
        while a:
            lsb = a & (-a)
            v = lsb.bit_length() - 1
            d = bin(adj[v] & available).count('1')
            if d > best_d:
                best_d = d
                best_v = v
            a ^= lsb
        if best_v < 0:
            return False
        # Branch 1: include best_v
        new_avail = available & ~(1 << best_v) & ~adj[best_v]
        chosen_list.append(best_v)
        if find_independent_set(new_avail, chosen_bits | (1 << best_v), chosen_list, current_size + 1, target):
            return True
        chosen_list.pop()
        # Branch 2: exclude best_v
        new_avail2 = available & ~(1 << best_v)
        if find_independent_set(new_avail2, chosen_bits, chosen_list, current_size, target):
            return True
        return False

    # Iterative deepening over target sizes.
    # Start from best_size+1 up to MAX_K.
    # For each, reset available to full set (all n vertices).
    full_avail = (1 << n) - 1

    # We'll also seed the search with the greedy set as a starting partial solution to speed up.
    # Actually, simpler: search from scratch for each target.

    improved = False
    for target in range(best_size + 1, MAX_K + 1):
        if time.time() > time_budget:
            break
        # Reset
        node_budget[0] = 0
        # Try to find an independent set of size >= target.
        # Use the greedy set as an initial partial to reduce work:
        # If greedy already has size >= target, we're done (but target > best_size, so no).
        # Just run search.
        chosen_list = []
        found = find_independent_set(full_avail, 0, chosen_list, 0, target)
        if found:
            # chosen_list now has an independent set of size >= target.
            # Update best.
            if len(chosen_list) > best_size:
                best_size = len(chosen_list)
                best_set = [vals[i] for i in chosen_list]
                improved = True
                # Continue to higher target? We can, but each is expensive. 
                # Since we found size best_size, try best_size+1 next.
                # Loop will continue to next target = old best_size+2? No, target increments.
                # Actually we found size >= target, which may be > target. Let's set best_size and 
                # continue searching for even larger. But to be safe, continue the loop.
                pass
        else:
            # No independent set of size target exists. Since we search increasing target,
            # and we found none for target, can there be one for larger target? 
            # No! If no independent set of size target, then none of size > target.
            # So we can stop.
            break

    # Local search refinement: try to swap elements to increase size or maintain.
    # Convert best_set to index set.
    # We'll do simple hill-climbing: try adding a random unused element and removing one used element
    # if it keeps validity and increases size. Since we want max size, we only accept if size increases.
    # But our set is already maximal-ish. Try to find a larger independent set via swaps is complex.
    # Instead, do a few random restarts of greedy with the current best as a seed.

    # Try random restarts of greedy to see if we can beat best_size.
    remaining = deadline - time.time()
    if remaining > 2.0:
        for _ in range(2000):
            if time.time() > deadline - 3.0:
                break
            o = list(range(n))
            random.shuffle(o)
            g = greedy_iset(o)
            if len(g) > best_size:
                best_size = len(g)
                best_set = [vals[i] for i in g]

    # Final: ensure we return a valid set of size best_size, capped at 48.
    if len(best_set) > MAX_K:
        best_set = best_set[:MAX_K]

    # Validate the final set (safety check).
    def is_valid(S):
        Sset = set(S)
        for a in S:
            others = [x for x in S if x != a]
            if len(others) >= 3:
                for tri in combinations(others, 3):
                    if sum(tri) % a == 0:
                        return False
        return True

    if not is_valid(best_set):
        # Fallback to a known valid set: just take the greedy from descending order.
        g = greedy_iset(list(range(n - 1, -1, -1)))
        best_set = [vals[i] for i in g]
        best_size = len(best_set)
        if len(best_set) > MAX_K:
            best_set = best_set[:MAX_K]

    return best_set
