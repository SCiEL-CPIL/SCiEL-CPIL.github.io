import time
import random
import math
import sys
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit

    n = 12
    m = 40
    N = 1 << n

    def is_subset(a, b):
        return (a & b) == a and a != b

    def count_comparable(family):
        count = 0
        len_fam = len(family)
        for i in range(len_fam):
            ai = family[i]
            for j in range(i + 1, len_fam):
                aj = family[j]
                if is_subset(ai, aj) or is_subset(aj, ai):
                    count += 1
        return count

    def greedy_start():
        """Greedy construction: add set that maximizes comparability with current family."""
        family = []
        used = set()
        for _ in range(m):
            best_mask = -1
            best_gain = -1
            for mask in range(N):
                if mask in used:
                    continue
                gain = 0
                for f in family:
                    if (mask & f) == mask or (f & mask) == f:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_mask = mask
            if best_mask == -1:
                while True:
                    mask = random.randint(0, N - 1)
                    if mask not in used:
                        best_mask = mask
                        break
            family.append(best_mask)
            used.add(best_mask)
        return family

    def random_start():
        fam = []
        used = set()
        while len(fam) < m:
            mask = random.randint(0, N - 1)
            if mask not in used:
                used.add(mask)
                fam.append(mask)
        return fam

    def local_search(initial_family, deadline):
        family = initial_family[:]
        family_set = set(family)
        current_score = count_comparable(family)
        best_family = family[:]
        best_score = current_score
        temp = 10.0
        no_improve_count = 0

        iter_count = 0
        while time.time() < deadline:
            iter_count += 1

            if iter_count % 1000 == 0 and temp > 0.1:
                temp *= 0.995
                if no_improve_count > 5000:
                    no_improve_count = 0
                    temp = 5.0
                    continue

            idx = random.randint(0, m - 1)
            old_mask = family[idx]

            if random.random() < 0.6:
                new_mask = random.randint(0, N - 1)
            elif random.random() < 0.5:
                new_mask = old_mask ^ (1 << random.randint(0, n - 1))
            else:
                b1 = random.randint(0, n - 1)
                b2 = random.randint(0, n - 1)
                while b2 == b1:
                    b2 = random.randint(0, n - 1)
                new_mask = old_mask ^ (1 << b1) ^ (1 << b2)

            if new_mask == old_mask:
                continue
            if new_mask in family_set:
                continue

            lost = 0
            gained = 0
            for j in range(m):
                if j == idx:
                    continue
                other = family[j]
                if is_subset(old_mask, other) or is_subset(other, old_mask):
                    lost += 1
                if is_subset(new_mask, other) or is_subset(other, new_mask):
                    gained += 1

            delta = gained - lost

            accept = False
            if delta > 0:
                accept = True
            elif temp > 0.1:
                if random.random() < math.exp(delta / temp):
                    accept = True

            if accept:
                family[idx] = new_mask
                family_set.remove(old_mask)
                family_set.add(new_mask)
                current_score += delta
                no_improve_count = 0

                if current_score > best_score:
                    best_score = current_score
                    best_family = family[:]
            else:
                no_improve_count += 1

        return best_family, best_score

    # ---- Maximum-weight clique via branch-and-bound ----
    # Nodes are the 4096 subsets. Edge between u,v if one is a strict subset of the other.
    # We want a 40-clique maximizing total comparable pairs.
    # Precompute neighbor lists and degrees.

    # For speed, use a bitset representation for neighbors.
    # N = 4096, so we can use Python ints as bitsets (4096 bits = 512 bytes each).
    # Build adjacency as a list of Python int bitsets.

    # First, precompute for each mask its "comparable" set (all masks strictly containing or contained in it).
    # Two masks a,b are comparable iff a & b == a or a & b == b, and a != b.

    # Build adj bitsets
    adj = [0] * N
    for a in range(N):
        # all supersets of a (strict)
        # iterate over supersets
        # submasks of complement
        comp = (N - 1) ^ a
        # iterate all submasks of comp
        sub = comp
        while True:
            b = a | sub
            if b != a:
                adj[a] |= (1 << b)
            if sub == 0:
                break
            sub = (sub - 1) & comp
        # all subsets of a (strict)
        sub = a
        while True:
            b = sub
            if b != a:
                adj[a] |= (1 << b)
            if sub == 0:
                break
            sub = (sub - 1) & a

    # Degree of each node
    degree = [0] * N
    for i in range(N):
        # popcount of adj[i]
        x = adj[i]
        c = 0
        while x:
            c += 1
            x &= x - 1
        degree[i] = c

    # Weight of a clique of size k: number of comparable pairs = k*(k-1)/2
    # Since every pair in a clique is comparable, the weight is fixed for a given size.
    # So we just need the largest clique, or a 40-clique.
    # Actually we want exactly 40 nodes forming a clique, maximizing comparable pairs.
    # But if we have a clique of size > 40, we can take any 40 of them and they form a clique too.
    # So we want to find a clique of size at least 40.
    # The maximum number of comparable pairs for 40 sets is 40*39/2 = 780.
    # So the goal is to find 40 mutually comparable sets (a chain? no, a clique in comparability graph).
    # Wait: a clique in the comparability graph means every pair is comparable.
    # But in a Boolean lattice, if every pair is comparable, the sets form a chain (total order).
    # Because comparability is transitive? No, comparability is not transitive in general.
    # Actually, if A < B, B < C, then A < C. So if every pair is comparable, the sets are totally ordered, i.e., a chain.
    # So a clique in the comparability graph is exactly a chain.
    # The longest chain in B12 has size 13 (from empty set to full set).
    # So we cannot have a 40-clique! The maximum clique size is 13.
    # This means the problem is NOT to find a clique, but to maximize comparable pairs.
    # So the branch-and-bound for max clique is not directly applicable.
    # Let me reconsider.

    # The problem: maximize the number of comparable pairs among 40 sets.
    # This is equivalent to: choose 40 nodes to maximize the number of edges in the induced subgraph.
    # This is the "maximum induced subgraph" problem, or "densest subgraph" with fixed size.
    # This is NP-hard in general, but for this specific graph structure (comparability graph of Boolean lattice),
    # we can use the fact that the graph is a comparability graph (transitively orientable).

    # Alternative approach: the comparability graph of the Boolean lattice B_n is a well-studied graph.
    # The maximum number of comparable pairs in a family of k subsets is known.
    # For k = 40, n = 12, we can try to construct a family that maximizes comparable pairs.

    # Known result: the family that maximizes comparable pairs is a "nested" family, i.e., a union of chains.
    # Specifically, take as many chains as possible, each of length up to 13.
    # For 40 sets, we can take 3 chains of length 13 (39 sets) and 1 more set.
    # But chains must be disjoint in terms of elements? No, chains can share elements.
    # Actually, we can take multiple chains that are not necessarily disjoint.
    # But to maximize comparable pairs, we want the chains to be as long as possible and as "parallel" as possible.

    # Let me try a different approach: use the existing local search but with a better initialization.
    # The greedy_start already tries to maximize comparability. Let me try to improve it.

    # Actually, let me try to construct a family based on chains.
    # A chain in B12 is a sequence of sets where each is a subset of the next.
    # The longest chain has 13 sets.
    # If we take 40 sets, we can form them as 3 full chains (39 sets) + 1 extra.
    # But the chains should be chosen to maximize cross-chain comparability.
    # Two sets from different chains are comparable if one is a subset of the other.
    # To maximize this, we want the chains to be "aligned" in some sense.

    # Let me try: take chains that differ by shifting which elements are added.
    # For example, chain 1: {}, {0}, {0,1}, {0,1,2}, ..., {0,1,...,11}
    # Chain 2: {}, {1}, {1,0}, {1,0,2}, ..., {1,0,...,11}
    # But these chains share many comparable pairs.

    # Actually, let me just use the local search with a better greedy initialization.
    # The current greedy_start is already quite good. Let me try to add a chain-based initialization.

    def chain_based_start():
        """Create a family based on multiple chains."""
        family = []
        used = set()

        # Create chains by varying the order of adding elements
        # For each permutation of a subset of elements, create a chain
        # But we need 40 distinct sets.

        # Simple approach: create chains of length 13, 13, 13, 1
        # Chain 1: add elements in order 0,1,2,...,11
        for i in range(n + 1):
            mask = (1 << i) - 1  # {0, 1, ..., i-1}
            if mask not in used:
                family.append(mask)
                used.add(mask)

        # Chain 2: add elements in order 1,0,2,3,...,11
        # {1}, {1,0}, {1,0,2}, ...
        chain2 = []
        order2 = [1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        mask = 0
        for i in range(n + 1):
            if i > 0:
                mask |= (1 << order2[i - 1])
            if mask not in used:
                family.append(mask)
                used.add(mask)

        # Chain 3: add elements in order 2,0,1,3,4,...,11
        order3 = [2, 0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        mask = 0
        for i in range(n + 1):
            if i > 0:
                mask |= (1 << order3[i - 1])
            if mask not in used:
                family.append(mask)
                used.add(mask)

        # Add more sets to reach 40
        while len(family) < m:
            mask = random.randint(0, N - 1)
            if mask not in used:
                family.append(mask)
                used.add(mask)

        return family[:m]

    best_overall_family = None
    best_overall_score = 0

    # Try multiple initializations
    initializations = [
        greedy_start(),
        chain_based_start(),
        random_start(),
    ]

    for init in initializations:
        if time.time() > deadline - 5:
            break
        fam, score = local_search(init, deadline)
        if score > best_overall_score:
            best_overall_score = score
            best_overall_family = fam

    if best_overall_family is None:
        best_overall_family = list(range(m))

    result = []
    for mask in best_overall_family:
        result.append([b for b in range(n) if (mask >> b) & 1])

    return result
