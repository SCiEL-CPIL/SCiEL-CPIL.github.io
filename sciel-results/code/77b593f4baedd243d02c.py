import time
import random
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

n = P['n']
edge_set = set()
adj = [set() for _ in range(n)]
for u, v in P['edges']:
    edge_set.add((min(u, v), max(u, v)))
    adj[u].add(v)
    adj[v].add(u)

m = len(edge_set)
edge_list = list(edge_set)
edge_idx = {e: i for i, e in enumerate(edge_list)}

# --- Build maximal bicliques via edge-union enumeration ---
def maximal_bicliques():
    # For each edge (u,v), candidate maximal bicliques:
    # L = {u} ∪ (N(v) \ N(u))  , R = {v} ∪ (N(u) \ N(v))
    # (and the symmetric swap). These are the only maximal bicliques
    # that can contain that edge.
    seen = set()
    bicliques = []
    for (u, v) in edge_list:
        for Lcand, Rcand in [
            ({u} | (adj[v] - adj[u]), {v} | (adj[u] - adj[v])),
            ({v} | (adj[u] - adj[v]), {u} | (adj[v] - adj[u])),
        ]:
            Lcand = frozenset(Lcand)
            Rcand = frozenset(Rcand)
            if Lcand & Rcand:
                continue
            key = (Lcand, Rcand)
            if key in seen:
                continue
            # verify it is a valid biclique
            ok = True
            for a in Lcand:
                for b in Rcand:
                    if (min(a, b), max(a, b)) not in edge_set:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            seen.add(key)
            bicliques.append((Lcand, Rcand))
    return bicliques

bicliques = maximal_bicliques()
nb = len(bicliques)

# For each biclique, the frozenset of edge indices it covers
bic_edges = []
for L, R in bicliques:
    es = set()
    for a in L:
        for b in R:
            e = (min(a, b), max(a, b))
            es.add(edge_idx[e])
    bic_edges.append(es)

# --- Set cover branch and bound ---
# Universe = set of edge indices. We need to cover all m edges.
# Greedy upper bound first.
def greedy_cover():
    uncovered = set(range(m))
    chosen = []
    remaining = [set(es) for es in bic_edges]
    while uncovered:
        best_i = -1
        best_gain = 0
        for i in range(nb):
            if not remaining[i]:
                continue
            gain = len(remaining[i] & uncovered)
            if gain > best_gain:
                best_gain = gain
                best_i = i
        if best_i == -1:
            break
        chosen.append(best_i)
        chosen_set = remaining[best_i]
        uncovered -= chosen_set
        for i in range(nb):
            if i != best_i:
                remaining[i] -= chosen_set
    return chosen

ub = greedy_cover()
ub_count = len(ub)

# Precompute per-edge list of bicliques covering it (for lower bound)
edge_to_bic = [set() for _ in range(m)]
for i, es in enumerate(bic_edges):
    for e in es:
        edge_to_bic[e].add(i)

def lower_bound(uncovered, chosen_set):
    # disjoint-edges bound: each chosen biclique covers at most one edge
    # from any set of pairwise non-co-biclique edges.
    # Cheap bound: greedy set-cover LP-ish via maximal packing.
    # Use: number of edges that are each in a distinct "essential" group.
    # Simpler: count edges; each biclique covers at most max_size edges.
    # Stronger: find a set of edges no two share a biclique -> each needs own.
    # We do a greedy packing:
    pack = 0
    used_bics = set()
    rem = list(uncovered)
    # sort by rarity
    rem.sort(key=lambda e: len(edge_to_bic[e]))
    for e in rem:
        cand = [b for b in edge_to_bic[e] if b not in used_bics]
        if not cand:
            continue
        # pick the biclique covering fewest other uncovered edges
        b = min(cand, key=lambda b: len(bic_edges[b] & uncovered))
        pack += 1
        used_bics.add(b)
    return pack

best_solution = list(ub)
best_count = ub_count
nodes = [0]
start = time.time()
DEADLINE = 80.0

def solve(uncovered, chosen, count):
    global best_solution, best_count
    nodes[0] += 1
    if time.time() > DEADLINE:
        return
    if count >= best_count:
        return
    if not uncovered:
        best_solution = list(chosen)
        best_count = count
        return
    lb = lower_bound(uncovered, set(chosen))
    if count + lb >= best_count:
        return
    # pick the rarest uncovered edge
    e = min(uncovered, key=lambda x: len(edge_to_bic[x]))
    cand = sorted([b for b in edge_to_bic[e]],
                  key=lambda b: -len(bic_edges[b] & uncovered))
    for b in cand:
        new_unc = uncovered - bic_edges[b]
        chosen.append(b)
        solve(new_unc, chosen, count + 1)
        chosen.pop()
        if time.time() > DEADLINE:
            return

full = set(range(m))
solve(full, [], 0)

# Fallback to greedy if B&B found nothing better
if best_count > ub_count:
    best_solution = list(ub)
    best_count = ub_count

# Convert biclique indices back to [left, right] lists
result = []
for b in best_solution:
    L, R = bicliques[b]
    result.append([sorted(int(x) for x in L), sorted(int(x) for x in R)])

# Verify validity; if invalid, fall back to a trivially valid per-edge cover
def valid(result):
    seen = set()
    for L, R in result:
        if not L or not R:
            return False
        for a in L:
            if a in R:
                return False
        for a in L:
            for b in R:
                e = (min(a, b), max(a, b))
                if e not in edge_set:
                    return False
                if e in seen:
                    return False
                seen.add(e)
    return seen == edge_set

if not valid(result):
    # trivial cover: each edge its own biclique
    result = [[ [u], [v] ] for (u, v) in edge_list]

def construct():
    return result

# Build result eagerly so construct() is fast and deterministic
result = result  # already computed above
