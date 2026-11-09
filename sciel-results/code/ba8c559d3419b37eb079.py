import time
import random
from itertools import combinations, permutations

N = 16
# Cube vertices 0..7 as 3-bit; edges differ by one bit
CUBE = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
# Automorphism group of Q3 (48 elements) as permutations of {0..7}
def _gen_automorphisms():
    g = []
    for perm in permutations(range(8)):
        p = list(perm)
        ok = all(abs(bin(p[u] ^ p[v]).count('1')) == 1 for u, v in CUBE)
        if ok:
            g.append(p)
    return g
AUTOS = _gen_automorphisms()  # 48 elements

# All distinct cube embeddings on labels 0..7 (mod automorphism)
def _all_embeddings():
    seen = set()
    out = []
    for perm in permutations(range(8)):
        p = list(perm)
        emb = tuple(sorted((min(p[u], p[v]), max(p[u], p[v])) for u, v in CUBE))
        if emb in seen:
            continue
        # check automorphism-equivalent to an existing one
        dup = False
        for a in AUTOS:
            emb2 = tuple(sorted((min(a[u], a[v]), max(a[u], a[v])) for u, v in CUBE))
            # apply p to emb2? Actually check if p(emb) == existing
        # simpler: canonical form under autos
        canon = min(
            tuple(sorted((min(a[p[u]], a[p[v]]), max(a[p[u]], a[p[v]])) for u, v in CUBE))
            for a in AUTOS
        )
        if canon in seen:
            continue
        seen.add(canon)
        out.append(p)
    return out
EMBEDS = _all_embeddings()  # 840 embeddings (permutations p: cube_label -> subset_label)

# Precompute for each embedding: for each cube-edge index, the pair of subset-labels
EMB_PAIRS = [[(p[u], p[v]) for u, v in CUBE] for p in EMBEDS]

def _build_adj(edge_set):
    adj = [0]*N
    for u, v in edge_set:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    return adj

def _has_q3(edge_set, adj):
    """Return True if graph contains Q3 as a subgraph."""
    for subset in combinations(range(N), 8):
        sset = set(subset)
        # For each embedding, check if all 12 cube-edges are present in G[subset]
        for pairs in EMB_PAIRS:
            ok = True
            for a, b in pairs:
                if not (adj[subset[a]] >> subset[b]) & 1:
                    ok = False
                    break
            if ok:
                return True
    return False

def _has_q3_with_edge(edge_set, adj, u, v):
    """Check if adding edge (u,v) to current graph creates a Q3 where (u,v) is a cube edge.
    Only checks embeddings where (u,v) appears as a cube edge."""
    # (u,v) must be a cube edge. In cube, edges connect labels differing by 1 bit.
    # For each pair of cube-vertices (cu, cv) that form a cube edge, and for each embedding,
    # if embedding maps cu->u and cv->v (or cu->v and cv->u), check the rest.
    # Simpler: iterate embeddings, and for each, check if there's an assignment where u,v map to a cube edge.
    # Even simpler: for each 6-subset T of the other 14 vertices, the 8-set is {u,v}+T.
    # For that 8-set, check all embeddings.
    others = [x for x in range(N) if x != u and x != v]
    for T in combinations(others, 6):
        subset = (u, v) + T
        sset = set(subset)
        # check all embeddings on this subset
        for pairs in EMB_PAIRS:
            ok = True
            for a, b in pairs:
                sa, sb = subset[a], subset[b]
                if sa == u and sb == v:
                    continue  # this is the new edge
                if sa == v and sb == u:
                    continue
                if not (adj[sa] >> sb) & 1:
                    ok = False
                    break
            if ok:
                return True
    return False

def construct():
    start = time.time()
    deadline = start + 115  # 35s safety margin before 150s kill

    # Start with K16
    all_edges = [(i, j) for i in range(N) for j in range(i+1, N)]
    edge_set = set(all_edges)
    adj = _build_adj(edge_set)

    # Phase 1: Greedy removal. While Q3 exists, remove an edge that's in many Q3s.
    # For speed, just remove a random edge from a found Q3.
    def find_q3():
        """Return a Q3 as a set of 8 vertices + embedding, or None."""
        for subset in combinations(range(N), 8):
            for pairs in EMB_PAIRS:
                ok = True
                for a, b in pairs:
                    if not (adj[subset[a]] >> subset[b]) & 1:
                        ok = False
                        break
                if ok:
                    return subset, pairs
        return None

    # Greedy: repeatedly find a Q3 and remove one of its edges
    removed_count = 0
    while time.time() < deadline - 25:
        found = find_q3()
        if found is None:
            break
        subset, pairs = found
        # Remove a random edge from this Q3
        idx = random.randrange(12)
        a, b = pairs[idx]
        e = (min(subset[a], subset[b]), max(subset[a], subset[b]))
        edge_set.discard(e)
        adj[subset[a]] &= ~(1 << subset[b])
        adj[subset[b]] &= ~(1 << subset[a])
        removed_count += 1

    # Phase 2: Local search - try to add back removed edges
    removed_edges = [e for e in all_edges if e not in edge_set]
    random.shuffle(removed_edges)

    def try_add(e):
        u, v = e
        if (adj[u] >> v) & 1:
            return True
        if _has_q3_with_edge(edge_set, adj, u, v):
            return False
        edge_set.add(e)
        adj[u] |= 1 << v
        adj[v] |= 1 << u
        return True

    # Add back edges greedily
    progress = True
    while progress and time.time() < deadline - 10:
        progress = False
        for e in removed_edges:
            if e in edge_set:
                continue
            if try_add(e):
                progress = True
                removed_edges.remove(e)
                if time.time() > deadline - 10:
                    break

    # Phase 3: 1-swap local search - remove one edge, add another
    if time.time() < deadline - 5:
        for _ in range(500):
            if time.time() > deadline - 5:
                break
            # Try a random swap: remove e1, add e2
            if not edge_set:
                break
            e1 = random.choice(list(edge_set))
            e2 = random.choice(removed_edges) if removed_edges else None
            if e2 is None:
                break
            # remove e1
            u1, v1 = e1
            edge_set.discard(e1)
            adj[u1] &= ~(1 << v1)
            adj[v1] &= ~(1 << u1)
            # try add e2
            u2, v2 = e2
            if not _has_q3_with_edge(edge_set, adj, u2, v2):
                edge_set.add(e2)
                adj[u2] |= 1 << v2
                adj[v2] |= 1 << u2
                removed_edges.remove(e2)
                removed_edges.append(e1)
            else:
                # revert
                edge_set.add(e1)
                adj[u1] |= 1 << v1
                adj[v1] |= 1 << u1

    # Final verification (should be Q3-free by construction)
    # Build result
    result = [[u, v] for u, v in edge_set]
    return result

if __name__ == "__main__":
    # Quick self-test
    import sys
    sys.setrecursionlimit(10000)
    edges = construct()
    print(f"Edges: {len(edges)}")
    # Verify no Q3
    es = set((min(u,v), max(u,v)) for u, v in edges)
    adj = _build_adj(es)
    assert not _has_q3(es, adj), "Q3 found!"
    print("Q3-free verified.")
