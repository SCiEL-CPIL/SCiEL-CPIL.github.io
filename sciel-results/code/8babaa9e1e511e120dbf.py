import time
import random
import heapq
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve_instance(n, p, seed):
        edges = random_graph(n, p, seed)
        if not edges:
            return []

        # Adjacency sets
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        m = len(edges)
        # edge index
        edge_to_idx = {}
        for idx, e in enumerate(edges):
            edge_to_idx[e] = idx

        # Line graph adjacency: two edges conflict (can't share a clique) if they don't share
        # a common vertex... wait, no. Two edges CAN share a clique if they are in a common triangle.
        # Actually for a clique partition: a set of edges forms a clique in G iff they are pairwise
        # "compatible" meaning each pair shares a common vertex AND that common vertex is the SAME?
        # No. A set of edges forms a clique in G iff all their endpoints form a clique.
        # For two edges e1=(a,b), e2=(c,d): they can be in the same clique iff {a,b,c,d} is a clique
        # in G, i.e., all 6 pairs are edges. But that's too strict for greedy.
        #
        # Actually, the constraint is: the set of edges must be the edge set of a clique in G.
        # A clique in G on vertex set S has all C(|S|,2) edges. So a "clique" in the partition
        # is determined by its vertex set S (|S|>=2), and includes ALL edges within S.
        #
        # So the partition is: choose a set of vertex-cliques S_1, S_2, ... such that every edge
        # belongs to exactly one S_i, and within each S_i, all edges are present (they must be,
        # since S_i is a clique).
        #
        # This is equivalent to: partition the edge set into cliques where each clique's edge set
        # is exactly the complete graph on some vertex subset.
        #
        # Greedy: repeatedly pick a large clique in the remaining-edge graph, add all its edges.
        # The "line graph coloring" framing: two edges are "compatible" (can be in same clique)
        # if they share a vertex AND the union of their endpoints is a clique. But simpler:
        # just find max clique in the subgraph induced by remaining edges.

        remaining = set(edges)

        def subgraph_adj(rem):
            a = [set() for _ in range(n)]
            for u, v in rem:
                a[u].add(v)
                a[v].add(u)
            return a

        def find_max_clique(vertices, adj_sub):
            """Bron-Kerbosch with pivot for max clique."""
            best = []
            def bk(R, P, X):
                nonlocal best
                if not P and not X:
                    if len(R) > len(best):
                        best = R[:]
                    return
                if not P:
                    return
                # pivot
                pivot = None
                max_cnt = -1
                for u in list(P) + list(X):
                    c = len(adj_sub[u] & P)
                    if c > max_cnt:
                        max_cnt = c
                        pivot = u
                if pivot is None:
                    return
                for v in list(P - adj_sub[pivot]):
                    bk(R + [v], P & adj_sub[v], X & adj_sub[v])
                    P.remove(v)
                    X.add(v)
            bk([], set(vertices), set())
            return best

        partition = []
        while remaining:
            if time.time() > deadline:
                break
            # active vertices
            active = set()
            for u, v in remaining:
                active.add(u)
                active.add(v)
            if not active:
                break
            adj_sub = subgraph_adj(remaining)
            clique = find_max_clique(list(active), adj_sub)
            if not clique:
                u, v = next(iter(remaining))
                clique = [u, v]
            partition.append(clique)
            # remove all edges within clique
            cs = set(clique)
            for u in clique:
                for v in clique:
                    if u < v:
                        remaining.discard((u, v))
                    elif u > v:
                        remaining.discard((v, u))

        # Local improvement: try to merge two cliques if their union is a clique in G
        # (i.e., all vertices in the union form a clique). This reduces count.
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            new_part = []
            used = [False] * len(partition)
            for i in range(len(partition)):
                if used[i]:
                    continue
                # try to merge with later cliques
                best_j = -1
                for j in range(i + 1, len(partition)):
                    if used[j]:
                        continue
                    union = set(partition[i]) | set(partition[j])
                    # check if union is a clique in G
                    ok = True
                    ul = list(union)
                    for a in range(len(ul)):
                        for b in range(a + 1, len(ul)):
                            if ul[b] not in adj[ul[a]]:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        best_j = j
                        break
                if best_j >= 0:
                    new_part.append(list(set(partition[i]) | set(partition[best_j])))
                    used[best_j] = True
                    improved = True
                else:
                    new_part.append(partition[i])
            partition = new_part

        return partition

    result1 = solve_instance(32, 0.6, 601)
    result2 = solve_instance(64, 0.55, 602)

    return [result1, result2]
