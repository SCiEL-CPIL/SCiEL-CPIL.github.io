import time
from itertools import combinations

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    n = 12

    # adjacency helpers
    def build_adj(edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def is_valid(adj):
        # min degree >= 3
        for i in range(n):
            if len(adj[i]) < 3:
                return False
        # triangle-free
        for i in range(n):
            for j in adj[i]:
                for k in adj[j]:
                    if k != i and i in adj[k] and i < j and j < k:
                        return False
        return True

    def all_cycles(adj):
        # enumerate all simple cycles via DFS with canonical start = min vertex
        cycles = []
        for start in range(n):
            # only start from the minimum vertex in a cycle to avoid duplicates
            stack = [(start, [start], {start})]
            while stack:
                node, path, seen = stack.pop()
                for nb in adj[node]:
                    if nb == start and len(path) >= 4:
                        # close the cycle
                        cy = tuple(sorted(path))
                        cycles.append(cy)
                    elif nb not in seen and nb > start:
                        # only expand to vertices > start to keep start as min
                        stack.append((nb, path + [nb], seen | {nb}))
        return cycles

    def distinct_lengths(adj):
        cycles = all_cycles(adj)
        if not cycles:
            return set()
        return set(len(c) for c in cycles)

    def score_of(adj):
        if not is_valid(adj):
            return None
        sl = distinct_lengths(adj)
        if not sl:
            return None
        return len(sl)

    # Candidate generators
    def two_k33():
        e = []
        for a in [0,1,2]:
            for b in [3,4,5]:
                e.append((a,b))
        for a in [6,7,8]:
            for b in [9,10,11]:
                e.append((a,b))
        return e

    def two_4prisms():
        # 4-prism on vertices 0..5: two 4-cycles (0-1-2-3-0) and (4-5-6-7-4) plus rungs
        # Use vertices 0-5 for first, 6-11 for second
        e = []
        for base in (0, 6):
            # square 1: base, base+1, base+2, base+3
            sq1 = [base, base+1, base+2, base+3]
            sq2 = [base+4, base+5, base+2+3-2+3, base+3+4-3+4]  # messy, just hardcode
        # Let's hardcode cleanly:
        e = []
        # First prism on 0..5
        # top square: 0-1-2-3-0
        e += [(0,1),(1,2),(2,3),(3,0)]
        # bottom square: 4-5-? need 6 vertices total for a prism (2 squares of 4 would be 8)
        # A 4-prism actually has 8 vertices. So two 4-prisms need 16 vertices. Not possible on 12.
        # Instead use a single 8-cycle with perfect matching chords (cube) + 4 leftover? No.
        # So scrap 4-prism; use 8-cycle with chords.
        return None

    def single_8cycle_with_chords():
        # 8-cycle on 0..7 with 4 rungs connecting opposite? That gives triangles.
        # Instead: 8-cycle 0-1-2-3-4-5-6-7-0 plus chords (0,4),(1,5),(2,6),(3,7) -> creates 4-cycles? (0,1,5,4,0) is a 4-cycle, good, but also (0,4,5,1,0)? same. And 8-cycle still present -> lengths 4 and 8.
        # Leftover 4 vertices 8,9,10,11 must have degree >=3 but they're isolated -> invalid.
        return None

    def ring_12_with_chords():
        # 12-cycle 0-1-...-11-0 plus chords (i, i+4) for i in 0..3 -> creates 4-cycles? (0,1,5,4,0) yes 4-cycle. Also 12-cycle present. lengths {4,12}. All degree: each vertex has 2 cycle neighbors + 1 chord = 3. Good.
        e = []
        for i in range(12):
            e.append((i, (i+1) % 12))
        for i in range(4):
            e.append((i, i+4))
        return e

    def ring_12_step2():
        # 12-cycle plus chords (i, i+6) for i in 0..5 -> opposite chords. Each vertex deg 3. Creates 6-cycles? (0,1,7,6,0) is a 4-cycle! (0-1 edge, 1-7? no, 1's chord is 1+6=7, so 1-7 edge, 7-6 edge (cycle), 6-0? 6's chord is 6+6=0 mod 12 =0, so 6-0 edge. So 0-1-7-6-0 is a 4-cycle. So lengths {4,6,12}? The 12-cycle: 0-1-2-3-4-5-6-7-8-9-10-11-0. Is there a 6-cycle? 0-1-7-6-5-11-0? 11-0 yes, 5-11? no direct. Hmm. Anyway multiple lengths.
        e = []
        for i in range(12):
            e.append((i, (i+1) % 12))
        for i in range(6):
            e.append((i, (i+6) % 12))
        return e

    def k4_4():
        # K_{4,4} on 8 vertices 0..7, plus 4 vertices 8..11 as... need degree 3. Attach each to 3 vertices of K_{4,4}.
        # K_{4,4}: A={0,1,2,3}, B={4,5,6,7}
        e = []
        for a in [0,1,2,3]:
            for b in [4,5,6,7]:
                e.append((a,b))
        # Now vertices 8,9,10,11 need degree 3, triangle-free.
        # Connect 8 to 0,1,4 (all in different parts? 0,1 in A, 4 in B -> 8-0,8-1,8-4. No triangle since 0-1 not edge, 0-4 is edge but 8-1-4-8? 1-4 is edge, 4-8 edge, 8-1 edge -> triangle 1-4-8! Bad.
        # Connect 8 to 0,4,5: 0-4 edge, 4-5 no, 0-5 no, 4-5 no. 8-0,8-4,8-5. Triangle? 0-4-8: 0-4 yes,4-8 yes,8-0 yes -> triangle! Bad.
        # This is hard. Skip.
        return None

    # Best priors:
    priors = [two_k33(), ring_12_with_chords(), ring_12_step2()]
    priors = [p for p in priors if p is not None]

    best_edges = None
    best_score = 999
    best_lengths = None

    for e in priors:
        adj = build_adj(e)
        s = score_of(adj)
        if s is not None and s < best_score:
            best_score = s
            best_edges = e
            best_lengths = distinct_lengths(adj)

    # Local search: 1-switch and edge-swap moves
    import random
    random.seed(42)

    edges_set = set()
    for u,v in best_edges:
        edges_set.add((min(u,v), max(u,v)))

    def adj_from_set(es):
        adj = [set() for _ in range(n)]
        for u,v in es:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    cur_adj = adj_from_set(edges_set)
    cur_score = best_score

    while time.time() < deadline - 5:
        improved = False
        # Generate candidate moves
        moves = []
        es = set((min(u,v), max(u,v)) for u,v in cur_adj if u < v)

        # 1-switch: remove edge (a,b), add edge (c,d) where c in adj[a], d in adj[b], a!=c, b!=d, and no triangle
        for a in range(n):
            for b in list(cur_adj[a]):
                if a >= b: continue
                for c in cur_adj[a]:
                    if c == b: continue
                    for d in cur_adj[b]:
                        if d == a or d == c: continue
                        if min(c,d) > max(c,d): c,d = d,c
                        if (c,d) in es: continue
                        # Check triangle-freeness of new edge (c,d)
                        common = cur_adj[c] & cur_adj[d]
                        if common: continue
                        # Check min degree after removal
                        # Removing (a,b): deg a and b drop by 1; adding (c,d): deg c and d up by 1
                        # a's new neighbors: cur_adj[a] - {b} + {c} -> but we're just testing validity
                        # Simulate: new_adj = copy
                        new_es = set(es)
                        new_es.discard((min(a,b), max(a,b)))
                        new_es.add((min(c,d), max(c,d)))
                        new_adj = adj_from_set(new_es)
                        s = score_of(new_adj)
                        if s is not None and s < cur_score:
                            moves.append((s, new_es, '1switch'))
                            if not improved: improved = True

        # Edge-swap: remove (a,b),(c,d), add (a,c),(b,d) etc. (Wagner-like)
        edge_list = list(es)
        for i in range(len(edge_list)):
            for j in range(i+1, len(edge_list)):
                a1,b1 = edge_list[i]
                a2,b2 = edge_list[j]
                # Try reconnection: (a1,a2)+(b1,b2) or (a1,b2)+(b1,a2)
                for pair in [((a1,a2),(b1,b2)), ((a1,b2),(b1,a2))]:
                    n1,n2 = pair
                    if n1[0] >= n1[1]: n1 = (n1[1], n1[0])
                    if n2[0] >= n2[1]: n2 = (n2[1], n2[0])
                    if n1 == n2: continue
                    if n1 in es or n2 in es: continue
                    new_es = set(es)
                    new_es.discard(edge_list[i])
                    new_es.discard(edge_list[j])
                    new_es.add(n1)
                    new_es.add(n2)
                    new_adj = adj_from_set(new_es)
                    s = score_of(new_adj)
                    if s is not None and s < cur_score:
                        moves.append((s, new_es, 'swap'))
                        if not improved: improved = True

        if moves:
            moves.sort()
            s, new_es, _ = moves[0]
            cur_adj = adj_from_set(new_es)
            cur_score = s
            best_score = s
            best_edges = [(u,v) for u,v in new_es]
            best_lengths = distinct_lengths(cur_adj)
        else:
            # Try random restarts among priors
            for e in priors:
                adj = build_adj(e)
                s = score_of(adj)
                if s is not None and s < cur_score:
                    cur_adj = adj
                    cur_score = s
                    best_score = s
                    best_edges = e
                    best_lengths = distinct_lengths(adj)
                    break
            else:
                break

    # If we achieved score 1, great. Otherwise return best.
    # But ensure we always have a valid fallback: two K3,3 (score 2)
    if best_edges is None or best_score > 2:
        best_edges = two_k33()

    result = [[int(u), int(v)] for u,v in best_edges]
    result.sort()
    return result

