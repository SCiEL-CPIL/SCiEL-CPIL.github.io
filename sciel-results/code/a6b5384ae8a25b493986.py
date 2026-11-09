import time
import sys
import random
from collections import deque
from itertools import combinations

_START = time.time()
_DEADLINE = None


def _check_validity(edges):
    """Check all constraints: 12 vertices, min degree >= 3, no triangles, has a cycle."""
    n = 12
    adj = [set() for _ in range(n)]
    for u, v in edges:
        if u == v:
            return False
        if not (0 <= u < n and 0 <= v < n):
            return False
        adj[u].add(v)
        adj[v].add(u)
    for i in range(n):
        if len(adj[i]) < 3:
            return False
    # Triangle-free check
    for u in range(n):
        for v in adj[u]:
            for w in adj[v]:
                if w != u and w in adj[u]:
                    return False
    return True


def _get_distinct_cycle_lengths(edges):
    """
    Computes the set of distinct lengths of all simple cycles in the graph.
    """
    n = 12
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    cycle_lengths = set()
    
    def dfs(start, current, visited):
        if len(current) > 12:
            return
        for neighbor in adj[current[-1]]:
            if neighbor == start and len(current) >= 4:
                cycle_lengths.add(len(current))
            elif neighbor > start and neighbor not in visited:
                visited.add(neighbor)
                dfs(start, current + [neighbor], visited)
                visited.remove(neighbor)

    for v in range(n):
        dfs(v, [v], {v})

    return cycle_lengths


def _find_cycle_lengths_fast(adj, n=12):
    """
    Efficiently find all distinct cycle lengths using DFS with pruning.
    adj: list of sets of neighbors
    """
    cycle_lengths = set()
    adj_list = [list(s) for s in adj]
    
    # For each vertex, find cycles where this vertex is the smallest
    for start in range(n):
        visited = {start}
        
        def dfs(current, length):
            if length > n:
                return
            for neighbor in adj_list[current]:
                if neighbor == start and length >= 4:
                    cycle_lengths.add(length)
                elif neighbor > start and neighbor not in visited:
                    visited.add(neighbor)
                    dfs(neighbor, length + 1)
                    visited.remove(neighbor)
        
        dfs(start, 1)
    
    return cycle_lengths


def construct():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 85.0

    def make_k33(offset):
        edges = []
        for a in range(3):
            for b in range(3):
                u = offset + a
                v = offset + 3 + b
                edges.append([int(u), int(v)])
        return edges

    # Fallback: Two disjoint K_{3,3} components -> cycle lengths {4, 6}, count = 2
    raw_edges = make_k33(0) + make_k33(6)
    seen = set()
    fallback = []
    for u, v in raw_edges:
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        if key not in seen:
            seen.add(key)
            fallback.append([key[0], key[1]])
    fallback.sort(key=lambda x: (x[0], x[1]))

    best_result = fallback
    best_score = 2

    def check_and_score(edges):
        """Check validity and return set of cycle lengths, or None if invalid."""
        n = 12
        adj = [set() for _ in range(n)]
        for u, v in edges:
            if u == v or not (0 <= u < n and 0 <= v < n):
                return None
            adj[u].add(v)
            adj[v].add(u)
        for i in range(n):
            if len(adj[i]) < 3:
                return None
        # Triangle-free
        for u in range(n):
            for v in adj[u]:
                for w in adj[v]:
                    if w != u and w in adj[u]:
                        return None
        # Find all cycle lengths
        cycle_lengths = _find_cycle_lengths_fast(adj, n)
        return cycle_lengths

    rng = random.Random(42)

    # Strategy 1: Try specific known constructions
    # Petersen graph (10 vertices, 3-regular, girth 5) + 2 more vertices
    petersen_edges = [
        [0,1],[1,2],[2,3],[3,4],[4,0],  # outer 5-cycle
        [5,7],[7,9],[9,6],[6,8],[8,5],  # inner star
        [0,5],[1,6],[2,7],[3,8],[4,9]   # spokes
    ]
    
    # Try to extend Petersen to 12 vertices with min degree 3
    for extra_edges in [
        [[10,0],[10,1],[10,2],[11,3],[11,4],[11,5]],
        [[10,0],[10,1],[10,5],[11,2],[11,3],[11,6]],
        [[10,0],[10,5],[10,11],[11,1],[11,6],[11,10]],
        [[10,0],[10,1],[10,11],[11,2],[11,3],[11,4]],
        [[10,5],[10,6],[10,7],[11,8],[11,9],[11,0]],
        [[10,0],[10,2],[10,4],[11,1],[11,3],[11,5]],
    ]:
        candidate = petersen_edges + extra_edges
        seen_e = set()
        clean = []
        for u, v in candidate:
            if u == v:
                continue
            k = (min(u,v), max(u,v))
            if k not in seen_e:
                seen_e.add(k)
                clean.append([k[0], k[1]])
        
        if len(clean) < 18:
            continue
        
        cl = check_and_score(clean)
        if cl is not None and len(cl) >= 1:
            if len(cl) == 1:
                best_result = sorted(clean, key=lambda x: (x[0], x[1]))
                best_score = 1
                break
            elif len(cl) < best_score:
                best_score = len(cl)
                best_result = sorted(clean, key=lambda x: (x[0], x[1]))

    if best_score == 1:
        return best_result

    # Strategy 2: Systematic search for 3-regular bipartite graphs
    # with few cycle lengths. Use structured generation.
    # Key insight: For bipartite graphs, all cycles are even length.
    # If we can make all cycles length 4, that's ideal (score 1).
    # A 3-regular bipartite graph on 12 vertices (6+6) with all cycles length 4
    # would be a union of C4s, but 3-regular bipartite with 6+6 vertices
    # that's a union of C4s is impossible (each C4 uses 2+2 vertices, 
    # and 3-regular means each vertex has degree 3, not 2).
    # So we need at least 2 cycle lengths. Let's try to get {4, 6} or {4, 8} etc.
    
    # Try random 3-regular bipartite graphs with more trials and better checking
    for trial in range(100000):
        if time.time() > _DEADLINE - 5:
            break

        a_side = list(range(6))
        b_side = list(range(6, 12))

        a_stubs = []
        for v in a_side:
            a_stubs.extend([v] * 3)
        b_stubs = []
        for v in b_side:
            b_stubs.extend([v] * 3)

        rng.shuffle(a_stubs)
        rng.shuffle(b_stubs)

        edges = []
        valid = True
        seen_pairs = set()
        for u, v in zip(a_stubs, b_stubs):
            if u == v:
                valid = False
                break
            key = (min(u, v), max(u, v))
            if key in seen_pairs:
                valid = False
                break
            seen_pairs.add(key)
            edges.append([u, v])

        if not valid or len(edges) != 18:
            continue

        cl = check_and_score(edges)
        if cl is not None and len(cl) >= 1:
            if len(cl) == 1:
                best_result = sorted(edges, key=lambda x: (x[0], x[1]))
                best_score = 1
                break
            elif len(cl) < best_score:
                best_score = len(cl)
                best_result = sorted(edges, key=lambda x: (x[0], x[1]))

    if best_score == 1:
        return best_result

    # Strategy 3: Try 4-regular bipartite graphs (more edges, potentially fewer cycle lengths)
    for trial in range(30000):
        if time.time() > _DEADLINE - 5:
            break

        a_side = list(range(6))
        b_side = list(range(6, 12))

        a_stubs = []
        for v in a_side:
            a_stubs.extend([v] * 4)
        b_stubs = []
        for v in b_side:
            b_stubs.extend([v] * 4)

        rng.shuffle(a_stubs)
        rng.shuffle(b_stubs)

        edges = []
        valid = True
        seen_pairs = set()
        for u, v in zip(a_stubs, b_stubs):
            if u == v:
                valid = False
                break
            key = (min(u, v), max(u, v))
            if key in seen_pairs:
                valid = False
                break
            seen_pairs.add(key)
            edges.append([u, v])

        if not valid or len(edges) != 24:
            continue

        cl = check_and_score(edges)
        if cl is not None and len(cl) >= 1:
            if len(cl) == 1:
                best_result = sorted(edges, key=lambda x: (x[0], x[1]))
                best_score = 1
                break
            elif len(cl) < best_score:
                best_score = len(cl)
                best_result = sorted(edges, key=lambda x: (x[0], x[1]))

    if best_score == 1:
        return best_result

    # Strategy 4: Try 5-regular bipartite graphs
    for trial in range(15000):
        if time.time() > _DEADLINE - 5:
            break

        a_side = list(range(6))
        b_side = list(range(6, 12))

        a_stubs = []
        for v in a_side:
            a_stubs.extend([v] * 5)
        b_stubs = []
        for v in b_side:
            b_stubs.extend([v] * 5)

        rng.shuffle(a_stubs)
        rng.shuffle(b_stubs)

        edges = []
        valid = True
        seen_pairs = set()
        for u, v in zip(a_stubs, b_stubs):
            if u == v:
                valid = False
                break
            key = (min(u, v), max(u, v))
            if key in seen_pairs:
                valid = False
                break
            seen_pairs.add(key)
            edges.append([u, v])

        if not valid or len(edges) != 30:
            continue

        cl = check_and_score(edges)
        if cl is not None and len(cl) >= 1:
            if len(cl) == 1:
                best_result = sorted(edges, key=lambda x: (x[0], x[1]))
                best_score = 1
                break
            elif len(cl) < best_score:
                best_score = len(cl)
                best_result = sorted(edges, key=lambda x: (x[0], x[1]))

    if best_score == 1:
        return best_result

    # Strategy 5: Random non-bipartite triangle-free graphs
    for trial in range(20000):
        if time.time() > _DEADLINE - 5:
            break

        n = 12
        stubs = []
        for v in range(n):
            stubs.extend([v] * 3)
        rng.shuffle(stubs)
        
        edges = []
        valid = True
        seen_pairs = set()
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                valid = False
                break
            key = (min(u, v), max(u, v))
            if key in seen_pairs:
                valid = False
                break
            seen_pairs.add(key)
            edges.append([u, v])

        if not valid or len(edges) != 18:
            continue

        cl = check_and_score(edges)
        if cl is not None and len(cl) >= 1:
            if len(cl) == 1:
                best_result = sorted(edges, key=lambda x: (x[0], x[1]))
                best_score = 1
                break
            elif len(cl) < best_score:
                best_score = len(cl)
                best_result = sorted(edges, key=lambda x: (x[0], x[1]))

    if best_score == 1:
        return best_result

    # Strategy 6: Try complete bipartite K_{3,3} x K_{3,3} with modifications
    # Actually, let's try a different approach: use a known graph with few cycle lengths
    # The complete bipartite graph K_{3,3} has cycle lengths {4, 6}
    # Two copies give {4, 6}
    # What about a 3-regular bipartite graph that's a single connected component?
    # The 3-cube Q3 is 3-regular bipartite on 8 vertices with cycle lengths {4, 6, 8}
    # That's 3 distinct lengths, worse than 2.
    
    # Let's try to find a 3-regular bipartite graph on 12 vertices with only {4, 6}
    # by being more systematic. Generate all possible 3-regular bipartite graphs
    # on 6+6 vertices and check.
    
    # Actually, let's just return the best we have
    if best_score == 2:
        return fallback
    return best_result
