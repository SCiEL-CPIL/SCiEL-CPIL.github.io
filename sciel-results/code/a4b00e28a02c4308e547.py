"""Triangle-free graphs without independent 10-sets (Ramsey number R(3,10))."""
import time
import random

_DEADLINE = 120.0  # seconds; evaluator kills at 150s, keep safety margin.


def _deadline_guard(start, margin=5.0):
    return (time.time() - start) > (_DEADLINE - margin)


def construct():
    start = time.time()
    random.seed(42)

    # We want a triangle-free graph on n vertices with no independent set of size 10.
    # The known best is 39. We'll try to construct one via a greedy/randomized approach
    # and verify with bitset-based checks.

    # Strategy: Use a known construction or search for a valid graph.
    # The Andrásfai graph And(9) has 26 vertices. We need larger.
    # 
    # For R(3,10) > 39, we need a triangle-free graph on 39+ vertices with alpha < 10.
    # 
    # Approach: Build a graph greedily. Start with n=40 and try to find a triangle-free
    # graph with independence number <= 9.
    #
    # We'll use a randomized local search:
    # 1. Start with a random graph or a structured graph.
    # 2. Ensure no triangles (remove edges from triangles).
    # 3. Check independence number; if >= 10, add edges to break independent sets.
    # 4. Iterate.

    # Better approach: Use the fact that the complement of a triangle-free graph with
    # alpha <= 9 means the complement has no K10 and the original has no K3.
    # So we need a graph G such that G is K3-free and G_bar is K10-free.
    # Equivalently, G is a (K3, K10)-free graph in the sense of Ramsey.

    # Let's try a structured approach: circulant graphs.
    # For n vertices, consider G where i~j if (j-i) mod n is in some set S.
    # Triangle-free means no a,b,c with a-b, b-c, c-a in S (mod n).
    # Alpha <= 9 means no 10 vertices with all pairwise differences not in S.

    # Let's try n=40 first, then 39, etc.

    best_n = 26  # Andrásfai as fallback
    best_edges = [[i, j] for i in range(26) for j in range(i + 1, 26) if (j - i) % 26 % 3 == 1]

    for n in range(40, 25, -1):
        if _deadline_guard(start):
            break
        if n <= best_n:
            break
        found = _search_circulant(n, start)
        if found:
            best_n = n
            best_edges = found
            break
        # Also try random local search
        found = _random_local_search(n, start)
        if found:
            best_n = n
            best_edges = found
            break

    return {'n': best_n, 'edges': best_edges}


def _has_triangle_bitset(adj):
    """Check if graph with adjacency bitsets has a triangle."""
    n = len(adj)
    for i in range(n):
        # For each neighbor j of i, check if any neighbor of j is also a neighbor of i
        ai = adj[i]
        # Get neighbors of i
        j = 0
        while j < n:
            if (ai >> j) & 1:
                # Check if adj[j] & ai has any bit set (common neighbor)
                if adj[j] & ai:
                    return True
            j += 1
    return False


def _independence_number_bitset(adj, n, target=10):
    """Check if there's an independent set of size >= target using branch and bound."""
    # Independent set in G = clique in complement
    # Complement adjacency: adj_comp[i] = ~adj[i] & ((1 << n) - 1) & ~(1 << i)
    comp = [0] * n
    full_mask = (1 << n) - 1
    for i in range(n):
        comp[i] = full_mask ^ adj[i] ^ (1 << i)

    # Branch and bound to find clique of size >= target in complement
    # Use a simple recursive approach with pruning
    best = [0]

    def dfs(clique, candidates, size):
        if size >= target:
            return True
        if not candidates:
            return False
        if size + candidates.bit_count() <= best[0]:
            return False
        # Try vertices in candidates
        c = candidates
        while c:
            v_bit = c & (-c)
            v = v_bit.bit_length() - 1
            c ^= v_bit
            new_candidates = candidates & comp[v]
            if size + 1 + new_candidates.bit_count() <= best[0]:
                continue
            if size + 1 > best[0]:
                best[0] = size + 1
            if dfs(clique | v_bit, new_candidates, size + 1):
                return True
        return False

    return dfs(0, full_mask, 0)


def _is_valid(n, edges):
    """Check if graph is triangle-free and has no independent set of size 10."""
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    # Check triangle-free
    for i in range(n):
        ai = adj[i]
        j = 0
        while j < n:
            if (ai >> j) & 1:
                if adj[j] & ai:
                    return False
            j += 1
    # Check independence number < 10
    if _independence_number_bitset(adj, n, 10):
        return False
    return True


def _search_circulant(n, start):
    """Search for a valid circulant graph on n vertices."""
    # S is the set of offsets. i~j iff (j-i) mod n in S.
    # By symmetry, S = -S mod n, so we only need offsets 1..n//2.
    # Triangle-free: no a in S such that there exist b, c in S with a+b+c ≡ 0 mod n
    #   (considering the triangle i, i+a, i+a+b; we need a, b, -(a+b) all in S or similar)
    # Actually for circulant: triangle i, i+a, i+b exists if a, b, a-b (mod n) all in S ∪ -S... 
    # More carefully: vertices i, i+a, i+b form triangle if a∈S, b∈S, and (b-a) mod n ∈ S ∪ -S.
    # Since S is symmetric, (b-a) mod n ∈ S.
    # So: for all a,b in S, (b-a) mod n not in S.

    # Independence: no 10 vertices with all pairwise differences not in S.
    # This is harder to check directly.

    # Let's just try random symmetric S sets and verify.
    max_off = n // 2
    for trial in range(2000):
        if _deadline_guard(start):
            return None
        # Random symmetric set
        S = set()
        for off in range(1, max_off + 1):
            if random.random() < 0.3:
                S.add(off)
                S.add(n - off)
        if not S:
            continue
        # Build adjacency
        adj = [0] * n
        for i in range(n):
            for s in S:
                j = (i + s) % n
                adj[i] |= (1 << j)
        # Check triangle-free
        valid = True
        for i in range(n):
            ai = adj[i]
            j = 0
            while j < n:
                if (ai >> j) & 1:
                    if adj[j] & ai:
                        valid = False
                        break
                j += 1
            if not valid:
                break
        if not valid:
            continue
        # Check independence
        if _independence_number_bitset(adj, n, 10):
            continue
        # Convert to edge list
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges
    return None


def _random_local_search(n, start):
    """Random local search for a valid graph."""
    for trial in range(500):
        if _deadline_guard(start):
            return None
        # Start with random graph
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.3:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        # Remove triangles
        changed = True
        while changed and not _deadline_guard(start):
            changed = False
            for i in range(n):
                ai = adj[i]
                j = 0
                while j < n:
                    if (ai >> j) & 1:
                        common = adj[j] & ai
                        if common:
                            # Remove one edge to break triangle
                            # Remove edge (i, j)
                            adj[i] &= ~(1 << j)
                            adj[j] &= ~(1 << i)
                            changed = True
                            ai = adj[i]
                    j += 1
        # Now triangle-free. Check independence.
        if not _independence_number_bitset(adj, n, 10):
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (adj[i] >> j) & 1:
                        edges.append([i, j])
            return edges
    return None


if __name__ == "__main__":
    result = construct()
    print(f"n={result['n']}, edges={len(result['edges'])}")
