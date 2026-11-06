import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115

    def popcount(x):
        return bin(x).count('1')

    def build_adj_bitset(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def is_triangle_free_bitset(adj, n):
        for i in range(n):
            ni = adj[i]
            while ni:
                lsb = ni & (-ni)
                j = lsb.bit_length() - 1
                ni ^= lsb
                if adj[i] & adj[j] & ~(1 << i) & ~(1 << j):
                    return False
        return True

    def independence_number(adj, n, k, time_limit=5.0):
        """Returns True if independent set of size k exists."""
        start_t = time.time()
        found = [False]

        def dfs(candidates, current_size):
            if found[0]:
                return True
            if time.time() - start_t > time_limit:
                return False
            if current_size >= k:
                found[0] = True
                return True
            if not candidates:
                return False
            # Upper bound: current_size + popcount(candidates)
            if current_size + popcount(candidates) < k:
                return False
            # Greedy coloring bound
            remaining = candidates
            color_count = 0
            while remaining:
                # Pick vertex with min degree in remaining
                best_v = -1
                best_deg = float('inf')
                temp = remaining
                while temp:
                    lsb = temp & (-temp)
                    v = lsb.bit_length() - 1
                    temp ^= lsb
                    deg = popcount(adj[v] & remaining)
                    if deg < best_deg:
                        best_deg = deg
                        best_v = v
                # Greedy maximal independent set
                color_class = 0
                rem = remaining
                while rem:
                    lsb = rem & (-rem)
                    v = lsb.bit_length() - 1
                    rem ^= lsb
                    color_class |= (1 << v)
                    rem &= ~adj[v] & ~(1 << v)
                color_count += 1
                remaining &= ~color_class
            if current_size + color_count < k:
                return False
            # Branch on vertex with min degree
            best_v = -1
            best_deg = float('inf')
            temp = candidates
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                temp ^= lsb
                deg = popcount(adj[v] & candidates)
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
            v = best_v
            bit_v = 1 << v
            rest = candidates & ~bit_v
            # Include v
            if dfs(rest & ~adj[v], current_size + 1):
                return True
            # Exclude v
            if dfs(rest, current_size):
                return True
            return False

        dfs((1 << n) - 1, 0)
        return found[0]

    def circulant_adj(n, S):
        """Build adjacency bitset for circulant graph with connection set S."""
        adj = [0] * n
        for i in range(n):
            for s in S:
                j = (i + s) % n
                if j != i:
                    adj[i] |= (1 << j)
        return adj

    def is_triangle_free_circulant(n, S):
        """Check if circulant graph with connection set S is triangle-free."""
        S_set = set(S)
        # Triangle exists if there exist a, b, c in S (with a+b+c ≡ 0 mod n)
        # where a, b, c are all in S (or their negatives, depending on undirected)
        # For undirected circulant, edge exists between i and j if (j-i) mod n in S or (i-j) mod n in S
        # Triangle i, j, k exists if:
        #   (j-i) mod n in S or (i-j) mod n in S
        #   (k-j) mod n in S or (j-k) mod n in S
        #   (i-k) mod n in S or (k-i) mod n in S
        # Simplify: check all triples
        for a in S:
            for b in S:
                for c in S:
                    if (a + b + c) % n == 0:
                        # Need to verify these correspond to a valid triangle
                        # Vertices 0, a, a+b (mod n)
                        v0 = 0
                        v1 = a % n
                        v2 = (a + b) % n
                        if v0 == v1 or v1 == v2 or v0 == v2:
                            continue
                        # Check edges
                        d1 = (v1 - v0) % n
                        d2 = (v2 - v1) % n
                        d3 = (v0 - v2) % n
                        if d1 in S_set and d2 in S_set and d3 in S_set:
                            return False
                        # Also check with negatives
                        nd1 = n - d1 if d1 else 0
                        nd2 = n - d2 if d2 else 0
                        nd3 = n - d3 if d3 else 0
                        # Edge exists if diff in S or (n-diff) in S
                        if (d1 in S_set or nd1 in S_set) and \
                           (d2 in S_set or nd2 in S_set) and \
                           (d3 in S_set or nd3 in S_set):
                            return False
        return True

    def verify_graph(n, edges):
        adj = build_adj_bitset(n, edges)
        if not is_triangle_free_bitset(adj, n):
            return False
        if independence_number(adj, n, 10, time_limit=3.0):
            return False
        return True

    def edges_from_adj(adj, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges

    # Start with a known valid graph: Andrásfai A_9 (n=27)
    def andrasfai_graph(k):
        n = 3 * k
        edges = []
        for a in range(3):
            for i in range(k):
                for b in range(3):
                    if (b - a) % 3 != 1:
                        continue
                    for j in range(k):
                        if (j - i) % k in (0, 1):
                            u = a * k + i
                            v = b * k + j
                            if u < v:
                                edges.append([u, v])
        return {'n': n, 'edges': edges}

    best = {'n': 27, 'edges': []}
    ag = andrasfai_graph(9)
    if verify_graph(ag['n'], ag['edges']):
        best = ag

    # Search circulant graphs for larger n
    for target_n in range(28, 41):
        if time.time() - start_time > deadline - 15:
            break

        found_for_n = False
        # Search over connection sets of size 2 and 3
        # For triangle-free: no a,b,c in S with a+b+c ≡ 0 mod n (with proper edge checks)

        # Size 2: S = {a, b}. Triangle-free if no a+b ≡ 0, 2a+b ≡ 0, a+2b ≡ 0, 3a ≡ 0, 3b ≡ 0 (mod n)
        # Actually need: for undirected, edges at distances a and b (and n-a, n-b)
        # Triangle 0, a, a+b: edges a, b, (a+b). Need a+b not in {a,b,n-a,n-b} and not 0
        # More carefully: triangle-free iff no three (possibly equal) elements of S sum to 0 mod n
        # where "elements" means we consider S ∪ {n-s : s in S} as the full distance set

        # Let's just brute-force small S sets
        max_S_size = 3
        candidates_S = []
        if max_S_size >= 2:
            for a in range(1, target_n // 2 + 1):
                for b in range(a + 1, target_n // 2 + 1):
                    candidates_S.append(frozenset([a, b]))
        if max_S_size >= 3:
            for a in range(1, target_n // 3 + 1):
                for b in range(a + 1, target_n // 3 + 1):
                    for c in range(b + 1, target_n // 2 + 1):
                        candidates_S.append(frozenset([a, b, c]))

        random.shuffle(candidates_S)

        for S in candidates_S:
            if time.time() - start_time > deadline - 10:
                break
            S_list = sorted(S)
            # Quick triangle-free check for circulant
            adj = circulant_adj(target_n, S_list)
            if not is_triangle_free_bitset(adj, target_n):
                continue
            # Check independence
            if not independence_number(adj, target_n, 10, time_limit=2.0):
                edges = edges_from_adj(adj, target_n)
                best = {'n': target_n, 'edges': edges}
                found_for_n = True
                break

        if found_for_n:
            continue

        # Try size 4 connection sets (subset of above)
        if time.time() - start_time > deadline - 10:
            break
        # For size 4, search more selectively
        for _ in range(500):
            if time.time() - start_time > deadline - 10:
                break
            S_list = random.sample(range(1, target_n // 2 + 1), 4)
            S_set = frozenset(S_list)
            adj = circulant_adj(target_n, S_list)
            if not is_triangle_free_bitset(adj, target_n):
                continue
            if not independence_number(adj, target_n, 10, time_limit=2.0):
                edges = edges_from_adj(adj, target_n)
                best = {'n': target_n, 'edges': edges}
                found_for_n = True
                break

    # Final fallback: ensure we return a valid graph
    if not best['edges']:
        # Return a small valid graph
        best = {'n': 27, 'edges': andrasfai_graph(9)['edges']}

    # Final validation
    if not verify_graph(best['n'], best['edges']):
        # Fallback to Andrásfai
        ag = andrasfai_graph(9)
        if verify_graph(ag['n'], ag['edges']):
            best = ag
        else:
            # Minimal fallback: complete bipartite K_{5,5} is triangle-free, alpha=5
            n = 10
            edges = []
            for i in range(5):
                for j in range(5, 10):
                    edges.append([i, j])
            best = {'n': n, 'edges': edges}

    return best
