"""Construct small 4-regular graphs of girth 10 using a random lift / greedy embedding approach."""
import random
import itertools
from collections import deque

def _verify(n, edges, k, g):
    """Verify simple, k-regular, no cycle < g via non-backtracking BFS from every vertex."""
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        if u == v:
            return False
        if v in adj[u]:
            return False
        adj[u].add(v)
        adj[v].add(u)
    for i in range(n):
        if len(adj[i]) != k:
            return False
    # Non-backtracking walk check: for every vertex and every starting neighbor,
    # do BFS on states (prev, curr) ensuring we never go back to prev.
    # If we reach a vertex at distance < g/2 ... actually we need to ensure no closed non-backtracking walk of length < g.
    # A non-backtracking walk of length L from v0 to v1 ... vL with vL = v0 and no immediate backtrack corresponds to a cycle of length L (if L>=3) or could be shorter.
    # We check all non-backtracking walks of length up to g-1 from every (v0, v1) state.
    # If any such walk returns to v0 at step L < g, that's a cycle of length L (or a shorter one).
    # Actually, a non-backtracking walk from (v0, v1) of length L ending at (v_{L-1}, v0) means v0 appears at step 0 and step L.
    # The subsequence v0..vL is a non-backtracking walk. If L < g, this implies a cycle of length <= L < g.
    # We'll do BFS from each (v0, v1) and see if we can reach a state (v', v0) where the path length is < g.
    # More efficiently: BFS from each v0, tracking (prev, curr, dist). But state space is n*(n-1).
    # For n=320, that's ~100k states per v0, times 320 v0s = 32M, which might be slow in Python.
    # Let's optimize: we only need to check distances up to g-1 = 9.
    # BFS depth 9 from each (v0, v1).
    for v0 in range(n):
        for v1 in adj[v0]:
            # State: (prev, curr)
            # Start: (v0, v1) at dist 1
            # We want to see if we can reach (x, v0) at dist L where 1 < L < g
            # i.e., curr == v0 and prev != v1 (non-backtracking) and L < g
            # Actually if curr == v0, the walk is v0 -> v1 -> ... -> v0. Length is L.
            # We need L < g.
            visited = set()
            visited.add((v0, v1))
            queue = deque([(v0, v1, 1)])
            while queue:
                prev, curr, dist = queue.popleft()
                if dist >= g:
                    continue
                for nxt in adj[curr]:
                    if nxt == prev:
                        continue
                    new_state = (curr, nxt)
                    if new_state in visited:
                        continue
                    visited.add(new_state)
                    if nxt == v0 and dist + 1 < g:
                        return False
                    queue.append((curr, nxt, dist + 1))
    return True

def construct(k, g):
    if k != 4 or g != 10:
        # Fallback to a known construction if parameters differ, though task says k=4, g=10
        # For now, just return a trivial invalid or use the old logic? 
        # The task says "the evaluator passes this task's degree k and girth g".
        # It implies k=4, g=10.
        pass
    
    # Strategy: Use the "random lift" or "greedy embedding" idea.
    # A well-known method for constructing cages is the "greedy embedding" or "random regular graph" with girth checking.
    # However, finding a 4-regular graph of girth 10 with < 320 vertices is hard.
    # The current seed uses Lazebnik-Ustimenko-Woldar which is quite large.
    # Let's try a different family: The "Tutte-Coxeter" graph is 3-regular, girth 9, 30 vertices.
    # For 4-regular, girth 10, we can try to use a "covering" of a smaller graph or a specific algebraic construction.
    
    # Alternative: Use the "randomized greedy" construction.
    # Start with an empty graph, add edges one by one ensuring regularity and girth.
    # This is very slow.
    
    # Better: Use a known small graph if possible?
    # The smallest known is 320. Can we beat it? Unlikely in 120s.
    # Can we match it with a different construction?
    # The seed program likely produces a graph larger than 320?
    # Let's check the seed: q=4, d = max(3, 10-5=5 if 5%2 else 10-4=6) -> 5 is odd, so d=5.
    # L-U-W graph with q=4, d=5.
    # Number of vertices in L-U-W(q, d) is roughly q * (q^d - 1) / (q-1) * 2?
    # For q=4, d=5: The graph is bipartite.
    # Points: (q-1)-tuples? No, the construction in the code uses vectors of length d+1?
    # Let's trace the code:
    # q=4, d=5.
    # start = (0, (None, 0, 0, 0, 0, 0)) -> length 6?
    # y = [None, first] + [0]*(d-1) -> length 1 + d = 6.
    # Indices 1..d are used.
    # The number of vertices in the connected component of the zero point.
    # For L-U-W, the size is q * (q^d - 1) / (q - 1) * 2?
    # Actually, the L-U-W graph CD(q, d) has 2 * q * (q^d - 1) / (q - 1) vertices?
    # For q=4, d=5: 2 * 4 * (1024 - 1) / 3 = 8 * 341 = 2728.
    # This is much larger than 320.
    # So the seed program is very weak (score 1.0 is just the baseline, but 2728 is huge).
    # We need to produce a graph much smaller than 2728.
    # The best known is 320.
    # Can we construct the 320-vertex graph?
    # The 320-vertex graph is often constructed via a "random" search or specific algebraic means.
    # One such construction is the "Higman-Sims" related or "M22" related?
    # Actually, the 320-vertex 4-regular graph of girth 10 is known as the "Klein graph" or similar?
    # No, the (3,5) cage is Petersen (10). (3,6) is Heawood (14). (3,7) is Tutte-Coxeter (30). (3,8) is McGee (24). (3,9) is (162). (3,10) is (102).
    # For (4,10), the cage is 320.
    
    # How to construct the 320-vertex graph in Python within 120s?
    # We can use a "random regular graph" generation and check girth, but that's unlikely to find girth 10 quickly.
    # However, we can use a "greedy" algorithm to embed a tree of depth 4 (since girth 10 means radius 4 is a tree) into a graph.
    # This is the "greedy embedding" algorithm.
    
    # Algorithm:
    # 1. Generate a random 4-regular graph on N vertices.
    # 2. Check girth.
    # 3. If girth < 10, try to "fix" it by swapping edges (2-opt).
    # This is heuristic.
    
    # Better: Use the "random lift" of a small graph.
    # Or, use a specific known construction for the 320-vertex cage.
    # The 320-vertex cage is often constructed as a "covering" of the 20-vertex 4-regular graph of girth 6?
    # No.
    
    # Let's try a different approach: Random search for a 4-regular graph on N vertices with girth >= 10.
    # We can start with N=250 (lower bound is 243) and increase N until we find one.
    # Generating a random 4-regular graph is easy.
    # Checking girth is O(N * 4^4) which is fast.
    # But the probability of a random 4-regular graph on 250 vertices having girth >= 10 is extremely low.
    
    # We need a smarter construction.
    # The "greedy embedding" algorithm (Babai, Sós, Szemerédi) is the standard way to find cages.
    # It works by starting with a tree of depth r = floor((g-1)/2) and greedily adding vertices and edges to close the graph while maintaining regularity and girth.
    
    # Implementation of Greedy Embedding:
    # 1. Start with a tree T of depth 4 (root has degree 4, internal nodes degree 4, leaves degree 1).
    #    Number of vertices in T: 1 + 4 + 16 + 64 + 256 = 341.
    #    Wait, the Moore bound for (4,10) is 1 + 4 + 16 + 64 + 256 = 341?
    #    Moore bound M(k,g) for even g=2r: 1 + k * sum_{i=0}^{r-1} (k-1)^i.
    #    r=5. 1 + 4 * (1 + 3 + 9 + 27 + 81) = 1 + 4 * 121 = 485.
    #    Wait, my previous calculation was wrong.
    #    M(4,10) = 1 + 4 * (1 + 3 + 9 + 27 + 81) = 1 + 484 = 485.
    #    So the lower bound is 485?
    #    The task says "best lower bound is 243".
    #    Let me re-read the task.
    #    "The smallest known such graph has 320 vertices ... and the best lower bound is 243."
    #    This contradicts the Moore bound of 485.
    #    Ah, the Moore bound is for *cages*? No, the Moore bound is a lower bound on the number of vertices of a *cage*.
    #    If the lower bound is 243, then my Moore bound calculation must be wrong or I am misinterpreting.
    #    Let's check: M(k, g) for even g=2r is 1 + k * ((k-1)^r - 1) / (k-2).
    #    For k=4, r=5: 1 + 4 * (3^5 - 1) / 2 = 1 + 2 * (243 - 1) = 1 + 484 = 485.
    #    So the Moore bound is 485.
    #    How can the lower bound be 243?
    #    Maybe the task is about *bipartite* cages?
    #    "4-regular graph with no cycle shorter than 10".
    #    If it's bipartite, the girth is even.
    #    The Moore bound for bipartite (k, 2r) is 2 * sum_{i=0}^{r-1} (k-1)^i.
    #    For k=4, r=5: 2 * (1 + 3 + 9 + 27 + 81) = 2 * 121 = 242.
    #    So 243 is consistent with the bipartite Moore bound (242) + 1?
    #    Or maybe the graph is not necessarily bipartite, but the lower bound 243 comes from a different argument?
    #    Actually, if the graph is not bipartite, it can have odd cycles. But girth 10 means no cycles of length 3,5,7,9.
    #    So the graph *must* be bipartite! Because any non-bipartite graph has an odd cycle.
    #    If the girth is 10, the shortest odd cycle is at least 10 (which is even), so there are NO odd cycles.
    #    Therefore, the graph MUST be bipartite.
    #    So the Moore bound is indeed 242.
    #    The smallest known is 320.
    
    # So we are looking for a bipartite 4-regular graph of girth 10 with as few vertices as possible.
    # The seed program produces a L-U-W graph which is bipartite.
    # Its size was ~2728.
    # We need to get close to 320.
    
    # How to construct the 320-vertex graph?
    # It is known that the (4,10) cage is unique? Or one of few?
    # We can use a "randomized" search with "local improvement".
    # Start with a random bipartite 4-regular graph on N vertices (N even, N >= 242).
    # Check girth. If < 10, find a short cycle and try to "break" it by swapping edges.
    # This is a standard heuristic for finding cages.
    
    # Let's implement this heuristic.
    # 1. Generate random bipartite 4-regular graph on N vertices.
    #    We can do this by generating a random 4-regular graph on N/2 vertices and "blowing up" each vertex to 2? No.
    #    We can generate a random 4-regular bipartite graph by taking two sets U and V of size N/2.
    #    Each u in U is connected to 4 distinct v in V, and each v in V is connected to 4 distinct u in U.
    #    This is equivalent to finding a random 4-regular bipartite graph.
    #    We can use the "configuration model" or "Havel-Hakimi" for bipartite graphs.
    #    Simpler: Use numpy to generate a random permutation and use it to build the graph?
    #    We can use the fact that a 4-regular bipartite graph is a union of 4 perfect matchings.
    #    So we can generate 4 random perfect matchings between U and V.
    #    1. Generate 4 random permutations of V.
    #    2. For each u in U, connect to the 4 corresponding v's in the permutations.
    #    This ensures 4-regularity.
    #    However, we might get multiple edges. We need to ensure the 4 permutations are distinct at each position?
    #    No, if two permutations map u to the same v, we get a double edge.
    #    We can retry if multiple edges occur, or use a more sophisticated method.
    #    For N/2 = 160, the probability of collision is low if we use random permutations.
    #    Actually, we can just use 4 random permutations and check for duplicates.
    
    # 2. Check girth.
    # 3. If girth < 10, find a cycle of length < 10.
    # 4. Try to break the cycle by swapping edges.
    #    A standard move: If we have a cycle u1-v1-u2-v2-...-uk-vk-u1, we can try to swap edges.
    #    But since it's bipartite, we can use "2-opt" swaps.
    #    This is complex to implement correctly.
    
    # Alternative: Use the "greedy embedding" algorithm directly.
    # This is more robust.
    # Reference: "Finding cages" by Babai et al.
    # The algorithm builds the graph layer by layer.
    # Layer 0: 1 vertex? No, bipartite.
    # Let's stick to the random heuristic, it's easier to code.
    
    # We will try N from 244 to 320 in steps of 2.
    # For each N, try to find a graph with girth >= 10.
    # If found, return it.
    # If not found after many tries, move to next N.
    
    # To speed up, we can use the fact that the 320-vertex graph exists.
    # We can hard-code a known 320-vertex graph?
    # The task says "construct", so we should generate it.
    # But if we can't find it quickly, we can return a valid graph with n=320 if we know one.
    # However, we don't have the edges memorized.
    
    # Let's try the random heuristic with a time limit.
    # If we don't find one, we can fall back to a larger graph, e.g., the L-U-W graph from the seed,
    # but that's huge.
    # Maybe we can find a smaller L-U-W variant?
    # The seed used q=4, d=5.
    # What if we use q=5, d=4?
    # q must be prime or 4. 5 is prime.
    # d=4. Girth is at least d+5 = 9? Or d+4?
    # The code says "girth is at least d+5 for odd d".
    # If d is even, maybe d+4?
    # Let's check the seed logic:
    # d = max(3, g-5 if (g-5)%2 else g-4)
    # For g=10, g-5=5 (odd), so d=5.
    # If we want a smaller graph, maybe we can use a different q?
    # q=4 is the smallest.
    # So L-U-W with q=4 is the smallest in that family.
    
    # Let's try the random heuristic.
    # We'll use a time budget.
    import time
    start_time = time.time()
    time_limit = 100  # seconds
    
    for N in range(244, 322, 2):
        if time.time() - start_time > time_limit:
            break
        # Try to find a graph on N vertices
        found = False
        attempts = 0
        while not found and time.time() - start_time < time_limit:
            attempts += 1
            # Generate random bipartite 4-regular graph
            # U = 0..N/2-1, V = N/2..N-1
            n_half = N // 2
            # Generate 4 random permutations
            perms = [random.sample(range(n_half), n_half) for _ in range(4)]
            # Build edges
            edges = []
            adj_u = [set() for _ in range(n_half)]
            adj_v = [set() for _ in range(n_half)]
            valid = True
            for u in range(n_half):
                for p in range(4):
                    v = perms[p][u]
                    v_idx = n_half + v
                    if v in adj_u[u]:
                        valid = False
                        break
                    adj_u[u].add(v)
                    adj_v[v].add(u)
                    edges.append((u, v_idx))
            if not valid:
                continue
            # Check degree of V
            if any(len(adj_v[v]) != 4 for v in range(n_half)):
                continue
            # Check girth
            if _verify(N, edges, 4, 10):
                return N, edges
            # If not, we could try to improve, but for now just retry
            # To save time, we can limit attempts per N
            if attempts > 1000:
                break
    
    # If we didn't find one, return the seed's construction (which is valid but large)
    # Or return a trivial invalid graph? No, must be valid.
    # Let's use the seed's logic as a fallback.
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
