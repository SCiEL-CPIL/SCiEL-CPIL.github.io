"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

N = 512
LIM = 512

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _lcm(a, b):
    return a // _gcd(a, b) * b

def construct():
    start = time.time()
    deadline = start + 75.0  # safety margin before 90s kill

    # Build compatibility graph: vertex i (1..512) is compatible with j if lcm(i,j) <= 512
    # Represent neighbors as Python int bitsets for fast intersection.
    # Index: use 0-based index for integer value v: idx = v-1
    size = N
    neigh = [0] * size
    for a in range(1, size + 1):
        ia = a - 1
        bits = 0
        for b in range(1, size + 1):
            if b == a:
                continue
            if _lcm(a, b) <= LIM:
                bits |= (1 << (b - 1))
        neigh[ia] = bits

    # Vertex order for branching: prefer higher-degree vertices first (heuristic)
    deg = [bin(neigh[i]).count('1') for i in range(size)]
    order = list(range(size))
    order.sort(key=lambda i: (-deg[i], i))

    best_clique = []  # list of vertex indices

    def _color_bound(P_bits, order_seq):
        # Greedy coloring of the subgraph induced by P_bits, using order_seq as vertex order.
        # Returns the number of colors used (upper bound on clique size in P).
        # We color vertices in the given order; a vertex can take the smallest color
        # not used by its already-colored neighbors in P.
        # Returns (num_colors, color_of_vertex_dict not needed, just count).
        # For pruning we only need the count.
        remaining = P_bits
        if remaining == 0:
            return 0
        # We'll assign colors greedily.
        # color[i] = color assigned to vertex i (0-based), -1 if uncolored.
        color = {}
        # Process vertices in order_seq but only those in P.
        colors_used = 0
        # For each color class, track which vertices have that color.
        # A vertex v can use color c if none of its neighbors in P have color c.
        # We'll do: for each vertex in order, assign smallest feasible color.
        # To check feasibility we need, for each color, the set of vertices with that color.
        # That's expensive. Instead use a simpler bound: greedy coloring by ordering.
        # Simpler: use the "first fit" with neighbor check.
        # Given size <= 512, and this is called many times, keep it fast.
        # Alternative fast bound: use a greedy independent set packing to get a lower bound
        # on chromatic number? No, we need upper bound on clique = lower bound on chromatic.
        # Actually clique size <= chromatic number. Greedy coloring gives an upper bound on
        # chromatic number, which is an upper bound on clique. Good.
        # Let's do first-fit coloring.
        assigned = {}  # vertex -> color
        color_classes = []  # list of sets (as ints) of vertices with that color
        for v in order_seq:
            vb = 1 << v
            if not (remaining & vb):
                continue
            # Find smallest color c such that v has no neighbor in color c
            c = 0
            while c < len(color_classes):
                if neigh[v] & color_classes[c]:
                    c += 1
                else:
                    break
            if c == len(color_classes):
                color_classes.append(0)
            color_classes[c] |= vb
            assigned[v] = c
            remaining &= ~vb
        return len(color_classes)

    # For efficiency, precompute for each vertex its neighbors restricted to a given set is
    # done via bit operations. The coloring above iterates over order_seq each time which
    # is O(size) per call. With pruning this should be acceptable.

    def _bkc(P_bits, X_bits, R):
        nonlocal best_clique
        if time.time() > deadline:
            return
        # Upper bound: |P| + |R|
        p_count = bin(P_bits).count('1')
        if p_count + len(R) <= len(best_clique):
            return
        if P_bits == 0:
            if len(R) > len(best_clique):
                best_clique = R[:]
            return

        # Coloring-based upper bound
        # Use a fixed order for coloring (the global order) for consistency
        cb = _color_bound(P_bits, order)
        if len(R) + cb <= len(best_clique):
            return

        # Pivot selection: choose u in P ∪ X with max degree in P
        PX = P_bits | X_bits
        # Find pivot with max |N(u) & P|
        best_u = -1
        best_deg = -1
        # Iterate over set bits in PX
        tmp = PX
        while tmp:
            lsb = tmp & (-tmp)
            u = lsb.bit_length() - 1
            d = bin(neigh[u] & P_bits).count('1')
            if d > best_deg:
                best_deg = d
                best_u = u
            tmp &= tmp - 1

        # Candidates: P \ N(u)
        if best_u >= 0:
            candidates = P_bits & ~neigh[best_u]
        else:
            candidates = P_bits

        # Branch on candidates in order
        tmp = candidates
        while tmp:
            if time.time() > deadline:
                return
            lsb = tmp & (-tmp)
            v = lsb.bit_length() - 1
            tmp &= tmp - 1

            R.append(v)
            _bkc(P_bits & neigh[v], X_bits, R)
            R.pop()

            P_bits &= ~lsb
            X_bits |= lsb

    # Initial call
    initial_P = (1 << size) - 1
    _bkc(initial_P, 0, [])

    # Convert best_clique (vertex indices) to values (1-based)
    result = sorted([v + 1 for v in best_clique])
    if not result:
        # Fallback: all 1..sqrt(512)
        result = list(range(1, math.isqrt(LIM) + 1))
    return result
# EVOLVE-BLOCK-END
