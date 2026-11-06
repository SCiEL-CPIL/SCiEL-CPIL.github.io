import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85

    n = 512
    # Build adjacency: node i (0-indexed) represents value i+1
    adj = [0] * n

    for i in range(n):
        a = i + 1
        for j in range(i + 1, n):
            b = j + 1
            l = a * b // gcd(a, b)
            if l <= 512:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    # Precompute degrees
    degrees = [adj[i].bit_count() for i in range(n)]

    # Initial best clique: divisors of a highly composite number
    # 360 = 2^3 * 3^2 * 5 has 24 divisors
    # 480 = 2^5 * 3 * 5 has 24 divisors
    # 360 divisors: 1,2,3,4,5,6,8,9,10,12,15,18,20,24,30,36,40,45,48,60,72,80,90,120,180,360
    # Check: all pairs have lcm <= 512? lcm(180,360)=360<=512, lcm(120,180)=360<=512, etc.
    # Actually let's use a greedy approach for initial clique
    def greedy_clique():
        P = (1 << n) - 1
        R = 0
        while P:
            best_node = -1
            max_deg = -1
            temp = P
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                temp ^= lsb
                deg = (P & adj[v]).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_node = v
            if best_node == -1:
                break
            R |= (1 << best_node)
            P &= adj[best_node]
        return R

    best_clique_mask = greedy_clique()
    best_size = best_clique_mask.bit_count()

    # Tomita-style: color the graph to get upper bounds
    # We'll use a simple greedy coloring for upper bound estimation
    def color_upper_bound(P):
        """Return a list of colors such that each color class is an independent set.
        The max color gives an upper bound on clique size in P."""
        if P == 0:
            return []
        colors = []
        remaining = P
        color = 0
        while remaining:
            color += 1
            # Greedily pick nodes for this color class (independent set)
            # A node can be added if it's not adjacent to any already in this color class
            color_class = 0
            temp = remaining
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                temp ^= lsb
                if (color_class & adj[v]) == 0:
                    color_class |= lsb
            remaining &= ~color_class
            colors.append(color_class)
        return colors

    # More efficient: sequential coloring that assigns colors in order
    def sequential_color(P):
        """Assign colors to nodes in P such that adjacent nodes get different colors.
        Returns list of color classes (masks). Max color = upper bound on clique size."""
        if P == 0:
            return []
        # Order nodes by degree descending for better coloring
        nodes = []
        temp = P
        while temp:
            lsb = temp & (-temp)
            v = lsb.bit_length() - 1
            temp ^= lsb
            nodes.append(v)
        nodes.sort(key=lambda v: -(P & adj[v]).bit_count())

        node_color = {}
        max_color = 0
        for v in nodes:
            # Find smallest available color
            used = set()
            for u in nodes:
                if u == v:
                    break
                if (adj[v] >> u) & 1:
                    if u in node_color:
                        used.add(node_color[u])
            c = 0
            while c in used:
                c += 1
            node_color[v] = c
            if c > max_color:
                max_color = c

        # Build color classes
        color_classes = []
        for c in range(max_color + 1):
            mask = 0
            for v in nodes:
                if node_color.get(v) == c:
                    mask |= (1 << v)
            color_classes.append(mask)
        return color_classes

    # Simpler and faster: use the fact that we just need an upper bound
    # Greedy coloring: process nodes, assign smallest available color
    def greedy_color_bound(P):
        """Returns upper bound on clique size in P using greedy coloring."""
        if P == 0:
            return 0
        # Get nodes in P ordered by degree in P descending
        nodes = []
        temp = P
        while temp:
            lsb = temp & (-temp)
            v = lsb.bit_length() - 1
            temp ^= lsb
            nodes.append(v)
        nodes.sort(key=lambda v: -(P & adj[v]).bit_count())

        # Assign colors
        node_color = {}
        for v in nodes:
            used = set()
            for u in nodes:
                if u == v:
                    break
                if (adj[v] >> u) & 1 and u in node_color:
                    used.add(node_color[u])
            c = 0
            while c in used:
                c += 1
            node_color[v] = c

        return max(node_color.values()) + 1 if node_color else 0

    # Even simpler: just use |P| as upper bound initially, refine with coloring
    # For performance, let's use a lighter approach: the number of nodes in P
    # is already an upper bound. We can use coloring to get tighter bounds.

    best_clique = [i + 1 for i in range(n) if (best_clique_mask >> i) & 1]

    def bronk(R, P, X):
        nonlocal best_clique, best_size, best_clique_mask
        if time.time() > deadline:
            return
        if not P and not X:
            sz = R.bit_count()
            if sz > best_size:
                best_size = sz
                best_clique = [i + 1 for i in range(n) if (R >> i) & 1]
                best_clique_mask = R
            return
        if not P:
            return

        # Upper bound: if |R| + |P| <= best_size, we can't improve
        if R.bit_count() + P.bit_count() <= best_size:
            return

        # Choose pivot u from P ∪ X to maximize |P ∩ N(u)|
        union = P | X
        if union == 0:
            return

        pivot = -1
        max_count = -1
        temp = union
        while temp:
            lsb = temp & (-temp)
            u = lsb.bit_length() - 1
            temp ^= lsb
            count = (P & adj[u]).bit_count()
            if count > max_count:
                max_count = count
                pivot = u
                if count == P.bit_count():
                    break

        if pivot == -1:
            lsb = union & (-union)
            pivot = lsb.bit_length() - 1

        # Iterate over P \ N(pivot)
        candidates = P & ~adj[pivot]
        while candidates:
            lsb = candidates & (-candidates)
            v = lsb.bit_length() - 1
            candidates ^= lsb

            bronk(R | lsb, P & adj[v], X & adj[v])
            P ^= lsb
            X |= lsb

    all_nodes = (1 << n) - 1
    bronk(0, all_nodes, 0)

    if not best_clique:
        return [1]

    return best_clique
