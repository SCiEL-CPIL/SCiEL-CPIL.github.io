import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin before 60s limit

    N = 256

    # Precompute perfect squares up to 2*N = 512
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1

    # A vertex v is "forbidden" if 2*v is a perfect square (diagonal constraint a=b)
    # Also, we only consider vertices 1..N
    # Build incompatibility graph: edge (u,v) if u+v is a perfect square
    # We want Maximum Independent Set in this incompatibility graph
    # Equivalently, Maximum Clique in the compatibility graph (complement of incompatibility, excluding self-loops)

    # Let's work with the incompatibility graph directly for MIS
    # Or better: work with compatibility graph for Max Clique (Tomita is well-studied for Max Clique)

    # Compatibility: two distinct vertices u, v are compatible if u+v is NOT a perfect square
    # Self-compatibility: v is compatible with itself if 2*v is NOT a perfect square

    # Filter out vertices that are self-incompatible (2*v is square)
    valid_vertices = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_vertices.append(v)

    num_valid = len(valid_vertices)
    # Map from value to index
    val_to_idx = {v: i for i, v in enumerate(valid_vertices)}

    # Build compatibility adjacency (as sets of indices)
    # adj[i] = set of j != i such that valid_vertices[i] + valid_vertices[j] is NOT a perfect square
    adj = [set() for _ in range(num_valid)]

    for i in range(num_valid):
        ui = valid_vertices[i]
        for j in range(i + 1, num_valid):
            uj = valid_vertices[j]
            if (ui + uj) not in squares:
                adj[i].add(j)
                adj[j].add(i)

    # Now solve Max Clique on this graph using Tomita's algorithm with coloring

    # Convert adj to a list of sorted lists for faster intersection?
    # Or keep as sets. For Tomita, we often use bitmasks or sorted lists.
    # With ~200 nodes, sets are fine.

    # Tomita's MCS (Maximum Clique Search)
    # We maintain:
    # P: set of candidate vertices
    # S: current clique (list of vertices)
    # best: best clique found so far

    best_clique = []
    best_size = 0

    # Degrees for ordering
    degrees = [len(adj[i]) for i in range(num_valid)]

    def color_sort(P):
        """
        Greedily color the subgraph induced by P.
        Returns:
        - sorted_list: list of vertices in P sorted by color (ascending)
        - colors: dict vertex -> color
        - max_color: int
        """
        if not P:
            return [], {}, 0

        # Sort P by degree in the induced subgraph (approximated by global degree or computed)
        # For better bounds, sort by degree in P
        P_list = list(P)
        # Compute degree in P
        # This is O(|P|^2) which might be slow if done repeatedly.
        # Let's use global degree for speed, or pre-compute?
        # Global degree is a decent heuristic.
        P_list.sort(key=lambda x: degrees[x], reverse=True)

        colors = {}
        color_classes = []  # list of sets, color_classes[c] = set of vertices with color c
        max_color = 0

        for v in P_list:
            # Find smallest color not used by neighbors of v in P
            used_colors = set()
            for u in adj[v]:
                if u in P:
                    if u in colors:
                        used_colors.add(colors[u])

            c = 1
            while c in used_colors:
                c += 1

            colors[v] = c
            if c > max_color:
                max_color = c

        # Sort vertices by color, then by degree (or index) for determinism
        # Tomita sorts by color, and within same color, often by degree or index
        # The key insight: vertices with higher color are more likely to be in a large clique?
        # Actually, the algorithm processes vertices in order of increasing color?
        # No, Tomita processes in order of decreasing color?
        # Let's check: "Sort the vertices in P by their color numbers in non-increasing order"
        # Then iterate from the end? Or from the beginning?
        # Standard Tomita:
        # 1. Color P.
        # 2. Sort P by color (non-increasing).
        # 3. Iterate v in this sorted order.
        #    If len(S) + color(v) <= best_size, break (prune).
        #    Recurse with P' = P \ {v} intersect N(v), S' = S U {v}.
        #    Remove v from P.

        # So we need the list sorted by color descending.
        sorted_P = sorted(P, key=lambda x: (colors[x], degrees[x]), reverse=True)
        return sorted_P, colors, max_color

    def expand(P, S):
        nonlocal best_clique, best_size

        if time.time() > deadline:
            return

        if not P:
            if len(S) > best_size:
                best_size = len(S)
                best_clique = S.copy()
            return

        sorted_P, colors, max_color = color_sort(P)

        # Iterate in reverse order (highest color first)
        # We maintain P as a list/set. We will modify it as we go.
        # To avoid expensive set operations, we can use a list and a set for membership.
        # But P is passed as a set or list? Let's use a set for O(1) lookup.
        # However, we need to iterate in a specific order.

        # Let's use a list for the current P candidates, and a set for quick lookup if needed.
        # Actually, we can just iterate over sorted_P and maintain a set of "active" vertices.
        # Or, simpler: iterate over sorted_P. For each v, if v is still in P (we remove as we go), process it.

        # We need to be careful: when we recurse, we pass a new P.
        # The "remove v from P" step is for the current level's loop.

        P_set = set(P)
        # sorted_P is a list. We iterate through it.
        # We need to check if v is still in P_set.
        # As we process v, we remove it from P_set.

        for v in sorted_P:
            if v not in P_set:
                continue

            # Pruning: if len(S) + colors[v] <= best_size, we can stop
            # because all remaining vertices have color <= colors[v]
            # and any clique in the remaining P has size <= max_color_remaining <= colors[v] (roughly)
            # More precisely, the bound is: len(S) + (number of vertices in remaining P with color >= colors[v])
            # But Tomita uses: if len(S) + colors[v] <= best_size, break.
            # This is valid because colors[v] is an upper bound on the clique size in the subgraph induced by
            # the remaining vertices (which all have color <= colors[v]... wait, no).
            # The coloring is such that any clique has at most one vertex of each color.
            # So max clique size in P <= max_color.
            # For the subset of P with color >= c, max clique size <= (max_color - c + 1)?
            # No. The bound used is: if we pick v, the new clique size is 1 + clique_size(P \cap N(v)).
            # The coloring bound says: clique_size(P) <= max_color.
            # So if len(S) + max_color <= best_size, we can prune the whole P.
            # Inside the loop, we use a tighter bound.
            # The standard Tomita bound:
            # Let k be the number of vertices in P with color >= color(v).
            # Then any clique in P containing v has size <= 1 + (k - 1)? No.
            # Actually, the bound is simply: if len(S) + colors[v] <= best_size, break.
            # Why? Because colors[v] is the color assigned to v.
            # In the greedy coloring, if v has color c, it means there are c-1 colors used by its neighbors in the processed set.
            # This implies that the subgraph induced by the unprocessed vertices (which have color <= c? No, we process high to low)
            # Let's stick to the standard implementation:
            # Sort by color descending.
            # For v in sorted_P:
            #   if len(S) + colors[v] <= best_size: break
            #   ...

            if len(S) + colors[v] <= best_size:
                break

            # Branch: Include v
            S.append(v)
            # New P: neighbors of v in P_set
            new_P = P_set & adj[v]
            expand(new_P, S)
            S.pop()

            # Remove v from P_set for the next iteration (Exclude v)
            P_set.remove(v)

    # Initial call
    initial_P = set(range(num_valid))
    # Order initial P by degree? color_sort handles it.
    expand(initial_P, [])

    # Convert indices back to values
    result = [valid_vertices[i] for i in best_clique]

    # Fallback if something went wrong
    if not result:
        # Simple greedy
        greedy_set = []
        remaining = set(valid_vertices)
        while remaining:
            # Pick vertex with max degree in remaining? Or min?
            # For MIS, picking high degree is often good (removes many conflicts).
            # For Max Clique, picking high degree is good.
            # Let's pick max degree in the compatibility graph induced by remaining.
            # We need to compute degrees in the induced subgraph.
            # This is O(n^2) per step, so O(n^3) total. n~200, so 8M ops. Fine.
            max_deg = -1
            best_v = None
            for v in remaining:
                deg = 0
                for u in remaining:
                    if u != v and u in adj[val_to_idx[v]]:
                        deg += 1
                if deg > max_deg:
                    max_deg = deg
                    best_v = v

            if best_v is None:
                break

            greedy_set.append(best_v)
            # Remove best_v and its incompatible neighbors from remaining
            # Incompatible neighbors are those NOT in adj[best_v]
            # So we keep u if u is in adj[best_v]
            bv_idx = val_to_idx[best_v]
            new_remaining = {u for u in remaining if u != best_v and u in adj[bv_idx]}
            remaining = new_remaining

        result = greedy_set

    return result

# Test locally
# if __name__ == "__main__":
#     res = construct()
#     print(f"Size: {len(res)}")
#     print(res)
