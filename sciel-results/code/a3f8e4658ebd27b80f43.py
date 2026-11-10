"""Find largest pairwise non-coprime subset of [1,255] containing 255."""
import time
import sys
from math import gcd

# EVOLVE-BLOCK-START
def construct():
    """
    Find the maximum subset S ⊆ [1, 255] containing 255 such that gcd(a,b) > 1
    for all distinct pairs a,b in S.
    
    Since 255 ∈ S, every x ∈ S must satisfy gcd(x, 255) > 1.
    Let C = {x ∈ [1,255] | gcd(x,255) > 1}. Then S ⊆ C and 255 ∈ S.
    Since gcd(255, x) > 1 for all x ∈ C, the problem reduces to finding
    the maximum clique in the graph induced by V = C \ {255}, then adding 255.
    
    We use Bron-Kerbosch with pivot (Tomita MCQ style) with greedy coloring
    for upper-bound pruning.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    # 1. Generate candidate set C
    C = [x for x in range(1, 256) if gcd(x, 255) > 1]
    V = [x for x in C if x != 255]
    n = len(V)  # 110 vertices

    # 2. Build adjacency as bitmasks for speed
    # adj_mask[i] is a bitmask where bit j is set iff gcd(V[i], V[j]) > 1
    adj_mask = [0] * n
    for i in range(n):
        vi = V[i]
        for j in range(i + 1, n):
            if gcd(vi, V[j]) > 1:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)

    # 3. Tomita MCQ: Bron-Kerbosch with pivot + greedy coloring upper bound
    best_clique = []
    best_size = [0]

    def color_sort(candidates_mask):
        """
        Greedily color vertices in candidates_mask and return them sorted by
        color descending. Returns list of (vertex_index, color).
        The color of vertex v gives an upper bound on clique size through v.
        """
        # Extract vertex list from bitmask
        verts = []
        m = candidates_mask
        while m:
            lb = m & (-m)
            v = lb.bit_length() - 1
            verts.append(v)
            m ^= lb

        if not verts:
            return []

        # Sort by degree within candidates (descending) to help coloring
        verts.sort(key=lambda v: (adj_mask[v] & candidates_mask).bit_count(), reverse=True)

        # Greedy coloring
        color_of = {}
        for v in verts:
            # Collect colors of already-colored neighbors of v
            neighbor_colors = set()
            nv = adj_mask[v] & candidates_mask
            while nv:
                lb = nv & (-nv)
                u = lb.bit_length() - 1
                if u in color_of:
                    neighbor_colors.add(color_of[u])
                nv ^= lb
            c = 1
            while c in neighbor_colors:
                c += 1
            color_of[v] = c

        # Sort by color descending
        sorted_verts = sorted(verts, key=lambda v: color_of[v], reverse=True)
        return [(v, color_of[v]) for v in sorted_verts]

    def expand(clique, candidates_mask):
        """Bron-Kerbosch with pivot and coloring bound."""
        if time.time() > deadline:
            return

        if candidates_mask == 0:
            if len(clique) > best_size[0]:
                best_size[0] = len(clique)
                best_clique[:] = clique[:]
            return

        # Upper bound: current clique size + number of candidates
        if len(clique) + candidates_mask.bit_count() <= best_size[0]:
            return

        # Color sort to get upper bounds
        colored = color_sort(candidates_mask)
        if not colored:
            return

        # Additional pruning: if len(clique) + max_color <= best_size, prune
        max_color = colored[0][1]
        if len(clique) + max_color <= best_size[0]:
            return

        # Pivot selection: choose vertex in candidates with max degree in candidates
        # to minimize branching
        pivot_v = None
        max_deg = -1
        m = candidates_mask
        while m:
            lb = m & (-m)
            v = lb.bit_length() - 1
            deg = (adj_mask[v] & candidates_mask).bit_count()
            if deg > max_deg:
                max_deg = deg
                pivot_v = v
            m ^= lb

        # Branch on candidates \ N(pivot_v)
        # These are vertices NOT adjacent to pivot
        if pivot_v is not None:
            branch_mask = candidates_mask & ~(adj_mask[pivot_v])
        else:
            branch_mask = candidates_mask

        # Process vertices in reverse color order (from color_sort)
        # We iterate through colored list in reverse (highest color first)
        for v, c in reversed(colored):
            if time.time() > deadline:
                return
            # Only branch on vertices in branch_mask
            if not (branch_mask & (1 << v)):
                continue
            # Prune: if len(clique) + c <= best_size, skip (and all subsequent lower-color)
            if len(clique) + c <= best_size[0]:
                break

            new_clique = clique + [v]
            new_candidates = candidates_mask & adj_mask[v]
            expand(new_clique, new_candidates)

            # Remove v from candidates for subsequent iterations
            candidates_mask &= ~(1 << v)
            branch_mask &= ~(1 << v)

            # Prune remaining
            if len(clique) + candidates_mask.bit_count() <= best_size[0]:
                break

    # Initialize with a good lower bound: greedy clique
    # Start with all multiples of 3 (they form a clique since all share factor 3)
    mult3 = [i for i in range(n) if V[i] % 3 == 0]
    if mult3:
        best_size[0] = len(mult3)
        best_clique[:] = mult3[:]

    # Run the exact algorithm
    all_mask = (1 << n) - 1
    expand([], all_mask)

    # 4. Construct final set
    result_values = [V[i] for i in best_clique]
    result_values.append(255)

    # 5. Verification
    s = set(result_values)
    assert 255 in s
    assert len(s) == len(result_values)
    for a in s:
        assert 1 <= a <= 255
    for i in range(len(result_values)):
        for j in range(i + 1, len(result_values)):
            assert gcd(result_values[i], result_values[j]) > 1

    return result_values
# EVOLVE-BLOCK-END
