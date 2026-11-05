import time
import random
import itertools

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

def _edge_idx(i, j):
    """Get linear index for edge (i,j) with i<j in lexicographic order."""
    # Edges are ordered: (0,1),(0,2),...,(0,15),(1,2),(1,3),...,(14,15)
    # For vertex i, edges to j>i start at position sum_{k=0}^{i-1} (N-1-k)
    # = sum_{k=0}^{i-1} (N-1-k) = i*(N-1) - i*(i-1)//2
    # Then within vertex i's edges, j-i-1 offset
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

def _get_edges():
    """Return list of (i,j) pairs in lexicographic order."""
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            edges.append((i, j))
    return edges

def construct():
    deadline = time.time() + 80  # 80 second deadline with margin
    
    edges = _get_edges()
    edge_to_idx = {(i, j): k for k, (i, j) in enumerate(edges)}
    
    # Precompute all 4-cycles: for each 4-set of vertices, 3 cycles
    # Each cycle is a set of 4 edge indices
    all_cycles = []
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        c1 = frozenset([edge_to_idx[(a,b)], edge_to_idx[(b,c)], edge_to_idx[(c,d)], edge_to_idx[(a,d)]])
        # Cycle 2: a-c-b-d-a  => edges (a,c),(c,b),(b,d),(d,a) = (a,c),(b,c),(b,d),(a,d)
        c2 = frozenset([edge_to_idx[(a,c)], edge_to_idx[(b,c)], edge_to_idx[(b,d)], edge_to_idx[(a,d)]])
        # Cycle 3: a-b-d-c-a  => edges (a,b),(b,d),(d,c),(c,a) = (a,b),(b,d),(c,d),(a,c)
        c3 = frozenset([edge_to_idx[(a,b)], edge_to_idx[(b,d)], edge_to_idx[(c,d)], edge_to_idx[(a,c)]])
        all_cycles.append(c1)
        all_cycles.append(c2)
        all_cycles.append(c3)
    
    # For each edge, precompute which cycles it participates in
    edge_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, cyc in enumerate(all_cycles):
        for e in cyc:
            edge_cycles[e].append(ci)
    
    def is_valid(coloring):
        """Check if coloring has no rainbow 4-cycles."""
        for cyc in all_cycles:
            colors = set()
            for e in cyc:
                colors.add(coloring[e])
                if len(colors) > 4:
                    break
            if len(colors) == 4:
                return False
        return True
    
    def count_colors(coloring):
        return len(set(coloring))
    
    # Start with a simple valid coloring: partition into 4 groups of 4
    # Color by group pair: if both in same group, color 0; else color based on group pair
    # Actually, let's use a simpler approach: color edge (i,j) by (i//4, j//4) pair
    # Groups: 0,1,2,3 each with 4 vertices
    # Color = 4*min(g1,g2) + max(g1,g2) where g1=i//4, g2=j//4
    # This gives colors 0..14, but many may be same
    # Actually for same group, use color 0; for different groups, use unique pair color
    
    # Simpler: start with all 0s, then greedily add colors
    coloring = [0] * NUM_EDGES
    best_coloring = coloring[:]
    best_count = 1
    
    # Try to improve by recoloring edges
    # Greedy: for each edge, try assigning a new color and check validity
    # To speed up, maintain a constraint counter
    
    # Better approach: use local search with simulated annealing-like approach
    # But first, let's try a structured approach
    
    # Partition vertices into groups and use group-based coloring
    # Try different partition sizes
    
    # Start: 2 groups of 8
    # Color (i,j): 0 if same group, 1 if different
    # This gives 2 colors. Then try to introduce more.
    
    # Let's use a more sophisticated initial coloring:
    # Split into 4 groups of 4. Within group: color 0. Between groups: unique color per pair.
    # 4 groups -> 6 inter-group pairs, so colors 1-6 for inter-group, 0 for intra.
    # Total 7 colors.
    
    group_size = 4
    num_groups = N // group_size
    
    def make_group_coloring(num_groups, group_size):
        coloring = [0] * NUM_EDGES
        for k, (i, j) in enumerate(edges):
            gi = i // group_size
            gj = j // group_size
            if gi == gj:
                coloring[k] = 0
            else:
                # Unique color for each unordered pair of groups
                # Map (gi,gj) to a color
                if gi > gj:
                    gi, gj = gj, gi
                # There are num_groups*(num_groups-1)//2 inter-group pairs
                # Assign sequential colors starting from 1
                # Precompute mapping
                pass
        return coloring
    
    # Precompute group pair to color mapping
    group_pair_colors = {}
    color_counter = 1
    for g1 in range(num_groups):
        for g2 in range(g1 + 1, num_groups):
            group_pair_colors[(g1, g2)] = color_counter
            color_counter += 1
    
    def initial_coloring():
        coloring = [0] * NUM_EDGES
        for k, (i, j) in enumerate(edges):
            gi = i // group_size
            gj = j // group_size
            if gi == gj:
                coloring[k] = 0
            else:
                if gi > gj:
                    gi, gj = gj, gi
                coloring[k] = group_pair_colors[(gi, gj)]
        return coloring
    
    coloring = initial_coloring()
    if is_valid(coloring):
        best_coloring = coloring[:]
        best_count = count_colors(coloring)
    
    # Now try to improve by recoloring individual edges to new colors
    # For each edge, try each color from 0 to 119, see if it's valid
    # This is O(120 * 120 * num_cycles_per_edge) per trial, too slow
    
    # Better: maintain a count of how many cycles each edge is in that would become rainbow
    # Actually, let's just do random local search
    
    random.seed(42)
    
    # Local search: repeatedly pick a random edge, try a random new color, accept if valid
    # and increases color count (or with some probability for diversity)
    
    current = best_coloring[:]
    current_count = best_count
    
    # Precompute: for each edge, the list of cycle indices it's in
    # And for each cycle, the 4 edge indices
    
    # To check validity quickly after changing one edge:
    # Only cycles containing that edge need rechecking
    # A cycle becomes rainbow if all 4 edges have distinct colors
    
    def try_recolor(edge_idx, new_color, coloring):
        """Return True if recoloring edge_idx to new_color keeps coloring valid."""
        old_color = coloring[edge_idx]
        if old_color == new_color:
            return True
        
        # Check all cycles containing this edge
        for ci in edge_cycles[edge_idx]:
            cyc = all_cycles[ci]
            # Check if this cycle would have 4 distinct colors
            colors = set()
            for e in cyc:
                if e == edge_idx:
                    colors.add(new_color)
                else:
                    colors.add(coloring[e])
                if len(colors) > 4:
                    break
            if len(colors) == 4:
                return False
        return True
    
    # Local search
    max_iters = 500000
    iter_count = 0
    
    while iter_count < max_iters and time.time() < deadline - 5:
        iter_count += 1
        
        # Pick random edge
        ei = random.randint(0, NUM_EDGES - 1)
        
        # Try a random new color
        nc = random.randint(0, 119)
        
        if not try_recolor(ei, nc, current):
            continue
        
        old_c = current[ei]
        current[ei] = nc
        
        new_count = count_colors(current)
        
        if new_count >= current_count:
            current_count = new_count
            if new_count > best_count:
                best_count = new_count
                best_coloring = current[:]
        else:
            # Revert
            current[ei] = old_c
    
    # Final: try to fill in missing colors
    # For each color not used, try to find an edge that can take it
    used_colors = set(best_coloring)
    for c in range(120):
        if c in used_colors:
            continue
        if time.time() > deadline - 3:
            break
        # Try each edge
        for ei in range(NUM_EDGES):
            if try_recolor(ei, c, best_coloring):
                best_coloring[ei] = c
                used_colors.add(c)
                break
    
    return best_coloring
