import time
import random
from itertools import combinations

DEADLINE = 150.0
START_TIME = time.time()

def _is_non_2colorable(n, edges):
    """
    Exact check for non-2-colorability using backtracking with unit propagation.
    Returns True if NOT 2-colorable.
    """
    m = len(edges)
    if m == 0:
        return False
    
    # Build edge masks and vertex-edge adjacency
    edge_masks = []
    vert_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
        for v in e:
            vert_edges[v].append(i)
    
    # State: assign[v] is -1, 0, or 1
    # e_c0[ei] is count of color 0 in edge ei
    # e_c1[ei] is count of color 1 in edge ei
    assign = [-1] * n
    e_c0 = [0] * m
    e_c1 = [0] * m
    
    def propagate():
        """Unit propagation. Returns False on contradiction."""
        worklist = list(range(m))
        in_work = [False] * m
        for ei in range(m):
            in_work[ei] = True
            
        while worklist:
            ei = worklist.pop()
            in_work[ei] = False
            c0, c1 = e_c0[ei], e_c1[ei]
            unassigned = 5 - c0 - c1
            
            if c0 == 5 or c1 == 5:
                return False  # Monochromatic edge found -> contradiction in 2-coloring attempt
            
            forced_color = -1
            if c0 == 4 and unassigned == 1:
                forced_color = 1
            elif c1 == 4 and unassigned == 1:
                forced_color = 0
            
            if forced_color != -1:
                # Find the unassigned vertex in this edge
                for v in edges[ei]:
                    if assign[v] == -1:
                        assign[v] = forced_color
                        # Update all edges containing v
                        for ej in vert_edges[v]:
                            if forced_color == 0:
                                e_c0[ej] += 1
                            else:
                                e_c1[ej] += 1
                            if not in_work[ej] and ej != ei:
                                worklist.append(ej)
                                in_work[ej] = True
                        break
                # Re-check this edge
                if not in_work[ei]:
                    worklist.append(ei)
                    in_work[ei] = True
        
        return True

    def backtrack():
        """Find unassigned vertex with most constrained edges, branch."""
        best_v = -1
        best_unassigned_count = 99
        for v in range(n):
            if assign[v] != -1:
                continue
            min_unassigned = 99
            for ei in vert_edges[v]:
                c0, c1 = e_c0[ei], e_c1[ei]
                unassigned = 5 - c0 - c1
                if unassigned < min_unassigned:
                    min_unassigned = unassigned
            if min_unassigned < best_unassigned_count:
                best_unassigned_count = min_unassigned
                best_v = v
        
        if best_v == -1:
            # All assigned; verify no monochromatic edges
            for ei in range(m):
                if e_c0[ei] == 5 or e_c1[ei] == 5:
                    return True  # Contradiction = not 2-colorable
            return False  # Valid coloring found = 2-colorable
        
        # Save state
        old_assign = assign[:]
        old_e_c0 = e_c0[:]
        old_e_c1 = e_c1[:]
        
        for color in (0, 1):
            assign[best_v] = color
            for ei in vert_edges[best_v]:
                if color == 0:
                    e_c0[ei] += 1
                else:
                    e_c1[ei] += 1
            
            if propagate():
                if backtrack():
                    return True
            
            # Restore state
            assign[:] = old_assign
            e_c0[:] = old_e_c0
            e_c1[:] = old_e_c1
        
        return False

    if not propagate():
        return True
    return backtrack()

def _generate_set_cover(n, rng):
    """
    Generate a set cover for the constraints S (|S| <= n/2).
    Universe: all subsets S of V with |S| <= n/2.
    Sets: For each edge E (5-subset of V), the set of S such that E cap S = empty.
    """
    vertices = list(range(n))
    
    # Generate all subsets S with |S| <= n//2
    # Note: If n is even, |S| can be up to n/2. If n is odd, |S| can be up to floor(n/2).
    # The condition is |S| <= n/2.
    max_s_size = n // 2
    
    universe = []
    for k in range(max_s_size + 1):
        for s in combinations(vertices, k):
            # Represent S as a bitmask
            mask = 0
            for v in s:
                mask |= (1 << v)
            universe.append(mask)
    
    # If universe is empty (n=0,1), trivial. But n>=5.
    # For n=5, max_s_size=2.
    
    # Generate all possible edges (5-subsets)
    all_edges = list(combinations(vertices, 5))
    
    # For each edge, determine which constraints it covers
    # Constraint i is covered by edge E if E cap S_i = empty.
    # This is equivalent to: S_i is a subset of V \ E.
    
    # We want to pick edges to cover all constraints.
    # Greedy Set Cover:
    # 1. Start with all constraints uncovered.
    # 2. At each step, pick the edge that covers the most uncovered constraints.
    # 3. Remove those constraints.
    # 4. Repeat until all covered.
    
    uncovered = set(universe)
    if not uncovered:
        return []
        
    # Precompute coverage for each edge to speed up?
    # Number of edges is C(n,5). For n=10, C(10,5)=252.
    # Number of constraints is sum_{k=0}^{n/2} C(n,k). For n=10, sum_{k=0}^5 C(10,k) = 1+10+45+120+210+252 = 638.
    # For n=12, sum_{k=0}^6 C(12,k) = 1+12+66+220+495+792+924 = 2510.
    # For n=14, sum_{k=0}^7 C(14,k) = ... ~ 10000.
    # For n=16, sum_{k=0}^8 C(16,k) = ... ~ 40000.
    # This might be too large for large n.
    # However, we only need to check if we can find a cover of size <= 51.
    # Smaller n might yield smaller covers?
    # Actually, smaller n means fewer vertices, so edges are "denser".
    # Let's try n from 9 to 16.
    
    # To make it faster, we can randomize the order of edges or use a heuristic.
    # Let's precompute the list of constraints covered by each edge.
    # edge_cover[edge_idx] = set of constraint masks covered by this edge.
    
    # Optimization: Instead of storing sets of masks, we can use bitsets if universe size is small.
    # But universe size grows exponentially.
    # Let's stick to sets of masks for simplicity, but limit n.
    
    # Let's try a different approach:
    # Randomly select edges and check if they cover all constraints.
    # Or, use the greedy algorithm but with a time limit.
    
    selected_edges = []
    edge_to_constraint = {}
    
    # Map constraint mask to index for faster lookup? No, just use set.
    
    # Precompute coverage
    # This step might be slow for large n.
    # Let's estimate time.
    # For each edge (C(n,5)), for each constraint (Sum C(n,k)), check intersection.
    # Intersection of two bitmasks is fast.
    
    # Let's build a list of (edge, covered_constraints)
    # But storing covered_constraints as a set is memory heavy.
    
    # Alternative:
    # Greedy step:
    # For each candidate edge, count how many *uncovered* constraints it covers.
    # This requires iterating over uncovered constraints for each edge.
    # |Uncovered| * |Edges|.
    
    # Let's implement the greedy step carefully.
    
    while uncovered:
        if time.time() > DEADLINE - 5:
            return None # Timeout
        
        best_edge = None
        max_covered = 0
        best_covered_set = set()
        
        # Shuffle edges to break ties randomly?
        # Or just iterate.
        for e in all_edges:
            if e in selected_edges:
                continue
                
            # Count uncovered constraints covered by e
            # Constraint s is covered if e & s == 0
            covered = 0
            for s in uncovered:
                if (e_mask := 0) or True: # dummy
                    pass
            # This loop is too slow.
            
        # We need a faster way to count coverage.
        # Let's precompute the coverage sets once.
        pass
        
    return None

def _greedy_set_cover(n, rng):
    vertices = list(range(n))
    max_s_size = n // 2
    
    universe = []
    for k in range(max_s_size + 1):
        for s in combinations(vertices, k):
            mask = 0
            for v in s:
                mask |= (1 << v)
            universe.append(mask)
            
    if not universe:
        return []

    all_edges = list(combinations(vertices, 5))
    
    # Precompute coverage for each edge
    # cover[edge_idx] = set of constraint indices covered
    # This is O(C(n,5) * Sum(C(n,k))) which is too big for n>12.
    
    # Let's try a randomized local search instead.
    # Start with a random set of edges.
    # Check if it covers all constraints.
    # If not, add edges that cover the most uncovered constraints.
    # If yes, try to remove edges.
    
    # This is essentially the same as greedy.
    
    # Let's limit n to 12 or 13 to keep the universe size manageable.
    # For n=12, |Universe| = 2510. C(12,5) = 792.
    # 792 * 2510 ~ 2 million operations per greedy step.
    # If we need 50 edges, that's 100 million operations. Might be too slow in Python.
    
    # We need a faster coverage check.
    # Use bitmasks for the universe?
    # If |Universe| <= 64, we can use an integer bitmask.
    # For n=9, |Universe| = Sum_{k=0}^4 C(9,k) = 1+9+36+84+126 = 256. Too big for 64-bit int.
    # For n=8, |Universe| = Sum_{k=0}^4 C(8,k) = 1+8+28+56+70 = 163. Still too big.
    # For n=7, |Universe| = Sum_{k=0}^3 C(7,k) = 1+7+21+35 = 64. Fits in 64-bit int!
    # For n=6, |Universe| = Sum_{k=0}^3 C(6,k) = 1+6+15+20 = 42.
    # For n=5, |Universe| = Sum_{k=0}^2 C(5,k) = 1+5+10 = 16.
    
    # So we can use bitmask acceleration for n <= 7.
    # But n must be at least 5.
    # Is it possible to have a non-2-colorable 5-uniform hypergraph with n=7?
    # Max edges C(7,5) = 21.
    # If we use all 21 edges, is it non-2-colorable?
    # Yes, likely.
    # But we want to minimize edges.
    
    # Let's try n=7, 8, 9 with the bitmask approach if possible, or sets.
    # For n=8, 163 constraints. We can use a set of ints.
    # For n=9, 256 constraints. Set of ints.
    # For n=10, 638 constraints. Set of ints.
    
    # Let's implement the greedy set cover with sets of ints.
    
    uncovered = set(universe)
    selected_edges = []
    
    # Precompute edge masks
    edge_masks = [0] * len(all_edges)
    for i, e in enumerate(all_edges):
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks[i] = mask
        
    # Precompute coverage sets?
    # cover[i] = {j for j, s_mask in enumerate(universe) if edge_masks[i] & s_mask == 0}
    # This is expensive to compute for large n.
    
    # Let's compute on the fly but optimize the inner loop.
    # For each edge, we want to find how many uncovered constraints it covers.
    
    while uncovered:
        if time.time() > DEADLINE - 5:
            return None
            
        best_idx = -1
        max_covered = 0
        best_covered_set = set()
        
        # Randomize edge order to avoid bias
        edge_indices = list(range(len(all_edges)))
        rng.shuffle(edge_indices)
        
        for i in edge_indices:
            if i in [x[0] for x in selected_edges]: # selected_edges stores (idx, mask)
                continue
                
            e_mask = edge_masks[i]
            covered = set()
            for s_mask in uncovered:
                if (e_mask & s_mask) == 0:
                    covered.add(s_mask)
                    
            if len(covered) > max_covered:
                max_covered = len(covered)
                best_idx = i
                best_covered_set = covered
                
        if best_idx == -1:
            # No edge covers any uncovered constraint?
            # This means the hypergraph is not non-2-colorable?
            # Wait, if no edge covers a constraint S, it means for all edges E, E & S != 0.
            # This implies S is a "hitting set" for the edges.
            # If S is a hitting set, then V\S is an independent set? No.
            # If E & S != 0 for all E, then no edge is contained in V\S.
            # This is exactly the condition for V\S to be a color class in a proper 2-coloring?
            # No, the condition for non-2-colorability is that for ALL S with |S|<=n/2, there IS an edge E subset of V\S.
            # If there is an S such that NO edge is subset of V\S, then S is a valid color class (along with V\S) for a proper 2-coloring?
            # Let's re-verify.
            # Proper 2-coloring: No edge is monochromatic.
            # Let C0, C1 be color classes.
            # No edge subset of C0. No edge subset of C1.
            # So for S=C0, there is NO edge subset of V\S = C1.
            # And for S=C1, there is NO edge subset of V\S = C0.
            # So if we find an S such that NO edge is subset of V\S, then S could be a color class.
            # But we also need to ensure the other class C1 has no monochromatic edges.
            # The set cover condition "For all S, exists E subset V\S" is SUFFICIENT for non-2-colorability.
            # Is it NECESSARY?
            # If the hypergraph is not 2-colorable, then for any partition (C0, C1), one of C0 or C1 contains a monochromatic edge.
            # WLOG, C0 contains a monochromatic edge E. So E subset C0.
            # This means for S=C1, E is NOT subset of V\S=C0? Wait.
            # E subset C0 means E & C1 = empty. So E subset V\C1.
            # So for S=C1, there IS an edge E subset V\S.
            # So if the hypergraph is not 2-colorable, then for ALL partitions, one side has a monochromatic edge.
            # This implies that for any S, either S has a monochromatic edge or V\S has a monochromatic edge.
            # If S has a monochromatic edge E, then E subset S.
            # This doesn't directly imply E subset V\S.
            
            # Let's re-read the standard reduction.
            # A hypergraph is NOT 2-colorable iff for every subset S of V, there is an edge E such that E subset S OR E subset V\S?
            # No.
            
            # Let's stick to the sufficient condition:
            # If for every S with |S| <= n/2, there is an edge E subset V\S, then the hypergraph is NOT 2-colorable.
            # Proof: Assume 2-coloring exists. Let C0 be the smaller color class. |C0| <= n/2.
            # By condition, there is an edge E subset V\C0 = C1.
            # So E is monochromatic (color 1). Contradiction.
            
            # So we just need to satisfy this sufficient condition.
            # If we fail to cover all S, it means there is some S with |S|<=n/2 such that NO edge is subset of V\S.
            # This does NOT mean the hypergraph is 2-colorable. It just means our sufficient condition is not met.
            # But if we can't satisfy it, we might still have a non-2-colorable hypergraph.
            # However, finding a set cover is a good heuristic.
            
            return None # Failed to find cover with this n
            
        # Add best edge
        selected_edges.append((best_idx, all_edges[best_idx]))
        uncovered -= best_covered_set
        
    return [e for idx, e in selected_edges]

def construct():
    best_edges = float('inf')
    best_result = None
    seed = 0
    
    # Try different n
    for n in range(7, 17):
        if time.time() > DEADLINE - 10:
            break
            
        rng = random.Random(seed)
        edges = _greedy_set_cover(n, rng)
        if edges is not None:
            # Verify
            if _is_non_2colorable(n, edges):
                if len(edges) < best_edges:
                    best_edges = len(edges)
                    best_result = {'n': n, 'edges': edges}
                    if best_edges <= 51:
                        # Try to improve with local search?
                        # For now, return if we hit 51?
                        # The task says minimize. 51 is the known lower bound.
                        # So 51 is optimal.
                        break
        seed += 1
        
    if best_result is None:
        # Fallback
        # Use a known construction or complete graph
        n = 9
        edges = [list(e) for e in combinations(range(n), 5)]
        best_result = {'n': n, 'edges': edges}
        
    return best_result
