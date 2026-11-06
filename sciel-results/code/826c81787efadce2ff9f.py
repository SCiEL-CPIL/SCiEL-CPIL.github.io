import time
import random
from itertools import combinations

# Precompute all Q3 subgraph edge patterns over 8 vertices
# Q3 = 3-cube: 8 vertices, 12 edges. Vertices labeled 0..7 as 3-bit binary.
# Edge between u,v if Hamming distance = 1.
_Q3_EDGES = []
for u in range(8):
    for v in range(u + 1, 8):
        if bin(u ^ v).count('1') == 1:
            _Q3_EDGES.append((u, v))
# 12 edges per Q3

# All 8-vertex subsets of 0..15
_ALL_8SETS = list(combinations(range(16), 8))
# For each 8-set, map its vertices to 0..7 and check if it forms a Q3
# Precompute: for each 8-set, the 12 "local" edge pairs (indices 0..7)
_Q3_LOCAL = set(_Q3_EDGES)  # frozenset of 12 local pairs

def construct():
    start = time.time()
    DEADLINE = start + 110.0  # safety margin before 120s kill
    random.seed(42)
    
    n = 16
    # All possible edges (120 total)
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_idx = {(i, j) if i < j else (j, i): k for k, (i, j) in enumerate(all_edges)}
    
    def edge_key(u, v):
        return (u, v) if u < v else (v, u)
    
    # For each 8-set, precompute the 12 local edge pairs as global edge indices
    # An 8-set S = (v0, v1, ..., v7) contains a Q3 iff the induced subgraph
    # on S has exactly the 12 Q3 edges (and no others among those 12 positions)
    # Actually, a Q3 subgraph means there EXISTS an isomorphism from Q3 to the
    # induced subgraph on S. Since |S|=8, the induced subgraph IS the Q3 iff
    # it has exactly 12 edges and they match the Q3 pattern.
    # But wait: a subgraph isomorphic to Q3 could use a subset of S's edges.
    # Since |V(Q3)|=8=|S|, any Q3 subgraph on exactly these 8 vertices must use
    # all 8 vertices and exactly 12 edges forming the cube structure.
    
    # For each 8-set, determine which global edges correspond to each local Q3 edge
    # q3_constraints[si] = list of 12 global edge indices that form a Q3 on that 8-set
    q3_constraints = []
    for si, S in enumerate(_ALL_8SETS):
        # S is a tuple of 8 vertex indices in order
        # local vertex i corresponds to S[i]
        local_edges = []
        for (lu, lv) in _Q3_EDGES:
            gu, gv = S[lu], S[lv]
            local_edges.append(edge_idx[edge_key(gu, gv)])
        q3_constraints.append(local_edges)
    
    # Start with a dense random graph
    def make_graph(edge_prob=0.9):
        present = set()
        for (u, v) in all_edges:
            if random.random() < edge_prob:
                present.add(edge_idx[edge_key(u, v)])
        return present
    
    def count_q3(present):
        """Count number of Q3 subgraphs in the graph."""
        cnt = 0
        for edges_12 in q3_constraints:
            # Check if all 12 edges are present
            if all(e in present for e in edges_12):
                cnt += 1
        return cnt
    
    def find_one_q3(present):
        """Find one Q3 subgraph (returns list of 12 edge indices or None)."""
        for edges_12 in q3_constraints:
            if all(e in present for e in edges_12):
                return edges_12
        return None
    
    def find_min_removing_set(present):
        """Find minimum set of edges to remove to destroy all Q3s.
        Use greedy: repeatedly find a Q3, remove one of its edges that appears
        in the most Q3s."""
        if not present:
            return set()
        
        # Count how many Q3s each edge participates in
        def edge_q3_count(present_set):
            counts = {}
            for edges_12 in q3_constraints:
                if all(e in present_set for e in edges_12):
                    for e in edges_12:
                        counts[e] = counts.get(e, 0) + 1
            return counts
        
        removed = set()
        while True:
            current = present - removed
            if not current:
                break
            # Check if any Q3 remains
            remaining_q3 = None
            for edges_12 in q3_constraints:
                if all(e in current for e in edges_12):
                    remaining_q3 = edges_12
                    break
            if remaining_q3 is None:
                break
            # Find the edge in this Q3 that's in the most Q3s
            counts = edge_q3_count(current)
            best_edge = max(remaining_q3, key=lambda e: counts.get(e, 0))
            removed.add(best_edge)
        
        return removed
    
    best_graph = None
    best_count = 0
    
    # Strategy 1: Start dense, iteratively remove edges to fix Q3 violations
    # Strategy 2: Start sparse, add edges while maintaining Q3-freeness
    
    # --- Strategy 1: Dense start, repair ---
    for trial in range(30):
        if time.time() > DEADLINE:
            break
        present = make_graph(edge_prob=0.95)
        
        # Iterative repair: find Q3s and remove edges
        for iteration in range(200):
            if time.time() > DEADLINE:
                break
            q3 = find_one_q3(present)
            if q3 is None:
                break
            # Remove the edge from this Q3 that's in the most Q3s
            # Quick heuristic: just remove a random edge from the Q3
            # Better: count how many Q3s each edge is in
            counts = {}
            for edges_12 in q3_constraints:
                if all(e in present for e in edges_12):
                    for e in edges_12:
                        counts[e] = counts.get(e, 0) + 1
            best_edge = max(q3, key=lambda e: counts.get(e, 0))
            present.discard(best_edge)
        
        # Try to add back edges
        for trial2 in range(50):
            if time.time() > DEADLINE:
                break
            # Try adding a random missing edge
            missing = [e for e in range(len(all_edges)) if e not in present]
            if not missing:
                break
            random.shuffle(missing)
            improved = False
            for e in missing:
                test_present = present | {e}
                if find_one_q3(test_present) is None:
                    present = test_present
                    improved = True
                    break
            if not improved:
                break
        
        cnt = len(present)
        if cnt > best_count:
            best_count = cnt
            best_graph = present.copy()
    
    # --- Strategy 2: Greedy edge addition ---
    for trial in range(20):
        if time.time() > DEADLINE:
            break
        present = set()
        # Random edge order
        edge_order = list(range(len(all_edges)))
        random.shuffle(edge_order)
        
        for e in edge_order:
            if time.time() > DEADLINE:
                break
            test_present = present | {e}
            if find_one_q3(test_present) is None:
                present.add(e)
        
        cnt = len(present)
        if cnt > best_count:
            best_count = cnt
            best_graph = present.copy()
    
    # --- Strategy 3: Local search / hill climbing on the best found ---
    if best_graph is not None:
        current = best_graph.copy()
        for step in range(500):
            if time.time() > DEADLINE:
                break
            # Try to add a random missing edge
            missing = [e for e in range(len(all_edges)) if e not in current]
            if not missing:
                break
            random.shuffle(missing)
            improved = False
            for e in missing:
                test = current | {e}
                if find_one_q3(test) is None:
                    current = test
                    improved = True
                    break
            if not improved:
                # Try removing one edge and adding two others (2-opt style)
                if len(current) < 2:
                    break
                present_list = list(current)
                random.shuffle(present_list)
                found_move = False
                for e_remove in present_list[:10]:
                    test = current - {e_remove}
                    # Now try to add 2 edges
                    missing_now = [e for e in range(len(all_edges)) if e not in test]
                    random.shuffle(missing_now)
                    for e1 in missing_now[:20]:
                        test1 = test | {e1}
                        if find_one_q3(test1) is not None:
                            continue
                        for e2 in missing_now[:20]:
                            if e2 == e1:
                                continue
                            test2 = test1 | {e2}
                            if find_one_q3(test2) is None:
                                current = test2
                                found_move = True
                                break
                        if found_move:
                            break
                    if found_move:
                        break
                if not found_move:
                    break
    
    # Convert to edge list
    if best_graph is None:
        # Fallback: the K3,13 from seed
        return [[a, b] for a in range(3) for b in range(3, 16)]
    
    result = []
    for e in sorted(best_graph):
        u, v = all_edges[e]
        result.append([u, v])
    return result
