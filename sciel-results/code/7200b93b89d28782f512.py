import itertools
import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    n = 16
    vertices = list(range(n))

    # Precompute all Q3 embeddings as frozensets of 12 edge indices
    # Edge index: e(u,v) where u < v, index = u*n + v - (u+1)*(u+2)//2
    # Simpler: use (u,v) tuples with u < v

    # Generate all 8-vertex subsets
    subsets_8 = list(itertools.combinations(vertices, 8))

    # For each 8-subset, generate all Q3 embeddings (bijections to cube vertices)
    # Q3 vertices: 0..7, edges connect vertices differing in exactly one bit
    # Q3 edges:
    q3_edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]

    # All permutations of 8 positions for mapping cube vertices to 8-subset positions
    # But we can reduce: fix vertex 0 of Q3 to position 0 of the subset
    # This reduces by factor of 8

    forbidden_sets = []  # list of frozensets of (u,v) edges

    for s in subsets_8:
        # Fix Q3 vertex 0 -> s[0]
        # Permute the remaining 7 cube vertices over s[1..7]
        remaining_cube = list(range(1, 8))
        remaining_pos = list(range(1, 8))

        for perm in itertools.permutations(remaining_pos):
            # mapping: cube_vertex -> position in s
            mapping = [0] + list(perm)  # cube vertex i -> s[mapping[i]]

            # Build the 12 edges as (u,v) with u<v
            edge_set = set()
            for a, b in q3_edges:
                u = s[mapping[a]]
                v = s[mapping[b]]
                if u > v:
                    u, v = v, u
                edge_set.add((u, v))

            forbidden_sets.append(frozenset(edge_set))

    # Remove duplicates
    forbidden_sets = list(set(forbidden_sets))

    # Build edge index mapping
    all_edge_tuples = [(u, v) for u in range(n) for v in range(u+1, n)]
    edge_to_idx = {e: i for i, e in enumerate(all_edge_tuples)}
    idx_to_edge = {i: e for i, e in enumerate(edge_to_idx.items())}
    num_edges = len(all_edge_tuples)  # 120

    # Convert forbidden sets to index sets
    forbidden_idx_sets = []
    for fs in forbidden_sets:
        idx_set = frozenset(edge_to_idx[e] for e in fs)
        forbidden_idx_sets.append(idx_set)

    # Build for each edge index, list of forbidden set indices it appears in
    edge_to_forbidden = [[] for _ in range(num_edges)]
    for fi, fset in enumerate(forbidden_idx_sets):
        for ei in fset:
            edge_to_forbidden[ei].append(fi)

    # Max-SAT style: start with all edges, greedily remove edges to satisfy all clauses
    # Each clause (forbidden set) is satisfied if at least one edge in it is removed

    def is_q3_free(edge_set_indices):
        """Check if the set of active edges (as a set of indices) is Q3-free."""
        active = set(edge_set_indices)
        for fset in forbidden_idx_sets:
            if fset.issubset(active):
                return False
        return True

    # Greedy removal: start with all edges, remove edges that appear in most unsatisfied clauses
    current_active = set(range(num_edges))

    # Count how many forbidden sets each edge is in
    edge_count = [len(edge_to_forbidden[i]) for i in range(num_edges)]

    # Greedy set cover: repeatedly remove the edge that covers the most unsatisfied clauses
    unsatisfied = set(range(len(forbidden_idx_sets)))

    while unsatisfied:
        if time.time() > deadline - 5:
            break
        # For each active edge, count how many unsatisfied clauses it covers
        best_edge = None
        best_count = -1
        for ei in sorted(current_active, key=lambda x: -edge_count[x]):
            count = 0
            for fi in edge_to_forbidden[ei]:
                if fi in unsatisfied:
                    count += 1
            if count > best_count:
                best_count = count
                best_edge = ei
            if count <= 0:
                break
        if best_edge is None or best_count <= 0:
            break
        current_active.discard(best_edge)
        # Remove all clauses satisfied by this edge
        for fi in edge_to_forbidden[best_edge]:
            unsatisfied.discard(fi)

    initial_edge_count = len(current_active)

    # Now try to add back edges via local search
    # An edge can be added if it doesn't complete any forbidden set
    def can_add_edge(ei, active_set):
        """Check if adding edge ei to active_set keeps Q3-free."""
        for fi in edge_to_forbidden[ei]:
            fset = forbidden_idx_sets[fi]
            if fset - {ei} <= active_set:
                return False
        return True

    # Local search: try to add edges back
    best_active = set(current_active)
    best_count = len(best_active)

    for restart in range(50):
        if time.time() > deadline - 10:
            break

        active = set(best_active)

        # Get candidate edges not in active
        candidates = [ei for ei in range(num_edges) if ei not in active]
        random.shuffle(candidates)

        # Try adding each candidate
        for ei in candidates:
            if time.time() > deadline - 5:
                break
            if ei in active:
                continue
            if can_add_edge(ei, active):
                active.add(ei)

        if len(active) > best_count:
            best_count = len(active)
            best_active = set(active)

    # Convert back to edge list
    result = [[u, v] for ei in best_active for u, v in [all_edge_tuples[ei]]]
    return result

if __name__ == "__main__":
    start = time.time()
    edges = construct()
    print(f"Found {len(edges)} edges in {time.time() - start:.2f}s")
