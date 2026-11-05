"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

# EVOLVE-BLOCK-START
def _contains_q3(adj_matrix):
    """
    Check if the graph (given as 16x16 numpy adjacency matrix) contains Q3 as a subgraph.
    Q3 has 8 vertices and 12 edges. We use a backtracking approach:
    - Pick an ordering of Q3 vertices (0..7) and Q3 edges.
    - Try to embed Q3 into the graph.
    
    For efficiency, we use a heuristic: try to find 8 vertices that induce at least the Q3 structure.
    A simpler approach: enumerate all 8-subsets (C(16,8)=12870) and check if any induces a supergraph of Q3.
    """
    n = adj_matrix.shape[0]
    # Q3 vertices and edges
    q3_vertices = list(range(8))
    q3_edges = []
    for i in range(8):
        for j in range(i+1, 8):
            # In Q3, two vertices are adjacent iff they differ in exactly one bit
            diff = bin(i ^ j).count('1')
            if diff == 1:
                q3_edges.append((i, j))
    # q3_edges has 12 edges
    
    # For each 8-subset of vertices, check if it can host Q3
    # We need to find a bijection from Q3 vertices to the 8 chosen vertices
    # such that all Q3 edges map to existing edges.
    
    # Precompute: for each pair of vertices in the subset, whether edge exists
    # Use backtracking with constraint propagation
    
    best_subset = None
    
    for subset in itertools.combinations(range(n), 8):
        subset_set = set(subset)
        # adj_subset[i][j] = 1 if edge between subset[i] and subset[j]
        # We'll use a 8x8 boolean matrix
        adj_sub = np.zeros((8, 8), dtype=bool)
        for i in range(8):
            for j in range(8):
                if i != j:
                    adj_sub[i, j] = adj_matrix[subset[i], subset[j]] > 0
        
        # Now check if Q3 can be embedded into this 8-vertex graph
        # Use backtracking: assign Q3 vertices 0..7 to positions 0..7 in subset
        # Order Q3 vertices by degree (all degree 3 in Q3, so any order)
        # But we can use a smarter ordering: pick vertex 0, then its neighbors, etc.
        
        # Let's use a simple backtracking with forward checking
        # q3_adj[i] = list of neighbors of i in Q3
        q3_adj = [[] for _ in range(8)]
        for u, v in q3_edges:
            q3_adj[u].append(v)
            q3_adj[v].append(u)
        
        # Order: we'll assign in order 0,1,2,...,7 but with pruning
        assignment = [-1] * 8  # q3 vertex -> subset position
        used = [False] * 8     # subset position used?
        
        def backtrack(qv):
            if qv == 8:
                return True
            for pos in range(8):
                if used[pos]:
                    continue
                # Check all already-assigned neighbors of qv
                ok = True
                for nb in q3_adj[qv]:
                    if nb < qv:  # already assigned
                        nb_pos = assignment[nb]
                        if not adj_sub[pos, nb_pos]:
                            ok = False
                            break
                if ok:
                    assignment[qv] = pos
                    used[pos] = True
                    if backtrack(qv + 1):
                        return True
                    assignment[qv] = -1
                    used[pos] = False
            return False
        
        if backtrack(0):
            return True
    
    return False


def _count_q3_subgraphs_fast(adj_matrix):
    """
    Count the number of subgraphs isomorphic to Q3.
    This is expensive, so we only use it for validation, not in the main loop.
    """
    n = adj_matrix.shape[0]
    count = 0
    # This is too slow for repeated calls; use _contains_q3 instead for checking
    # For counting, we'd need a more sophisticated approach
    # For now, just check existence
    return 1 if _contains_q3(adj_matrix) else 0


def construct():
    """
    Construct a Q3-free graph on 16 vertices with maximum edges.
    Strategy: Start with K16 and iteratively remove edges that are "most harmful"
    (i.e., appear in the most Q3 subgraphs) until no Q3 remains.
    
    We use a heuristic: track for each edge how many 4-cycles it's in, since Q3
    contains many 4-cycles. Removing edges in high-4-cycle-density regions helps.
    
    Actually, a better approach: 
    1. Start with a known Q3-free graph (e.g., K3,13 with 39 edges, or better constructions).
    2. Try to add edges one by one, checking Q3-freeness after each addition.
    3. Use random restarts with different initial orderings.
    
    Let's use a greedy edge-adding approach:
    - Start with empty graph.
    - Maintain a list of candidate edges (all C(16,2)=120 edges).
    - At each step, try adding each candidate edge (in some order), check if Q3-free.
    - If yes, add it; if no, skip it (or try later).
    - Use multiple random orderings and keep the best result.
    
    To speed up Q3 checking, we can use a necessary condition:
    Q3 is bipartite with parts of size 4. So any Q3 subgraph must have 4 vertices in one part
    and 4 in the other, with all edges between parts following the Q3 structure.
    
    A faster check: For Q3 to exist, there must be 8 vertices forming a specific pattern.
    We can precompute for each 8-subset whether it's "Q3-like" using a fast test.
    """
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    n = 16
    # All possible edges
    all_edges = list(itertools.combinations(range(n), 2))
    
    best_edges = []
    best_count = 0
    
    # Precompute adjacency for quick checks
    # We'll use a bitset representation for speed
    
    def make_adj(edge_set):
        """Create adjacency matrix from set of edges."""
        adj = np.zeros((n, n), dtype=np.uint8)
        for u, v in edge_set:
            adj[u, v] = 1
            adj[v, u] = 1
        return adj
    
    def is_q3_free(adj):
        """Check if graph is Q3-free using backtracking on 8-subsets."""
        # Quick necessary condition: chromatic number of Q3 is 2, so if graph is not
        # bipartite-like in some sense... actually Q3 is bipartite, so this doesn't help directly.
        
        # Use the backtracking approach but with early termination
        # Precompute for each pair whether edge exists
        # For each 8-subset, do a fast embedding check
        
        for subset in itertools.combinations(range(n), 8):
            # Fast check: does this subset contain a Q3?
            # Build local adjacency
            sub_adj = {}
            for i in range(8):
                for j in range(i+1, 8):
                    sub_adj[(subset[i], subset[j])] = adj[subset[i], subset[j]] > 0
            
            # Try to embed Q3: Q3 vertices 0-7, edges where Hamming distance = 1
            # Use backtracking
            assignment = [-1] * 8
            used = [False] * 8
            
            # Q3 adjacency
            q3_nb = [[] for _ in range(8)]
            for i in range(8):
                for j in range(i+1, 8):
                    if bin(i ^ j).count('1') == 1:
                        q3_nb[i].append(j)
                        q3_nb[j].append(i)
            
            def bt(qv):
                if qv == 8:
                    return True
                for pos in range(8):
                    if used[pos]:
                        continue
                    ok = True
                    for nb in q3_nb[qv]:
                        if nb < qv:
                            npos = assignment[nb]
                            a = subset[pos]
                            b = subset[npos]
                            if not (adj[a, b] > 0):
                                ok = False
                                break
                    if ok:
                        assignment[qv] = pos
                        used[pos] = True
                        if bt(qv + 1):
                            return True
                        assignment[qv] = -1
                        used[pos] = False
                return False
            
            if bt(0):
                return False  # Found Q3
        
        return True  # No Q3 found
    
    # Greedy edge addition with multiple random orderings
    import random
    
    num_trials = 0
    while time.time() < deadline:
        num_trials += 1
        # Random permutation of edges
        edge_order = all_edges[:]
        random.shuffle(edge_order)
        
        current_edges = set()
        adj = np.zeros((n, n), dtype=np.uint8)
        
        for u, v in edge_order:
            if time.time() > deadline:
                break
            
            # Try adding edge (u, v)
            adj[u, v] = 1
            adj[v, u] = 1
            
            if is_q3_free(adj):
                current_edges.add((u, v))
            else:
                # Remove the edge
                adj[u, v] = 0
                adj[v, u] = 0
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = list(current_edges)
        
        # If we've used most of the time, break
        if time.time() > deadline - 5:
            break
    
    # Convert to list of [u, v] format
    result = [[u, v] for u, v in best_edges]
    
    # Ensure it's valid (final check)
    if result:
        adj_final = np.zeros((n, n), dtype=np.uint8)
        for u, v in result:
            adj_final[u, v] = 1
            adj_final[v, u] = 1
        if not is_q3_free(adj_final):
            # Fallback to known valid graph: K3,13
            result = [[a, b] for a in range(3) for b in range(3, 16)]
    
    return result
# EVOLVE-BLOCK-END
