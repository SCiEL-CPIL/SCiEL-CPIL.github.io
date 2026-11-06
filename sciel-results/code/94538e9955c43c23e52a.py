import numpy as np
import itertools
import time
import random

# Precomputation for fast Q3 check
# We need to check if adding (u,v) creates a Q3.
# This requires checking all (U, V) pairs with u in U, v in V.
# U has 4 vertices, V has 4 vertices.
# The other 3 in U are chosen from 14.
# The other 3 in V are chosen from 11 (excluding u, v and the 3 in U? No, U and V are disjoint).
# Wait, U and V are disjoint sets of vertices.
# If u in U and v in V, then u not in V and v not in U.
# So we choose 3 from (16 - {u, v}) for U?
# No, U must be disjoint from V.
# Let S = {0..15} \ {u, v}. |S| = 14.
# We choose 3 from S for U_rest.
# We choose 3 from (S \ U_rest) for V_rest.
# So we are partitioning S into U_rest, V_rest, and W (remaining).
# This is equivalent to choosing U_rest and V_rest.

# Precompute all (U_rest, V_rest) pairs for a fixed u, v?
# There are 16*15 = 240 pairs (u,v).
# For each, 60k combinations.
# Total 14.4 million precomputed tuples.
# Memory: 14.4M * (tuple overhead) is too high.
# Better to generate on the fly or cache per (u,v) if called often.
# Or just generate on the fly. 60k iterations is fast enough.

def creates_q3(graph, u, v):
    # graph is 16x16 boolean array
    # Check if adding (u,v) creates a Q3 containing (u,v)
    # u in U, v in V.
    # U = {u} + U_rest
    # V = {v} + V_rest
    # U_rest, V_rest disjoint, from S = {0..15} \ {u, v}
    
    S = [i for i in range(16) if i != u and i != v]
    
    # Iterate over all ways to pick 3 for U_rest from S
    for U_rest in itertools.combinations(S, 3):
        U = (u,) + U_rest
        # V_rest must be chosen from S \ U_rest
        # S \ U_rest has 11 elements.
        # We need to pick 3 for V_rest.
        # The remaining 8 form W (ignored).
        # But we also need to check the structure.
        # The structure is:
        # Edges in Q3:
        # 1. (u, v) - assumed added
        # 2. (u, w) for w in V_rest? No.
        # Q3 is 3-regular.
        # u is connected to 3 vertices in V. One is v. So u is connected to 2 others in V_rest?
        # No, u is connected to 3 vertices in V.
        # V has 4 vertices: v and V_rest (3 vertices).
        # So u must be connected to v AND 2 of the V_rest?
        # Or v AND 3 of V_rest?
        # u has degree 3 in Q3.
        # So u is connected to 3 vertices in V.
        # Since v is one, u is connected to 2 others in V_rest?
        # No, V has 4 vertices. u is connected to 3 of them.
        # So u is connected to v and 2 vertices in V_rest.
        # The vertex in V_rest NOT connected to u is the "matched" one in the complement?
        # Yes.
        # Similarly, v is connected to u and 2 vertices in U_rest.
        
        # So for a specific (U, V), the Q3 is defined by:
        # - u is not connected to exactly 1 vertex in V_rest.
        # - v is not connected to exactly 1 vertex in U_rest.
        # - The other 8 edges form a specific pattern?
        # Actually, it's easier to check the full 4x4 matrix.
        
        # Let's just iterate V_rest combinations.
        S_minus_U = [x for x in S if x not in U_rest]
        for V_rest in itertools.combinations(S_minus_U, 3):
            V = (v,) + V_rest
            
            # Now check if the induced subgraph on U, V (plus edge u-v) is Q3.
            # We need to check if there exists a permutation P such that:
            # A[i][j] = 1 if j != P[i]
            # This is equivalent to: The complement of A is a permutation matrix.
            # i.e., A has exactly 3 ones in each row and column.
            
            # Let's check row/col sums.
            # Row u: edges to V.
            # u-v is 1.
            # u-V_rest: need 2 ones.
            # So exactly 2 of the 3 edges (u, V_rest[i]) must be 1.
            
            # Row v: edges to U.
            # v-u is 1.
            # v-U_rest: need 2 ones.
            # So exactly 2 of the 3 edges (v, U_rest[i]) must be 1.
            
            # If these conditions fail, skip.
            
            # Count edges from u to V_rest
            cnt_u = 0
            for w in V_rest:
                if graph[u, w]:
                    cnt_u += 1
            if cnt_u != 2:
                continue
                
            # Count edges from v to U_rest
            cnt_v = 0
            for w in U_rest:
                if graph[v, w]:
                    cnt_v += 1
            if cnt_v != 2:
                continue
            
            # Now we have candidates where u and v have correct degree in the subgraph.
            # We need to check if the whole 4x4 matrix is J - P.
            # This means the 12 edges of Q3 exist.
            # The 12 edges are:
            # 1. (u, v)
            # 2. 2 edges from u to V_rest
            # 3. 2 edges from v to U_rest
            # 4. 6 edges between U_rest and V_rest
            
            # We know u is connected to 2 specific vertices in V_rest. Let's call them V_u_conn.
            # The one NOT connected is V_u_miss.
            # Similarly, v is connected to 2 in U_rest. Let U_v_conn.
            # The one NOT connected is U_v_miss.
            
            V_u_miss = None
            for w in V_rest:
                if not graph[u, w]:
                    V_u_miss = w
            U_v_miss = None
            for w in U_rest:
                if not graph[v, w]:
                    U_v_miss = w
            
            # The Q3 structure implies:
            # The missing edges in the 4x4 grid are (u, V_u_miss), (U_v_miss, v), and 6 more?
            # No, the missing edges form a perfect matching.
            # So (u, V_u_miss) is one pair.
            # (U_v_miss, v) is another pair?
            # Note: U_v_miss is in U_rest. v is in V.
            # So (U_v_miss, v) is an edge in the grid.
            # Is it missing?
            # Yes, because v is not connected to U_v_miss.
            # So the matching includes (u, V_u_miss) and (U_v_miss, v).
            
            # The remaining 2 pairs of the matching must be between U_rest \ {U_v_miss} and V_rest \ {V_u_miss}.
            # Let U_rem = U_rest \ {U_v_miss} (2 vertices)
            # Let V_rem = V_rest \ {V_u_miss} (2 vertices)
            # The matching on U_rem, V_rem must be such that the edges in the matching are MISSING in the graph.
            # And all other edges between U_rem, V_rem are PRESENT.
            
            # There are 2! = 2 possible matchings on U_rem, V_rem.
            # We need to check if EITHER matching corresponds to missing edges in the graph.
            
            U_rem = [w for w in U_rest if w != U_v_miss]
            V_rem = [w for w in V_rest if w != V_u_miss]
            
            # Matching 1: (U_rem[0], V_rem[0]) and (U_rem[1], V_rem[1])
            # Matching 2: (U_rem[0], V_rem[1]) and (U_rem[1], V_rem[0])
            
            # Check Matching 1:
            # Edges (U_rem[0], V_rem[0]) and (U_rem[1], V_rem[1]) must be MISSING (0 in graph).
            # Edges (U_rem[0], V_rem[1]) and (U_rem[1], V_rem[0]) must be PRESENT (1 in graph).
            # Wait, are there other edges?
            # Total edges between U_rest and V_rest is 3x3=9.
            # Q3 has 6 edges between U_rest and V_rest.
            # So 3 edges are missing.
            # The missing edges are the matching edges.
            # So yes, the 2 matching edges must be 0, and the other 7 must be 1?
            # No.
            # The 4x4 matrix has 16 entries.
            # 12 are 1 (edges of Q3).
            # 4 are 0 (matching).
            # We identified 2 zeros: (u, V_u_miss) and (U_v_miss, v).
            # The other 2 zeros must be in the U_rest x V_rest block.
            # And they must form a matching.
            
            # So, for Matching 1:
            # (U_rem[0], V_rem[0]) == 0
            # (U_rem[1], V_rem[1]) == 0
            # AND all other 6 edges in U_rest x V_rest must be 1.
            
            # Let's check this.
            
            valid = True
            # Check zeros
            if graph[U_rem[0], V_rem[0]]: valid = False
            if graph[U_rem[1], V_rem[1]]: valid = False
            
            if valid:
                # Check ones
                for i in range(2):
                    for j in range(2):
                        if i == j: continue # These are the zeros
                        if not graph[U_rem[i], V_rem[j]]:
                            valid = False
                # Also need to check edges from U_rest to V_u_miss?
                # No, V_u_miss is only connected to U_v_miss?
                # Wait. The matching covers all vertices.
                # (u, V_u_miss)
                # (U_v_miss, v)
                # (U_rem[0], V_rem[0])
                # (U_rem[1], V_rem[1])
                # This covers u, v, U_v_miss, V_u_miss, U_rem, V_rem.
                # Are there other zeros?
                # No, the matching is perfect.
                # So all other 12 edges must be 1.
                # We already checked:
                # u to V_rest (2 ones, 1 zero) - OK
                # v to U_rest (2 ones, 1 zero) - OK
                # U_rest to V_rest (6 ones, 2 zeros) - OK if valid
                
            if valid:
                return True
            
            # Check Matching 2
            valid = True
            if graph[U_rem[0], V_rem[1]]: valid = False
            if graph[U_rem[1], V_rem[0]]: valid = False
            
            if valid:
                for i in range(2):
                    for j in range(2):
                        if (i == 0 and j == 1) or (i == 1 and j == 0):
                            continue
                        if not graph[U_rem[i], V_rem[j]]:
                            valid = False
                if valid:
                    return True
                    
    return False

def construct():
    # Start with complete graph
    graph = np.ones((16, 16), dtype=bool)
    np.fill_diagonal(graph, False)
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Phase 1: Remove edges to make Q3-free
    # Use fast Q3 finder from original code?
    # Or just use creates_q3? No, creates_q3 is for ADDING.
    # We need to find ANY Q3.
    # We can use the LUT method from the original code, it's fast enough for ~50 removals.
    
    # Re-use the LUT logic from the prompt's current program for the initial phase
    # But we need to define PATTERNS and LUT.
    
    def get_patterns():
        patterns = []
        for perm in itertools.permutations(range(4)):
            mask = 0
            for i in range(4):
                for j in range(4):
                    if j != perm[i]:
                        mask |= (1 << (i * 4 + j))
            patterns.append(mask)
        return patterns

    PATTERNS = get_patterns()
    LUT = [False] * (1 << 16)
    for m in range(1 << 16):
        for p in PATTERNS:
            if (m & p) == p:
                LUT[m] = True
                break

    SUBSETS = list(itertools.combinations(range(16), 8))
    ALL_BIPS = []
    for idx, sub in enumerate(SUBSETS):
        parts = list(itertools.combinations(sub, 4))
        for A in parts:
            B = tuple(x for x in sub if x not in A)
            ALL_BIPS.append((sub, A, B))

    def find_q3(graph):
        for sub, A, B in ALL_BIPS:
            mask = 0
            for i in range(4):
                for j in range(4):
                    if graph[A[i], B[j]]:
                        mask |= (1 << (i * 4 + j))
            if LUT[mask]:
                # Return a random edge from this Q3
                edges = []
                for i in range(4):
                    for j in range(4):
                        if graph[A[i], B[j]]:
                            edges.append((A[i], B[j]))
                if edges:
                    return random.choice(edges)
        return None

    while time.time() < deadline - 30.0: # Reserve time for Phase 2
        edge = find_q3(graph)
        if edge is None:
            break
        u, v = edge
        graph[u, v] = False
        graph[v, u] = False

    # Phase 2: Local Search to add edges
    # Iterated Local Search
    best_edges = sum(graph) // 2
    
    for restart in range(5):
        if time.time() > deadline - 10.0:
            break
            
        # Try to add all possible edges
        missing_edges = [(i, j) for i in range(16) for j in range(i+1, 16) if not graph[i, j]]
        random.shuffle(missing_edges)
        
        improved = True
        while improved:
            improved = False
            for u, v in missing_edges:
                if time.time() > deadline - 10.0:
                    break
                # Check if we can add (u, v)
                if not creates_q3(graph, u, v):
                    graph[u, v] = True
                    graph[v, u] = True
                    improved = True
                    # Remove from missing list? Or just mark as present?
                    # Since we iterate a static list, we might try to add it again?
                    # No, if we add it, it's no longer missing.
                    # But the list is static.
                    # We should check if it's already present before checking creates_q3?
                    # Yes.
                    pass
        
        # If we improved, update best
        current_edges = sum(graph) // 2
        if current_edges > best_edges:
            best_edges = current_edges
        
        # Perturbation: Remove a random edge
        if time.time() > deadline - 5.0:
            break
        present_edges = [(i, j) for i in range(16) for j in range(i+1, 16) if graph[i, j]]
        if present_edges:
            u, v = random.choice(present_edges)
            graph[u, v] = False
            graph[v, u] = False

    # Return list of edges
    edges = []
    for i in range(16):
        for j in range(i+1, 16):
            if graph[i, j]:
                edges.append([i, j])
                
    return edges
