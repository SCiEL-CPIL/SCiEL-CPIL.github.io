"""Q3-free graph construction on 16 vertices via local search."""
import time
import random
from itertools import combinations

N = 16
TOTAL_EDGES = N * (N - 1) // 2  # 120

# Precompute all Q3 embeddings as frozensets of edge indices
# Q3 vertices are (a,b,c) for a,b,c in {0,1}; edges connect Hamming distance 1
Q3_VERTS = [(a, b, c) for a in range(2) for b in range(2) for c in range(2)]
Q3_EDGES = []
for i, v in enumerate(Q3_VERTS):
    for j, w in enumerate(Q3_VERTS):
        if i < j:
            dist = sum(1 for x, y in zip(v, w) if x != y)
            if dist == 1:
                Q3_EDGES.append((i, j))

# Edge index mapping for the 120 edges
def edge_idx(u, v):
    if u > v:
        u, v = v, u
    # u in [0,15], v in [u+1,15]
    # index = u*15 - u*(u+1)//2 + (v - u - 1)
    return u * (N - 1) - u * (u + 1) // 2 + (v - u - 1)

def idx_to_edge(idx):
    # Inverse of edge_idx
    u = 0
    remaining = idx
    while remaining >= (N - 1 - u):
        remaining -= (N - 1 - u)
        u += 1
    v = u + 1 + remaining
    return (u, v)

# Precompute all Q3 subgraph edge index sets
# A Q3 embedding is a function from Q3 vertices to graph vertices (injective)
# For each such embedding, the 12 edges of Q3 map to 12 edges in the graph
# We need all possible sets of 12 edge indices that form a Q3

# To speed up, precompute: for each tuple of 8 distinct vertices, check if they form a Q3
# This is C(16,8) = 12870 tuples, each with 120 possible edge assignments... too slow to enumerate all

# Better approach: precompute the list of all Q3 edge-sets (as frozensets of edge indices)
# There are at most 16!/(16-8)! = 16*15*14*13*12*11*10*9 = 518,918,400 embeddings, but many are equivalent.
# Actually, we can be smarter: enumerate all subsets of 8 vertices, and for each, check all 8! / 6 = 3360 labelings... still too many.

# Alternative: Use a bitset-based approach. Precompute for each pair of vertices (u,v) the set of Q3-edge-positions
# Actually, let's just do a fast check function and use local search.

def check_q3_present(adj_matrix):
    """Check if the graph (as adjacency matrix) contains a Q3 subgraph.
    Returns True if Q3 is present, False otherwise.
    Uses: for each 8-subset of vertices, check if the induced subgraph contains Q3 as a subgraph.
    This is slow, so we use a smarter approach.
    """
    # A Q3 has 8 vertices and 12 edges. 
    # For each 8-subset, we need to check if there's an isomorphism to Q3.
    # There are 12870 8-subsets. For each, we check if the induced subgraph has at least 12 edges
    # and then check all possible Q3 mappings.
    
    # Actually, a faster check: Q3 is bipartite with parts of size 4.
    # The bipartition is determined by parity of (a+b+c).
    # So for a Q3 to exist, we need 4 vertices U and 4 vertices V such that:
    # - No edges within U, no edges within V (wait, Q3 IS bipartite, so yes)
    # - Every vertex in U has exactly 3 neighbors in V
    # - Every vertex in V has exactly 3 neighbors in U
    # - The edges form a specific pattern
    
    # Even faster: enumerate all 4+4 splits. C(16,4)*C(12,4)/... but that's a lot.
    
    # Let's use a different approach: enumerate all ways to pick 8 vertices, then check.
    # But 12870 * (check cost) might be too slow.
    
    # Best approach for local search: maintain a count of Q3s and update incrementally.
    # But for simplicity in this rewrite, let's use a moderately fast check.
    
    # Check: for each subset of 4 vertices U and 4 vertices V (disjoint),
    # check if the bipartite graph between U and V is exactly Q3.
    # Q3 as bipartite: each vertex in U connects to exactly 3 in V, and the structure is specific.
    
    # Number of (U,V) pairs: C(16,4) * C(12,4) / 2 ≈ 1820 * 495 / 2 ≈ 450,000. Too many.
    
    # Alternative: just check all 8-subsets and for each, check if it's a Q3.
    # For an 8-subset, check if the induced subgraph is isomorphic to Q3.
    # Q3 has 12 edges. First check if induced subgraph has exactly 12 edges (or at least 12, then check).
    
    for combo in combinations(range(N), 8):
        combo_set = set(combo)
        # Count edges in induced subgraph
        edge_count = 0
        for i in range(8):
            for j in range(i+1, 8):
                if adj_matrix[combo[i]][combo[j]]:
                    edge_count += 1
        if edge_count < 12:
            continue
        # Check if the induced subgraph contains Q3 as a subgraph
        # Map combo vertices to 0..7
        # Check all 8! / (aut(Q3)) mappings... aut(Q3) has size 48 (hyperoctahedral group B3)
        # So 8!/48 = 1680 checks per 8-subset. 12870 * 1680 is too many.
        
        # Better: check if the bipartite structure matches Q3.
        # Q3 is bipartite with parts determined by parity.
        # For the 8 vertices, try all ways to split into two sets of 4 (the bipartition).
        # There are C(8,4) = 70 ways, but only 2 are valid for a given labeling... 
        # Actually, for a graph to contain Q3, there must exist a bipartition (A,B) of the 8 vertices
        # such that the bipartite graph between A and B is exactly Q3.
        
        # For each of the 70 ways to split into A (4) and B (4):
        for a_subset in combinations(range(8), 4):
            a_set = set(a_subset)
            b_set = set(range(8)) - a_set
            # Check: each vertex in A has exactly 3 neighbors in B, and vice versa
            # And the specific Q3 structure holds.
            
            # Check degrees
            valid = True
            for i in a_set:
                deg = 0
                for j in b_set:
                    if adj_matrix[combo[i]][combo[j]]:
                        deg += 1
                if deg != 3:
                    valid = False
                    break
            if not valid:
                continue
            for j in b_set:
                deg = 0
                for i in a_set:
                    if adj_matrix[combo[i]][combo[j]]:
                        deg += 1
                if deg != 3:
                    valid = False
                    break
            if not valid:
                continue
            
            # Now check the specific Q3 structure.
            # In Q3, the bipartite graph between the two parts of size 4:
            # Part A: vertices with even parity (0,0,0), (1,1,0), (1,0,1), (0,1,1)
            # Part B: vertices with odd parity (1,0,0), (0,1,0), (0,0,1), (1,1,1)
            # Each vertex in A is connected to 3 vertices in B (all except the one at Hamming distance 3).
            # Equivalently, each vertex in A is NOT connected to exactly 1 vertex in B.
            
            # Check: for each vertex in A, it is non-adjacent to exactly 1 vertex in B.
            # And the non-adjacency pattern should be a perfect matching (each B vertex is non-adjacent to exactly 1 A vertex).
            
            non_adj = []
            for i in a_set:
                non_adj_to = []
                for j in b_set:
                    if not adj_matrix[combo[i]][combo[j]]:
                        non_adj_to.append(j)
                if len(non_adj_to) != 1:
                    valid = False
                    break
                non_adj.append(non_adj_to[0])
            if not valid:
                continue
            
            # Check that non_adj is a permutation of b_set
            if sorted(non_adj) != sorted(b_set):
                valid = False
            
            if valid:
                return True
    
    return False


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with a random dense graph
    best_adj = None
    best_edges = 0
    
    # Greedy initialization: start with complete graph and remove edges to break Q3s
    # Or start with a random graph with p=0.8 and add edges greedily.
    
    # Strategy: local search
    # 1. Start with a high-density random graph
    # 2. If Q3 present, remove edges to break Q3s
    # 3. Try to add edges back
    # 4. Repeat with perturbations
    
    best_score = 0
    best_adj_list = None
    
    for restart in range(50):
        if time.time() > deadline - 10.0:
            break
        
        # Initialize: random graph with p=0.75
        adj = [[0]*N for _ in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                if random.random() < 0.75:
                    adj[i][j] = 1
                    adj[j][i] = 1
        
        # Phase 1: Remove edges to make Q3-free
        # Repeatedly find a Q3 and remove an edge from it
        for _ in range(200):
            if time.time() > deadline - 5.0:
                break
            if not check_q3_present(adj):
                break
            # Find a Q3 and remove a random edge from it
            # We need to find one Q3. Let's find it during the check.
            # For efficiency, let's find a Q3 and return one of its edges.
            found = False
            for combo in combinations(range(N), 8):
                combo_set = set(combo)
                edge_count = 0
                for i in range(8):
                    for j in range(i+1, 8):
                        if adj[combo[i]][combo[j]]:
                            edge_count += 1
                if edge_count < 12:
                    continue
                for a_subset in combinations(range(8), 4):
                    a_set = set(a_subset)
                    b_set = set(range(8)) - a_set
                    valid = True
                    for i in a_set:
                        deg = 0
                        for j in b_set:
                            if adj[combo[i]][combo[j]]:
                                deg += 1
                        if deg != 3:
                            valid = False
                            break
                    if not valid:
                        continue
                    for j in b_set:
                        deg = 0
                        for i in a_set:
                            if adj[combo[i]][combo[j]]:
                                deg += 1
                        if deg != 3:
                            valid = False
                            break
                    if not valid:
                        continue
                    non_adj = []
                    for i in a_set:
                        non_adj_to = []
                        for j in b_set:
                            if not adj[combo[i]][combo[j]]:
                                non_adj_to.append(j)
                        if len(non_adj_to) != 1:
                            valid = False
                            break
                        non_adj.append(non_adj_to[0])
                    if not valid:
                        continue
                    if sorted(non_adj) != sorted(b_set):
                        valid = False
                    if valid:
                        # Found a Q3. Remove a random edge from it.
                        # The edges of this Q3 are all (combo[i], combo[j]) where adj is 1
                        q3_edges = []
                        for i in range(8):
                            for j in range(i+1, 8):
                                if adj[combo[i]][combo[j]]:
                                    q3_edges.append((combo[i], combo[j]))
                        # Remove a random edge
                        u, v = random.choice(q3_edges)
                        adj[u][v] = 0
                        adj[v][u] = 0
                        found = True
                        break
                if found:
                    break
            if not found:
                break
        
        # Count edges
        edge_count = sum(sum(row) for row in adj) // 2
        if edge_count > best_score:
            best_score = edge_count
            best_adj_list = [row[:] for row in adj]
        
        # Phase 2: Try to add edges
        for _ in range(500):
            if time.time() > deadline - 5.0:
                break
            # Try adding a random missing edge
            missing = []
            for i in range(N):
                for j in range(i+1, N):
                    if not adj[i][j]:
                        missing.append((i, j))
            if not missing:
                break
            u, v = random.choice(missing)
            adj[u][v] = 1
            adj[v][u] = 1
            if check_q3_present(adj):
                adj[u][v] = 0
                adj[v][u] = 0
            else:
                edge_count = sum(sum(row) for row in adj) // 2
                if edge_count > best_score:
                    best_score = edge_count
                    best_adj_list = [row[:] for row in adj]
        
        # Phase 3: Perturbation - remove some random edges and re-add
        if best_adj_list is not None:
            adj = [row[:] for row in best_adj_list]
            # Remove 5-10 random edges
            for _ in range(random.randint(5, 10)):
                u = random.randint(0, N-1)
                v = random.randint(0, N-1)
                if u != v and adj[u][v]:
                    adj[u][v] = 0
                    adj[v][u] = 0
            # Try to re-add
            for _ in range(300):
                if time.time() > deadline - 5.0:
                    break
                missing = []
                for i in range(N):
                    for j in range(i+1, N):
                        if not adj[i][j]:
                            missing.append((i, j))
                if not missing:
                    break
                u, v = random.choice(missing)
                adj[u][v] = 1
                adj[v][u] = 1
                if check_q3_present(adj):
                    adj[u][v] = 0
                    adj[v][u] = 0
                else:
                    edge_count = sum(sum(row) for row in adj) // 2
                    if edge_count > best_score:
                        best_score = edge_count
                        best_adj_list = [row[:] for row in adj]
    
    # Convert best_adj_list to edge list
    if best_adj_list is None:
        # Fallback: K3,13
        return [[a, b] for a in range(3) for b in range(3, 16)]
    
    edges = []
    for i in range(N):
        for j in range(i+1, N):
            if best_adj_list[i][j]:
                edges.append([i, j])
    return edges
