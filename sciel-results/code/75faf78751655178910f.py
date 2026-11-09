"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from functools import lru_cache

def construct():
    """
    Find a maximum independent set in the conflict graph on {1,...,256}.
    Edge (a,b) exists iff (a+b) divides a*b, i.e., (a*b) % (a+b) == 0.
    We want the largest subset with no such conflicts.
    
    Strategy:
    1. Build conflict graph as bitsets (Python ints for 256 bits).
    2. Use branch-and-bound with greedy coloring upper bound (Tomita-style).
    3. Order vertices by degree descending for better pruning.
    4. Deadline guard: return best found if time runs low.
    """
    start_time = time.time()
    DEADLINE = 85.0  # seconds, leaving 5s margin before 90s kill
    
    N = 256
    
    # Build conflict adjacency as list of ints (bitsets)
    # adj[i] has bit j set iff i and j conflict (i.e., (i+1)*(j+1) % ((i+1)+(j+1)) == 0)
    # Vertices indexed 0..255 correspond to numbers 1..256
    adj = [0] * N
    for i in range(N):
        a = i + 1
        for j in range(i + 1, N):
            b = j + 1
            if (a * b) % (a + b) == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Degree of each vertex
    def popcount(x):
        return bin(x).count('1')
    
    degrees = [popcount(adj[i]) for i in range(N)]
    
    # Order vertices by degree descending
    order = sorted(range(N), key=lambda i: -degrees[i])
    
    # Remap: new_index[i] = position of vertex i in the ordered list
    # new_vertex[pos] = original vertex index
    new_vertex = [order[k] for k in range(N)]
    
    # Build adjacency in new ordering
    adj_new = [0] * N
    for new_i in range(N):
        orig_i = new_vertex[new_i]
        # For each original neighbor
        mask = adj[orig_i]
        # Convert to new indices
        # We need to map bit positions from old to new
        # Build inverse mapping: old_pos -> new_pos
        old_to_new = [0] * N
        for k in range(N):
            old_to_new[order[k]] = k
        # Now build adj_new[new_i]
        m = mask
        bits = []
        while m:
            lsb = m & (-m)
            bit_pos = lsb.bit_length() - 1
            bits.append(bit_pos)
            m ^= lsb
        for bp in bits:
            ni = old_to_new[bp]
            adj_new[new_i] |= (1 << ni)
    
    # Greedy coloring for upper bound in branch-and-bound
    # For a set of candidate vertices (as a bitset), color them greedily
    # to get an upper bound on clique size in the complement (independent set size)
    
    # For MIS: we're finding max independent set in G = max clique in complement of G
    # But it's easier to work directly with independent set formulation.
    
    # Let's reformulate: we want max set S such that no two elements in S have an edge in G.
    # This is exactly a maximum independent set.
    
    # Branch and bound for MIS:
    # - Candidates: set of vertices not yet decided
    # - Current set: vertices already in the independent set
    # - For upper bound: we can use greedy coloring on the induced subgraph on candidates
    #   Wait, for independent set, the upper bound is different.
    #   
    # Actually, let me think again. For maximum independent set:
    # Upper bound = current_size + (number of candidates that could potentially be added)
    # A simple bound: current_size + len(candidates)
    # Better: use the fact that in any independent set, no two candidates adjacent to each other
    # can both be in the set. So upper bound <= current_size + alpha(G[candidates])
    # where alpha is independence number. We can bound alpha by greedy coloring of G[candidates]:
    # color the subgraph G[candidates] greedily, the max color class size is an upper bound on
    # the size of a clique in G[candidates]... wait no.
    #
    # For independent set in G: upper bound on how many more we can add from candidates
    # = size of max independent set in G[candidates].
    # A simple upper bound: use greedy coloring of G[candidates]. In a proper coloring,
    # each color class is an independent set in G[candidates]... no wait, each color class
    # is an independent set in G[candidates] means vertices in same color class have NO edges
    # between them in G[candidates], so they form an independent set. So the largest color class
    # gives a LOWER bound, not upper.
    #
    # For upper bound on independence number: the minimum number of colors needed (chromatic number)
    # is a lower bound on... hmm.
    #
    # Actually: if we color G[candidates] with k colors, then the independence number alpha(G[candidates])
    # <= |V| / 1 ... that's not helpful.
    #
    # Better approach: use the fact that in any graph, alpha(G) <= |V| - maximum matching... complex.
    #
    # Simpler: use a greedy algorithm to find a maximal independent set in G[candidates],
    # its size is a lower bound. For upper bound, note that if we greedily pick vertices
    # and remove their neighbors, the number of picks is a lower bound on alpha.
    #
    # For upper bound, we can use: alpha(G) <= |V| - tau(G) where tau is min vertex cover...
    # or alpha(G) <= |V| / (1 + min_degree) roughly... 
    #
    # Actually, the standard approach for MIS branch-and-bound is:
    # Upper bound = current_size + (number of vertices in candidates that are not adjacent to each other in a greedy sense)
    # 
    # Let me use a different, well-known approach:
    # For each candidate vertex v, either include it (and remove its neighbors from candidates) 
    # or exclude it. The upper bound at a node is:
    #   current_size + greedy_max_independent_set(candidates)
    # where greedy_max_independent_set uses a simple heuristic.
    #
    # Even simpler upper bound: current_size + number of candidates
    # This is weak but correct.
    #
    # Better: use coloring. In the graph G[candidates], if we can color with k colors,
    # then alpha(G[candidates]) <= |V| ... no.
    #
    # I'll use: upper bound = current_size + (size of a greedy independent set in G[candidates] as lower bound)
    # For pruning, we need UPPER bounds. So:
    # upper_bound = current_size + len(candidates)
    # This is always valid but weak.
    #
    # Let me try a different approach: since N=256 is small, use a more direct method.
    #
    # Actually, for a 256-vertex graph, branch and bound with a decent heuristic should work.
    # Let me use the following:
    # - Sort candidates by degree in the remaining graph (descending)
    # - At each step, branch on the highest-degree vertex
    # - Upper bound: current_size + number of vertices in candidates (weak but simple)
    # - Additionally, use a greedy coloring bound: 
    #   In G[candidates], greedily color. If we use c colors, then any independent set in G[candidates]
    #   has at most... hmm, this doesn't directly give an upper bound on alpha.
    #
    # Wait, I think I was confusing things. Let me look at this differently.
    #
    # For MAX CLIQUE (which is what Tomita's algorithm solves):
    # - Upper bound via greedy coloring: if the remaining candidates can be colored with k colors,
    #   then no clique larger than k exists in them, so upper bound = current_clique_size + k.
    #
    # For MAX INDEPENDENT SET in G = MAX CLIQUE in complement of G:
    # - We can run max clique algorithm on the complement graph!
    #
    # Let's do that. Build complement graph, run Tomita's max clique algorithm.
    
    # Build complement adjacency
    all_mask = (1 << N) - 1
    adj_comp = [0] * N
    for i in range(N):
        # In complement, i is adjacent to all except itself and its neighbors in G
        adj_comp[i] = all_mask & ~(adj_new[i] | (1 << i))
    
    # Tomita's algorithm for maximum clique
    best_clique_size = 0
    best_clique = []
    
    # Greedy coloring for upper bound
    # Colors assigned to candidates in order
    def greedy_color_bound(candidates_mask):
        """
        Greedily color the subgraph induced by candidates in the COMPLEMENT graph.
        Returns an upper bound on clique size = max color used.
        We process vertices in order of decreasing degree in the complement subgraph.
        """
        # Get list of vertices in candidates
        verts = []
        m = candidates_mask
        while m:
            lsb = m & (-m)
            pos = lsb.bit_length() - 1
            verts.append(pos)
            m ^= lsb
        
        if not verts:
            return 0
        
        # Compute degrees in the complement subgraph
        degs = []
        for v in verts:
            degs.append(popcount(adj_comp[v] & candidates_mask))
        
        # Sort by degree descending
        order_idx = sorted(range(len(verts)), key=lambda i: -degs[i])
        sorted_verts = [verts[order_idx[i]] for i in range(len(verts))]
        
        # Greedy coloring: assign each vertex the smallest available color
        color_of = {}
        max_color = 0
        for v in sorted_verts:
            # Find colors used by already-colored neighbors of v
            neighbor_colors = set()
            neigh_mask = adj_comp[v] & candidates_mask
            nm = neigh_mask
            while nm:
                lsb = nm & (-nm)
                np = lsb.bit_length() - 1
                if np in color_of:
                    neighbor_colors.add(color_of[np])
                nm ^= lsb
            
            # Assign smallest non-conflicting color
            c = 0
            while c in neighbor_colors:
                c += 1
            color_of[v] = c
            max_color = max(max_color, c + 1)
        
        return max_color
    
    # Branch and bound for maximum clique
    sys.setrecursionlimit(10000)
    
    def branch_and_bound(candidates_mask, current_clique, current_size):
        nonlocal best_clique_size, best_clique, start_time
        
        if time.time() - start_time > DEADLINE - 5:
            return
        
        if not candidates_mask:
            if current_size > best_clique_size:
                best_clique_size = current_size
                best_clique = current_clique[:]
            return
        
        # Upper bound
        ub = greedy_color_bound(candidates_mask)
        if current_size + ub <= best_clique_size:
            return
        
        # Pick a vertex to branch on - highest degree in candidates
        verts = []
        m = candidates_mask
        while m:
            lsb = m & (-m)
            pos = lsb.bit_length() - 1
            verts.append(pos)
            m ^= lsb
        
        # Sort by degree in candidates, descending
        degs = [popcount(adj_comp[v] & candidates_mask) for v in verts]
        order_idx = sorted(range(len(verts)), key=lambda i: -degs[i])
        
        for idx in order_idx:
            if time.time() - start_time > DEADLINE - 5:
                return
            
            v = verts[idx]
            
            # Include v
            new_clique = current_clique + [v]
            # New candidates: those in candidates that are adjacent to v in complement
            new_candidates = candidates_mask & adj_comp[v]
            # Remove v from candidates
            new_candidates &= ~(1 << v)
            
            branch_and_bound(new_candidates, new_clique, current_size + 1)
            
            # Exclude v
            candidates_mask &= ~(1 << v)
            if not candidates_mask:
                break
    
    # Initial call
    initial_candidates = (1 << N) - 1
    branch_and_bound(initial_candidates, [], 0)
    
    # Convert clique indices back to original numbers
    # The clique is in the complement graph, which corresponds to an independent set in G
    # The indices are in the "new" ordering
    result_set = set()
    for v in best_clique:
        orig_idx = new_vertex[v]
        result_set.add(orig_idx + 1)  # Convert back to 1-based
    
    if not result_set:
        # Fallback: return all odds
        result_set = set(range(1, 257, 2))
    
    return list(result_set)

# EVOLVE-BLOCK-END
