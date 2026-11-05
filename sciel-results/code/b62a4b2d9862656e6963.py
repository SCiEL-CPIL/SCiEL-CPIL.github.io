import random
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    best_n = 39
    best_edges = []
    
    # We want to maximize n. Try from 47 downwards.
    # If we find a valid graph for n, that's the best possible for that n.
    # We stop as soon as we find a valid graph for the largest n.
    
    for n in range(47, 38, -1):
        if time.time() > deadline:
            break
            
        # Target minimum degree
        min_deg = n - 39
        
        # If min_deg is 0, empty graph works, but we are looking for larger n.
        # For n=47, min_deg = 8.
        # For n=46, min_deg = 7.
        
        # We will try to construct such a graph using random restarts.
        # We use a bitmask representation for neighbors to quickly check for C4.
        # A C4 exists if there are two distinct vertices u, v that share two common neighbors.
        # When adding edge (u, v), we check if N(u) & N(v) has more than 1 bit set.
        # Actually, if we add (u, v), we must ensure no C4 is formed.
        # A C4 is u-w-x-v-u. This requires w and x to be common neighbors of u and v.
        # So if we add edge (u, v), we must check if there are already two common neighbors between u and v.
        # Wait, if we add (u, v), the new cycles involving (u, v) are of length 4 if there is a path of length 3 between u and v.
        # Path u-w-x-v. w and x are neighbors of u and v respectively? No.
        # Path u-w-x-v implies w is neighbor of u, x is neighbor of v, and w is neighbor of x.
        # This is not simply common neighbors.
        
        # Correct C4 check:
        # A graph has a C4 if and only if there exist two distinct vertices u, v such that |N(u) \cap N(v)| >= 2.
        # So, when adding edge (u, v), we don't just check common neighbors of u and v.
        # We must ensure that for all w, v, the number of common neighbors doesn't exceed 1.
        
        # Let's refine the check.
        # If we add edge (u, v), we update N(u) to include v and N(v) to include u.
        # This change might create a C4 involving u or v.
        # A C4 involving u must be of the form u-w-x-y-u.
        # Since we only added one edge (u, v), any new C4 must use this edge.
        # So the C4 must be u-v-w-x-u.
        # This requires:
        # 1. v is neighbor of u (just added)
        # 2. w is neighbor of v
        # 3. x is neighbor of w
        # 4. x is neighbor of u
        # So, we need to check if there exists a path v-w-x of length 2 such that x is in N(u).
        # Equivalently, for each w in N(v), check if N(w) intersects N(u) in any vertex x (x != v, x != u).
        # If such an x exists, then u-v-w-x-u is a C4.
        
        # So the check for adding (u, v) is:
        # For all w in N(v) (before adding v to N(u) and u to N(v)), 
        # if N(w) & N(u) is non-empty (excluding v and u if they were already connected, but they aren't), 
        # then adding (u, v) creates a C4.
        
        # Wait, N(w) & N(u) might contain v? No, v is not in N(u) yet.
        # Might it contain u? No, u is not in N(w) usually (simple graph, no loops).
        # So we just check if (N(w) & N(u)) is not empty.
        
        # Complexity: O(deg(v) * n/word_size) or O(deg(v) * deg(u)) if iterating.
        # Using bitmasks, N(w) & N(u) is a single bitwise AND and check for 0.
        
        def can_add_edge(u, v, neighbors):
            # neighbors is a list of ints (bitmasks)
            # Check if adding (u, v) creates a C4
            # For each w in N(v), check if N(w) & N(u) != 0
            # Note: u and v are not neighbors yet.
            mask_u = neighbors[u]
            mask_v = neighbors[v]
            
            # Iterate through set bits in mask_v
            m = mask_v
            while m:
                lsb = m & -m
                w = lsb.bit_length() - 1
                # Check if N(w) intersects N(u)
                if neighbors[w] & mask_u:
                    return False
                m ^= lsb
            return True

        def try_construct(n, min_deg):
            # Random restarts
            # We want min degree >= min_deg.
            # Greedy approach:
            # 1. Start with empty graph.
            # 2. Repeatedly add random edges that don't create C4.
            # 3. Check if min degree is satisfied.
            # 4. If not, maybe remove some edges from high-degree vertices to allow others to grow?
            #    Or use a different heuristic.
            
            # A better heuristic:
            # Start with a random graph where each vertex has degree exactly min_deg?
            # No, random regular graphs likely have C4s.
            
            # Let's stick to greedy addition but with a twist:
            # Prioritize adding edges to vertices with low degree.
            
            for restart in range(2000):
                if time.time() > deadline:
                    return False, []
                
                neighbors = [0] * n
                degrees = [0] * n
                edges = []
                
                # Shuffle vertices
                verts = list(range(n))
                random.shuffle(verts)
                
                # We want to reach min_deg for all.
                # Total edges needed approx n * min_deg / 2.
                # Let's try to add edges until we can't or time runs out.
                
                # To ensure we try to satisfy min_deg, we can bias edge selection.
                # Or simply add random valid edges and hope.
                # Given the high density (min_deg ~ 8 for n=47), random greedy might get stuck.
                
                # Let's try a specific strategy:
                # Build a "near-regular" graph.
                # For each vertex u, try to connect it to min_deg random vertices.
                # But we must check C4 constraints.
                
                # Alternative: Start with a known C4-free structure?
                # Polar graph of elliptic quadric?
                # For n=47, this is specific.
                
                # Let's try random greedy with "pressure" on low-degree nodes.
                # At each step, pick a random valid edge (u, v) where min(deg(u), deg(v)) is small?
                # Or just pick random u, v and check.
                
                # Optimization: Pre-compute possible edges? No, dynamic.
                
                # Let's iterate.
                # Max steps: n * n / 2.
                # We can stop early if all degrees >= min_deg.
                
                success = False
                # To speed up, we can try to fill degrees.
                # Let's shuffle the list of all potential edges and try to add them.
                # But checking C4 is expensive if we do it for all.
                
                # Randomized Greedy:
                # Keep a list of candidates?
                # Just random sampling.
                
                # We will do a "random walk" on the space of valid graphs?
                # No, just random restarts of greedy construction.
                
                # To improve success rate, we can try to construct a graph where degrees are as equal as possible.
                
                # Let's try:
                # 1. Shuffle vertices.
                # 2. For each u in shuffled order:
                #    - Try to add min_deg edges from u to random other vertices.
                #    - If can't add enough, retry with different random choices.
                
                # This "vertex-by-vertex" approach might fail early if we block ourselves.
                # But it's fast.
                
                # Let's try multiple strategies.
                
                # Strategy 1: Random edge addition
                # Generate all pairs (u, v) with u < v.
                # Shuffle them.
                # Iterate and add if valid.
                # Check min degree at the end.
                
                pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
                random.shuffle(pairs)
                
                for u, v in pairs:
                    if time.time() > deadline:
                        return False, []
                    
                    # If both already have enough degree, skip?
                    # No, adding more edges doesn't hurt C4-free property if it's valid, 
                    # but it might block future additions?
                    # Actually, adding an edge can only increase the risk of C4 for future edges.
                    # So we should be careful.
                    # If we just add ANY valid edge, we might end up with a graph that is C4-free 
                    # but has low min degree because we "used up" the valid connections.
                    
                    # So, we should prioritize edges that help low-degree vertices.
                    # Let's modify:
                    # Instead of shuffling all pairs once, we can dynamically select.
                    # But that's complex.
                    
                    # Let's stick to shuffling pairs, but maybe bias the shuffle?
                    # Or just try many restarts.
                    
                    if can_add_edge(u, v, neighbors):
                        neighbors[u] |= (1 << v)
                        neighbors[v] |= (1 << u)
                        degrees[u] += 1
                        degrees[v] += 1
                        edges.append((u, v))
                
                if min(degrees) >= min_deg:
                    return True, edges
                    
            return False, []

        # Try to find a graph for this n
        found, edges = try_construct(n, min_deg)
        if found:
            return {'n': n, 'edges': edges}
            
    # If no n > 39 found, return 39
    return {'n': 39, 'edges': []}
