import random
import time
import itertools
from collections import Counter

def construct():
    n = 16
    total_edges = n * (n - 1) // 2  # 120
    edge_list = [(u, v) for u in range(n) for v in range(u + 1, n)]
    edge_index = {e: i for i, e in enumerate(edge_list)}
    
    # Precompute all 8-vertex subsets
    all_subsets = list(itertools.combinations(range(n), 8))
    
    # For each subset, precompute which edge indices are within it
    # subset_edges_mask[subset_idx] = list of edge indices in that subset
    subset_edge_indices = []
    for subset in all_subsets:
        sset = set(subset)
        inds = []
        for e in edge_list:
            if e[0] in sset and e[1] in sset:
                inds.append(edge_index[e])
        subset_edge_indices.append(inds)
    
    # Q3 structure: 8 vertices, 12 edges
    # Vertices 0-7, edges:
    q3_local_edges = [
        (0,1), (0,2), (0,4),
        (1,3), (1,5),
        (2,3), (2,6),
        (4,5), (4,6),
        (5,7), (6,7), (3,7)
    ]
    
    # All automorphisms of Q3 (24 of them)
    # Generate by trying all 8! permutations is too slow.
    # Instead, we can generate them by fixing the image of vertex 0 and 1, etc.
    # Or just hardcode a few key ones or generate on the fly.
    # Actually, for checking if a set of 12 edges on 8 vertices forms Q3,
    # we can just check if the graph is isomorphic to Q3.
    # But we want to check if the induced subgraph CONTAINS Q3.
    # Since we are maximizing edges, we will likely have induced subgraphs with > 12 edges.
    # If an induced subgraph has > 12 edges, it definitely contains Q3?
    # No. A graph with 8 vertices and 13 edges might not contain Q3?
    # Actually, the Turan number for Q3 on 8 vertices...
    # If we have all 28 edges, it contains Q3.
    # The condition "contains Q3" is equivalent to "is not Q3-free".
    # The maximum number of edges in a Q3-free graph on 8 vertices is 11?
    # Let's check. Q3 has 12 edges.
    # If we remove 1 edge from Q3, it's Q3-free. So 11 edges is possible.
    # Can we have 12 edges without Q3?
    # If a graph has 12 edges on 8 vertices, does it necessarily contain Q3?
    # No. For example, K_{4,4} has 16 edges and contains Q3.
    # What about K_{3,5}? 15 edges.
    # It is known that ex(8, Q3) = 11? Or 12?
    # If ex(8, Q3) = 11, then any graph with 12 edges contains Q3.
    # Let's assume for now that if edge_count >= 12, we need to check carefully.
    # If edge_count <= 11, it's safe.
    
    # To be safe and fast, we will precompute all "bad" edge sets.
    # A "bad" set is a set of 12 edges on 8 vertices that forms a Q3.
    # There are many such sets.
    # Number of Q3 subgraphs in K8:
    # Number of automorphisms of Q3 is 48? No, 24?
    # The number of labeled Q3s in K8 is 8! / 48?
    # Let's just generate all subsets of 12 edges from the 28 possible edges on 8 vertices.
    # C(28, 12) is huge.
    
    # Alternative approach:
    # We maintain a count of "violations" for each edge.
    # A violation is a Q3 subgraph that includes that edge.
    # When we remove an edge, we decrement the count for all Q3s containing it.
    # When we add an edge, we increment.
    
    # But we need to know which Q3s exist.
    # This is the hard part.
    
    # Let's go back to the heuristic:
    # If induced subgraph on 8 vertices has >= 12 edges, it's "bad".
    # This is a sufficient condition for containing Q3?
    # If ex(8, Q3) = 11, then yes.
    # If ex(8, Q3) = 12, then a graph with 12 edges might be Q3-free.
    # Let's verify ex(8, Q3).
    # Q3 is bipartite.
    # The maximum edges in a Q3-free graph on 8 vertices.
    # If we take K4,4 and remove a perfect matching, we have 12 edges.
    # Does K4,4 minus perfect matching contain Q3?
    # Q3 is a subgraph of K4,4.
    # Q3 vertices can be partitioned into two sets of 4 (bipartition).
    # In Q3, each vertex has degree 3.
    # In K4,4 minus perfect matching, each vertex has degree 3.
    # Is K4,4 minus perfect matching isomorphic to Q3?
    # Yes! Q3 is exactly K4,4 minus a perfect matching.
    # So a graph with 12 edges CAN be Q3-free (if it's not Q3).
    # Wait, if it IS Q3, it's not Q3-free.
    # So if the induced subgraph is isomorphic to Q3, it's bad.
    # If it has 12 edges but is not isomorphic to Q3, it might be good?
    # But if it has 12 edges and is not Q3, does it contain Q3 as a subgraph?
    # Since it has 12 edges, any subgraph with 12 edges must be the whole graph.
    # So if it's not isomorphic to Q3, it doesn't contain Q3?
    # Wait, a subgraph can have fewer edges.
    # "Contains Q3" means there exists a subgraph isomorphic to Q3.
    # Q3 has 12 edges.
    # So the subgraph must have at least 12 edges.
    # If the induced subgraph has exactly 12 edges, the only candidate is the induced subgraph itself.
    # So if it's not isomorphic to Q3, it's safe.
    # If it has > 12 edges, it might contain Q3.
    
    # So the condition "edge_count >= 12" is too strict?
    # No, it's a safe heuristic (removes more edges than necessary) but might lose optimality.
    # To maximize edges, we should only remove edges if a Q3 is actually present.
    
    # Let's implement a check for "contains Q3" for an 8-vertex induced subgraph.
    # 1. If edge_count < 12, return False.
    # 2. If edge_count == 12, check if isomorphic to Q3.
    # 3. If edge_count > 12, check if contains Q3.
    
    # For speed, we will use the heuristic "edge_count >= 12" in the main loop,
    # but we will try to add edges back if they don't actually create a Q3.
    
    # Actually, let's just use the heuristic for now, as it's fast.
    # The score 0.84 suggests we are far from optimal.
    # The inspirations scored ~1.4-1.5.
    # This suggests the heuristic is too weak or the search is bad.
    
    # Let's try a different strategy:
    # Start with a known dense Q3-free graph?
    # Or start with K16 and remove edges to break all Q3s.
    
    # Let's use the bitmask approach for speed.
    
    # Adjacency matrix as bitmasks
    adj = [0] * n
    for e in edge_list:
        u, v = e
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    def is_q3_free(adj):
        # Check all 8-subsets
        for subset in all_subsets:
            # Check if induced subgraph contains Q3
            # Fast check: count edges
            # We can precompute the edge count for each subset quickly?
            # Or just iterate.
            edge_count = 0
            for i in range(8):
                u = subset[i]
                mask = adj[u]
                # Count bits in mask that correspond to subset vertices
                # subset_mask = 0
                # for v in subset: subset_mask |= (1 << v)
                # edge_count += popcount(mask & subset_mask)
                # This is O(8) per subset, 12870 * 8 = ~100k ops. Fast.
                pass
            
            # Let's implement the popcount check properly
            # We need a subset_mask for each subset
            pass
        
        return True
    
    # Precompute subset masks
    subset_masks = []
    for subset in all_subsets:
        m = 0
        for v in subset:
            m |= (1 << v)
        subset_masks.append(m)
    
    def count_edges_in_subset(adj, subset_mask):
        # Sum of degrees in subset / 2
        # But we need to be careful.
        # Let's just iterate vertices in subset.
        # We need to know which vertices are in the subset.
        # We can extract them from the mask.
        count = 0
        m = subset_mask
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            count += (adj[v] & subset_mask).bit_count()
            m ^= lsb
        return count // 2
    
    def contains_q3(adj, subset, subset_mask):
        ec = count_edges_in_subset(adj, subset_mask)
        if ec < 12:
            return False
        if ec > 12:
            # Likely contains Q3
            # For now, assume True to be safe
            return True
        # ec == 12
        # Check if isomorphic to Q3
        # Q3 is 3-regular.
        # Check if all degrees in induced subgraph are 3.
        # If not, return False.
        # If yes, it's a 3-regular graph on 8 vertices.
        # There are only a few 3-regular graphs on 8 vertices.
        # Q3 is one of them.
        # K4,4 minus perfect matching is Q3.
        # Are there others?
        # The cube graph is the only 3-regular bipartite graph on 8 vertices?
        # Actually, there are others.
        # But if it's 3-regular and bipartite, is it Q3?
        # Q3 is the unique 3-regular bipartite graph on 8 vertices?
        # No, there are two: Q3 and the "circular ladder" CL4?
        # CL4 is also 3-regular on 8 vertices.
        # Does CL4 contain Q3?
        # CL4 is not isomorphic to Q3.
        # So if the graph is CL4, it is Q3-free.
        # So we need to distinguish.
        
        # For now, if ec == 12, we assume it contains Q3 (safe but suboptimal).
        return True

    # Simulated Annealing
    start_time = time.time()
    deadline = start_time + 110
    
    # Initialize with K16
    # This will have many violations.
    # We need to remove edges.
    
    # Let's start with a random graph or K16 and remove edges greedily.
    # Or start with a sparse graph and add edges.
    # Adding edges is better for maximization.
    
    # Start with empty graph? No, too slow to add 120 edges.
    # Start with K16 and remove edges.
    
    # Let's use the heuristic: remove edge if it's part of a "bad" subset.
    
    # We'll maintain a list of "active" bad subsets.
    # Initially, all subsets with >= 12 edges are bad.
    
    # Actually, let's just do a simple loop:
    # While there is a bad subset, remove a random edge from it.
    
    # To speed up, we can keep a list of bad subsets.
    
    bad_subsets = []
    for i, subset in enumerate(all_subsets):
        if contains_q3(adj, subset, subset_masks[i]):
            bad_subsets.append(i)
    
    random.shuffle(bad_subsets)
    
    while bad_subsets and time.time() < deadline:
        # Pick a random bad subset
        idx = random.choice(bad_subsets)
        subset = all_subsets[idx]
        subset_mask = subset_masks[idx]
        
        # Find edges in this subset
        # Remove one randomly
        # We need to find an edge (u,v) in subset such that adj[u] has v.
        # Pick random u in subset, then random v in adj[u] & subset_mask.
        
        u = random.choice(subset)
        neighbors = adj[u] & subset_mask
        if neighbors:
            lsb = neighbors & (-neighbors)
            v = lsb.bit_length() - 1
            # Remove edge (u, v)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            
            # Update bad_subsets?
            # This is expensive.
            # Let's just re-scan or keep a dirty flag.
            # For now, remove from list and re-scan periodically?
            # Or just keep the list and check if still bad when picked.
            
            bad_subsets.remove(idx)
            
            # Check if other subsets became good?
            # No, removing edges only makes subsets "more good" (fewer edges).
            # So no new bad subsets are created.
            # But some bad subsets might become good.
            # We don't know which ones.
            # So we just keep the list and filter.
            
            # To avoid list growing too large with stale entries,
            # we can periodically clean it.
            pass
    
    # Now, try to add edges back?
    # This is the "repair" phase.
    # Try to add each missing edge if it doesn't create a Q3.
    
    # This is complex.
    # Let's just return the current graph.
    
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                edges.append([u, v])
    
    return edges

# Wait, the above logic has a flaw.
# "bad_subsets.remove(idx)" is O(N).
# And we don't update the list when subsets become good.
# So the list will contain many stale entries.
# When we pick a stale entry, contains_q3 will return False (or we check).
# We should check if it's still bad before removing an edge.

def construct():
    n = 16
    edge_list = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    all_subsets = list(itertools.combinations(range(n), 8))
    subset_masks = []
    for subset in all_subsets:
        m = 0
        for v in subset:
            m |= (1 << v)
        subset_masks.append(m)
    
    def count_edges_in_subset(adj, subset_mask):
        count = 0
        m = subset_mask
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            count += (adj[v] & subset_mask).bit_count()
            m ^= lsb
        return count // 2
    
    def contains_q3(adj, subset_mask):
        ec = count_edges_in_subset(adj, subset_mask)
        if ec < 12:
            return False
        # Heuristic: if ec >= 12, assume Q3 present
        # This is safe but might be suboptimal
        return True
    
    # Initialize with K16
    adj = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    
    # Find all bad subsets
    bad_indices = [i for i in range(len(all_subsets)) if contains_q3(adj, subset_masks[i])]
    
    start_time = time.time()
    deadline = start_time + 110
    
    while bad_indices and time.time() < deadline:
        # Pick random index
        idx = random.randrange(len(bad_indices))
        subset_mask = subset_masks[idx]
        
        # Check if still bad
        if not contains_q3(adj, subset_mask):
            # Remove from list
            bad_indices[idx] = bad_indices[-1]
            bad_indices.pop()
            continue
        
        # Find an edge to remove
        # Pick random vertex in subset
        m = subset_mask
        lsb = m & (-m)
        u = lsb.bit_length() - 1
        
        neighbors = adj[u] & subset_mask
        if neighbors:
            lsb2 = neighbors & (-neighbors)
            v = lsb2.bit_length() - 1
            
            # Remove edge (u, v)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            
            # Don't update bad_indices immediately, just let stale entries be filtered
            # But we should remove this one? No, it might still be bad if it has other Q3s?
            # No, removing an edge reduces edge count.
            # If it was bad because ec >= 12, it might now be ec = 11 (good) or still >= 12 (bad).
            # So we leave it in the list.
            
            # To prevent infinite loop on a subset that is "bad" but we can't fix?
            # We always remove an edge, so eventually it will be good.
            
    # Construct output
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                edges.append([u, v])
    
    return edges
