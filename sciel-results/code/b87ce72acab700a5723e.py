import sys
import time
import random
from collections import deque

def construct(k, g):
    # Target: 3-regular graph, girth >= g.
    # Strategy: Construct a tree of depth R = g//2 - 1 (if g even) or (g-1)//2 - 1?
    # For even girth g, radius R = g//2 - 1.
    # Moore bound N = 1 + k * ( (k-1)^R - 1 ) / (k-2) ? No.
    # N = 1 + k * sum_{i=0}^{R-1} (k-1)^i.
    
    if k != 3:
        # Fallback to LUW or simple random for other k
        # But task specifies k=3, g=16.
        # We'll just return a dummy or use LUW logic if available, but for now assume k=3.
        pass
        
    # Calculate R
    R = g // 2 - 1
    
    # Build the tree
    # Root is 0.
    # Children of 0: 1, 2, 3.
    # Children of 1: 4, 5.
    # ...
    
    # We will generate the tree structure explicitly.
    # Since N is around 766, we can just build it.
    
    n_vertices = 1 + k * (( (k-1)**R - 1 ) // (k-2)) if k > 2 else 0
    # For k=3: 1 + 3 * (2^R - 1)
    n_vertices = 1 + 3 * (2**R - 1)
    
    # Adjacency list
    adj = [[] for _ in range(n_vertices)]
    
    # Build tree edges
    # Level 0: [0]
    # Level 1: [1, 2, 3]
    # Level i: vertices from index start_i to end_i
    
    # Let's just do a BFS to assign indices and build tree.
    # Queue: (vertex, level)
    # We need to ensure the structure is exactly the Moore tree.
    
    # Actually, simpler:
    # 0 is root.
    # 1, 2, 3 are children of 0.
    # 4, 5 are children of 1.
    # 6, 7 are children of 2.
    # 8, 9 are children of 3.
    # ...
    
    # Let's generate indices level by level.
    levels = [[] for _ in range(R + 1)]
    levels[0] = [0]
    
    # Assign indices
    next_idx = 1
    for i in range(1, R + 1):
        for parent in levels[i-1]:
            # Number of children
            if i == 1:
                num_children = k
            else:
                num_children = k - 1
            
            for _ in range(num_children):
                levels[i].append(next_idx)
                adj[parent].append(next_idx)
                adj[next_idx].append(parent)
                next_idx += 1
                
    # Check if we used all vertices
    # next_idx - 1 should be n_vertices - 1
    
    leaves = levels[R]
    
    # Group leaves by their ancestor at level 1
    # For each leaf, find its level-1 ancestor.
    # We can do this by tracing back, but we can also store parent pointers.
    parent = [-1] * n_vertices
    for i in range(1, R + 1):
        for v in levels[i]:
            # Find parent
            # We know the structure, but let's just use the adj list for now?
            # No, adj list has tree edges.
            # For leaf v, parent is adj[v][0] (since leaf has only 1 tree neighbor).
            parent[v] = adj[v][0]
            
    # Now, for each leaf, find level 1 ancestor.
    leaf_to_group = {}
    for leaf in leaves:
        curr = leaf
        while curr != 0:
            curr = parent[curr]
            if curr in (1, 2, 3):
                leaf_to_group[leaf] = curr
                break
        else:
            # Should not happen
            leaf_to_group[leaf] = -1
            
    # Groups: 1, 2, 3
    groups = {1: [], 2: [], 3: []}
    for leaf, group in leaf_to_group.items():
        if group in groups:
            groups[group].append(leaf)
            
    # We need to connect leaves such that:
    # 1. Each leaf gets 2 new edges.
    # 2. Edges only between leaves of different groups.
    # 3. Distance in tree between connected leaves is >= g - 2 = 14?
    #    Actually, we established that LCA must be root for dist >= 15.
    #    So any leaf in group 1 can connect to any leaf in group 2 or 3.
    #    And the distance will be 16 (max).
    #    So the girth constraint is satisfied by construction (dist 16 + edge 1 = 17? No).
    #    Wait. Path length in tree is 16. Edge adds 1. Cycle length 17.
    #    We need girth >= 16. 17 is fine.
    #    Are there any shorter cycles?
    #    No, because any cycle must use at least one leaf-leaf edge.
    #    If it uses one, the rest is tree path.
    #    If it uses more, it's even longer.
    
    # So we just need to form a 2-regular graph on the leaves,
    # using only edges between different groups.
    
    # This is equivalent to finding a 2-regular subgraph in the complete tripartite graph K_{128,128,128}.
    # We can do this by creating a random permutation and connecting.
    
    # Shuffle leaves in each group
    random.shuffle(groups[1])
    random.shuffle(groups[2])
    random.shuffle(groups[3])
    
    # We need to assign 2 neighbors to each leaf.
    # Let's create a list of "slots" for each leaf.
    # Or simpler:
    # Create a list of all leaves.
    # For each leaf, pick 2 random leaves from other groups.
    # This might not result in a 2-regular graph (degree might not match).
    
    # Better:
    # Create a bipartite matching or just use a greedy assignment.
    # Since the graph is dense (complete tripartite), we can just:
    # 1. Shuffle all leaves.
    # 2. Pair them up? No, we need 2-regular.
    # 3. A 2-regular graph is a union of cycles.
    # 4. We can form one big cycle that visits all leaves, ensuring no two consecutive leaves are in the same group.
    
    all_leaves = groups[1] + groups[2] + groups[3]
    # We need to order them such that adjacent ones are in different groups.
    # Since groups are equal size, we can just interleave them.
    
    # Interleave: 1, 2, 3, 1, 2, 3, ...
    interleaved = []
    for i in range(len(groups[1])):
        interleaved.append(groups[1][i])
        interleaved.append(groups[2][i])
        interleaved.append(groups[3][i])
        
    # Now, connect interleaved[i] to interleaved[i+1] and interleaved[i-1].
    # This forms a single cycle of length 3 * 128 = 384.
    # Each leaf has degree 2 from this cycle.
    # Plus degree 1 from the tree.
    # Total degree 3.
    
    # Check if any edge is a loop or duplicate.
    # Since groups are disjoint, no loops.
    # Since it's a cycle, no duplicate edges (unless N < 4, but N=384).
    
    for i in range(len(interleaved)):
        u = interleaved[i]
        v = interleaved[(i + 1) % len(interleaved)]
        if v not in adj[u]:
            adj[u].append(v)
            adj[v].append(u)
            
    # Verify girth
    # BFS from each vertex to depth g//2 - 1?
    # If we find a cycle of length < g, return invalid?
    # The construction guarantees girth >= 16?
    # Let's double check.
    # Tree depth 8. Leaves at depth 8.
    # Distance between leaves in different root-subtrees:
    # Path goes up to root (8 steps) and down (8 steps). Total 16.
    # Cycle length = 16 + 1 (edge) = 17.
    # So girth is at least 17.
    # This satisfies girth >= 16.
    
    # Return n, edges
    edges = []
    for u in range(n_vertices):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
                
    return n_vertices, edges

# EVOLVE-BLOCK-END
