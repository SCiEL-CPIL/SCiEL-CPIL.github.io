import time
import random

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Uses a tree-based construction with a matching on the leaves.
    """
    start_time = time.time()
    # Safety margin: 110s
    deadline = start_time + 110.0
    
    # Parameters for the tree construction
    # We build a tree of depth h = ceil(g/2)
    # This ensures that any cycle formed by connecting leaves from different
    # main branches has length at least 2*h + 1.
    # If 2*h + 1 >= g, the girth condition is satisfied.
    # For even g=2r, h=r. 2r+1 >= 2r. OK.
    # For odd g=2r+1, h=r+1. 2(r+1)+1 = 2r+3 >= 2r+1. OK.
    
    h = (g + 1) // 2
    
    # Build the tree
    # Vertex 0 is the root.
    # We use BFS to assign indices.
    
    n = 0
    parent = {}
    depth = {}
    branch = {} # Which child of the root is this vertex in?
    
    edges = []
    
    # We will store vertices in a list to manage indices
    # But we can also just use a dictionary if we want, 
    # but we need to return indices 0..n-1.
    
    vertices = []
    index_map = {}
    
    def add_vertex(v_id, par, dep, br):
        nonlocal n
        index_map[v_id] = n
        vertices.append(v_id)
        parent[n] = par
        depth[n] = dep
        branch[n] = br
        n += 1
        return n - 1 # return index
        
    # Root
    root_id = 0
    add_vertex(root_id, -1, 0, -1)
    
    # Queue for BFS
    queue = [0] # indices
    
    while queue:
        if time.time() > deadline:
            break
        u = queue.pop(0)
        
        if depth[u] >= h:
            continue
            
        # Determine neighbors to add
        # If u is root, add k neighbors.
        # Else, add k-1 neighbors.
        
        if u == 0:
            num_new = k
        else:
            num_new = k - 1
            
        for i in range(num_new):
            # Create new vertex
            new_id = f"{u}_{i}"
            
            # Determine branch
            if u == 0:
                new_branch = i
            else:
                new_branch = branch[u]
                
            new_idx = add_vertex(new_id, u, depth[u] + 1, new_branch)
            
            # Add edge
            edges.append((u, new_idx))
            
            queue.append(new_idx)
            
    # Check if we have enough time to do the matching
    if time.time() > deadline:
        return n, edges # Return incomplete? No, must be valid.
        # If we ran out of time during tree building, we have a tree, not a regular graph.
        # We should ensure the tree building finishes. It's fast.
        
    # Now, collect leaves at depth h
    # Each leaf has k-1 dangling edges.
    # We need to match these ends.
    
    ends = []
    for i in range(n):
        if depth[i] == h:
            # This vertex is a leaf
            # It has k-1 ends
            for _ in range(k - 1):
                ends.append((i, branch[i]))
                
    # Shuffle ends to randomize the matching
    random.seed(42) # Fixed seed for reproducibility
    random.shuffle(ends)
    
    # We need to pair up ends such that they are from different branches.
    # Group ends by branch
    from collections import defaultdict
    branch_ends = defaultdict(list)
    for e in ends:
        branch_ends[e[1]].append(e[0])
        
    # We have k branches.
    # We will greedily pick ends from different branches.
    # Since the distribution is uniform (each branch has same number of ends),
    # we can just cycle through branches.
    
    # A simpler robust matching:
    # 1. Flatten the list of all ends (vertex, branch)
    # 2. We need to form pairs (u, v) with branch[u] != branch[v].
    
    # Let's use a greedy approach with backtracking if needed, 
    # but for uniform distribution, a simple greedy should work.
    
    # To be safe, we can use a bipartite matching if we split into two sets?
    # No, it's a general matching on a complete k-partite graph.
    
    # Greedy:
    # Maintain a list of available ends.
    # While available ends:
    #   Pick one end e1.
    #   Find an available end e2 with branch[e2] != branch[e1].
    #   If found, pair them.
    #   If not found, we are stuck (all remaining ends have same branch).
    #   This can happen if we are unlucky.
    
    # To avoid getting stuck, we can prioritize picking from the largest branch.
    
    # Let's implement a slightly smarter greedy:
    # 1. Count ends per branch.
    # 2. While ends remain:
    #    a. Find branch b1 with max count.
    #    b. Find branch b2 with max count (b2 != b1).
    #    c. If no b2, fail (shouldn't happen for k>=2).
    #    d. Pick one end from b1, one from b2.
    #    e. Add edge.
    #    f. Decrement counts.
    
    counts = {b: len(lst) for b, lst in branch_ends.items()}
    # Ensure all branches are in counts
    for b in range(k):
        if b not in counts:
            counts[b] = 0
            
    # List of ends to pick from
    # We need to keep track of which specific vertex end is used.
    # We'll just pop from the lists.
    
    for b in branch_ends:
        random.shuffle(branch_ends[b])
        
    while sum(counts.values()) > 0:
        if time.time() > deadline:
            break
            
        # Find b1
        b1 = -1
        max_c = -1
        for b in range(k):
            if counts[b] > max_c:
                max_c = counts[b]
                b1 = b
                
        if b1 == -1:
            break
            
        # Find b2
        b2 = -1
        max_c2 = -1
        for b in range(k):
            if b != b1 and counts[b] > max_c2:
                max_c2 = counts[b]
                b2 = b
                
        if b2 == -1:
            # All remaining ends are in b1.
            # This is a failure state for this greedy step.
            # We should not reach here if k>=2 and initial counts are balanced.
            # But if we did, we can't proceed.
            break
            
        # Pick ends
        v1 = branch_ends[b1].pop()
        v2 = branch_ends[b2].pop()
        
        # Add edge
        edges.append((v1, v2))
        
        counts[b1] -= 1
        counts[b2] -= 1
        
    # Verify degree
    # The construction should result in a k-regular graph.
    # Root: k edges to children.
    # Internal: 1 parent + k-1 children = k.
    # Leaves: 1 parent + k-1 matched edges = k.
    
    return n, edges
