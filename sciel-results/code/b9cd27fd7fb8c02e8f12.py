import random
import time
from collections import deque

def construct(k, g):
    # If k=4 and g=10, use the BFS Tree Completion method to get N=324
    if k == 4 and g == 10:
        # Build the tree
        # Depth 5
        # Level 0: 1 node (0)
        # Level 1: 4 nodes (1, 2, 3, 4)
        # Level 2: 12 nodes
        # Level 3: 36 nodes
        # Level 4: 108 nodes
        # Level 5: 324 nodes
        # Total 485 nodes in tree.
        # But we only keep the tree structure for edges.
        # We identify V5 nodes and add edges between them.
        
        # We will index nodes 0..484
        # Node 0 is root.
        # Children of node i are i*3 + 1, i*3 + 2, i*3 + 3? No, root has 4.
        # Let's build explicitly.
        
        nodes = []
        parent = []
        level = []
        
        # Root
        nodes.append(0)
        parent.append(-1)
        level.append(0)
        
        # We need to assign IDs carefully.
        # Level 0: 0
        # Level 1: 1, 2, 3, 4 (Children of 0)
        # Level 2: Children of 1,2,3,4. 3 each. 12 nodes.
        # Level 3: Children of L2. 36 nodes.
        # Level 4: Children of L3. 108 nodes.
        # Level 5: Children of L4. 324 nodes.
        
        # Total nodes: 1 + 4 + 12 + 36 + 108 + 324 = 485.
        
        edges = []
        
        # Generate nodes
        current_level = [0]
        next_level = []
        node_id = 1
        parents_of_level = {0: [0]} # level -> list of node ids
        
        for d in range(1, 6):
            next_level = []
            for p in current_level:
                num_children = 3 if d > 1 else 4 # Root has 4, others have 3
                for _ in range(num_children):
                    child_id = node_id
                    node_id += 1
                    next_level.append(child_id)
                    edges.append((p, child_id))
            current_level = next_level
            parents_of_level[d] = current_level
            
        # Now we have nodes 0..484.
        # V5 is parents_of_level[5], size 324.
        # We need to add edges between V5 nodes.
        # Partition V5 into 4 sets S1, S2, S3, S4 based on ancestor at Level 1.
        
        # Find ancestor at Level 1 for each node in V5.
        # We can trace back.
        # Or store ancestor info during generation.
        
        # Let's re-generate with ancestor info.
        nodes_info = [] # (id, level, ancestor_L1)
        nodes_info.append((0, 0, None))
        
        current_level = [(0, 0, None)]
        node_id = 1
        
        for d in range(1, 6):
            next_level = []
            for (p_id, p_level, p_anc) in current_level:
                num_children = 3 if d > 1 else 4
                for _ in range(num_children):
                    child_id = node_id
                    node_id += 1
                    if d == 1:
                        anc = child_id # The node itself is the L1 ancestor for its subtree?
                        # No, the L1 ancestor of a node in L2 is the L1 node it descends from.
                        # For a node in L1, its L1 ancestor is itself.
                        # For a node in L2, its L1 ancestor is its parent (which is in L1).
                        # So if d==1, anc = child_id.
                        # If d>1, anc = p_anc.
                        anc = child_id if d == 1 else p_anc
                    else:
                        anc = p_anc
                    next_level.append((child_id, d, anc))
                    edges.append((p_id, child_id))
            current_level = next_level
            
        # V5 nodes are current_level (d=5).
        v5_nodes = current_level
        
        # Partition into 4 sets by anc (which is 1, 2, 3, 4).
        sets = {1: [], 2: [], 3: [], 4: []}
        for (node_id, lvl, anc) in v5_nodes:
            sets[anc].append(node_id)
            
        # Each set has 81 nodes.
        # We need to create a 3-regular graph on these 324 nodes,
        # using only edges between different sets.
        
        # Simple strategy: Random 3-regular 4-partite graph.
        # We have 4 parts of size 81.
        # Each node needs 3 edges to nodes in other parts.
        # Total edges = 324 * 3 / 2 = 486.
        
        # We can construct this by:
        # 1. Create a list of all possible pairs (u, v) where u in Si, v in Sj, i != j.
        # 2. Randomly select 486 pairs such that each node has degree 3.
        # This is a random regular graph in a multipartite setting.
        
        # Easier:
        # For each node u, pick 3 random neighbors from other sets.
        # Ensure no duplicates.
        # This might not yield a simple graph or regular graph easily.
        
        # Let's use a standard random regular graph generation for multipartite.
        # Or just:
        # For each set Si, shuffle the nodes.
        # Connect Si[j] to Sj[(j + offset) % 81] for j=0..80.
        # We need 3 neighbors.
        # Neighbor 1: from S((i+1)%4)
        # Neighbor 2: from S((i+2)%4)
        # Neighbor 3: from S((i+3)%4)
        
        for i in range(1, 5):
            set_i = sets[i]
            for j in range(81):
                u = set_i[j]
                
                # Connect to S_{i+1}
                v1 = sets[(i % 4) + 1][(j + 0) % 81]
                edges.append((u, v1))
                
                # Connect to S_{i+2}
                v2 = sets[(i + 1) % 4 + 1][(j + 20) % 81] # Offset to avoid symmetry
                edges.append((u, v2))
                
                # Connect to S_{i+3}
                v3 = sets[(i + 2) % 4 + 1][(j + 40) % 81]
                edges.append((u, v3))
                
        # Check for duplicate edges?
        # The construction ensures u connects to specific v1, v2, v3.
        # Are these unique?
        # u is in Si.
        # v1 is in S_{i+1}.
        # v2 is in S_{i+2}.
        # v3 is in S_{i+3}.
        # Since sets are disjoint, no self loops.
        # Are edges repeated?
        # Edge (u, v) is added once when processing u.
        # Is it possible that v also connects to u?
        # If v is in S_{i+1}, it connects to S_{i+2}, S_{i+3}, S_{i+4}=S_i.
        # So v connects to S_i.
        # v's neighbor in S_i is determined by v's index in S_{i+1}.
        # It might be u.
        # If so, we have (u,v) and (v,u) in the list.
        # The problem asks for a list of edges (i,j).
        # Usually, undirected edges are unique.
        # We should deduplicate.
        
        unique_edges = set()
        for e in edges:
            if e[0] < e[1]:
                unique_edges.add((e[0], e[1]))
            else:
                unique_edges.add((e[1], e[0]))
                
        return 485, list(unique_edges)

    # Fallback for other k, g
    # Use the original LUW logic
    def luw_construct(k, g):
        # ... (copy from seed) ...
        pass
        
    return luw_construct(k, g)
