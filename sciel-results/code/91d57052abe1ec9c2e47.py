import numpy as np
import random
import time

def construct(k, g):
    # Target: 4-regular, girth 10
    # k=4, g=10
    if k != 4 or g != 10:
        # Fallback to original LUW for other parameters
        return _luw_construct(k, g)

    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer
    
    # Construct the tree of depth 5
    # Root at 0
    # Level 0: 1 node
    # Level 1: 4 nodes
    # Level 2: 12 nodes
    # Level 3: 36 nodes
    # Level 4: 108 nodes
    # Level 5: 324 nodes
    # Total 485 nodes
    
    # We will build the graph on these 485 nodes.
    # The tree edges are fixed.
    # We need to add 3 edges per leaf (Level 5 nodes).
    
    # Identify leaves and their "branch" (first child of root)
    leaves = []
    leaf_branch = []
    
    # Build tree
    # Node 0 is root
    # Children of 0: 1, 2, 3, 4
    # We can generate the tree structure explicitly
    
    N = 485
    edges = []
    
    # Tree construction
    # 0
    # 1-4
    # 5-16
    # 17-52
    # 53-160
    # 161-484
    
    # Let's just create the edges of the tree
    # Level 0: [0]
    # Level 1: [1, 2, 3, 4]
    # Level 2: [5..16]
    # Level 3: [17..52]
    # Level 4: [53..160]
    # Level 5: [161..484]
    
    tree_edges = []
    # Level 0 to 1
    for i in range(1, 5):
        tree_edges.append((0, i))
        
    # Level 1 to 2
    node_id = 5
    for parent in range(1, 5):
        for _ in range(3):
            tree_edges.append((parent, node_id))
            node_id += 1
            
    # Level 2 to 3
    for parent in range(5, 17):
        for _ in range(3):
            tree_edges.append((parent, node_id))
            node_id += 1
            
    # Level 3 to 4
    for parent in range(17, 53):
        for _ in range(3):
            tree_edges.append((parent, node_id))
            node_id += 1
            
    # Level 4 to 5
    for parent in range(53, 161):
        for _ in range(3):
            tree_edges.append((parent, node_id))
            node_id += 1
            
    # Leaves are 161..484
    # Branch of a leaf is determined by the first child of root
    # We can trace back
    # Or simply:
    # Branch 0: leaves under 1
    # Branch 1: leaves under 2
    # Branch 2: leaves under 3
    # Branch 3: leaves under 4
    
    # Calculate number of leaves per branch
    # Each branch has 3^4 = 81 leaves
    
    # Assign branches
    branch_of_leaf = {}
    # We can determine branch by the path
    # But we can just assign ranges if we generated them in order
    # The generation above added children in order.
    # So leaves under 1 are the first 81 leaves?
    # Let's check.
    # Parent 1 has children 5,6,7.
    # Parent 2 has children 8,9,10.
    # ...
    # So yes, the leaves are grouped by branch.
    
    # Leaf indices: 161 to 484
    # Branch 0: 161..241 (81 leaves)
    # Branch 1: 242..322
    # Branch 2: 323..403
    # Branch 3: 404..484
    
    # We need to add a 3-regular graph on these 324 leaves,
    # with no edges within the same branch.
    
    leaf_nodes = list(range(161, 485))
    branches = [list(range(161, 242)), list(range(242, 323)), list(range(323, 404)), list(range(404, 485))]
    
    # Initialize leaf edges
    leaf_edges = _generate_valid_leaf_graph(branches, 3)
    
    if leaf_edges is None:
        return _luw_construct(k, g)
        
    all_edges = tree_edges + leaf_edges
    
    # Verify girth
    # We can do a quick check
    # If valid, return
    # If not, try to repair? 
    # Given the constraints, it's likely valid if generated correctly.
    # But we must ensure no short cycles.
    
    # To be safe, we run a girth check
    # And if it fails, we fall back to LUW
    
    if _has_girth_ge_edges(N, all_edges, g):
        return N, all_edges
    else:
        return _luw_construct(k, g)

def _generate_valid_leaf_graph(branches, degree):
    # Generate a 'degree'-regular graph on the union of branches
    # Such that no edge is within the same branch
    
    all_leaves = []
    for b in branches:
        all_leaves.extend(b)
        
    n_leaves = len(all_leaves)
    total_stubs = n_leaves * degree
    
    # Create stubs
    # Each leaf has 'degree' stubs
    # We need to pair stubs from different branches
    
    # We can do this by creating a list of (leaf, branch_id)
    # And shuffling, then pairing, ensuring branch_id differs
    
    # This is a bipartite-like matching problem generalized to 4 parts.
    # We can use a greedy random approach with backtracking or just random retries.
    
    for _ in range(1000): # Retry loop
        edges = []
        used = set()
        
        # Create list of stubs
        stubs = []
        for i, leaf in enumerate(all_leaves):
            # Determine branch
            b_id = 0
            for j, b in enumerate(branches):
                if leaf in b:
                    b_id = j
                    break
            for _ in range(degree):
                stubs.append((leaf, b_id))
                
        random.shuffle(stubs)
        
        valid = True
        for i in range(0, len(stubs), 2):
            l1, b1 = stubs[i]
            l2, b2 = stubs[i+1]
            
            if b1 == b2:
                valid = False
                break
            if l1 == l2:
                valid = False
                break
            if (l1, l2) in used or (l2, l1) in used:
                valid = False
                break
                
            used.add((l1, l2))
            edges.append((l1, l2))
            
        if valid:
            return edges
            
    return None

def _has_girth_ge_edges(N, edges, g):
    # Build adjacency list
    adj = [[] for _ in range(N)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Check for cycles of length < g
    # Use BFS from each vertex? Too slow for N=485.
    # Use matrix multiplication?
    # A = np.zeros((N, N))
    # for u, v in edges:
    #     A[u, v] = 1
    #     A[v, u] = 1
    # A_k = A.copy()
    # for k in range(3, g):
    #     A_k = A_k @ A
    #     if np.any(np.diag(A_k) > 0):
    #         return False
    # return True
        
    # Since N is small (485), matrix mult is fast.
    A = np.zeros((N, N), dtype=np.int8)
    for u, v in edges:
        A[u, v] = 1
        A[v, u] = 1
        
    A_k = A.copy()
    for k in range(3, g):
        A_k = A_k @ A
        if np.any(np.diag(A_k) > 0):
            return False
    return True

# Fallback LUW construction (copied from original)
def _luw_construct(k, g):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
