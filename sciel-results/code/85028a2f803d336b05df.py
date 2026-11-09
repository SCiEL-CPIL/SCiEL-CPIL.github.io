import numpy as np
from scipy.optimize import minimize

def squares_in_square10() -> tuple:
        n = 10
        best_val = 0
        best_centers = None
        best_sides = None
        
        # Generate several random guillotine topologies
        np.random.seed(42)
        
        for trial in range(100):
            # Generate a random binary tree with n leaves
            # We can represent the tree by a list of split decisions
            # Let's use a recursive approach to build the tree structure
            # Node: (type, left, right)
            # Leaves are indices 0..n-1
            
            tree_nodes = []
            next_id = 0
            
            def build_tree(k):
                nonlocal next_id
                if k == 1:
                    return next_id
                next_id += 1
                # Randomly decide split direction: 0 for H, 1 for V
                split_dir = np.random.randint(0, 2)
                # Randomly decide how to split k items
                # i in [1, k-1]
                i = np.random.randint(1, k)
                left = build_tree(i)
                right = build_tree(k - i)
                tree_nodes.append((split_dir, left, right, next_id))
                return next_id
            
            build_tree(n)
            
            # tree_nodes is a list of (dir, left, right, id)
            # We need to map ids to nodes
            # Actually, let's store the tree in a dictionary for easy access
            # Or just use the list index as the id if we append in order?
            # The recursive build appends in post-order? No, it appends after children.
            # So the list index is the node id.
            
            # Parameters to optimize: split ratios for each internal node
            # There are n-1 = 9 internal nodes.
            # The order in tree_nodes corresponds to the creation order.
            # We need to ensure we optimize the correct parameter for each node.
            
            # Let's restructure:
            # We will create a list of internal nodes.
            # Each internal node has a parameter 'split' in [0.05, 0.95].
            
            internal_nodes = []
            
            def build_tree_v2(k):
                if k == 1:
                    return -1 # Leaf marker, but we need to know which leaf
                    # Actually, we need to map leaves to specific squares?
                    # The problem doesn't care which square is which, just the set.
                    # So leaves are just "a square".
                    # But we need to return the center and side.
                    # So we need to track the path to compute the final rectangle.
                    
                    # Let's change strategy:
                    # The tree defines the *structure*.
                    # The leaves are just "square slots".
                    # We don't need to label them 1..10 specifically, just produce 10 squares.
                    
                    pass

            # Let's use a simpler representation:
            # A list of 9 splits.
            # We need to define the topology.
            # Topology: A binary tree where internal nodes are splits.
            
            # Let's generate the topology as a list of tuples (left_size, right_size, direction)
            # But the sizes are not fixed, they are determined by the split ratio?
            # No, the split ratio determines the *geometric* split.
            # The number of squares in each branch is fixed by the topology.
            
            # Let's generate a random topology:
            # A list of 9 operations.
            # We can represent the tree by a stack or recursive build.
            
            # Let's try a different, more direct approach:
            # We know the optimal packing for 10 squares is likely a 3x3 grid with one cell split, or 2x5, etc.
            # But the directive says "Search over binary tree topologies".
            
            # Let's implement the optimizer for a specific topology.
            # Topology is defined by the sequence of splits.
            
            # We will generate 20 random topologies.
            # For each, we optimize the 9 split ratios.
            
            def generate_topology(n):
                # Returns a list of (direction, left_count, right_count) for each internal node
                # The order is post-order (children before parent) or pre-order?
                # Let's use a recursive generator that returns a tree object.
                
                class Node:
                    def __init__(self, is_leaf, left=None, right=None, dir=None):
                        self.is_leaf = is_leaf
                        self.left = left
                        self.right = right
                        self.dir = dir # 0: H, 1: V
                        self.param_idx = -1 # Index in the split array

                def build(k):
                    if k == 1:
                        return Node(True)
                    left = build(np.random.randint(1, k))
                    right = build(k - np.random.randint(1, k)) # Wait, left+right must be k
                    # Correction:
                    i = np.random.randint(1, k)
                    left = build(i)
                    right = build(k - i)
                    dir = np.random.randint(0, 2)
                    return Node(False, left, right, dir)

                root = build(n)
                
                # Assign parameter indices
                params = []
                def assign(node):
                    if not node.is_leaf:
                        node.param_idx = len(params)
                        params.append(0.5) # Initial guess
                        assign(node.left)
                        assign(node.right)
                assign(root)
                return root, params

            # Optimization function for a given tree
            def optimize_tree(root, n_params):
                # x: array of n_params split ratios
                def objective(x):
                    # Compute sum of sides
                    # Recursively compute w, h for each node
                    # We need to return the sum of min(w,h) for leaves
                    # And also we need to be able to extract centers later
                    # For now, just sum.
                    
                    total_sum = 0
                    # We need a helper that returns (w, h, sum_sides, centers, sides)
                    # But for optimization, we just need the sum to be fast.
                    # We can compute w, h recursively.
                    
                    def compute(node, w, h):
                        nonlocal total_sum
                        if node.is_leaf:
                            side = min(w, h)
                            total_sum += side
                            return
                        split = x[node.param_idx]
                        if node.dir == 0: # Horizontal split (top/bottom)
                            h_top = h * split
                            h_bot = h * (1 - split)
                            compute(node.left, w, h_top)
                            compute(node.right, w, h_bot)
                        else: # Vertical split (left/right)
                            w_left = w * split
                            w_right = w * (1 - split)
                            compute(node.left, w_left, h)
                            compute(node.right, w_right, h)
                    
                    total_sum = 0
                    compute(root, 1.0, 1.0)
                    return -total_sum

                x0 = np.ones(n_params) * 0.5
                bounds = [(0.05, 0.95)] * n_params
                res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000})
                return res.x

            # Main loop
            for trial in range(50):
                root, params = generate_topology(n)
                n_params = len(params)
                if n_params == 0: continue
                x_opt = optimize_tree(root, n_params)
                
                # Calculate final centers and sides
                centers = np.zeros((n, 2))
                sides = np.zeros(n)
                leaf_idx = 0
                
                def extract(node, w, h, cx, cy):
                    nonlocal leaf_idx
                    if node.is_leaf:
                        side = min(w, h)
                        # Center of the square within the rectangle
                        # The square is centered in the rectangle?
                        # The problem says "Square i is axis-aligned...".
                        # In a guillotine packing, the square usually fills the min dimension.
                        # If w > h, square is h x h, centered vertically?
                        # The problem allows any position inside.
                        # To maximize the "fit" or just valid, we can center it.
                        # Centering is always valid if it fits.
                        # Does centering maximize the sum? No, sum is determined by size.
                        # Size is determined by min(w,h).
                        # So we just place it centered in the rectangle.
                        centers[leaf_idx] = [cx, cy]
                        sides[leaf_idx] = side
                        leaf_idx += 1
                        return
                    
                    split = x_opt[node.param_idx]
                    if node.dir == 0: # H
                        h_top = h * split
                        h_bot = h * (1 - split)
                        # Top child
                        cy_top = cy - h/2 + h_top/2
                        extract(node.left, w, h_top, cx, cy_top)
                        # Bottom child
                        cy_bot = cy - h/2 + h_top + h_bot/2
                        extract(node.right, w, h_bot, cx, cy_bot)
                    else: # V
                        w_left = w * split
                        w_right = w * (1 - split)
                        # Left child
                        cx_left = cx - w/2 + w_left/2
                        extract(node.left, w_left, h, cx_left, cy)
                        # Right child
                        cx_right = cx - w/2 + w_left + w_right/2
                        extract(node.right, w_right, h, cx_right, cy)

                extract(root, 1.0, 1.0, 0.5, 0.5)
                
                # Validate
                if leaf_idx != n: continue
                
                # Check validity
                tol = 1e-7
                valid = True
                for i in range(n):
                    cx, cy = centers[i]
                    s = sides[i]
                    if cx - s/2 < -tol or cx + s/2 > 1 + tol or cy - s/2 < -tol or cy + s/2 > 1 + tol:
                        valid = False
                        break
                    if s <= 0:
                        valid = False
                        break
                
                if valid:
                    for i in range(n):
                        for j in range(i+1, n):
                            xi, yi = centers[i]
                            xj, yj = centers[j]
                            si = sides[i]
                            sj = sides[j]
                            t = (si + sj) / 2.0 - tol
                            if abs(xi - xj) < t and abs(yi - yj) < t:
                                valid = False
                                break
                        if not valid: break
                
                if valid:
                    val = np.sum(sides)
                    if val > best_val:
                        best_val = val
                        best_centers = centers.copy()
                        best_sides = sides.copy()

            if best_centers is None:
                # Fallback
                centers = np.zeros((n, 2))
                sides = np.zeros(n)
                for i in range(n):
                    col = i % 4
                    row = i // 4
                    centers[i] = [(col + 0.5) / 4.0, (row + 0.5) / 3.0]
                    sides[i] = 1.0 / 4.0
                return centers, sides

            return best_centers, best_sides
