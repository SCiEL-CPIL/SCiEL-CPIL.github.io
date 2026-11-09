# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import itertools


def squares_in_square10() -> tuple:
    """
    Erdos "squares in a square" problem. Place n squares with pairwise disjoint
    interiors inside the unit square [0,1]^2 so as to MAXIMIZE the sum of their
    side lengths. Here n = 10.

    Returns:
        centers: np.ndarray of shape (10, 2) with the (x, y) center of each square.
        sides:   np.ndarray of shape (10,) with the side length of each square.
    The i-th axis-aligned square occupies
        [cx - s/2, cx + s/2] x [cy - s/2, cy + s/2].
    """
    n = 10
    
    # We will use a recursive guillotine cut approach.
    # A guillotine cut tree is a binary tree where each internal node represents
    # a rectangular region split into two sub-regions by a vertical or horizontal line.
    # Leaves correspond to the squares.
    
    # Since n=10 is not a power of 2, the tree will be unbalanced.
    # We will define a few specific tree structures (topologies) and optimize
    # the split ratios for each.
    
    # Representation of the tree:
    # We'll use a list of nodes. Each node is a tuple:
    # (type, ...)
    # type 0: Leaf, containing square index
    # type 1: Internal, containing split direction (0=vert, 1=horiz), split ratio (0-1), left_child, right_child
    
    # However, for optimization, it's easier to parameterize the splits.
    # Let's define a function that takes a tree structure description and split parameters,
    # and returns the centers and sides.
    
    # Tree Structure 1: "Balanced-ish"
    # Root splits vertically.
    # Left child (5 squares) splits horizontally.
    #   Left-Left (2 squares) splits vertically.
    #   Left-Right (3 squares) splits horizontally.
    #     Left-Right-Left (1 square)
    #     Left-Right-Right (2 squares) splits vertically.
    # Right child (5 squares) splits horizontally.
    #   Right-Left (2 squares) splits vertically.
    #   Right-Right (3 squares) splits horizontally.
    #     Right-Right-Left (1 square)
    #     Right-Right-Right (2 squares) splits vertically.
    
    # This specific topology has 9 internal nodes (splits).
    # We need to optimize 9 split ratios.
    
    def build_tree_topology_1():
        """
        Returns a list of operations to build the tree.
        Each operation is (node_id, type, child1, child2, split_dir)
        type 0: leaf (square_idx)
        type 1: internal
        """
        # We'll just hardcode the recursive logic for optimization.
        # Instead of building a complex tree object, we'll write a specific
        # evaluation function for this topology.
        pass

    # Let's try a simpler, more robust approach:
    # Use a recursive function that takes a rectangle and a list of square indices.
    # If len(indices) == 1, it's a leaf.
    # Otherwise, split the list into two sublists and the rectangle into two sub-rectangles.
    # The split point is a parameter.
    
    # We will define a few topologies by specifying how the list of 10 indices is split at each node.
    # Topology A: 
    # Root: Split [0..9] into [0..4] and [5..9] vertically.
    # Node [0..4]: Split into [0..1] and [2..4] horizontally.
    # Node [0..1]: Split into [0] and [1] vertically.
    # Node [2..4]: Split into [2] and [3..4] horizontally.
    # Node [3..4]: Split into [3] and [4] vertically.
    # Node [5..9]: Split into [5..6] and [7..9] horizontally.
    # Node [5..6]: Split into [5] and [6] vertically.
    # Node [7..9]: Split into [7] and [8..9] horizontally.
    # Node [8..9]: Split into [8] and [9] vertically.
    
    # This topology has 9 splits.
    # Parameters: 9 split ratios (0 to 1).
    
    def evaluate_topology_A(params):
        """
        params: array of 9 split ratios.
        Returns centers, sides.
        """
        splits = params
        # Index mapping for splits
        # 0: Root (vert)
        # 1: Left-Child [0..4] (horiz)
        # 2: Left-Left [0..1] (vert)
        # 3: Left-Right [2..4] (horiz)
        # 4: Left-Right-Right [3..4] (vert)
        # 5: Right-Child [5..9] (horiz)
        # 6: Right-Left [5..6] (vert)
        # 7: Right-Right [7..9] (horiz)
        # 8: Right-Right-Right [8..9] (vert)
        
        centers = np.zeros((10, 2))
        sides = np.zeros(10)
        
        # Recursive helper
        def recurse(rect, indices, split_idx):
            # rect: (x_min, y_min, x_max, y_max)
            # indices: list of square indices in this node
            # split_idx: index into splits array for this node's split, or -1 if leaf
            
            x_min, y_min, x_max, y_max = rect
            w = x_max - x_min
            h = y_max - y_min
            
            if len(indices) == 1:
                idx = indices[0]
                s = min(w, h)
                cx = (x_min + x_max) / 2.0
                cy = (y_min + y_max) / 2.0
                centers[idx] = [cx, cy]
                sides[idx] = s
                return
            
            # Internal node
            if split_idx < 0 or split_idx >= len(splits):
                # Should not happen if topology is correct
                return
            
            ratio = splits[split_idx]
            
            # Determine split direction based on topology
            # We need to know the direction for each node.
            # Let's encode directions in a separate list or infer from topology.
            # For Topology A:
            # Node 0 (Root): Vert
            # Node 1 (Left): Horiz
            # Node 2 (Left-Left): Vert
            # Node 3 (Left-Right): Horiz
            # Node 4 (Left-Right-Right): Vert
            # Node 5 (Right): Horiz
            # Node 6 (Right-Left): Vert
            # Node 7 (Right-Right): Horiz
            # Node 8 (Right-Right-Right): Vert
            
            # This hardcoding is fragile. Let's pass direction as well.
            # Better: Define the tree structure explicitly.
            
            pass # Placeholder
        
        # This recursive approach with hardcoded indices is error-prone.
        # Let's use a more generic tree representation.
        
        # Tree Node: (left, right, direction)
        # direction: 0 for vertical split, 1 for horizontal split.
        
        # Topology A Tree Structure:
        # Root (0, vert) -> Left: Node1, Right: Node2
        # Node1 (1, horiz) -> Left: Node3, Right: Node4
        # Node3 (2, vert) -> Left: Leaf0, Right: Leaf1
        # Node4 (3, horiz) -> Left: Leaf2, Right: Node5
        # Node5 (4, vert) -> Left: Leaf3, Right: Leaf4
        # Node2 (5, horiz) -> Left: Node6, Right: Node7
        # Node6 (6, vert) -> Left: Leaf5, Right: Leaf6
        # Node7 (7, horiz) -> Left: Leaf7, Right: Node8
        # Node8 (8, vert) -> Left: Leaf8, Right: Leaf9
        
        # We can represent this as a list of tuples:
        # (node_id, left_child, right_child, direction)
        # Leaves are -1.
        
        tree = [
            (0, 1, 2, 0), # Root: Left=1, Right=2, Dir=Vert
            (1, 3, 4, 1), # Node1: Left=3, Right=4, Dir=Horiz
            (2, 6, 7, 1), # Node2: Left=6, Right=7, Dir=Horiz
            (3, -1, -1, 0), # Node3: Leaf0, Leaf1, Dir=Vert (irrelevant for leaves, but need to map indices)
            (4, -1, -1, 1), # Node4: Leaf2, Node5, Dir=Horiz
            (5, -1, -1, 0), # Node5: Leaf3, Leaf4, Dir=Vert
            (6, -1, -1, 0), # Node6: Leaf5, Leaf6, Dir=Vert
            (7, -1, -1, 1), # Node7: Leaf7, Node8, Dir=Horiz
            (8, -1, -1, 0), # Node8: Leaf8, Leaf9, Dir=Vert
        ]
        
        # Wait, the indices in the tree above don't match the leaf indices directly.
        # I need a mapping from tree nodes to square indices.
        
        # Let's simplify. I will define the split logic directly in the optimization function
        # for a few specific, well-known good topologies.
        
        # Topology 1: "Two columns of 5"
        # Split unit square vertically at x=r0.
        # Left col (width r0) has 5 squares. Right col (width 1-r0) has 5 squares.
        # Left col: Split horizontally.
        #   Top part (height h1) has 2 squares. Bottom part (height 1-h1) has 3 squares.
        #   Top part: Split vertically into 1 and 1.
        #   Bottom part: Split horizontally.
        #     Top sub-part (height h2) has 1 square.
        #     Bottom sub-part (height 1-h1-h2) has 2 squares. Split vertically into 1 and 1.
        # Right col: Similar structure.
        
        # This is getting complicated to hardcode.
        
        # Let's try a different, more flexible approach:
        # Use a "Shelf" packing heuristic as the base, then optimize positions/sizes?
        # No, the directive says "recursive guillotine cut tree".
        
        # Let's stick to the tree but implement it cleanly.
        
        # Define a generic tree evaluator.
        # Tree is a list of nodes.
        # Each node: (left, right, direction)
        # If left == right == -1, it's a leaf. The leaf index is determined by traversal order.
        
        # Let's generate a few trees.
        
        def gen_tree_balanced(n):
            """Generates a balanced guillotine tree for n squares."""
            # This is tricky for n=10.
            # Let's manually define a few good trees.
            pass

        # Tree 1: 
        # Root (Vert)
        #   Left (Horiz)
        #     Left (Vert) -> L0, L1
        #     Right (Horiz)
        #       Left -> L2
        #       Right (Vert) -> L3, L4
        #   Right (Horiz)
        #     Left (Vert) -> L5, L6
        #     Right (Horiz)
        #       Left -> L7
        #       Right (Vert) -> L8, L9
        
        # This tree has 9 internal nodes.
        # Split parameters: 9 values.
        
        def eval_tree_1(params):
            # params: [r0, r1, r2, r3, r4, r5, r6, r7, r8]
            # r0: Root vert split
            # r1: Left child horiz split
            # r2: Left-Left vert split
            # r3: Left-Right horiz split
            # r4: Left-Right-Right vert split
            # r5: Right child horiz split
            # r6: Right-Left vert split
            # r7: Right-Right horiz split
            # r8: Right-Right-Right vert split
            
            centers = np.zeros((10, 2))
            sides = np.zeros(10)
            
            # Root
            x0, y0, x1, y1 = 0, 0, 1, 1
            w = x1 - x0
            h = y1 - y0
            
            # Root split: Vert at x0 + r0*w
            x_split = x0 + params[0] * w
            
            # Left Subtree: [x0, x_split] x [y0, y1]
            # Right Subtree: [x_split, x1] x [y0, y1]
            
            # Left Subtree (Horiz split)
            lx0, ly0, lx1, ly1 = x0, y0, x_split, y1
            lw = lx1 - lx0
            lh = ly1 - ly0
            y_split_l = ly0 + params[1] * lh
            
            # Left-Left (Vert split)
            llx0, lly0, llx1, lly1 = lx0, ly0, lx1, y_split_l
            llw = llx1 - llx0
            llh = lly1 - lly0
            x_split_ll = llx0 + params[2] * llw
            
            # Leaf 0
            s0 = min(x_split_ll - llx0, lly1 - lly0)
            centers[0] = [(llx0 + x_split_ll)/2, (lly0 + lly1)/2]
            sides[0] = s0
            
            # Leaf 1
            s1 = min(llx1 - x_split_ll, lly1 - lly0)
            centers[1] = [(x_split_ll + llx1)/2, (lly0 + lly1)/2]
            sides[1] = s1
            
            # Left-Right (Horiz split)
            lrx0, lry0, lrx1, lry1 = lx0, y_split_l, lx1, ly1
            lrw = lrx1 - lrx0
            lrh = lry1 - lry0
            y_split_lr = lry0 + params[3] * lrh
            
            # Leaf 2
            s2 = min(lrw, y_split_lr - lry0)
            centers[2] = [(lrx0 + lrx1)/2, (lry0 + y_split_lr)/2]
            sides[2] = s2
            
            # Left-Right-Right (Vert split)
            lrrx0, lrry0, lrrx1, lrry1 = lrx0, y_split_lr, lrx1, lry1
            lrrw = lrrx1 - lrrx0
            lrrh = lrry1 - lrry0
            x_split_lrr = lrrx0 + params[4] * lrrw
            
            # Leaf 3
            s3 = min(x_split_lrr - lrrx0, lrrh)
            centers[3] = [(lrrx0 + x_split_lrr)/2, (lrry0 + lrry1)/2]
            sides[3] = s3
            
            # Leaf 4
            s4 = min(lrrx1 - x_split_lrr, lrrh)
            centers[4] = [(x_split_lrr + lrrx1)/2, (lrry0 + lrry1)/2]
            sides[4] = s4
            
            # Right Subtree (Horiz split)
            rx0, ry0, rx1, ry1 = x_split, y0, x1, y1
            rw = rx1 - rx0
            rh = ry1 - ry0
            y_split_r = ry0 + params[5] * rh
            
            # Right-Left (Vert split)
            rlx0, rly0, rlx1, rly1 = rx0, ry0, rx1, y_split_r
            rlw = rlx1 - rlx0
            rlh = rly1 - rly0
            x_split_rl = rlx0 + params[6] * rlw
            
            # Leaf 5
            s5 = min(x_split_rl - rlx0, rlh)
            centers[5] = [(rlx0 + x_split_rl)/2, (rly0 + rly1)/2]
            sides[5] = s5
            
            # Leaf 6
            s6 = min(rlx1 - x_split_rl, rlh)
            centers[6] = [(x_split_rl + rlx1)/2, (rly0 + rly1)/2]
            sides[6] = s6
            
            # Right-Right (Horiz split)
            rrx0, rry0, rrx1, rry1 = rx0, y_split_r, rx1, ry1
            rrw = rrx1 - rrx0
            rrh = rry1 - rry0
            y_split_rr = rry0 + params[7] * rrh
            
            # Leaf 7
            s7 = min(rrw, y_split_rr - rry0)
            centers[7] = [(rrx0 + rrx1)/2, (rry0 + y_split_rr)/2]
            sides[7] = s7
            
            # Right-Right-Right (Vert split)
            rrrx0, rrry0, rrrx1, rrry1 = rrx0, y_split_rr, rrx1, rry1
            rrrw = rrrx1 - rrrx0
            rrrh = rrry1 - rrry0
            x_split_rrr = rrrx0 + params[8] * rrrw
            
            # Leaf 8
            s8 = min(x_split_rrr - rrrx0, rrrh)
            centers[8] = [(rrrx0 + x_split_rrr)/2, (rrry0 + rrry1)/2]
            sides[8] = s8
            
            # Leaf 9
            s9 = min(rrrx1 - x_split_rrr, rrrh)
            centers[9] = [(x_split_rrr + rrrx1)/2, (rrry0 + rrry1)/2]
            sides[9] = s9
            
            return centers, sides

        # Objective function for Tree 1
        def obj_tree_1(params):
            c, s = eval_tree_1(params)
            return -np.sum(s)
        
        # Optimize Tree 1
        best_c, best_s = None, None
        best_val = -np.inf
        
        # Initial guess: all splits at 0.5
        x0 = np.full(9, 0.5)
        bounds = [(0.0, 1.0)] * 9
        
        np.random.seed(42)
        for trial in range(20):
            if trial == 0:
                x_init = x0
            else:
                x_init = x0 + np.random.randn(9) * 0.2
                x_init = np.clip(x_init, 0.0, 1.0)
            
            try:
                res = minimize(obj_tree_1, x_init, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500})
                if res.fun < best_val:
                    best_val = res.fun
                    best_c, best_s = eval_tree_1(res.x)
            except:
                pass
        
        if best_c is not None:
            return best_c, best_s

        # Fallback: Simple grid
        centers = np.zeros((10, 2))
        sides = np.zeros(10)
        for i in range(10):
            col = i % 4
            row = i // 4
            centers[i] = [(col + 0.5) / 4.0, (row + 0.5) / 3.0]
            sides[i] = 0.25
        return centers, sides


# EVOLVE-BLOCK-END
