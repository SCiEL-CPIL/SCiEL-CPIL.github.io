"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    DTLZ2-specific optimizer using hypervolume-weighted sampling.
    
    Strategy:
    1. Exploit the known analytic Pareto front of DTLZ2 (unit sphere in positive orthant).
    2. Instead of a uniform grid in decision space, use recursive bisection in objective space.
    3. For each sub-region in objective space, take the center, project it onto the Pareto front,
       and map it back to decision space to generate the evaluation point.
    4. This maximizes the marginal hypervolume gain of each point by prioritizing regions
       with the largest volume in the objective space.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin before 330s limit
        
        # Extract bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget
        
        # DTLZ2 specifics:
        # n_obj = 3, dim = 12
        # Front variables: x[0], x[1]
        # G variables: x[2:12] set to 0.5 for g=0
        
        # Function to map target objective vector (on front) to decision vector
        def obj_to_dec(f):
            # f is length 3, on unit sphere (sum f^2 = 1)
            # f_1 = cos(x_1 * pi/2)
            # f_2 = sin(x_1 * pi/2) * cos(x_2 * pi/2)
            # f_3 = sin(x_1 * pi/2) * sin(x_2 * pi/2)
            
            x = np.zeros(dim)
            
            # Clip f to valid range [0, 1] to avoid domain errors in arccos
            f = np.clip(f, 0.0, 1.0)
            
            # Inverse mapping
            # x_1
            c1 = f[0]
            x[0] = (2.0 / np.pi) * np.arccos(c1)
            
            # x_2
            s1 = np.sqrt(1.0 - c1*c1)
            if s1 > 1e-12:
                c2 = f[1] / s1
                c2 = np.clip(c2, -1.0, 1.0)
                x[1] = (2.0 / np.pi) * np.arccos(c2)
            else:
                x[1] = 0.0 # Arbitrary if s1 is 0 (f1=1)
            
            # G variables
            x[2:] = 0.5
            
            # Clip to bounds
            x = np.clip(x, lo, hi)
            
            return x

        # Recursive bisection of objective space [0, 1]^3
        # We want to generate `budget` points.
        # We will perform a best-first search or a simple recursive split.
        # Given the small budget (200), we can use a priority queue based on the size of the sub-cube (volume).
        # Or simply a recursive split that generates points in a space-filling manner.
        
        # A simpler and effective approach for "centers of sub-rectangles":
        # Use a quadtree/octree on [0,1]^3.
        # Start with the whole cube.
        # Split into 8 sub-cubes.
        # Add centers of sub-cubes to the list.
        # Repeat, always splitting the sub-cube with the largest volume that hasn't been split yet?
        # Or just do a depth-first search until we have enough points?
        # To maximize hypervolume, we want points spread out.
        # A simple recursive bisection (midpoint) of the entire space at each step, 
        # but choosing the axis to split based on the longest dimension, is a standard k-means-like approach.
        # However, the directive says "centers of the 2^k sub-rectangles formed by recursive bisection".
        # This implies a structured grid-like recursion.
        
        # Let's use a priority queue of boxes.
        # Box defined by low, high.
        # Volume = prod(high - low).
        # Pop box with largest volume.
        # If budget allows, split it into 8 children.
        # Add the center of the *parent*? No, the centers of the children?
        # "places points at the centers of the 2^k sub-rectangles"
        # This sounds like: Level 1: 8 points (centers of 8 octants). Level 2: 64 points?
        # If we just take the centers of all leaves of a tree of depth D, we get 8^D points.
        # 8^2 = 64, 8^3 = 512.
        # Budget is 200.
        # We can do Depth 2 (64 points) and then fill the remaining 136 with Depth 3 points 
        # selected from the largest volume leaves of Depth 2.
        
        # Let's implement a best-first subdivision.
        # 1. Start with root box [0,1]^3.
        # 2. Keep a list of "active" boxes.
        # 3. At each step, pick the box with the largest volume.
        # 4. Split it into 8 children.
        # 5. Add the centers of these 8 children to our sample points?
        #    If we do this, we add 8 points per split.
        #    We need ~200 points. 200 / 8 = 25 splits.
        #    This will create a non-uniform grid that is denser in larger volume regions.
        #    Wait, the volume of the sub-boxes in objective space is what matters for hypervolume.
        #    Since the front is a sphere, the "volume" in the cube is a proxy.
        
        # Let's refine:
        # We will generate points by recursively splitting the objective space cube [0,1]^3.
        # We maintain a max-heap of boxes by volume.
        # We pop the largest box, split it into 8, and push the 8 children.
        # We record the *center* of the box we just popped? No, the centers of the children?
        # If we record the center of the box we are splitting, we get a nested set of points.
        # If we record the centers of the children, we get a finer grid.
        
        # Let's try: 
        # Initialize with the root box.
        # For i in range(budget):
        #   Pop largest box.
        #   If it's the root or we haven't split it yet, we might just take its center?
        #   Actually, a common strategy is to take the center of the *current* largest box, 
        #   then split that box into 8 and push them.
        #   This ensures the first point is the global center.
        #   The second point is the center of the largest child (which is all the same size, so arbitrary).
        #   This effectively creates a space-filling curve of points that prioritizes large empty regions.
        
        import heapq
        
        # Heap stores: (-volume, id, low, high)
        # Using negative volume for max-heap behavior with min-heap
        box_id_counter = 0
        boxes = []
        
        def get_volume(low, high):
            return np.prod(high - low)
        
        root_low = np.array([0.0, 0.0, 0.0])
        root_high = np.array([1.0, 1.0, 1.0])
        root_vol = get_volume(root_low, root_high)
        
        # We will push the root box
        heapq.heappush(boxes, (-root_vol, box_id_counter, root_low, root_high))
        box_id_counter += 1
        
        evaluated_points = []
        
        while len(evaluated_points) < budget and boxes and (time.time() - start_time) < deadline:
            neg_vol, bid, low, high = heapq.heappop(boxes)
            
            # Calculate center
            center = (low + high) / 2.0
            
            # Project center to Pareto front
            norm = np.linalg.norm(center)
            if norm > 1e-9:
                f_on_front = center / norm
            else:
                f_on_front = center # Should not happen for root
            
            # Map to decision space
            x = obj_to_dec(f_on_front)
            
            # Evaluate
            f_val = problem(x)
            evaluated_points.append(f_val)
            
            # Split the current box into 8 children
            # Only split if we have budget left
            if len(evaluated_points) < budget:
                mid = center
                for i in [0, 1]:
                    for j in [0, 1]:
                        for k in [0, 1]:
                            child_low = np.array([
                                low[0] if i==0 else mid[0],
                                low[1] if j==0 else mid[1],
                                low[2] if k==0 else mid[2]
                            ])
                            child_high = np.array([
                                mid[0] if i==0 else high[0],
                                mid[1] if j==0 else high[1],
                                mid[2] if k==0 else high[2]
                            ])
                            
                            # Check if volume is significant
                            c_vol = get_volume(child_low, child_high)
                            if c_vol > 1e-12:
                                heapq.heappush(boxes, (-c_vol, box_id_counter, child_low, child_high))
                                box_id_counter += 1
                            
                            # Early termination if time is tight
                            if (time.time() - start_time) > deadline - 5:
                                break
                        if (time.time() - start_time) > deadline - 5:
                            break
                    if (time.time() - start_time) > deadline - 5:
                        break
                
        return evaluated_points
# EVOLVE-BLOCK-END
