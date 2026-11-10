# EVOLVE-BLOCK-START
import numpy as np

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the respective outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    n = 11
    # Normals for the hexagon edges (pointing outwards)
    # Vertices at 0, 60, ... deg. Edges perpendicular to 30, 90, ... deg.
    angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180.0
    normals = np.stack([np.cos(angles), np.sin(angles)], axis=1) # shape (6, 2)
    
    def get_required_side(centers):
        """
        Calculates the minimum side length S of the outer hexagon (angle 0) 
        required to contain all inner hexagons (angle 0) centered at 'centers'.
        """
        # For a regular hexagon with side 1 (circumradius 1), the support function in any direction is 1.
        # The outer hexagon has side S, so its apothem (distance from center to edge midpoint) is S * sqrt(3)/2.
        # The condition is: max projection of any inner vertex onto any outer normal <= S * sqrt(3)/2.
        # Max projection of inner hex i onto normal j is centers[i] . normals[j] + 1.
        # So S >= (max_i (centers[i] . normals[j]) + 1) * 2 / sqrt(3) for all j.
        
        projections = centers @ normals.T
        max_projections = np.max(projections, axis=0) # shape (6,)
        required_Ss = (max_projections + 1.0) * 2.0 / np.sqrt(3.0)
        return np.max(required_Ss)

    def check_overlap(centers, eps=1e-6):
        """
        Returns True if any pair of hexagons overlaps or are too close.
        """
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        proj = diff @ normals.T
        max_abs_proj = np.max(np.abs(proj), axis=2)
        
        # Set diagonal to infinity to ignore self-overlap
        np.fill_diagonal(max_abs_proj, np.inf)
        
        # Separation is max_abs_proj - sqrt(3).
        # Valid if min separation > eps.
        # Overlap/Invalid if min separation <= eps.
        min_sep = np.min(max_abs_proj)
        return min_sep <= np.sqrt(3.0) + eps

    def calculate_penalty(centers, S):
        """
        Calculate a penalty score based on overlaps and containment violations.
        Lower is better.
        """
        # Overlap penalty
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        proj = diff @ normals.T
        max_abs_proj = np.max(np.abs(proj), axis=2)
        np.fill_diagonal(max_abs_proj, np.inf)
        
        # Overlap amount is sqrt(3) - max_abs_proj (if positive)
        overlaps = np.maximum(0, np.sqrt(3.0) - max_abs_proj)
        overlap_penalty = np.sum(overlaps)
        
        # Containment penalty
        projections = centers @ normals.T
        max_projections = np.max(projections, axis=0)
        required_Ss = (max_projections + 1.0) * 2.0 / np.sqrt(3.0)
        containment_penalty = np.sum(np.maximum(0, required_Ss - S))
        
        return overlap_penalty + 10.0 * containment_penalty

    def simulated_annealing(initial_centers, T_start=1.0, T_end=0.01, alpha=0.999, iterations=50000):
        current_centers = initial_centers.copy()
        current_S = get_required_side(current_centers)
        current_penalty = calculate_penalty(current_centers, current_S)
        
        best_centers = current_centers.copy()
        best_S = current_S
        best_penalty = current_penalty
        
        for i in range(iterations):
            T = T_start * (alpha ** i)
            
            # Pick a random hexagon to move
            idx = np.random.randint(n)
            
            # Random perturbation
            # Scale step size with temperature? Or fixed?
            # Fixed small step is often better with annealing.
            step_size = 0.5
            delta = np.random.normal(0, step_size, size=2)
            
            new_centers = current_centers.copy()
            new_centers[idx] += delta
            
            new_S = get_required_side(new_centers)
            new_penalty = calculate_penalty(new_centers, new_S)
            
            # Acceptance criteria
            # We primarily care about S, but penalty helps guide us.
            # Let's define cost as S + large_penalty * penalty
            # Or just use S if penalty is 0, and penalty if S is not the only concern?
            # The objective is to minimize S.
            # If penalty > 0, the configuration is invalid.
            # We want to minimize S among valid configs.
            
            # Heuristic: 
            # If current is valid (penalty 0) and new is valid, accept if new_S < current_S.
            # If current is valid and new is invalid, accept with prob exp(-(new_penalty)/T).
            # If current is invalid and new is valid, accept.
            # If both invalid, accept if new_penalty < current_penalty with prob exp(-(diff)/T).
            
            if current_penalty == 0 and new_penalty == 0:
                if new_S < current_S:
                    current_centers = new_centers
                    current_S = new_S
                    current_penalty = new_penalty
                else:
                    # Accept worse S with small prob to escape local minima?
                    # Usually not needed if we have many iterations, but let's allow it.
                    delta_S = new_S - current_S
                    if np.random.rand() < np.exp(-delta_S / T):
                        current_centers = new_centers
                        current_S = new_S
                        current_penalty = new_penalty
            elif current_penalty > 0 and new_penalty == 0:
                # Moving from invalid to valid is always good
                current_centers = new_centers
                current_S = new_S
                current_penalty = new_penalty
            elif current_penalty == 0 and new_penalty > 0:
                # Moving from valid to invalid: only if it reduces S significantly? 
                # Or just accept with low prob to explore.
                # Since we care about S, maybe accept if new_S is much smaller?
                # But if it's invalid, it's not a solution. 
                # Let's accept with prob exp(-new_penalty/T)
                if np.random.rand() < np.exp(-new_penalty / T):
                    current_centers = new_centers
                    current_S = new_S
                    current_penalty = new_penalty
            else:
                # Both invalid
                delta_pen = new_penalty - current_penalty
                if delta_pen < 0 or np.random.rand() < np.exp(-delta_pen / T):
                    current_centers = new_centers
                    current_S = new_S
                    current_penalty = new_penalty
            
            # Update best
            if current_penalty == 0 and current_S < best_S:
                best_S = current_S
                best_centers = current_centers.copy()
            
            # Early stop if we've been stuck for a long time?
            # Not implemented for simplicity, rely on T_end.

        return best_centers, best_S

    # Initial guess: 7-cluster + 4 in gaps
    # 7-cluster:
    c0 = np.array([0, 0])
    c1 = np.array([2, 0])
    c2 = np.array([1, np.sqrt(3)])
    c3 = np.array([-1, np.sqrt(3)])
    c4 = np.array([-2, 0])
    c5 = np.array([-1, -np.sqrt(3)])
    c6 = np.array([1, -np.sqrt(3)])
    
    # 4 more: place them at distance ~3.5 in the 4 "corner" gaps?
    r = 3.5
    c7 = np.array([r * np.cos(np.pi/6), r * np.sin(np.pi/6)])
    c8 = np.array([r * np.cos(5*np.pi/6), r * np.sin(5*np.pi/6)])
    c9 = np.array([r * np.cos(7*np.pi/6), r * np.sin(7*np.pi/6)])
    c10 = np.array([r * np.cos(11*np.pi/6), r * np.sin(11*np.pi/6)])
    
    initial_centers = np.array([c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10])
    
    # Run Simulated Annealing
    best_centers, best_S = simulated_annealing(initial_centers, T_start=2.0, T_end=0.001, alpha=0.9995, iterations=100000)
    
    # Construct return values
    # Inner hexagons: (x, y, 0)
    inner_hex_data = np.hstack([best_centers, np.zeros((n, 1))])
    # Outer hexagon: (0, 0, 0)
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = best_S
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
