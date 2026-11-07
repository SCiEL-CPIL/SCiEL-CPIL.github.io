"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin for 90s limit
    
    # Objective: Minimize max(number of distinct edge sums, number of distinct edge products)
    # Variables:
    # values: 14 distinct positive integers <= 10000
    # edges: 35 pairs of indices (0-13)
    
    # Pre-compute all possible edges
    all_edges = list(combinations(range(14), 2))
    N_EDGES = 35
    
    def evaluate(values, edge_indices):
        """
        Vectorized evaluation.
        values: np.array of shape (14,)
        edge_indices: np.array of shape (35, 2) containing indices into all_edges
        """
        # Get the actual edge pairs
        # all_edges is a list of tuples. 
        # To vectorize, we can pre-store all_edges as a numpy array.
        # Let's assume all_edges_np is available in closure or passed.
        # For simplicity in this inner function, we'll use the global all_edges_np.
        
        u = all_edges_np[edge_indices, 0]
        v = all_edges_np[edge_indices, 1]
        
        val_u = values[u]
        val_v = values[v]
        
        sums = val_u + val_v
        prods = val_u * val_v
        
        n_distinct_sums = len(np.unique(sums))
        n_distinct_prods = len(np.unique(prods))
        
        return max(n_distinct_sums, n_distinct_prods)

    # Pre-store edges as numpy array for fast indexing
    all_edges_np = np.array(all_edges, dtype=np.int32)
    
    def random_state():
        """Generate a random valid initial state."""
        # Values: distinct random integers
        # To encourage collisions, we can sample from a smaller range or use structured sets
        # Let's try a mix: some small, some random
        vals = set()
        while len(vals) < 14:
            if random.random() < 0.7:
                vals.add(random.randint(1, 100))
            else:
                vals.add(random.randint(1, 10000))
        values = np.array(list(vals), dtype=np.int64)
        
        # Edges: random subset of 35
        edge_indices = np.random.choice(len(all_edges), N_EDGES, replace=False)
        
        return values, edge_indices

    def make_move(values, edge_indices):
        """
        Perform a random perturbation.
        Returns new_values, new_edge_indices, move_type
        """
        new_values = values.copy()
        new_edge_indices = edge_indices.copy()
        
        move_type = random.random()
        
        if move_type < 0.5:
            # Change a value
            idx = random.randint(0, 13)
            # Try to keep values somewhat small to allow collisions, but allow full range
            if random.random() < 0.8:
                new_val = random.randint(1, 200)
            else:
                new_val = random.randint(1, 10000)
            
            # Ensure distinctness
            # Check if new_val is already in values
            if new_val in values:
                # Swap with another index? Or just retry?
                # Let's just pick a different random value until distinct
                while new_val in values:
                    new_val = random.randint(1, 10000)
            
            new_values[idx] = new_val
            return new_values, new_edge_indices, "value"
            
        elif move_type < 0.8:
            # Change an edge (swap one edge index with another)
            edge_pos = random.randint(0, N_EDGES - 1)
            new_edge_idx = random.randint(0, len(all_edges) - 1)
            
            # Ensure the new edge is not already in the set (to keep exactly 35 distinct edges)
            # If it is, we would have a duplicate edge, which is invalid.
            # Check if new_edge_idx is in edge_indices
            if new_edge_idx in edge_indices:
                # Retry or pick another
                # For efficiency, just reject this move? Or pick another.
                # Let's pick another random one.
                attempts = 0
                while new_edge_idx in edge_indices and attempts < 10:
                    new_edge_idx = random.randint(0, len(all_edges) - 1)
                    attempts += 1
                if new_edge_idx in edge_indices:
                    # Fallback: return original state (no move)
                    return values, edge_indices, "none"
            
            new_edge_indices[edge_pos] = new_edge_idx
            return new_values, new_edge_indices, "edge"
            
        else:
            # 2-swap: Change one value AND one edge
            val_idx = random.randint(0, 13)
            if random.random() < 0.8:
                new_val = random.randint(1, 200)
            else:
                new_val = random.randint(1, 10000)
                
            if new_val in values:
                while new_val in values:
                    new_val = random.randint(1, 10000)
            
            new_values[val_idx] = new_val
            
            edge_pos = random.randint(0, N_EDGES - 1)
            new_edge_idx = random.randint(0, len(all_edges) - 1)
            if new_edge_idx in edge_indices:
                attempts = 0
                while new_edge_idx in edge_indices and attempts < 10:
                    new_edge_idx = random.randint(0, len(all_edges) - 1)
                    attempts += 1
                if new_edge_idx in edge_indices:
                    return values, edge_indices, "none"
            
            new_edge_indices[edge_pos] = new_edge_idx
            return new_values, new_edge_indices, "both"

    # Simulated Annealing
    best_values, best_edges = random_state()
    best_score = evaluate(best_values, best_edges)
    
    current_values, current_edges = best_values.copy(), best_edges.copy()
    current_score = best_score
    
    T = 100.0
    T_min = 0.01
    cooling_rate = 0.999
    
    steps = 0
    max_steps = 50000 # Adjust based on speed
    
    while time.time() < deadline and T > T_min and steps < max_steps:
        new_values, new_edges, move_type = make_move(current_values, current_edges)
        
        if move_type == "none":
            T *= cooling_rate
            steps += 1
            continue
            
        new_score = evaluate(new_values, new_edges)
        
        delta = new_score - current_score
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            # Accept
            current_values = new_values
            current_edges = new_edges
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_values = current_values.copy()
                best_edges = current_edges.copy()
        
        T *= cooling_rate
        steps += 1
        
        # Periodic restarts if stuck?
        # If no improvement in last 1000 steps, maybe restart with a new random state?
        # For now, rely on SA.
        
    # Final cleanup: convert numpy arrays to Python lists
    final_values = [int(v) for v in best_values]
    final_edges = [list(all_edges_np[idx]) for idx in best_edges]
    
    # Ensure edges are lists of ints
    final_edges = [[int(u), int(v)] for u, v in final_edges]
    
    # Verification
    assert len(final_values) == 14
    assert len(set(final_values)) == 14
    assert all(isinstance(v, int) for v in final_values)
    assert all(1 <= v <= 10000 for v in final_values)
    
    assert len(final_edges) == 35
    edge_set = set(tuple(sorted(e)) for e in final_edges)
    assert len(edge_set) == 35, f"Duplicate edges found: {len(final_edges)} vs {len(edge_set)}"
    
    return {'values': final_values, 'edges': final_edges}

# EVOLVE-BLOCK-END
