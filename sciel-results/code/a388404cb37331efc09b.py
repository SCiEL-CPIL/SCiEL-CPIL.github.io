import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    def get_violations(adj_masks, n, limit=200):
        """
        Returns a list of violating subsets (tuples of vertices).
        Uses bitmask representation for fast degree calculation.
        """
        violations = []
        # Check subsets of size 6 to n
        # Start with size 6 as they are most common and "fundamental"
        for k in range(6, n + 1):
            for S in itertools.combinations(range(n), k):
                mask = 0
                for v in S:
                    mask |= (1 << v)
                
                first_deg = -1
                is_reg = True
                for u in S:
                    deg = (adj_masks[u] & mask).bit_count()
                    if first_deg == -1:
                        first_deg = deg
                    elif deg != first_deg:
                        is_reg = False
                        break
                if is_reg:
                    violations.append(S)
                    if len(violations) >= limit:
                        return violations
        return violations

    def try_find_graph(n, time_budget):
        """
        Attempts to find a valid graph on n vertices within time_budget.
        Returns the adjacency mask list if found, else None.
        """
        end_time = time.time() + time_budget
        best_violations = float('inf')
        best_masks = None
        
        # Multiple restarts
        max_restarts = 50
        for r in range(max_restarts):
            if time.time() > end_time:
                break
            
            # Random initialization
            adj_masks = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj_masks[i] |= (1 << j)
                        adj_masks[j] |= (1 << i)
            
            current_violations = get_violations(adj_masks, n)
            current_count = len(current_violations)
            
            if current_count == 0:
                return adj_masks
            
            if current_count < best_violations:
                best_violations = current_count
                best_masks = adj_masks.copy()
                
            # Local Search
            no_improve = 0
            max_no_improve = 50
            while no_improve < max_no_improve and time.time() < end_time:
                if not current_violations:
                    return adj_masks
                    
                # Pick a random violation
                S = random.choice(current_violations)
                S_list = list(S)
                
                # Pick a random pair in S to flip
                if len(S_list) < 2:
                    break
                u, v = random.sample(S_list, 2)
                
                # Flip edge (u, v)
                # Check if edge exists
                has_edge = (adj_masks[u] >> v) & 1
                if has_edge:
                    adj_masks[u] &= ~(1 << v)
                    adj_masks[v] &= ~(1 << u)
                else:
                    adj_masks[u] |= (1 << v)
                    adj_masks[v] |= (1 << u)
                
                new_violations = get_violations(adj_masks, n)
                new_count = len(new_violations)
                
                if new_count < current_count:
                    current_count = new_count
                    current_violations = new_violations
                    no_improve = 0
                    if current_count < best_violations:
                        best_violations = current_count
                        best_masks = adj_masks.copy()
                    if current_count == 0:
                        return adj_masks
                else:
                    # Revert
                    if has_edge:
                        adj_masks[u] |= (1 << v)
                        adj_masks[v] |= (1 << u)
                    else:
                        adj_masks[u] &= ~(1 << v)
                        adj_masks[v] &= ~(1 << u)
                    no_improve += 1
                    # Optional: Simulated Annealing step - accept bad move with low prob?
                    # For now, just revert.
                    
        return best_masks

    # Strategy: Try n=21 first with a time limit, then n=20 with remaining time.
    # If n=21 is impossible, we shouldn't waste all time on it.
    # Let's give n=21 40 seconds.
    masks_21 = try_find_graph(21, 40)
    if masks_21 is not None:
        edges = []
        for i in range(21):
            for j in range(i + 1, 21):
                if (masks_21[i] >> j) & 1:
                    edges.append([i, j])
        return {'n': 21, 'edges': edges}
        
    # Fallback to n=20
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        masks_20 = try_find_graph(20, remaining_time - 5)
        if masks_20 is not None:
            edges = []
            for i in range(20):
                for j in range(i + 1, 20):
                    if (masks_20[i] >> j) & 1:
                        edges.append([i, j])
            return {'n': 20, 'edges': edges}
            
    # Final fallback: n=5 (seed)
    return {'n': 5, 'edges': []}
