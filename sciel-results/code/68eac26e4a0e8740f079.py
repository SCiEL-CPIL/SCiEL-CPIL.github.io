import random
import itertools
import time
import math

# Precompute Q3 properties for fast checking
q3_adj = [0]*8
for i in range(8):
    adj = 0
    for j in range(8):
        if i != j and bin(i ^ j).count('1') == 1:
            adj |= (1 << j)
    q3_adj[i] = adj

def contains_q3_fast(subset, adj_bits):
    """
    Fast check if induced subgraph on 8 vertices contains Q3.
    adj_bits[v] is a bitmask of neighbors of vertex v.
    subset is a tuple/list of 8 vertex ids.
    """
    # Build local adjacency bitmasks
    local_adj = [0]*8
    for i in range(8):
        vi = subset[i]
        # Map global neighbors to local indices
        # We need to check which of the 8 subset vertices are neighbors of vi
        for j in range(8):
            if i != j and (adj_bits[vi] >> subset[j]) & 1:
                local_adj[i] |= (1 << j)
    
    # Degree check: all must have degree >= 3 in local graph
    for i in range(8):
        if bin(local_adj[i]).count('1') < 3:
            return False
    
    # Edge count check
    total_deg = sum(bin(local_adj[i]).count('1') for i in range(8))
    if total_deg < 24:  # 12 edges * 2
        return False
    
    # Backtracking
    mapping = [-1]*8
    
    def backtrack(q3_v):
        if q3_v == 8:
            return True
        
        # Try each unmapped local vertex
        used_mask = 0
        for u in range(q3_v):
            used_mask |= (1 << mapping[u])
        
        # Optimization: If q3_v has neighbors already mapped, restrict candidates
        candidates = []
        for s_idx in range(8):
            if (used_mask >> s_idx) & 1:
                continue
            # Check compatibility with already mapped neighbors
            valid = True
            for u in range(q3_v):
                if (q3_adj[u] >> q3_v) & 1:
                    if not ((local_adj[mapping[u]] >> s_idx) & 1):
                        valid = False
                        break
            if valid:
                candidates.append(s_idx)
        
        if not candidates:
            return False
            
        # Heuristic: Try candidates with lower degree in local graph first? 
        # Or just iterate.
        for s_idx in candidates:
            mapping[q3_v] = s_idx
            if backtrack(q3_v + 1):
                return True
            mapping[q3_v] = -1
        return False
    
    # Try all 8 vertices as the first mapping for Q3 vertex 0
    for first in range(8):
        mapping[0] = first
        if backtrack(1):
            return True
        mapping[0] = -1
    
    return False

def construct():
    n = 16
    start_time = time.time()
    time_limit = 115
    
    # Initialize graph: Start with a fairly dense random graph
    adj_bits = [0]*n
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.8:
                adj_bits[i] |= (1 << j)
                adj_bits[j] |= (1 << i)
    
    # Precompute all 8-vertex subsets for potential use, though we will sample
    all_subsets = list(itertools.combinations(range(n), 8))
    
    def get_violation_count(sample_size=100):
        """Estimate number of Q3 violations by sampling."""
        count = 0
        for _ in range(sample_size):
            subset = random.choice(all_subsets)
            if contains_q3_fast(subset, adj_bits):
                count += 1
        # Scale up to estimate total violations
        # Total subsets is 12870
        return (count / sample_size) * len(all_subsets)
    
    def get_exact_violation_count():
        """Exact count of Q3 violations. Slow."""
        count = 0
        for subset in all_subsets:
            if contains_q3_fast(subset, adj_bits):
                count += 1
        return count

    def get_edge_count():
        return sum(bin(adj_bits[i]).count('1') for i in range(n)) // 2

    # Simulated Annealing
    current_edges = get_edge_count()
    current_violations = get_exact_violation_count() # Start with exact to know where we are
    # If we start with violations, we need to handle them.
    # Objective: Maximize edges, Minimize violations.
    # Energy = -edges + P * violations
    
    P = 10.0 # Penalty weight
    T = 5.0  # Initial temperature
    T_min = 0.01
    alpha = 0.999 # Cooling rate
    
    best_edges = current_edges
    best_adj_bits = adj_bits[:]
    if current_violations == 0:
        best_violations = 0
    else:
        best_violations = current_violations

    steps = 0
    while time.time() - start_time < time_limit - 10:
        T *= alpha
        if T < T_min:
            T = T_min
            # If T is too low, we might be stuck. Restart with higher T?
            # Or just break and do local search.
            pass
        
        # Pick a random edge to flip
        i = random.randint(0, n-1)
        j = random.randint(i+1, n-1)
        
        # Flip edge (i, j)
        has_edge = (adj_bits[i] >> j) & 1
        if has_edge:
            adj_bits[i] &= ~(1 << j)
            adj_bits[j] &= ~(1 << i)
            new_edges = current_edges - 1
        else:
            adj_bits[i] |= (1 << j)
            adj_bits[j] |= (1 << i)
            new_edges = current_edges + 1
        
        # Estimate new violations
        # To save time, we can just sample
        new_violations = get_violation_count(50)
        
        # Calculate energy change
        # E = -edges + P * violations
        delta_E = -(new_edges - current_edges) + P * (new_violations - current_violations)
        
        # Acceptance criterion
        if delta_E < 0:
            accept = True
        else:
            # Metropolis criterion
            # Probability = exp(-delta_E / T)
            if T > 0:
                prob = math.exp(-delta_E / T)
                accept = random.random() < prob
            else:
                accept = False
        
        if accept:
            current_edges = new_edges
            current_violations = new_violations
            
            # Update best if valid or better
            if new_violations == 0:
                if new_edges > best_edges:
                    best_edges = new_edges
                    best_adj_bits = adj_bits[:]
            else:
                # Keep track of best valid solution found so far
                # We don't update best if invalid, unless we want to track progress
                pass
        else:
            # Revert
            if has_edge:
                adj_bits[i] |= (1 << j)
                adj_bits[j] |= (1 << i)
            else:
                adj_bits[i] &= ~(1 << j)
                adj_bits[j] &= ~(1 << i)
        
        steps += 1
        if steps % 1000 == 0:
            # Periodic exact check to update best if we are actually valid
            # This is expensive but necessary to ensure we don't miss a valid high-edge solution
            # due to sampling error in violation count.
            if time.time() - start_time < time_limit - 15:
                exact_v = get_exact_violation_count()
                if exact_v == 0:
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj_bits = adj_bits[:]
                current_violations = exact_v # Sync with reality

    # Final cleanup: Ensure best_adj_bits is Q3-free
    # Use best_adj_bits as the starting point for cleanup
    adj_bits = best_adj_bits[:]
    
    # Remove edges to destroy any remaining Q3s
    while time.time() - start_time < time_limit - 1:
        # Find a Q3
        found = False
        for subset in all_subsets:
            if contains_q3_fast(subset, adj_bits):
                found = True
                # Remove a random edge from this subset
                edges_in_q3 = []
                for u in range(8):
                    for v in range(u+1, 8):
                        if (adj_bits[subset[u]] >> subset[v]) & 1:
                            edges_in_q3.append((subset[u], subset[v]))
                if edges_in_q3:
                    u, v = random.choice(edges_in_q3)
                    adj_bits[u] &= ~(1 << v)
                    adj_bits[v] &= ~(1 << u)
                break
        
        if not found:
            break
    
    # Build edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (adj_bits[i] >> j) & 1:
                edges.append([i, j])
    
    return edges

# EVOLVE-BLOCK-END
