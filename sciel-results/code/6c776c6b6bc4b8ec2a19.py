import time
import math
import sys

# Increase recursion limit for deep branch-and-bound
sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    N = 512
    # 1. Build the conflict graph
    # Edge (u, v) exists if lcm(u, v) > N
    # We represent the graph using adjacency lists or bitmasks.
    # Bitmasks are efficient for set operations.
    # There are 512 nodes, which fits in a Python int (arbitrary precision).
    
    # adj[i] is a bitmask of all nodes j such that lcm(i, j) > N
    adj = [0] * N
    
    # Precompute primes or just use math.gcd
    # lcm(a, b) = a * b // gcd(a, b)
    # We can optimize by checking if a * b // gcd(a, b) > N
    # Which is equivalent to a // gcd(a, b) > N // b
    
    for i in range(N):
        ui = i + 1
        mask = 0
        for j in range(i + 1, N):
            uj = j + 1
            g = math.gcd(ui, uj)
            # Check lcm
            # ui * uj // g > N
            # ui // g * uj > N
            if (ui // g) * uj > N:
                mask |= (1 << j)
                adj[j] |= (1 << i)
        adj[i] = mask
    
    # 2. Find an initial independent set using a greedy heuristic
    # This provides a lower bound for the B&B
    def greedy_mis(adj, N):
        remaining = (1 << N) - 1
        result = []
        while remaining:
            # Pick a vertex with minimum degree in the remaining graph?
            # Or just the first one?
            # A better heuristic: pick vertex with min degree in induced subgraph to leave more options?
            # Actually, for MIS, picking a vertex with high degree in conflict graph (low degree in compatibility) is often good for speed, 
            # but for size, we want to pick vertices that don't conflict with many others.
            # Let's just pick the first available vertex for simplicity and speed of initial bound.
            # We can improve this if needed.
            v_bit = remaining & (-remaining)
            v = v_bit.bit_length() - 1
            result.append(v)
            # Remove v and its neighbors from remaining
            # Neighbors are in adj[v]. 
            # We need to remove v and adj[v] from remaining.
            # remaining = remaining & ~(adj[v] | v_bit)
            remaining &= ~(adj[v] | v_bit)
        return len(result)

    best_size = greedy_mis(adj, N)
    
    # 3. Branch and Bound for Maximum Independent Set
    # State: current set of available vertices (bitmask), current path (list of vertices)
    # We don't need to store the path if we just track size, but we need to return the set.
    # To save memory/time, we can store the best set found.
    
    best_set = [0] * N # Store the actual set as a list of booleans or a list of indices
    # Let's store best_set as a list of indices
    best_indices = []
    
    # We need to recover the set from the greedy step to have a valid best_set initially
    # Re-run greedy to get indices
    def greedy_mis_indices(adj, N):
        remaining = (1 << N) - 1
        result = []
        while remaining:
            v_bit = remaining & (-remaining)
            v = v_bit.bit_length() - 1
            result.append(v)
            remaining &= ~(adj[v] | v_bit)
        return result

    best_indices = greedy_mis_indices(adj, N)
    best_size = len(best_indices)
    
    # Helper to update best
    def update_best(current_indices):
        nonlocal best_size, best_indices
        if len(current_indices) > best_size:
            best_size = len(current_indices)
            best_indices = current_indices[:]

    # B&B Search
    # We use a stack for iterative DFS to avoid recursion overhead/limits if depth is high, 
    # but recursion is easier to write. With N=512, depth <= 512. Recursion is fine.
    
    def search(remaining_mask, current_indices):
        nonlocal best_size, best_indices
        nonlocal deadline, start_time
        
        # Time check
        if time.time() > deadline:
            return False # Signal to stop
            
        # Pruning
        # Upper bound is the number of bits in remaining_mask
        if len(current_indices) + remaining_mask.bit_count() <= best_size:
            return True # Pruned, but not a timeout
        
        if remaining_mask == 0:
            update_best(current_indices)
            return True
        
        # Heuristic: Pick a vertex
        # Choosing the vertex with the highest degree in the induced subgraph (remaining_mask)
        # might prune faster.
        # Let's pick the lowest index for determinism and speed of bit ops, 
        # but high degree is better for pruning.
        
        # Let's find a vertex in remaining_mask.
        # To improve pruning, select v such that |N(v) ∩ remaining| is maximized.
        # This is expensive to compute for all v.
        # Alternative: Just pick the first bit.
        v_bit = remaining_mask & (-remaining_mask)
        v = v_bit.bit_length() - 1
        
        # Branch 1: Include v
        # New remaining: remove v and its neighbors
        new_remaining = remaining_mask & ~(adj[v] | v_bit)
        current_indices.append(v)
        
        # Check time
        if time.time() > deadline:
            current_indices.pop()
            return False
            
        search(new_remaining, current_indices)
        current_indices.pop()
        
        if time.time() > deadline:
            return False
            
        # Branch 2: Exclude v
        new_remaining = remaining_mask & ~v_bit
        search(new_remaining, current_indices)
        
        return True

    # Start search
    # Initial mask is all bits set
    initial_mask = (1 << N) - 1
    search(initial_mask, [])
    
    # Return the best set found
    # The problem asks for a subset of [1, 512]. 
    # Our indices are 0-based. We need to convert to 1-based values.
    result_set = [i + 1 for i in best_indices]
    
    # Ensure non-empty (greedy guarantees at least 1, as 1 is always compatible with 1? 
    # lcm(1,1)=1<=512. So 1 is always in some MIS. Greedy picks 1 first? 
    # v_bit = 1 (index 0). Yes.)
    
    return result_set

# Fallback if something goes wrong, though unlikely
if __name__ == "__main__":
    # Quick test
    s = construct()
    print(len(s))
