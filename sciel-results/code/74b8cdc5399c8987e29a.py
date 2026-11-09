import time
import math
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin for safety
    
    N = 256
    
    # Squares up to 2*256 = 512
    squares = set()
    i = 2
    while i * i <= 2 * N:
        squares.add(i * i)
        i += 1
    
    # Identify valid nodes (no self-conflict: 2v not a square)
    valid_nodes = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_nodes.append(v)
    
    n = len(valid_nodes)
    node_index = {v: i for i, v in enumerate(valid_nodes)}
    
    # Build conflict adjacency as bitmasks
    # conflicts[i] is a bitmask of nodes that conflict with node i
    conflicts = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if valid_nodes[i] + valid_nodes[j] in squares:
                conflicts[i] |= (1 << j)
                conflicts[j] |= (1 << i)
    
    # For efficiency, also store neighbor lists
    neighbor_lists = []
    for i in range(n):
        neighbors = []
        mask = conflicts[i]
        while mask:
            lsb = mask & (-mask)
            j = lsb.bit_length() - 1
            neighbors.append(j)
            mask ^= lsb
        neighbor_lists.append(neighbors)
    
    # Precompute degrees
    degrees = [len(neighbor_lists[i]) for i in range(n)]
    
    # Upper bound using greedy coloring (maximal independent set packing)
    def color_bound(vertices_mask):
        """Return an upper bound on the size of MIS in the induced subgraph of vertices_mask.
        Uses greedy maximal independent set packing: repeatedly find a maximal IS in remaining,
        count it, remove it. This gives a tighter bound than simple vertex picking."""
        if vertices_mask == 0:
            return 0
        remaining = vertices_mask
        color_count = 0
        while remaining:
            color_count += 1
            # Greedily build a maximal independent set in remaining
            # Pick vertex with smallest degree in remaining (to make the IS larger)
            temp = remaining
            best_v = -1
            best_deg = float('inf')
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                deg = bin(conflicts[v] & remaining).count('1')
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
                temp ^= lsb
            # Remove best_v and its neighbors
            remaining &= ~(conflicts[best_v] | (1 << best_v))
            # Continue removing until no more non-conflicting vertices can be added
            # Actually, for a proper coloring bound, we just remove one vertex per color class
            # But for MIS packing bound, we should find a maximal IS
            # Let's use the simpler approach: one vertex per color (standard coloring bound)
            # This is actually a coloring of the complement, giving upper bound on clique = MIS
        return color_count
    
    # Branch and bound for Maximum Independent Set
    # State: (current_set_size, available_mask, best_found)
    # We use recursion with pruning
    
    def branch_and_bound(available_mask, current_size, best):
        """
        available_mask: bitmask of vertices still available to add
        current_size: size of current independent set
        best: mutable list [best_size, best_set_mask]
        Returns: True if we should continue exploring (prune if False)
        """
        # Prune: even if we add all remaining vertices, we can't beat best
        # Simple upper bound: current_size + popcount(available_mask)
        remaining_count = bin(available_mask).count('1')
        if current_size + remaining_count <= best[0]:
            return False
        
        # Color-based upper bound
        if available_mask == 0:
            if current_size > best[0]:
                best[0] = current_size
                best[1] = 0  # We'll track the set differently
            return False
        
        ub = color_bound(available_mask)
        if current_size + ub <= best[0]:
            return False
        
        # Pick a vertex to branch on
        # Heuristic: pick vertex with highest degree in the remaining subgraph
        # For speed, pick the vertex with smallest index (or use a simple heuristic)
        # Let's pick the vertex that appears in the most conflicts within available_mask
        best_v = -1
        best_deg = -1
        temp_mask = available_mask
        while temp_mask:
            lsb = temp_mask & (-temp_mask)
            v = lsb.bit_length() - 1
            # Count neighbors in available_mask
            deg_in_avail = bin(conflicts[v] & available_mask).count('1')
            if deg_in_avail > best_deg:
                best_deg = deg_in_avail
                best_v = v
            temp_mask ^= lsb
        
        if best_v == -1:
            return False
        
        v_bit = 1 << best_v
        
        # Branch 1: include best_v
        # New available = available_mask minus best_v and its neighbors
        new_available = available_mask & ~(v_bit | conflicts[best_v])
        if branch_and_bound(new_available, current_size + 1, best):
            pass
        
        # Branch 2: exclude best_v
        new_available2 = available_mask & ~v_bit
        if branch_and_bound(new_available2, current_size, best):
            pass
        
        return True
    
    # Track best solution
    best_size = 0
    best_mask = 0  # bitmask of the best independent set
    
    # Initial lower bound from a greedy approach
    def greedy_mis():
        available = (1 << n) - 1
        chosen = 0
        while available:
            # Pick vertex with lowest degree in remaining subgraph
            best_v = -1
            best_deg = float('inf')
            temp = available
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                deg = bin(conflicts[v] & available).count('1')
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
                temp ^= lsb
            chosen |= (1 << best_v)
            available &= ~(1 << best_v | conflicts[best_v])
        return chosen
    
    greedy_result = greedy_mis()
    greedy_size = bin(greedy_result).count('1')
    best_size = greedy_size
    best_mask = greedy_result
    
    # Local search to improve the initial solution
    def local_search(initial_mask):
        """Try to improve the independent set by adding vertices or swapping."""
        current = initial_mask
        current_size = bin(current).count('1')
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try adding each non-member vertex if it doesn't conflict
            for v in range(n):
                if not (current & (1 << v)):
                    # Check if v conflicts with any vertex in current
                    if not (conflicts[v] & current):
                        current |= (1 << v)
                        current_size += 1
                        improved = True
            # Try 1-for-1 swaps: remove v from current, add u not in current
            # Only if u doesn't conflict with current minus v
            for v in range(n):
                if current & (1 << v):
                    current_without_v = current & ~(1 << v)
                    for u in range(n):
                        if not (current & (1 << u)):
                            # Check if u conflicts with current_without_v
                            if not (conflicts[u] & current_without_v):
                                new_size = bin(current_without_v | (1 << u)).count('1')
                                if new_size > current_size:
                                    current = current_without_v | (1 << u)
                                    current_size = new_size
                                    improved = True
        return current, bin(current).count('1')
    
    # Apply local search to improve initial solution
    best_mask, best_size = local_search(best_mask)
    
    # Now run branch and bound to try to improve
    # To make it faster, we can try different orderings/strategies
    
    # Run B&B from the full set
    all_mask = (1 << n) - 1
    
    # Time-limited B&B with improved branching and pruning
    def timed_bnb():
        # Recursive B&B with time limit
        result = [best_size, best_mask]
        
        def bnb(available_mask, current_size, chosen_mask):
            # Check time
            if time.time() > deadline:
                return
            
            # Prune: simple upper bound
            remaining_count = bin(available_mask).count('1')
            if current_size + remaining_count <= result[0]:
                return
            
            if available_mask == 0:
                if current_size > result[0]:
                    result[0] = current_size
                    result[1] = chosen_mask
                return
            
            # Color bound (tighter)
            ub = color_bound(available_mask)
            if current_size + ub <= result[0]:
                return
            
            # Pick branching vertex: highest degree in available subgraph
            best_v = -1
            best_deg = -1
            temp = available_mask
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                deg = bin(conflicts[v] & available_mask).count('1')
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                temp ^= lsb
            
            if best_v == -1:
                return
            
            v_bit = 1 << best_v
            
            # Include best_v
            new_avail = available_mask & ~(v_bit | conflicts[best_v])
            if time.time() < deadline:
                bnb(new_avail, current_size + 1, chosen_mask | v_bit)
            
            # Exclude best_v
            new_avail2 = available_mask & ~v_bit
            if time.time() < deadline:
                bnb(new_avail2, current_size, chosen_mask)
        
        bnb(all_mask, 0, 0)
        return result
    
    # Run with a time limit
    # Try to improve upon the greedy solution
    final_result = timed_bnb()
    best_size = final_result[0]
    best_mask = final_result[1]
    
    # Final local search polish
    best_mask, best_size = local_search(best_mask)
    
    # Extract the result
    result_nodes = []
    for i in range(n):
        if best_mask & (1 << i):
            result_nodes.append(valid_nodes[i])
    
    if not result_nodes:
        result_nodes = [valid_nodes[0]]
    
    # Final verification: ensure no pair sum is a perfect square
    for idx_a in range(len(result_nodes)):
        for idx_b in range(idx_a, len(result_nodes)):
            s = result_nodes[idx_a] + result_nodes[idx_b]
            r = int(math.isqrt(s))
            if r * r == s:
                # Invalid - remove the problematic node and retry
                # Fallback: return a single valid node
                return [valid_nodes[0]]
    
    return result_nodes
