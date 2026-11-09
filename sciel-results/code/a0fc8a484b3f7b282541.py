import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 256
    
    # Perfect squares up to 2*N
    squares = set()
    i = 2
    while i * i <= 2 * N:
        squares.add(i * i)
        i += 1
    
    # Self-conflict: 2a is a perfect square
    self_conflict = set()
    for a in range(1, N + 1):
        if 2 * a in squares:
            self_conflict.add(a)
    
    # Valid nodes (no self-conflict)
    valid_nodes = [v for v in range(1, N + 1) if v not in self_conflict]
    node_set = set(valid_nodes)
    
    # Build adjacency: adj[v] = set of u in valid_nodes, u != v, such that u+v in squares
    adj = {v: set() for v in valid_nodes}
    for i_idx, v in enumerate(valid_nodes):
        for u in valid_nodes[i_idx + 1:]:
            if v + u in squares:
                adj[v].add(u)
                adj[u].add(v)
    
    best_set = set()
    best_size = 0
    
    # Greedy independent set with multiple random orderings
    def greedy_independent_set(order):
        selected = set()
        for v in order:
            if v not in selected:
                # Check if v conflicts with any already selected
                conflict = False
                for u in adj[v]:
                    if u in selected:
                        conflict = True
                        break
                if not conflict:
                    selected.add(v)
        return selected
    
    # Try several random orderings
    rng = random.Random(42)
    for trial in range(500):
        if time.time() > deadline - 5.0:
            break
        order = valid_nodes[:]
        rng.shuffle(order)
        s = greedy_independent_set(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = set(s)
    
    # Local search: try to add nodes, swap nodes
    def is_valid_set(s):
        for a in s:
            if 2 * a in squares:
                return False
            for b in s:
                if a < b and (a + b) in squares:
                    return False
        return True
    
    # Verify best_set
    if not is_valid_set(best_set):
        # Fallback: simple greedy
        best_set = greedy_independent_set(valid_nodes)
        best_size = len(best_set)
    
    # Local search with swaps
    current_set = set(best_set)
    current_size = len(current_set)
    
    def conflicts_with_set(v, s):
        """Check if v conflicts with any element in s"""
        if 2 * v in squares:
            return True
        for u in s:
            if v != u and (v + u) in squares:
                return True
        return False
    
    # Iterative local search
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        
        # Try adding a node
        for v in valid_nodes:
            if v not in current_set and not conflicts_with_set(v, current_set):
                current_set.add(v)
                current_size += 1
                improved = True
                if time.time() > deadline - 5.0:
                    break
        
        if time.time() > deadline - 5.0:
            break
        
        # Try swaps: remove one, add two
        # For efficiency, try removing each node and see if we can add more
        removed_list = list(current_set)
        rng.shuffle(removed_list)
        for v_remove in removed_list:
            if time.time() > deadline - 5.0:
                break
            if v_remove not in current_set:
                continue
            # Temporarily remove
            current_set.remove(v_remove)
            
            # Try to add as many nodes as possible
            added = []
            for v_add in valid_nodes:
                if v_add not in current_set and not conflicts_with_set(v_add, current_set):
                    current_set.add(v_add)
                    added.append(v_add)
            
            if len(added) > 1:
                improved = True
            else:
                # Revert
                current_set.add(v_remove)
                for v_add in added:
                    current_set.discard(v_add)
    
    if len(current_set) > best_size:
        best_set = current_set
        best_size = len(current_set)
    
    # Final verification
    if not is_valid_set(best_set):
        # Rebuild from scratch with simple approach
        best_set = greedy_independent_set(valid_nodes)
    
    return sorted(best_set)
