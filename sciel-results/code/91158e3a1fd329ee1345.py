"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    """
    Find the largest subset S of [1..256] such that for all a<=b in S,
    a+b is never a perfect square (including a=b, i.e., 2a is never a square).
    
    Uses min-degree greedy with randomized restarts plus local search (swap/add).
    """
    N = 256
    start_time = time.time()
    deadline = start_time + 75.0  # safety margin before 90s kill
    
    # Precompute: for each x in [1..256], which y in [1..256] conflict with x?
    # Conflict: x + y is a perfect square. Note: since we only consider a<=b,
    # but the constraint is symmetric in the sense that if a+b is a square,
    # both a and b conflict. We build an undirected conflict graph.
    # Also, a conflicts with itself if 2a is a perfect square.
    
    max_sum = 2 * N  # 512
    # Perfect squares up to 512
    squares = set()
    for s in range(1, int(math.isqrt(max_sum)) + 1):
        squares.add(s * s)
    
    # conflict[i] = set of j such that i + j is a perfect square
    # i and j are in range 1..N
    conflict = [set() for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            if i + j in squares:
                conflict[i].add(j)
    
    # Self-conflict: i conflicts with itself if 2*i is a square
    # This is already captured: conflict[i] contains i if 2*i is a square.
    
    # We want a maximum independent set in this graph.
    # Strategy: min-degree greedy with randomized restarts + local search.
    
    best_set = set()
    best_size = 0
    
    def is_valid_independent(s):
        """Check if set s is an independent set."""
        for a in s:
            if a in conflict[a]:  # self-conflict
                return False
            for b in s:
                if b > a and b in conflict[a]:
                    return False
        return True
    
    def greedy_min_degree(nodes, rng):
        """
        Greedy: repeatedly pick the node with minimum degree (within remaining nodes),
        with random tie-breaking.
        Returns the set of selected nodes.
        """
        remaining = set(nodes)
        selected = set()
        
        while remaining:
            # For each node in remaining, compute its degree within remaining
            # (including self-conflict: if node is in conflict[node], it has a self-loop)
            best_node = None
            best_deg = float('inf')
            candidates = []
            for node in remaining:
                # Degree = number of neighbors in remaining + (1 if self-conflict)
                deg = len(conflict[node] & remaining)
                if node in conflict[node]:
                    # self-conflict means this node can't be selected at all
                    # Actually if node has self-conflict, it can never be in an independent set
                    deg = float('inf')  # effectively unselectable
                if deg < best_deg:
                    best_deg = deg
                    candidates = [node]
                elif deg == best_deg:
                    candidates.append(node)
            
            if best_deg == float('inf'):
                # All remaining nodes have self-conflict or are fully connected
                break
            
            node = rng.choice(candidates)
            selected.add(node)
            # Remove node and all its neighbors from remaining
            remaining.discard(node)
            remaining -= conflict[node]
        
        return selected
    
    def local_search(s, rng, time_limit):
        """
        Try to improve the independent set by:
        1. Adding nodes that don't conflict.
        2. Swapping: remove a node, add a better one.
        """
        s = set(s)
        in_set = set(s)
        
        def try_add(node):
            """Try to add node to the set."""
            if node in in_set:
                return False
            if node in conflict[node]:
                return False
            if conflict[node] & in_set:
                return False
            in_set.add(node)
            return True
        
        def try_remove(node):
            in_set.discard(node)
        
        # First, try to add any node that can be added
        changed = True
        while changed:
            changed = False
            for node in range(1, N + 1):
                if try_add(node):
                    changed = True
        
        # Now try swaps: for each node in the set, remove it and see if we can add more
        improved = True
        while improved:
            improved = False
            current_size = len(in_set)
            # Shuffle order for randomization
            nodes_list = list(in_set)
            rng.shuffle(nodes_list)
            for node in nodes_list:
                if time.time() > time_limit:
                    return in_set
                # Temporarily remove node
                in_set.discard(node)
                # Try to add nodes that were previously blocked
                added = set()
                for candidate in range(1, N + 1):
                    if candidate in in_set:
                        continue
                    if candidate in conflict[candidate]:
                        continue
                    if conflict[candidate] & in_set:
                        continue
                    in_set.add(candidate)
                    added.add(candidate)
                
                if len(in_set) > current_size:
                    improved = True
                    # Keep this improvement
                else:
                    # Revert
                    in_set.discard(node)
                    for a in added:
                        in_set.discard(a)
        
        # Try to add more
        changed = True
        while changed:
            changed = False
            for node in range(1, N + 1):
                if try_add(node):
                    changed = True
        
        return in_set
    
    # Remove nodes with self-conflict from consideration
    valid_nodes = set(range(1, N + 1))
    for i in range(1, N + 1):
        if i in conflict[i]:
            valid_nodes.discard(i)
    
    # Multiple randomized restarts
    num_restarts = 200
    rng = random.Random(42)
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # For each restart, use a different random ordering
        nodes_list = list(valid_nodes)
        rng.shuffle(nodes_list)
        
        # Greedy min-degree
        s = greedy_min_degree(set(nodes_list), rng)
        
        # Local search with time limit
        ts_limit = min(time.time() + 0.3, deadline - 3.0)
        s = local_search(s, rng, ts_limit)
        
        if len(s) > best_size:
            best_size = len(s)
            best_set = set(s)
    
    # Final verification
    if not is_valid_independent(best_set):
        # Fallback: simple valid set
        best_set = set()
        for i in range(1, N + 1):
            if i in conflict[i]:
                continue
            if conflict[i] & best_set:
                continue
            best_set.add(i)
        best_size = len(best_set)
    
    if not best_set:
        # Shouldn't happen, but fallback
        best_set = {1}
    
    return list(best_set)
# EVOLVE-BLOCK-END
