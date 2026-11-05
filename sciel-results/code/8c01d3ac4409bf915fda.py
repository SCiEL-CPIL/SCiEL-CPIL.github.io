import time
import random

def construct():
    start_time = time.time()
    # Safety margin: evaluator kills at 90s, we stop at 50s to be safe
    deadline = start_time + 50.0
    
    # 1. Precomputation
    # Squares up to 512 (256+256)
    squares = set()
    k = 2
    while k * k <= 512:
        squares.add(k * k)
        k += 1
        
    # Forbidden nodes (diagonal constraint: 2*a is a square)
    forbidden = set()
    for a in range(1, 257):
        if 2 * a in squares:
            forbidden.add(a)
            
    valid_nodes = [a for a in range(1, 257) if a not in forbidden]
    valid_nodes_set = set(valid_nodes)
    
    # Build conflict graph
    # conflicts[v] contains u if u != v and u+v is a square
    conflicts = {v: set() for v in valid_nodes}
    for i, u in enumerate(valid_nodes):
        for v in valid_nodes[i+1:]:
            if u + v in squares:
                conflicts[u].add(v)
                conflicts[v].add(u)
                
    def can_add(node, current_set):
        # Check if node has no conflict with any node in current_set
        return conflicts[node].isdisjoint(current_set)
        
    def greedy_fill(current_set, available_nodes):
        """
        Greedily add nodes from available_nodes to current_set.
        available_nodes is modified in place.
        """
        added_any = True
        while added_any and time.time() < deadline:
            added_any = False
            # Iterate over a copy of available_nodes to allow modification
            # Sorting by degree might help, but O(N log N) per iteration might be costly if N is large.
            # Here N is small. Let's try sorting by degree ascending (least constrained first).
            # Actually, just random or arbitrary is fine for a heuristic.
            # Let's try to add the node with the smallest degree in the *remaining* graph?
            # That's expensive. 
            # Simple approach: iterate through list(available_nodes).
            # To improve quality, we can shuffle available_nodes before this call.
            for node in list(available_nodes):
                if can_add(node, current_set):
                    current_set.add(node)
                    available_nodes.remove(node)
                    added_any = True
                    if time.time() > deadline:
                        break
                    # Restart the scan because adding a node might block others, 
                    # but also might not. Restarting ensures we catch all possible adds.
                    break
                    
    def local_search(initial_set):
        """
        Perform local search starting from initial_set.
        Returns the best set found.
        """
        current_set = set(initial_set)
        current_available = valid_nodes_set - current_set
        
        # Initial greedy fill
        greedy_fill(current_set, current_available)
        
        best_set = set(current_set)
        best_size = len(current_set)
        
        # Iterative improvement
        # We try to remove a node and re-fill.
        # If size increases, we keep it.
        # To escape local optima, we use random restarts or random removals.
        
        stuck_count = 0
        while time.time() < deadline:
            improved = False
            
            # Try to add any available nodes
            greedy_fill(current_set, current_available)
            
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
                improved = True
                stuck_count = 0
            else:
                stuck_count += 1
                
            # If we are not improving, try a swap (remove one, add more)
            if current_set and time.time() < deadline:
                # Pick a node to remove
                # Heuristic: Remove a node that is part of many conflicts?
                # Or just random.
                # Let's try random first.
                node_to_remove = random.choice(list(current_set))
                
                # Remove it
                current_set.remove(node_to_remove)
                current_available.add(node_to_remove)
                
                # Re-fill
                greedy_fill(current_set, current_available)
                
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
                    improved = True
                    stuck_count = 0
                else:
                    # Revert
                    current_set.add(node_to_remove)
                    current_available.remove(node_to_remove)
                    
                    # If stuck for a while, do a random restart
                    if stuck_count > 50:
                        # Random restart: Construct a new random greedy solution
                        # Shuffle valid nodes
                        shuffled_nodes = valid_nodes[:]
                        random.shuffle(shuffled_nodes)
                        
                        new_set = set()
                        # Greedy construction
                        for node in shuffled_nodes:
                            if can_add(node, new_set):
                                new_set.add(node)
                        
                        # Refine
                        new_avail = valid_nodes_set - new_set
                        greedy_fill(new_set, new_avail)
                        
                        if len(new_set) > len(current_set):
                            current_set = new_set
                            current_available = new_avail
                        elif len(new_set) > best_size:
                            best_size = len(new_set)
                            best_set = new_set
                            
                        stuck_count = 0

        return best_set

    # 2. Generate initial candidates
    # Candidate 1: Mod 3 == 1
    cand1 = set(a for a in valid_nodes if a % 3 == 1)
    # Candidate 2: Mod 3 == 2 (excluding forbidden)
    cand2 = set(a for a in valid_nodes if a % 3 == 2)
    # Candidate 3: Mod 3 == 0 (excluding forbidden)
    cand3 = set(a for a in valid_nodes if a % 3 == 0)
    
    # Candidate 4: Random
    cand4 = set()
    
    best_overall = set()
    best_size_overall = 0
    
    # Run local search on each candidate
    for cand in [cand1, cand2, cand3]:
        if time.time() > deadline:
            break
        res = local_search(cand)
        if len(res) > best_size_overall:
            best_size_overall = len(res)
            best_overall = res
            
    # If time remains, run a few random restarts
    while time.time() < deadline - 5.0:
        # Random initial set
        random.shuffle(valid_nodes)
        res = local_search(set(valid_nodes[:10])) # Start small?
        # Or just random greedy
        shuffled = valid_nodes[:]
        random.shuffle(shuffled)
        new_set = set()
        for node in shuffled:
            if can_add(node, new_set):
                new_set.add(node)
        
        # Refine
        new_avail = valid_nodes_set - new_set
        greedy_fill(new_set, new_avail)
        
        # Local search on this
        res = local_search(new_set)
        
        if len(res) > best_size_overall:
            best_size_overall = len(res)
            best_overall = res

    return list(best_overall)
