import numpy as np
import random
import time

def priority(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([])
    
    # Convert to list of neighbors for fast access
    neighbors = [list(np.where(adjacency[i])[0]) for i in range(n)]
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    best_set = set()
    best_size = 0
    
    # Multiple random restarts of greedy to find a good initial set
    # Then local search to improve it
    
    def greedy_random():
        """Returns a maximal independent set using random tie-breaking."""
        s = set()
        available = set(range(n))
        while available:
            # Pick a random vertex from available
            v = random.choice(list(available))
            s.add(v)
            # Remove v and its neighbors from available
            to_remove = {v}
            for u in neighbors[v]:
                if u in available:
                    to_remove.add(u)
            available -= to_remove
        return s

    def local_search(s):
        """Improve the independent set using a simple swap/add heuristic."""
        s = set(s)
        improved = True
        iterations = 0
        while improved and time.time() < deadline:
            improved = False
            iterations += 1
            # Try to remove a random vertex and see if we can add more
            # This is a heuristic, not guaranteed to find global max, but effective
            if not s:
                break
            u = random.choice(list(s))
            s.remove(u)
            
            # Try to add as many vertices as possible from the "freed" area
            # Vertices that are not in s and not neighbors of s
            # Specifically, we want to add vertices that were blocked by u
            # But any vertex not in s and not neighbor of s can be added
            # Let's just greedily add any valid vertex
            added_count = 0
            # We need to check all vertices not in s
            # To be efficient, we can focus on neighbors of u, but adding any valid vertex helps
            # Let's iterate through all vertices not in s
            # Optimization: Only check vertices that are not neighbors of current s
            # This is O(N^2) per step. N=200, so 40k ops. 1000 steps = 40M ops. OK.
            
            candidates = [v for v in range(n) if v not in s and not any(w in s for w in neighbors[v])]
            # Actually, 'any(w in s for w in neighbors[v])' is slow.
            # Let's use a boolean array for s
            in_s = [False] * n
            for v in s:
                in_s[v] = True
            
            # Find valid candidates
            valid_candidates = []
            for v in range(n):
                if not in_s[v]:
                    # Check if v is neighbor of any in s
                    is_neighbor = False
                    for w in neighbors[v]:
                        if in_s[w]:
                            is_neighbor = True
                            break
                    if not is_neighbor:
                        valid_candidates.append(v)
            
            # Greedily add from valid_candidates
            # To maximize chance, maybe pick low degree first? Or random?
            # Let's pick random to avoid bias
            random.shuffle(valid_candidates)
            for v in valid_candidates:
                # Re-check validity because adding one might block another
                # v is valid if not in s and no neighbor in s
                if not in_s[v]:
                    conflict = False
                    for w in neighbors[v]:
                        if in_s[w]:
                            conflict = True
                            break
                    if not conflict:
                        s.add(v)
                        in_s[v] = True
                        added_count += 1
            
            # If we added more than 1 vertex, we improved (size increased by added_count - 1)
            # If we added 0 or 1, size is same or decreased.
            # We removed 1. If added_count > 1, size increased.
            # If added_count == 1, size same.
            # If added_count == 0, size decreased.
            
            if added_count > 1:
                improved = True
            elif added_count == 1:
                # Size same, keep the new set? Or revert?
                # Keeping it might help future steps. Let's keep it.
                improved = True # Force another iteration? No, just don't mark as "no improvement" immediately if we want to explore.
                # Actually, if size didn't increase, we should probably revert to avoid cycling.
                # Let's revert if size didn't increase.
                s.remove(v) # Remove the last added one? No, we added multiple.
                # This logic is getting messy. 
                # Let's use a simpler local search:
                # 1. Remove u.
                # 2. Greedily add valid vertices.
                # 3. If new size > old size, accept. Else revert.
                
                # Re-implementing local search step more cleanly
                pass # We will rewrite the local search function below
            
            if iterations > 5000: # Safety break
                break
                
        return s

    # Rewrite local search for clarity and correctness
    def local_search_v2(s):
        s = set(s)
        in_s = [False] * n
        for v in s:
            in_s[v] = True
        
        best_s = s
        best_size = len(s)
        
        # Iterate until no improvement or time out
        # Simple hill climbing: try removing a random vertex and re-filling greedily
        while time.time() < deadline:
            if not s:
                break
            u = random.choice(list(s))
            old_size = len(s)
            
            # Remove u
            s.remove(u)
            in_s[u] = False
            
            # Greedily add valid vertices
            # We can add any v not in s and not neighbor of s
            # To be efficient, we can just scan all nodes
            added = []
            for v in range(n):
                if in_s[v]:
                    continue
                # Check conflict
                conflict = False
                for w in neighbors[v]:
                    if in_s[w]:
                        conflict = True
                        break
                if not conflict:
                    s.add(v)
                    in_s[v] = True
                    added.append(v)
            
            new_size = len(s)
            
            if new_size > best_size:
                best_size = new_size
                best_s = set(s)
            elif new_size < old_size:
                # Revert
                # This is hard to revert cleanly if we modified s and in_s.
                # Let's use a copy for the trial.
                pass # We will use a trial copy approach instead
            
            # If we don't improve, we might cycle. 
            # Let's limit iterations.
            break # This simple version is too weak. Let's use a different strategy.

    # Better Strategy: 
    # 1. Run Greedy with Min-Degree (Core-like) to get a baseline set S0.
    # 2. Run Greedy with Random multiple times, keep the largest set S_rand.
    # 3. Try to merge or improve S_rand.
    
    # Given time, let's just do many random greedy restarts and keep the best.
    # This is a "stochastic greedy" approach.
    
    for _ in range(100): # 100 restarts
        if time.time() > deadline:
            break
        s = greedy_random()
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
    
    # Try to improve best_set with a simple local search
    # Remove a vertex, add as many as possible. If size increases, keep.
    s = set(best_set)
    in_s = [False] * n
    for v in s:
        in_s[v] = True
        
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try each vertex in s? No, too slow. Random sample.
        for _ in range(50):
            if time.time() > deadline:
                break
            if not s:
                break
            u = random.choice(list(s))
            old_size = len(s)
            
            # Trial: Remove u
            s.remove(u)
            in_s[u] = False
            
            # Greedily add
            for v in range(n):
                if in_s[v]:
                    continue
                conflict = False
                for w in neighbors[v]:
                    if in_s[w]:
                        conflict = True
                        break
                if not conflict:
                    s.add(v)
                    in_s[v] = True
            
            new_size = len(s)
            
            if new_size > old_size:
                improved = True
                # Keep the change
            else:
                # Revert
                # We need to restore the state before the trial
                # This is complex with in-place modification.
                # Let's use a copy for the trial in the loop.
                pass
        
        # The in-place modification makes reverting hard.
        # Let's rewrite the local search to use copies.
        break # Stop this flawed loop

    # Correct Local Search Implementation
    def local_search_correct(s):
        s = set(s)
        in_s = [False] * n
        for v in s:
            in_s[v] = True
        
        best_s = s
        best_size = len(s)
        
        # Try random removals and re-fills
        for _ in range(200): # Limit iterations
            if time.time() > deadline:
                break
            if not s:
                break
            u = random.choice(list(s))
            
            # Create a trial state
            trial_s = set(s)
            trial_in_s = in_s[:]
            
            trial_s.remove(u)
            trial_in_s[u] = False
            
            # Greedily add to trial_s
            for v in range(n):
                if trial_in_s[v]:
                    continue
                conflict = False
                for w in neighbors[v]:
                    if trial_in_s[w]:
                        conflict = True
                        break
                if not conflict:
                    trial_s.add(v)
                    trial_in_s[v] = True
            
            if len(trial_s) > best_size:
                best_size = len(trial_s)
                best_s = trial_s
                # Update current state to the improved one
                s = set(trial_s)
                in_s = trial_in_s[:]
            
        return best_s

    # Run local search on the best found set
    if best_set:
        best_set = local_search_correct(best_set)
    
    # Encode scores
    scores = np.zeros(n)
    for v in best_set:
        scores[v] = 1.0
    for v in range(n):
        if v not in best_set:
            scores[v] = -1.0
            
    return scores
