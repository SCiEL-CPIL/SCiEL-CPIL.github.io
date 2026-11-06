import time
import random
import sys

N = 9
SIZE = 1 << N
MASK = SIZE - 1

# Precompute neighbors for each vertex
NEIGHBORS = [[] for _ in range(SIZE)]
for v in range(SIZE):
    for b in range(N):
        NEIGHBORS[v].append(v ^ (1 << b))

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # We will try to find the longest snake.
    # Strategy: 
    # 1. Generate a good initial snake using a fast greedy heuristic (to have a baseline).
    # 2. Use a Backtracking Search with Forward Checking to try to extend or find longer snakes.
    #    Since finding the *longest* is NP-hard, we use a time-limited search that tries to maximize length.
    #    We can model this as: Find a path of length L. Binary search on L? Or just try to extend greedily 
    #    but with backtracking when stuck?
    
    # A more effective approach for "longest path" in small graphs with time limits:
    # Use a heuristic search (like A* or DFS with pruning) to find a long path.
    # However, the "snake" constraint (no chords) is very restrictive.
    
    # Let's use a randomized backtracking approach with "Look-Ahead" (Forward Checking).
    # State: (current_vertex, visited_set, path_list)
    # To make it fast, we represent visited_set as a bitset (integer) or a boolean array.
    # Given SIZE=512, a boolean array or a set of ints is fine. A bitset integer is very fast for "is visited".
    
    # We will perform multiple restarts with different random seeds/priorities.
    # In each run, we do a depth-first search, but we prioritize moves that leave the most options for the future.
    
    def greedy_initial():
        """Generate a reasonably long snake quickly using a greedy heuristic."""
        # Start at 0
        path = [0]
        visited = {0}
        current = 0
        
        # To make it deterministic but varied, we can shuffle neighbors slightly
        # But for a baseline, a simple greedy works.
        for _ in range(200): # Max possible is 512, but we won't get there
            # Find valid neighbors
            valid_nexts = []
            for nb in NEIGHBORS[current]:
                if nb in visited:
                    continue
                # Check if nb is adjacent to any vertex in path[:-1]
                # If yes, adding nb would create a chord (edge between nb and some path[k] where k < len(path)-1)
                # Actually, the condition is: non-consecutive vertices must differ in >= 2 bits.
                # So nb must NOT be adjacent to any path[k] for k in [0, len(path)-2].
                is_chord = False
                for v in path[:-1]:
                    if (nb ^ v).bit_count() == 1:
                        is_chord = True
                        break
                if not is_chord:
                    valid_nexts.append(nb)
            
            if not valid_nexts:
                break
            
            # Heuristic: Choose the neighbor that has the fewest valid *future* options?
            # Or just pick one randomly to allow diversity in restarts?
            # For the initial seed, let's pick one that minimizes the degree of the next node?
            # Simple: Pick the first one.
            next_v = valid_nexts[0]
            path.append(next_v)
            visited.add(next_v)
            current = next_v
            
        return path

    best_path = greedy_initial()
    best_len = len(best_path) - 1
    
    # Now, try to improve using Backtracking with Forward Checking.
    # We will try to find a snake of length > best_len.
    # We can do this by trying to extend the current best path, or restarting from scratch with a smarter search.
    
    # Let's implement a recursive DFS with pruning.
    # To avoid recursion depth issues and overhead, we might use an explicit stack, 
    # but Python recursion is okay for depth ~200 if we set limit.
    sys.setrecursionlimit(1000)
    
    def search(path, visited_set, current, depth, max_depth, time_limit):
        """
        Try to extend the path.
        Returns the longest path found in this branch.
        """
        if time.time() > time_limit:
            return path
        
        # Heuristic: Order neighbors by "safety" - how many valid moves they leave for the next step?
        # This is the "Most Constrained Value" heuristic.
        
        valid_nexts = []
        for nb in NEIGHBORS[current]:
            if nb in visited_set:
                continue
            
            # Check chord condition against path[:-1]
            # Optimization: We only need to check against the "boundary" of the path?
            # No, any previous vertex can form a chord.
            # However, if nb is adjacent to path[-2], it's invalid.
            # If nb is adjacent to path[-3], it's invalid.
            # ...
            # We can precompute for each vertex v, the set of vertices that would be "forbidden" 
            # if v is added? No, the forbidden set depends on the *entire* path.
            
            is_valid = True
            # We can optimize this check.
            # A vertex u is forbidden if it is in visited OR it is adjacent to any v in path[:-1].
            # Let's just do the check. It's O(L) per neighbor.
            for i in range(len(path) - 1):
                if (nb ^ path[i]).bit_count() == 1:
                    is_valid = False
                    break
            
            if is_valid:
                # Look-ahead: How many valid options does nb have?
                # This is expensive. Let's just collect valid nexts first.
                valid_nexts.append(nb)
        
        if not valid_nexts:
            return path
        
        # Sort valid_nexts by a heuristic:
        # Prefer neighbors that have fewer valid *subsequent* neighbors?
        # This helps us "force" the path into a dead end earlier, allowing backtracking to happen sooner
        # and explore other branches? Or does it help us find long paths?
        # Usually, "fail-first" (choose the most constrained variable) is good for finding *any* solution.
        # For *longest* path, we might want to keep options open.
        # Let's try: Sort by the number of valid neighbors of the neighbor (degree in remaining graph).
        
        def count_options(v):
            count = 0
            for nb2 in NEIGHBORS[v]:
                if nb2 in visited_set:
                    continue
                # Check if nb2 is adjacent to any vertex in path (excluding v, but v is not in path yet)
                # Actually, if we add v, the new path is path + [v].
                # The next step from v must not be adjacent to any vertex in path.
                # So for a candidate next step w from v, w must not be in visited_set U {v} 
                # and w must not be adjacent to any vertex in path.
                
                is_valid_w = True
                for p in path: # Check against all current path vertices
                    if (w ^ p).bit_count() == 1:
                        is_valid_w = False
                        break
                if is_valid_w and w not in visited_set and w != v:
                    count += 1
            return count
        
        # This look-ahead is O(N * L * N) per neighbor. 
        # L ~ 100, N=9. 9*100*9 = 8100 ops per neighbor. 9 neighbors. ~70k ops per step.
        # 100 steps -> 7 million ops. In Python, this might be too slow for deep searches within 115s.
        
        # Let's use a simpler heuristic: Random order, but maybe bias towards lower degree?
        # Or just random order to allow diversity.
        
        random.shuffle(valid_nexts)
        
        # Try each option
        for nb in valid_nexts:
            if time.time() > time_limit:
                break
            
            new_path = path + [nb]
            new_visited = visited_set | {nb}
            
            # Recursive call
            res = search(new_path, new_visited, nb, depth + 1, max_depth, time_limit)
            
            if len(res) > len(best_path_global):
                best_path_global[:] = res
                best_len_global[0] = len(res) - 1
            
            # If we found a very long path, we might stop early?
            # No, we want the longest.
            
            # Pruning: If we hit a dead end (len(res) == len(new_path)), 
            # and we've tried all, we return.
            
            # If we found a solution that is longer than current best, we update.
            # We continue searching to see if we can go even longer.
            
        return path

    # Global variables for the search to update
    best_path_global = list(best_path)
    best_len_global = [best_len]
    
    # We will run multiple independent searches with different random seeds and starting points.
    # To make the search effective, we can use "Iterative Deepening" or just random restarts.
    # Given the time limit, random restarts with a "fail-first" heuristic is often best.
    
    # Let's refine the search function to be iterative or use a better heuristic.
    # The recursive function above updates globals, which is messy. Let's return the best path found in the subtree.
    
    def search_best(path, visited_set, current, time_limit):
        """
        Returns the longest path found starting from this state.
        """
        if time.time() > time_limit:
            return path
        
        valid_nexts = []
        for nb in NEIGHBORS[current]:
            if nb in visited_set:
                continue
            
            is_valid = True
            for i in range(len(path) - 1):
                if (nb ^ path[i]).bit_count() == 1:
                    is_valid = False
                    break
            
            if is_valid:
                valid_nexts.append(nb)
        
        if not valid_nexts:
            return path
        
        # Heuristic: Sort by number of valid next-next steps (look-ahead depth 1)
        # This is the "Most Constrained" heuristic.
        # We calculate the number of valid moves from each candidate nb.
        
        def get_score(nb):
            count = 0
            for nb2 in NEIGHBORS[nb]:
                if nb2 in visited_set:
                    continue
                if nb2 == current: # Already visited
                    continue
                
                # Check if nb2 is adjacent to any vertex in path
                # If we add nb, the path becomes path + [nb].
                # The next vertex nb2 must not be adjacent to any vertex in path.
                # Note: nb2 is adjacent to nb (by definition). 
                # If nb2 is adjacent to any p in path, it creates a chord.
                # So we check against all p in path.
                
                is_valid = True
                for p in path:
                    if (nb2 ^ p).bit_count() == 1:
                        is_valid = False
                        break
                
                if is_valid:
                    count += 1
            return count
        
        # Calculate scores
        scored_nexts = [(nb, get_score(nb)) for nb in valid_nexts]
        
        # Sort by score ascending (fail-first: try the most constrained ones first)
        # This allows us to find dead ends quickly and backtrack.
        scored_nexts.sort(key=lambda x: x[1])
        
        best_sub_path = path
        max_len = len(path)
        
        for nb, _ in scored_nexts:
            if time.time() > time_limit:
                break
            
            new_path = path + [nb]
            new_visited = visited_set | {nb}
            
            res = search_best(new_path, new_visited, nb, time_limit)
            
            if len(res) > max_len:
                max_len = len(res)
                best_sub_path = res
            
            # If we found a very long path, we might want to explore other branches 
            # to see if we can go longer.
            
        return best_sub_path

    # Run multiple restarts
    # We allocate time to each restart.
    num_restarts = 20
    time_per_restart = 110.0 / num_restarts
    
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Random start
        start_v = random.randint(0, SIZE - 1)
        initial_path = [start_v]
        initial_visited = {start_v}
        
        # We can also try to extend the *current best path* by picking a random point and restarting?
        # No, that's complex. Just random restarts.
        
        # To add diversity, we can shuffle the neighbor order in the search function?
        # The search function currently uses a deterministic heuristic (sort by score).
        # We can add a small random tie-breaker.
        
        # Let's modify search_best to accept a random seed or use global random.
        # The current search_best uses `random` module implicitly if we shuffle? 
        # No, it sorts deterministically.
        # Let's add a random perturbation to the scores to break ties and introduce randomness.
        
        def search_best_perturbed(path, visited_set, current, time_limit, rnd):
            if time.time() > time_limit:
                return path
            
            valid_nexts = []
            for nb in NEIGHBORS[current]:
                if nb in visited_set:
                    continue
                
                is_valid = True
                for k in range(len(path) - 1):
                    if (nb ^ path[k]).bit_count() == 1:
                        is_valid = False
                        break
                
                if is_valid:
                    valid_nexts.append(nb)
            
            if not valid_nexts:
                return path
            
            def get_score(nb):
                count = 0
                for nb2 in NEIGHBORS[nb]:
                    if nb2 in visited_set:
                        continue
                    if nb2 == current:
                        continue
                    
                    is_valid = True
                    for p in path:
                        if (nb2 ^ p).bit_count() == 1:
                            is_valid = False
                            break
                    
                    if is_valid:
                        count += 1
                return count
            
            scored_nexts = [(nb, get_score(nb) + rnd.uniform(0, 0.1)) for nb in valid_nexts]
            scored_nexts.sort(key=lambda x: x[1])
            
            best_sub_path = path
            max_len = len(path)
            
            for nb, _ in scored_nexts:
                if time.time() > time_limit:
                    break
                
                new_path = path + [nb]
                new_visited = visited_set | {nb}
                
                res = search_best_perturbed(new_path, new_visited, nb, time_limit, rnd)
                
                if len(res) > max_len:
                    max_len = len(res)
                    best_sub_path = res
            
            return best_sub_path
        
        rnd = random.Random(i)
        
        # Give each restart a specific time slice
        # But we want to keep the global best.
        # So we run the search, and if it finds something longer, we update.
        
        # To avoid deep recursion taking too long in one branch, 
        # we can limit the depth of the *search tree* per restart?
        # No, the time limit handles it.
        
        # However, if one restart gets stuck in a deep, unproductive branch, it wastes time.
        # The "fail-first" heuristic helps with this.
        
        current_limit = time.time() + time_per_restart
        # But we also have the global deadline.
        current_limit = min(current_limit, deadline - 2)
        
        found_path = search_best_perturbed(initial_path, initial_visited, start_v, current_limit, rnd)
        
        if len(found_path) - 1 > best_len_global[0]:
            best_len_global[0] = len(found_path) - 1
            best_path_global = found_path
            
            # If we found a significantly better path, we might want to reset the time allocation?
            # No, just keep going.
            
            # Optional: Try to extend this new best path with a few more quick greedy steps?
            # The search should have done that.

    # Final check
    if best_path_global:
        return best_path_global
    else:
        return [0]

if __name__ == "__main__":
    p = construct()
    print(f"Length: {len(p)-1}")
    # Validate
    valid = True
    for i in range(len(p)-1):
        if (p[i] ^ p[i+1]).bit_count() != 1:
            valid = False
            break
    for i in range(len(p)):
        for j in range(len(p)):
            if abs(i-j) > 1 and (p[i] ^ p[j]).bit_count() == 1:
                valid = False
                break
        if not valid:
            break
    print(f"Valid: {valid}")
