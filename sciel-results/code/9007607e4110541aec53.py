import time
import random
import sys
from collections import deque

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
    
    # Best path found so far
    best_path = None
    best_len = -1
    
    def is_valid_snake(path):
        """Check if a list of vertices forms a valid snake."""
        if len(path) <= 1:
            return True
        if len(set(path)) != len(path):
            return False
        for i in range(len(path)):
            for j in range(i+1, len(path)):
                dist = (path[i] ^ path[j]).bit_count()
                if i + 1 == j:
                    if dist != 1:
                        return False
                else:
                    if dist < 2:
                        return False
        return True
    
    def greedy_extend(path, randomize=False):
        """Greedily extend a snake path."""
        path = list(path)
        visited = set(path)
        current = path[-1] if path else 0
        
        max_steps = 500
        for _ in range(max_steps):
            valid_nexts = []
            for nb in NEIGHBORS[current]:
                if nb in visited:
                    continue
                is_chord = False
                # Check chord condition: nb must not be adjacent to any path vertex except the last one
                for v in path[:-1]:
                    if (nb ^ v).bit_count() == 1:
                        is_chord = True
                        break
                if not is_chord:
                    valid_nexts.append(nb)
            
            if not valid_nexts:
                break
            
            if randomize:
                next_v = random.choice(valid_nexts)
            else:
                # Heuristic: pick the neighbor with fewest valid options (most constrained)
                def count_options(nb):
                    count = 0
                    for nb2 in NEIGHBORS[nb]:
                        if nb2 in visited:
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
                valid_nexts.sort(key=lambda x: count_options(x))
                next_v = valid_nexts[0]
            
            path.append(next_v)
            visited.add(next_v)
            current = next_v
        
        return path
    
    def backtracking_search(path, time_limit):
        """
        Constraint-based backtracking with MCV (Most Constrained Value) heuristic.
        Returns the longest snake found starting from the given path.
        """
        path = list(path)
        visited = set(path)
        
        # Precompute for each vertex in path, its neighbors (for fast chord check)
        # Actually, we'll just do the check inline for simplicity.
        
        def get_valid_nexts(current):
            """Get valid next vertices from current."""
            valids = []
            for nb in NEIGHBORS[current]:
                if nb in visited:
                    continue
                # Check chord: nb must not be adjacent to any vertex in path[:-1]
                is_valid = True
                for v in path[:-1]:
                    if (nb ^ v).bit_count() == 1:
                        is_valid = False
                        break
                if is_valid:
                    valids.append(nb)
            return valids
        
        # Iterative deepening with time limit
        # We'll use a stack-based approach to avoid recursion depth issues
        
        best_local = path[:]
        max_local_len = len(path)
        
        # Use a simple DFS with heuristic ordering
        # Stack items: (path, visited, current, valid_nexts, next_index)
        # To save memory, we'll just do recursive DFS with a time check
        
        sys.setrecursionlimit(10000)
        
        def dfs(path, visited, current, depth):
            nonlocal best_local, max_local_len
            
            if time.time() > time_limit:
                return
            
            # Update best if this path is longer
            if len(path) > max_local_len:
                max_local_len = len(path)
                best_local = path[:]
            
            # Get valid nexts
            valids = []
            for nb in NEIGHBORS[current]:
                if nb in visited:
                    continue
                is_valid = True
                for v in path[:-1]:
                    if (nb ^ v).bit_count() == 1:
                        is_valid = False
                        break
                if is_valid:
                    valids.append(nb)
            
            if not valids:
                return
            
            # Heuristic: sort by number of options from each candidate (MCV)
            def score(nb):
                count = 0
                for nb2 in NEIGHBORS[nb]:
                    if nb2 in visited:
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
            
            valids.sort(key=lambda x: (score(x), random.random()))
            
            # Try each valid next
            for nb in valids:
                if time.time() > time_limit:
                    return
                path.append(nb)
                visited.add(nb)
                dfs(path, visited, nb, depth + 1)
                path.pop()
                visited.remove(nb)
        
        dfs(path, visited, path[-1] if path else 0, 0)
        return best_local
    
    def local_search(path, time_limit):
        """
        Local search: try to improve the path by finding better extensions 
        or by reordering segments.
        """
        path = list(path)
        visited = set(path)
        best = path[:]
        best_len = len(path)
        
        # Try extending from the end
        ext = greedy_extend(path, randomize=True)
        if len(ext) > best_len:
            best = ext
            best_len = len(ext)
        
        # Try backtracking from the current path
        if time.time() < time_limit - 5:
            bt = backtracking_search(path, min(time_limit, time.time() + 10))
            if len(bt) > best_len:
                best = bt
                best_len = len(bt)
        
        return best
    
    # Phase 1: Generate initial snakes using greedy with different starts
    for i in range(50):
        if time.time() > deadline - 10:
            break
        
        start_v = random.randint(0, SIZE - 1)
        path = [start_v]
        path = greedy_extend(path, randomize=(i % 3 != 0))
        
        if len(path) - 1 > best_len:
            best_len = len(path) - 1
            best_path = path[:]
            
            # Validate
            if not is_valid_snake(best_path):
                print("Invalid snake found during initial generation!")
                best_path = [0]
                best_len = 0
    
    # Phase 2: Local search on the best path
    if time.time() < deadline - 15 and best_path:
        improved = local_search(best_path, min(deadline - 5, time.time() + 30))
        if len(improved) - 1 > best_len:
            best_len = len(improved) - 1
            best_path = improved
    
    # Phase 3: More backtracking searches from different starting points
    for i in range(20):
        if time.time() > deadline - 10:
            break
        
        # Start from a random vertex, do a quick greedy then backtrack
        start_v = random.randint(0, SIZE - 1)
        path = [start_v]
        path = greedy_extend(path, randomize=True)
        
        # Try to extend with backtracking
        if time.time() < deadline - 5:
            bt = backtracking_search(path, min(deadline - 2, time.time() + 5))
            if len(bt) - 1 > best_len:
                best_len = len(bt) - 1
                best_path = bt[:]
    
    # Phase 4: Final greedy extension on the best path
    if best_path and time.time() < deadline - 2:
        ext = greedy_extend(best_path, randomize=False)
        if len(ext) - 1 > best_len:
            best_len = len(ext) - 1
            best_path = ext
    
    if best_path is None:
        return [0]
    
    # Final validation
    if not is_valid_snake(best_path):
        # Fallback to a simple valid snake
        return [0, 1]
    
    return best_path

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
        for j in range(i+1, len(p)):
            if abs(i-j) > 1 and (p[i] ^ p[j]).bit_count() == 1:
                valid = False
                break
        if not valid:
            break
    print(f"Valid: {valid}")
