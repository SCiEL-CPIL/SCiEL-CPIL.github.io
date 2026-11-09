"""
Combined strategy: Heuristic DFS (Warnsdorff-ordered) with Backtracking + Refinement.
This structural change moves from pure greedy to a search that can escape local optima.
"""
import time
import random
import sys

# Increase recursion limit for DFS
sys.setrecursionlimit(10000)

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

N = P['n']
_adj = [set() for _ in range(N)]
for u, v in P['edges']:
    _adj[u].add(v)
    _adj[v].add(u)

# Precompute degrees and neighbor lists
_deg = [len(_adj[i]) for i in range(N)]
_nbrs = [list(_adj[i]) for i in range(N)]

# Find connected components
def find_components():
    visited = [False] * N
    comps = []
    for i in range(N):
        if not visited[i]:
            comp = []
            stack = [i]
            visited[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for nb in _adj[node]:
                    if not visited[nb]:
                        visited[nb] = True
                        stack.append(nb)
            comps.append(comp)
    return comps

_components = find_components()
# Sort by size, largest first
_components.sort(key=len, reverse=True)
_largest_comp = _components[0] if _components else []

# Global best path
_best_path = []
_deadline = 0

def _warnsdorff_score(node, seen):
    """Calculate the number of unvisited neighbors for a node."""
    return sum(1 for nb in _nbrs[node] if nb not in seen)

def _dfs_search(start, seen, path, depth=0):
    """
    Heuristic DFS with backtracking.
    Orders neighbors by Warnsdorff score (fewest unvisited neighbors first).
    """
    global _best_path, _deadline
    
    # Update best path if current is longer
    if len(path) > len(_best_path):
        _best_path = list(path)
        
    # Time check
    if time.time() > _deadline:
        return

    # Get unvisited neighbors
    cur = path[-1]
    cands = [v for v in _nbrs[cur] if v not in seen]
    
    if not cands:
        return

    # Sort candidates by Warnsdorff heuristic:
    # Prefer neighbors with fewer unvisited neighbors (to avoid getting stuck in high-degree areas)
    # Tie-break by node ID for determinism in sorting, though we might shuffle later if needed.
    # For DFS, we want to explore the "most constrained" first.
    cands.sort(key=lambda v: (_warnsdorff_score(v, seen), v))
    
    # To add randomness and escape local optima, we can sometimes shuffle or pick randomly from top K.
    # However, pure heuristic DFS is often very strong. Let's stick to strict heuristic first,
    # but we call this function from multiple random starts.
    
    for v in cands:
        seen.add(v)
        path.append(v)
        _dfs_search(v, seen, path, depth + 1)
        path.pop()
        seen.remove(v)
        
        # Early termination if we hit time limit
        if time.time() > _deadline:
            return

def _refine_path(path):
    """
    Refine the path by trying to extend ends and insert unvisited vertices.
    """
    if not path:
        return path
        
    seen = set(path)
    improved = True
    while improved:
        improved = False
        
        # 1. Try to extend from both ends using Warnsdorff
        # Extend tail
        while True:
            cur = path[-1]
            cands = [v for v in _nbrs[cur] if v not in seen]
            if not cands:
                break
            best = min(cands, key=lambda v: (_warnsdorff_score(v, seen), v))
            path.append(best)
            seen.add(best)
            improved = True
            
        # Extend head
        while True:
            cur = path[0]
            cands = [v for v in _nbrs[cur] if v not in seen]
            if not cands:
                break
            best = min(cands, key=lambda v: (_warnsdorff_score(v, seen), v))
            path.insert(0, best)
            seen.add(best)
            improved = True

        # 2. Try to insert an unvisited vertex into the path
        unvisited = [v for v in range(N) if v not in seen]
        inserted = False
        for v in unvisited:
            for i in range(len(path) - 1):
                a, b = path[i], path[i+1]
                if v in _adj[a] and v in _adj[b]:
                    path.insert(i+1, v)
                    seen.add(v)
                    inserted = True
                    break
            if inserted:
                break
        
        if inserted:
            improved = True
            
    return path

def construct():
    global _best_path, _deadline
    
    t0 = time.time()
    # Safety margin: 90s limit, stop at 85s to ensure return
    _deadline = t0 + 85.0
    
    _best_path = []
    
    if not _largest_comp:
        return [0] if N > 0 else []
        
    # Phase 1: Heuristic DFS from multiple random starts
    # We prioritize starts that are in the largest component
    # We use a fixed number of starts or run until time limit
    # Since DFS is fast for N=100 with good heuristics, we can do many restarts
    
    rng = random.Random(748) # Use task seed for reproducibility of search order? Or just 0?
    # The task says "pinned graph", so graph is fixed. Search strategy can be deterministic or random.
    # Randomness helps escape local optima in the search space.
    
    # Strategy: 
    # 1. Try DFS from high-degree nodes first (likely in the core of the giant component)
    # 2. Then random nodes.
    
    starts = list(_largest_comp)
    # Sort by degree descending for initial batch
    starts.sort(key=lambda x: -_deg[x])
    
    # We will iterate through starts. If a start yields a path of length L, 
    # and we have a better one, we continue.
    # To avoid spending too much time on one start, we can limit the DFS depth or just rely on time check.
    # The time check inside DFS is global, so it will stop any DFS if time is up.
    
    # We'll do a few passes.
    # Pass 1: High degree starts
    for start in starts[:20]: # Top 20 high degree
        if time.time() > _deadline - 5.0:
            break
        _dfs_search(start, {start}, [start])
        
    # Pass 2: Random starts from the largest component
    # Shuffle the rest
    remaining = starts[20:]
    rng.shuffle(remaining)
    
    for start in remaining:
        if time.time() > _deadline - 5.0:
            break
        _dfs_search(start, {start}, [start])
        
    # Phase 2: Refine the best path found
    if _best_path:
        refined = _refine_path(_best_path)
        if len(refined) > len(_best_path):
            _best_path = refined
            
    # Phase 3: If we still have time, try more random restarts with randomized neighbor ordering
    # This helps if the heuristic ordering was too deterministic and got stuck.
    if time.time() < _deadline - 10.0:
        # Modify DFS to use randomization? 
        # Instead of modifying the core DFS, we can just run more restarts.
        # Or we can implement a randomized version.
        
        def _dfs_randomized(start, seen, path):
            global _best_path, _deadline
            if len(path) > len(_best_path):
                _best_path = list(path)
            if time.time() > _deadline:
                return
                
            cur = path[-1]
            cands = [v for v in _nbrs[cur] if v not in seen]
            if not cands:
                return
            
            # Sort by heuristic, but pick randomly from top 3 if available
            cands.sort(key=lambda v: (_warnsdorff_score(v, seen), v))
            
            # Randomize the order of the top few to explore different branches
            top_k = min(3, len(cands))
            # We will try all candidates, but in a randomized order for the top ones?
            # No, standard DFS tries all. The order determines which path is found first.
            # To escape local optima, we can shuffle the entire candidate list with some probability
            # or just rely on multiple restarts.
            # Let's just do more restarts.
            
            for v in cands:
                seen.add(v)
                path.append(v)
                _dfs_randomized(v, seen, path)
                path.pop()
                seen.remove(v)
                if time.time() > _deadline:
                    return

        # Run randomized restarts
        for _ in range(50): # Limit iterations to avoid overhead
            if time.time() > _deadline - 2.0:
                break
            start = rng.choice(_largest_comp)
            _dfs_randomized(start, {start}, [start])
            
        # Final refinement
        if _best_path:
            refined = _refine_path(_best_path)
            if len(refined) > len(_best_path):
                _best_path = refined

    if not _best_path:
        # Fallback: return a single vertex or a simple edge if available
        if _largest_comp:
            _best_path = [_largest_comp[0]]
        else:
            _best_path = [0]
            
    return _best_path

# Ensure the function is named construct and returns a list
# The code above defines construct()

# For local testing/debugging, one might call construct()
if __name__ == "__main__":
    path = construct()
    print(f"Path length: {len(path)}")
    print(f"Path: {path}")
    # Verify validity
    for i in range(len(path) - 1):
        if path[i+1] not in _adj[path[i]]:
            raise ValueError(f"Invalid edge: {path[i]} - {path[i+1]}")
    if len(set(path)) != len(path):
        raise ValueError("Path is not simple")
