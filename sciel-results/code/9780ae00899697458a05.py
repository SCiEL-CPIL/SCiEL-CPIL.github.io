# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adj(n, edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _edge_key(u, v):
    return (min(u, v), max(u, v))

def _hamiltonian_cycle_dfs(vertices, adj, used_edges, start, vset, k):
    """
    Find a Hamiltonian cycle on 'vertices' (size k) using only unused edges.
    Returns a list of vertices representing the cycle, or None.
    """
    path = [start]
    visited = {start}
    
    # Pre-compute adjacency within vset for speed
    # adj_local[v] = set of neighbors in vset
    
    # We use a simple DFS with backtracking.
    # Heuristic: order neighbors by degree in the remaining graph (Warnsdorff's rule-like)
    
    def get_candidates(current):
        candidates = []
        for nxt in adj[current]:
            if nxt not in visited and nxt in vset:
                e = _edge_key(current, nxt)
                if e not in used_edges:
                    # Calculate "degree" of nxt in the unvisited, unused-edge subgraph
                    deg = 0
                    for nnxt in adj[nxt]:
                        if nnxt not in visited and nnxt in vset:
                            e2 = _edge_key(nxt, nnxt)
                            if e2 not in used_edges:
                                deg += 1
                    
                    # Special check for the last step
                    if len(path) == k - 1:
                        # If we add nxt, we must be able to close to start
                        if start in adj[nxt]:
                            e3 = _edge_key(nxt, start)
                            if e3 not in used_edges:
                                candidates.append((0, nxt)) # High priority
                                continue
                        else:
                            # Cannot close, so this branch is a dead end if we are at the last step
                            # But we might not be at the last step yet? No, len(path)==k-1 means one more node fills it.
                            # So if we can't close, we shouldn't even consider this candidate?
                            # Actually, if we pick this candidate, path becomes length k. Then we check closure.
                            # If closure fails, we backtrack. So it's safe to include, but inefficient.
                            # Better to filter out if we know it fails.
                            # Since we are at k-1, picking nxt makes path length k.
                            # If start not in adj[nxt] or edge used, it will fail.
                            # So we can skip adding it to candidates to save time.
                            continue
                    
                    candidates.append((deg, nxt))
        
        # Sort by degree (ascending) - prefer nodes with fewer options (more constrained)
        candidates.sort(key=lambda x: x[0])
        return [c[1] for c in candidates]

    def dfs(current):
        if len(path) == k:
            # Check closure
            e = _edge_key(current, start)
            if start in adj[current] and e not in used_edges:
                return list(path)
            return None
        
        for nxt in get_candidates(current):
            e = _edge_key(current, nxt)
            if e not in used_edges:
                visited.add(nxt)
                path.append(nxt)
                result = dfs(nxt)
                if result is not None:
                    return result
                path.pop()
                visited.remove(nxt)
        return None
    
    return dfs(start)

def _find_max_disjoint_cycles(vertices, adj, all_edges_set, time_limit, max_cycles=10, seed=42):
    """
    Greedily find edge-disjoint Hamiltonian cycles on the fixed vertex set.
    Uses random restarts for the first cycle if needed, and tries different orders.
    """
    vertices = list(vertices)
    vset = set(vertices)
    k = len(vertices)
    start_time = time.time()
    
    best_cycles = []
    
    # Try multiple random permutations of start nodes and potentially different heuristics
    # For simplicity, we just try different start nodes for the first cycle,
    # and then proceed greedily.
    
    # Pre-compute a list of potential start nodes
    start_nodes = list(range(k))
    random.seed(seed)
    random.shuffle(start_nodes)
    
    # We'll try up to a certain number of random start sequences
    # But since the graph is dense, one good start might work.
    # Let's try the natural order first (0, 1, ...), then random.
    
    attempts = []
    # Natural order
    attempts.append(list(range(k)))
    # Random orders
    for _ in range(5):
        s = list(range(k))
        random.shuffle(s)
        attempts.append(s)
    
    for start_seq in attempts:
        if time.time() - start_time > time_limit:
            break
            
        used_edges = set()
        cycles = []
        
        for i in range(max_cycles):
            if time.time() - start_time > time_limit:
                break
            
            # Pick a start node. 
            # For the first cycle, we can try the specific start from the sequence.
            # For subsequent cycles, the "start" node doesn't matter as much for existence,
            # but might matter for the specific cycle found.
            # Let's just use min(vertex) for subsequent cycles for consistency, 
            # or cycle through the start_seq?
            # Actually, the DFS is deterministic given the start node and the current state of used_edges.
            # So if we use min(vset) for all, we get a deterministic set of cycles.
            # If we vary the start, we might get different sets.
            
            # Let's use the i-th element of start_seq as the start node for the i-th cycle?
            # No, the start node for cycle i should be chosen to maximize success.
            # Let's just use min(vset) for all cycles in this attempt.
            # The variation comes from the *first* cycle's start node in the outer loop?
            # No, _find_max_disjoint_cycles takes the whole set.
            
            # Let's refine: In this attempt, we force the first cycle to start at start_seq[0].
            # For subsequent cycles, we use min(vset).
            
            start_node = start_seq[0] if i == 0 else min(vset)
            
            cycle = _hamiltonian_cycle_dfs(vertices, adj, used_edges, start_node, vset, k)
            
            if cycle is None:
                break
            
            # Add edges to used
            valid = True
            for j in range(len(cycle)):
                u = cycle[j]
                v = cycle[(j + 1) % len(cycle)]
                e = _edge_key(u, v)
                if e in used_edges:
                    valid = False
                    break
                used_edges.add(e)
            
            if not valid:
                break
            
            cycles.append(cycle)
        
        if len(cycles) > len(best_cycles):
            best_cycles = cycles
    
    return best_cycles

def construct():
    start_time = time.time()
    deadline = 85  # Leave 5s margin for 90s limit
    
    n = P['n']
    edges = P['edges']
    all_edges_set = set()
    for u, v in edges:
        all_edges_set.add(_edge_key(u, v))
    
    adj = _build_adj(n, edges)
    vertices = list(range(n))
    
    # Try to find as many as possible
    # The previous program found 3.
    # Let's try harder with more random restarts.
    
    best_cycles = []
    
    # Run 1: Natural order / min start
    cycles = _find_max_disjoint_cycles(vertices, adj, all_edges_set, 10, max_cycles=10, seed=0)
    if len(cycles) > len(best_cycles):
        best_cycles = cycles
    
    # Run 2-5: Random starts
    for seed in [1, 2, 3, 4, 5]:
        if time.time() - start_time > deadline - 5:
            break
        cycles = _find_max_disjoint_cycles(vertices, adj, all_edges_set, 10, max_cycles=10, seed=seed)
        if len(cycles) > len(best_cycles):
            best_cycles = cycles
    
    # If we still only have few, try a more aggressive search with more time per attempt?
    # Or try different vertex subsets? No, contract says "same vertex set".
    
    # Convert to list of lists (or tuples)
    # Contract: "list of distinct vertices... closing edge implicit"
    # "sequences must be list or tuple"
    # "integers must be Python int"
    
    result = []
    for cycle in best_cycles[:10]:
        # Ensure ints are python ints
        clean_cycle = [int(v) for v in cycle]
        result.append(clean_cycle)
        
    return result

# EVOLVE-BLOCK-END
