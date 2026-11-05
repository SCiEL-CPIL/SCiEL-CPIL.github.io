import time
from random import Random
from collections import deque
import itertools
import numpy as np

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    HAS_SCIPY_MILP = True
except ImportError:
    HAS_SCIPY_MILP = False

def instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def find_cycle_dfs(adj, start, remaining_edges, rng):
    """
    Finds a cycle in the graph using DFS.
    Returns a list of vertices [v0, v1, ..., vk] such that (vk, v0) is an edge.
    The cycle is simple.
    """
    visited = {start: -1}
    path = [start]
    
    def dfs(u):
        # Shuffle neighbors for randomization
        neighbors = list(adj[u])
        rng.shuffle(neighbors)
        
        for v in neighbors:
            e = (u, v) if u < v else (v, u)
            if e not in remaining_edges:
                continue
            
            if v == start and len(path) >= 2:
                # Found a cycle containing start
                return path[:]
            
            if v not in visited:
                visited[v] = u
                path.append(v)
                result = dfs(v)
                if result:
                    return result
                path.pop()
                del visited[v]
            
            # If v is visited and not parent, and not start, we have a cycle 
            # but extracting it from the current path is complex if it's not 
            # a back-edge to the immediate ancestor in a way that forms a simple 
            # cycle easily identifiable from the path list alone without 
            # checking if v is in path.
            # Since we want a simple cycle, and we are doing DFS, 
            # a back edge to any ancestor forms a cycle.
            # However, the above check `if v == start` only catches cycles 
            # returning to the root of this DFS tree.
            # To find ANY cycle, we should check if v is in visited and v != parent.
            # But extracting the cycle requires knowing the path from v to u.
            # Let's stick to finding cycles that close back to the start node 
            # of the DFS call, or use a more robust method.
            
            # A simpler robust method: 
            # If we encounter a visited node v (v != start) that is in the current path, 
            # we have a cycle.
            if v in visited and v != start:
                # Check if v is in current path
                if v in path:
                    # Extract cycle
                    idx = path.index(v)
                    cycle = path[idx:]
                    # The edge (u, v) closes the cycle
                    return cycle
            
    return dfs(start)

def generate_cycle_pool(adj, remaining_edges, n, num_cycles=200, rng=None):
    """
    Generates a pool of candidate cycles.
    """
    if rng is None:
        rng = Random(42)
    
    cycles = []
    seen_cycles = set()
    
    # Try to find cycles starting from each node multiple times
    for _ in range(num_cycles):
        # Pick a random start node with degree >= 2
        candidates = [u for u in range(n) if len(adj[u]) >= 2]
        if not candidates:
            break
        start = rng.choice(candidates)
        cycle = find_cycle_dfs(adj, start, remaining_edges, rng)
        if cycle and len(cycle) >= 3:
            # Normalize cycle representation for deduplication
            # Find the smallest vertex in the cycle and rotate
            min_idx = cycle.index(min(cycle))
            rotated = cycle[min_idx:] + cycle[:min_idx]
            # Check reverse direction
            rev = list(reversed(rotated))
            min_idx_rev = rev.index(min(rev))
            rotated_rev = rev[min_idx_rev:] + rev[:min_idx_rev]
            
            key = tuple(rotated) if tuple(rotated) < tuple(rotated_rev) else tuple(rotated_rev)
            
            if key not in seen_cycles:
                seen_cycles.add(key)
                cycles.append(cycle)
    
    # Also try to find longer cycles by "splicing" or just relying on the random DFS
    # A simple 2-opt improvement on the found cycles? 
    # Actually, for the ILP, we just need a diverse set of valid cycles.
    # The ILP will pick the best combination.
    
    return cycles

def solve_decomposition_ilp(edges, n, deadline):
    if not edges:
        return []
    
    remaining_edges = set(tuple(sorted(e)) for e in edges)
    adj = {u: set() for u in range(n)}
    for u, v in remaining_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Generate cycle pool
    # We want a mix of short and long cycles.
    # Let's generate a pool of size proportional to number of edges
    num_candidates = min(500, len(remaining_edges) * 2)
    
    rng = Random(123)
    candidate_cycles = generate_cycle_pool(adj, remaining_edges, n, num_cycles=num_candidates, rng=rng)
    
    # If no cycles found, return all edges as singletons
    if not candidate_cycles:
        return [[e[0], e[1]] for e in edges]
    
    # Build ILP
    # Variables:
    # x_i for each candidate cycle i (binary)
    # y_e for each edge e (binary, indicating if edge is left as singleton)
    #
    # Objective: Minimize sum(x_i) + sum(y_e)
    #
    # Constraints:
    # For each edge e:
    #   sum(x_i for i where e in cycle_i) + y_e = 1
    #
    # This ensures every edge is covered exactly once by either a cycle or a singleton.
    
    m = len(candidate_cycles)
    E = list(remaining_edges)
    k = len(E)
    
    # Map edge to index
    edge_to_idx = {e: i for i, e in enumerate(E)}
    
    # Cost vector
    # c = [1 for x_i] + [1 for y_e]
    c = np.zeros(m + k)
    c[:m] = 1.0
    c[m:] = 1.0
    
    # Constraint matrix A
    # A is k x (m + k)
    A = np.zeros((k, m + k))
    
    # Fill cycle columns
    for i, cycle in enumerate(candidate_cycles):
        for j in range(len(cycle) - 1):
            u, v = cycle[j], cycle[j+1]
            e = (u, v) if u < v else (v, u)
            if e in edge_to_idx:
                A[edge_to_idx[e], i] = 1.0
        # Closing edge
        u, v = cycle[-1], cycle[0]
        e = (u, v) if u < v else (v, u)
        if e in edge_to_idx:
            A[edge_to_idx[e], i] = 1.0
            
    # Fill singleton columns (identity matrix part)
    # y_e corresponds to column m + idx(e)
    for i, e in enumerate(E):
        A[i, m + i] = 1.0
        
    # Bounds
    # All variables are binary: [0, 1]
    integrality = np.ones(m + k)
    
    # Constraints: A * vars = 1
    # LinearConstraint(A, lb=1, ub=1)
    constraints = [LinearConstraint(A, lb=np.ones(k), ub=np.ones(k))]
    
    # Solve
    if HAS_SCIPY_MILP:
        res = milp(c=c, constraints=constraints, integrality=integrality, 
                   bounds=Bounds(lb=0, ub=1))
        
        if res.status == 0: # Optimal
            x = res.x
            pieces = []
            # Extract cycles
            for i in range(m):
                if x[i] > 0.5:
                    pieces.append(candidate_cycles[i])
            # Extract singletons
            for i in range(k):
                if x[m + i] > 0.5:
                    e = E[i]
                    pieces.append([e[0], e[1]])
            return pieces
        else:
            # Fallback if ILP fails
            pass
            
    # Fallback: Greedy approach if ILP is not available or fails
    # Use the generated cycles greedily
    remaining = set(remaining_edges)
    pieces = []
    
    # Sort cycles by length descending to pack more edges per piece?
    # Or maybe just iterate.
    # Let's try to pick cycles that cover the most "hard" edges?
    # Simple greedy: pick any valid cycle, remove edges, repeat.
    
    # Re-sort candidates to prioritize longer cycles
    candidate_cycles.sort(key=lambda c: len(c), reverse=True)
    
    used_cycles = []
    for cycle in candidate_cycles:
        # Check if all edges of this cycle are still in remaining
        valid = True
        for j in range(len(cycle) - 1):
            u, v = cycle[j], cycle[j+1]
            e = (u, v) if u < v else (v, u)
            if e not in remaining:
                valid = False
                break
        if not valid:
            continue
        # Also check closing edge
        u, v = cycle[-1], cycle[0]
        e = (u, v) if u < v else (v, u)
        if e not in remaining:
            continue
            
        if valid:
            # Use this cycle
            pieces.append(cycle)
            used_cycles.append(cycle)
            # Remove edges
            for j in range(len(cycle) - 1):
                u, v = cycle[j], cycle[j+1]
                e = (u, v) if u < v else (v, u)
                remaining.discard(e)
            u, v = cycle[-1], cycle[0]
            e = (u, v) if u < v else (v, u)
            remaining.discard(e)
            
    # Add remaining edges as singletons
    for e in remaining:
        pieces.append([e[0], e[1]])
        
    return pieces

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    
    for n, p, seed in specs:
        if time.time() > deadline:
            # Fallback: just return all edges as singletons
            edges = instance(n, p, seed)
            results.append([list(ed) for ed in edges])
            continue
            
        edges = instance(n, p, seed)
        
        # If ILP is available, use it. It should be fast for these sizes.
        # n=30, E~150. Variables ~ 150 + 500 = 650. Constraints 150.
        # This is very small for an ILP solver.
        
        pieces = solve_decomposition_ilp(edges, n, deadline)
        results.append(pieces)
        
    while len(results) < 3:
        results.append([])
        
    return results[:3]

if __name__ == "__main__":
    construct()
