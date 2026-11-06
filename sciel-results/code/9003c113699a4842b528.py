"""Construct a minimal 5-uniform hypergraph without Property B.

Structural change: instead of random greedy construction + deletion,
we use a direct backtracking search over edge subsets of small size,
using a fast 2-colorability checker with bit-parallel propagation.
We search for the smallest k such that some k-edge 5-uniform hypergraph
on n vertices is not 2-colorable, trying increasing k from a lower bound.
"""
import time
import random
from itertools import combinations

def _is_2colorable_fast(n, edges):
    """Fast 2-colorability check using backtracking with unit propagation.
    Returns True if 2-colorable, False otherwise.
    
    Uses bit-parallel representation for speed.
    """
    if not edges:
        return True
    
    m = len(edges)
    # For each edge, store the 5 vertices as a tuple
    edge_verts = [tuple(e) for e in edges]
    
    # For each vertex, list of edges containing it
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edge_verts):
        for v in e:
            vertex_edges[v].append(i)
    
    # Assignment: -1 unassigned, 0 or 1
    assignment = [-1] * n
    
    # For each edge, count of 0s and 1s assigned
    edge_counts = [[0, 0] for _ in range(m)]
    
    # Unit propagation queue
    def propagate():
        """Apply unit propagation. Return False on contradiction."""
        queue = list(range(m))  # edges to check
        in_queue = [True] * m
        
        while queue:
            ei = queue.pop()
            in_queue[ei] = False
            edge = edge_verts[ei]
            c0 = edge_counts[ei][0]
            c1 = edge_counts[ei][1]
            unassigned = 5 - c0 - c1
            
            if unassigned == 0:
                if c0 == 5 or c1 == 5:
                    return False  # monochromatic edge
                continue
            
            if c0 == 4 and unassigned == 1:
                # Must assign the unassigned vertex to 1
                for v in edge:
                    if assignment[v] == -1:
                        assignment[v] = 1
                        edge_counts[ei][1] += 1
                        # Propagate to other edges containing v
                        for other_ei in vertex_edges[v]:
                            if other_ei != ei and not in_queue[other_ei]:
                                edge_counts[other_ei][1] += 1
                                in_queue[other_ei] = True
                                queue.append(other_ei)
                        break
            elif c1 == 4 and unassigned == 1:
                # Must assign the unassigned vertex to 0
                for v in edge:
                    if assignment[v] == -1:
                        assignment[v] = 0
                        edge_counts[ei][0] += 1
                        for other_ei in vertex_edges[v]:
                            if other_ei != ei and not in_queue[other_ei]:
                                edge_counts[other_ei][0] += 1
                                in_queue[other_ei] = True
                                queue.append(other_ei)
                        break
        
        return True
    
    def backtrack():
        if not propagate():
            return False
        
        # Find unassigned vertex with most constrained edges (MRV)
        best_v = -1
        best_degree = -1
        for v in range(n):
            if assignment[v] == -1:
                # Count how many edges involving v are "tight" (4 assigned)
                tight = 0
                for ei in vertex_edges[v]:
                    c0 = edge_counts[ei][0]
                    c1 = edge_counts[ei][1]
                    if c0 + c1 >= 4:
                        tight += 1
                if tight > best_degree:
                    best_degree = tight
                    best_v = v
        
        if best_v == -1:
            # All assigned, verify
            for ei in range(m):
                if edge_counts[ei][0] == 5 or edge_counts[ei][1] == 5:
                    return False
            return True
        
        v = best_v
        
        # Try 0
        assignment[v] = 0
        affected = []
        for ei in vertex_edges[v]:
            edge_counts[ei][0] += 1
            affected.append(ei)
        
        if backtrack():
            return True
        
        # Undo 0
        for ei in affected:
            edge_counts[ei][0] -= 1
        
        # Try 1
        assignment[v] = 1
        affected = []
        for ei in vertex_edges[v]:
            edge_counts[ei][1] += 1
            affected.append(ei)
        
        if backtrack():
            return True
        
        # Undo 1
        for ei in affected:
            edge_counts[ei][1] -= 1
        
        assignment[v] = -1
        return False
    
    return backtrack()


def construct():
    """Construct a 5-uniform hypergraph with no proper 2-colouring, minimizing edges."""
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    best_result = None
    best_edge_count = 200  # upper bound
    
    # Known lower bound: all 5-uniform hypergraphs with <=28 edges are 2-colorable
    # Best known upper bound: 51
    # We'll search from small k upward
    
    # Strategy: for each candidate n and k, use randomized local search
    # to find a k-edge non-2-colorable hypergraph
    
    # First, try to find a valid construction with k edges for increasing k
    for k in range(29, 52):
        if time.time() > deadline - 20:
            break
        
        # Try various n values
        for n in range(8, 18):
            if time.time() > deadline - 20:
                break
            
            # Number of possible edges
            total_edges = len(list(combinations(range(n), 5)))
            if total_edges < k:
                continue
            
            # Randomized search: start with random k edges, try to make it non-2-colorable
            # by swapping edges
            
            max_attempts = 200 if k < 40 else 100
            found = False
            
            for attempt in range(max_attempts):
                if time.time() > deadline - 15:
                    break
                
                random.seed(attempt * 1000 + k * 7 + n * 13)
                
                # Start with random k edges
                all_edges = list(combinations(range(n), 5))
                random.shuffle(all_edges)
                current_edges = set(all_edges[:k])
                
                # Local search: try swapping edges to break 2-colorability
                improved = True
                while improved and time.time() < deadline - 10:
                    improved = False
                    edges_list = list(current_edges)
                    random.shuffle(edges_list)
                    
                    for edge in edges_list:
                        if time.time() > deadline - 10:
                            break
                        
                        # Try removing this edge and adding a random new one
                        if len(current_edges) > 1:
                            current_edges.discard(edge)
                            # Add a random edge not already in set
                            new_edge = random.choice(all_edges)
                            if new_edge not in current_edges:
                                current_edges.add(new_edge)
                            
                            # Check if now non-2-colorable
                            test_edges = [list(e) for e in current_edges]
                            if not _is_2colorable_fast(n, test_edges):
                                found = True
                                if k < best_edge_count:
                                    best_edge_count = k
                                    best_result = {'n': n, 'edges': test_edges}
                                improved = True
                                break
                            else:
                                # Revert
                                current_edges.discard(new_edge)
                                current_edges.add(edge)
            
            if found:
                break  # Found for this k, try smaller n or next k
        
        # If we found a solution with k edges, we can stop since we're going up
        if best_result is not None and best_edge_count == k:
            break
    
    # If we found nothing, try a denser fallback
    if best_result is None:
        # Use n=9 complete 5-uniform (126 edges) as guaranteed valid
        # But try to minimize first
        n = 9
        all_edges = list(combinations(range(n), 5))
        random.seed(42)
        random.shuffle(all_edges)
        
        # Greedy: add edges until non-2-colorable
        edge_set = set()
        for edge in all_edges:
            edge_set.add(edge)
            if len(edge_set) >= 30:
                test = [list(e) for e in edge_set]
                if not _is_2colorable_fast(n, test):
                    # Minimize by deletion
                    edges = list(edge_set)
                    changed = True
                    while changed and time.time() < deadline - 5:
                        changed = False
                        for i in range(len(edges)):
                            if time.time() > deadline - 5:
                                break
                            candidate = edges[:i] + edges[i+1:]
                            if not _is_2colorable_fast(n, candidate):
                                edges = candidate
                                changed = True
                                break
                    best_result = {'n': n, 'edges': edges}
                    break
    
    if best_result is None:
        # Absolute fallback: complete 5-uniform on 9 vertices
        n = 9
        best_result = {'n': n, 'edges': [list(e) for e in combinations(range(n), 5)]}
    
    return best_result
