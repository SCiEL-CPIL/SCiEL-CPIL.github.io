"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
from itertools import combinations

# EVOLVE-BLOCK-START

def _enumerate_all_cycles(adj):
    """Enumerate all simple cycles in an undirected graph."""
    n = len(adj)
    cycles = []
    
    def dfs(start, current, visited, path):
        for neighbor in adj[current]:
            if neighbor == start and len(path) >= 3:
                cycles.append(list(path))
            elif neighbor > start and neighbor not in visited:
                visited.add(neighbor)
                path.append(neighbor)
                dfs(start, neighbor, visited, path)
                path.pop()
                visited.remove(neighbor)
    
    for start in range(n):
        dfs(start, start, {start}, [start])
    
    return cycles

def _get_cycle_lengths(adj):
    """Get distinct cycle lengths in the graph."""
    cycles = _enumerate_all_cycles(adj)
    lengths = set(len(c) for c in cycles)
    return lengths

def construct():
    """
    Constructs a 3-regular bipartite graph on vertices 0..11 where every
    simple cycle has length exactly 6.
    
    Uses a backtracking search to find a 6x6 (0,1)-matrix with:
    - Row sums = 3 (each point connects to 3 lines)
    - Column sums = 3 (each line connects to 3 points)
    - No two rows share more than 1 column (C4-free)
    
    If found, this gives a graph with only 6-cycles (score improvement).
    Fallback: two K_3,3 components (score 2.5).
    """
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # Fallback: two K_3,3 components
    fallback_edges = []
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            fallback_edges.append([u, v])
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            fallback_edges.append([u, v])
    
    # Search for a 6x6 matrix with the desired properties
    # rows[i] = set of columns (lines) connected to point i
    rows = []
    col_counts = [0] * 6
    
    def backtrack(row_idx):
        nonlocal rows, col_counts, start_time, deadline
        if time.time() > deadline:
            return False
        
        if row_idx == 6:
            # Check all column sums are 3
            for c in range(6):
                if col_counts[c] != 3:
                    return False
            # Verify C4-free: no two rows share more than 1 column
            for i in range(6):
                for j in range(i+1, 6):
                    if len(rows[i] & rows[j]) > 1:
                        return False
            # Build adjacency and check all cycle lengths
            adj = [[] for _ in range(12)]
            for i in range(6):
                for j in rows[i]:
                    adj[i].append(j + 6)
                    adj[j + 6].append(i)
            
            cycle_lengths = _get_cycle_lengths(adj)
            # We want all cycles to be length 6
            if cycle_lengths == {6}:
                return True
            return False
        
        # Try all combinations of 3 columns for this row
        for combo in combinations(range(6), 3):
            # Check column capacity
            valid = True
            for c in combo:
                if col_counts[c] >= 3:
                    valid = False
                    break
            if not valid:
                continue
            
            # Check C4 constraint with existing rows
            row_set = set(combo)
            valid = True
            for existing in rows:
                if len(row_set & existing) > 1:
                    valid = False
                    break
            if not valid:
                continue
            
            # Assign
            rows.append(row_set)
            for c in combo:
                col_counts[c] += 1
            
            if backtrack(row_idx + 1):
                return True
            
            # Unassign
            rows.pop()
            for c in combo:
                col_counts[c] -= 1
        
        return False
    
    found = backtrack(0)
    
    if found:
        # Build edges from the found matrix
        edges = []
        for i in range(6):
            for j in sorted(rows[i]):
                edges.append([i, j + 6])  # Points 0-5, Lines 6-11
        
        # Ensure types
        final_edges = []
        for e in edges:
            u = int(e[0])
            v = int(e[1])
            final_edges.append([u, v])
        
        return final_edges
    else:
        # Use fallback
        final_edges = []
        for e in fallback_edges:
            u = int(e[0])
            v = int(e[1])
            final_edges.append([u, v])
        return final_edges

# EVOLVE-BLOCK-END
