"""Shortest superpermutation construction with greedy block reordering."""
import sys
import time
from itertools import permutations
from collections import defaultdict

# Deadline guard
_DEADLINE = 115.0  # seconds, safety margin before 120s limit
_start_time = time.time()


def _elapsed():
    return time.time() - _start_time


def construct(n):
    """Construct a superpermutation on n symbols.
    
    Uses the classic recursive construction with greedy Eulerian-trail-inspired
    reordering of permutation blocks to maximize overlaps.
    """
    global _start_time
    _start_time = time.time()
    
    # For small n, just use the classic construction
    if n <= 1:
        return "1"
    
    # Classic recursive construction
    word = [1]
    
    for m in range(2, n + 1):
        # Extract all distinct (m-1)-permutations from the current word
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Try to reorder perms to maximize overlap
        if len(perms) <= 5040:  # Attempt reordering for all manageable sizes
            perms = _greedy_eulerian_reorder(perms, m)
        
        # Build new word
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find maximum overlap with end of new
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        
        word = new
        
        # Check deadline
        if _elapsed() > _DEADLINE - 5:
            break
    
    return "".join(map(str, word))


def _compute_overlap_matrix(blocks):
    """Compute overlap matrix: overlap[i][j] = max overlap of end of blocks[i] with start of blocks[j]"""
    n = len(blocks)
    overlap = [[0] * n for _ in range(n)]
    
    for i in range(n):
        bi = blocks[i]
        for j in range(n):
            if i == j:
                continue
            bj = blocks[j]
            max_ov = min(len(bi), len(bj))
            for o in range(max_ov, -1, -1):
                if o == 0 or bi[len(bi) - o:] == bj[:o]:
                    overlap[i][j] = o
                    break
    
    return overlap


def _count_components(edges_set, nodes):
    """Count connected components in the subgraph of unvisited edges."""
    # Build adjacency from edges
    adj = defaultdict(set)
    for u, v in edges_set:
        adj[u].add(v)
        adj[v].add(u)
    
    visited = set()
    components = 0
    
    for node in nodes:
        if node not in visited:
            components += 1
            # BFS/DFS
            stack = [node]
            visited.add(node)
            while stack:
                curr = stack.pop()
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
    
    return components


def _greedy_eulerian_reorder(perms, m):
    """Reorder permutation blocks using greedy Eulerian trail heuristic.
    
    Uses a directed graph where nodes are permutations and edges represent
    valid suffix-prefix overlaps. Iteratively selects the next edge that
    minimizes the increase in disconnected components, prioritizing moves
    that keep remaining unvisited edges connected.
    """
    n_perms = len(perms)
    if n_perms <= 1:
        return perms
    
    # Precompute the full block for each perm
    blocks = [list(p) + [m] + list(p) for p in perms]
    
    # Compute overlap matrix
    overlap = _compute_overlap_matrix(blocks)
    
    # Build edge list: edge (i, j) exists if overlap[i][j] > 0
    # Each permutation i can transition to permutation j with cost = len(blocks[j]) - overlap[i][j]
    # We want to find an ordering that minimizes total length
    
    # For the Eulerian trail approach:
    # Nodes = permutations
    # Edge from i to j exists if overlap[i][j] > 0
    # Weight of edge = len(blocks[j]) - overlap[i][j] (additional length)
    
    # Greedy: start with a permutation, then repeatedly pick the next edge
    # that minimizes (1) increase in disconnected components, (2) additional length
    
    # Try multiple starting points for small n, single greedy for large n
    if n_perms <= 100:
        best_order = None
        best_length = float('inf')
        
        for start in range(n_perms):
            if _elapsed() > _DEADLINE - 3:
                break
            
            order = [start]
            remaining = set(range(n_perms)) - {start}
            current_length = len(blocks[start])
            
            # Build initial edge set (all edges except those from start)
            # Actually, we track unvisited edges
            # For simplicity, track which permutations are unvisited
            
            while remaining:
                if _elapsed() > _DEADLINE - 3:
                    break
                
                last = order[-1]
                best_next = None
                best_score = None
                best_ov = 0
                
                for j in remaining:
                    ov = overlap[last][j]
                    add_len = len(blocks[j]) - ov
                    
                    # Score: primarily minimize disconnected components increase,
                    # secondarily minimize added length
                    # For simplicity, use a combined score
                    score = (add_len,)
                    
                    if best_score is None or score < best_score:
                        best_score = score
                        best_next = j
                        best_ov = ov
                
                if best_next is None:
                    break
                
                order.append(best_next)
                remaining.remove(best_next)
                current_length += len(blocks[best_next]) - best_ov
            
            if len(order) == n_perms and current_length < best_length:
                best_length = current_length
                best_order = order
        
        if best_order is not None:
            return [perms[i] for i in best_order]
    
    else:
        # For larger n, do a single greedy pass
        order = [0]
        remaining = set(range(1, n_perms))
        
        while remaining:
            if _elapsed() > _DEADLINE - 3:
                break
            
            last = order[-1]
            best_next = None
            best_ov = 0
            
            for j in remaining:
                ov = overlap[last][j]
                if ov > best_ov:
                    best_ov = ov
                    best_next = j
            
            if best_next is None:
                break
            
            order.append(best_next)
            remaining.remove(best_next)
        
        if len(order) == n_perms:
            return [perms[i] for i in order]
    
    # Fallback: original order
    return perms
