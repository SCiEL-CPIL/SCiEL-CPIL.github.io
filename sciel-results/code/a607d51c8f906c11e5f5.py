import time
import random
import itertools
from collections import defaultdict

_DEADLINE = 140.0  # safety margin before 150s kill

def _all_k4s(n=16):
    """Generate all 4-subsets of {0,...,n-1}."""
    return list(itertools.combinations(range(n), 4))

def _edges_in_k4(k4):
    """Return the 6 edges (as frozensets or sorted tuples) within a K4."""
    a, b, c, d = k4
    return [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]

def _edge_index_map(n=16):
    """Map (i,j) with i<j to a linear index 0..119."""
    m = {}
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            m[(i,j)] = idx
            idx += 1
    return m

def construct():
    start_time = time.time()
    n = 16
    num_edges = 120
    k4s = _all_k4s(n)
    edge_map = _edge_index_map(n)
    
    # Precompute for each edge, which K4s it belongs to
    edge_to_k4s = defaultdict(list)
    for k4_idx, k4 in enumerate(k4s):
        for e in _edges_in_k4(k4):
            edge_to_k4s[edge_map[e]].append(k4_idx)
    
    # For each K4, store its 6 edge indices
    k4_edges = []
    for k4 in k4s:
        k4_edges.append([edge_map[e] for e in _edges_in_k4(k4)])
    
    # Try to find coloring with k colors for k in [13, 14, 15]
    for k_colors in [13, 14, 15]:
        if time.time() - start_time > _DEADLINE:
            break
        solution = _backtrack(n, num_edges, k4_edges, edge_to_k4s, k_colors, start_time)
        if solution is not None:
            # Verify
            if _verify(solution, k4_edges, n):
                return solution
    
    # Fallback: use the modular construction (guaranteed valid, uses ~16 colors)
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]

def _verify(colors, k4_edges, n=16):
    """Verify that every K4 has at least 5 distinct colors."""
    for edge_indices in k4_edges:
        cs = set(colors[e] for e in edge_indices)
        if len(cs) < 5:
            return False
    return True

def _backtrack(n, num_edges, k4_edges, edge_to_k4s, k_colors, start_time):
    """Backtracking with forward checking to find a valid k-coloring."""
    random.seed(42)
    
    # Order edges: prioritize edges that appear in many K4s (more constrained)
    edge_deg = [len(edge_to_k4s[e]) for e in range(num_edges)]
    edge_order = sorted(range(num_edges), key=lambda e: -edge_deg[e])
    
    # Reverse mapping: position in order -> edge index
    pos_to_edge = [edge_order[i] for i in range(num_edges)]
    edge_to_pos = [0] * num_edges
    for i, e in enumerate(edge_order):
        edge_to_pos[e] = i
    
    colors = [-1] * num_edges
    
    # For forward checking: for each K4, track color counts
    # k4_color_count[k4_idx][color] = count of edges in this K4 with that color
    k4_cc = [dict() for _ in range(len(k4_edges))]
    
    def is_feasible(k4_idx, new_color, edge_idx):
        """Check if adding new_color to edge_idx (in K4 k4_idx) keeps feasibility."""
        cc = k4_cc[k4_idx]
        cur_count = cc.get(new_color, 0)
        if cur_count >= 2:
            # Would make 3 edges of same color -> at most 4 distinct colors
            return False
        # Check if another color already has count 2
        if cur_count == 1:
            for c, cnt in cc.items():
                if c != new_color and cnt >= 2:
                    return False
        return True
    
    def assign(pos, edge_idx, color):
        colors[edge_idx] = color
        for k4_idx in edge_to_k4s[edge_idx]:
            cc = k4_cc[k4_idx]
            cc[color] = cc.get(color, 0) + 1
    
    def unassign(pos, edge_idx, color):
        colors[edge_idx] = -1
        for k4_idx in edge_to_k4s[edge_idx]:
            cc = k4_cc[k4_idx]
            cc[color] -= 1
            if cc[color] == 0:
                del cc[color]
    
    def backtrack(pos):
        if time.time() - start_time > _DEADLINE:
            return False
        if pos == num_edges:
            return True
        
        edge_idx = pos_to_edge[pos]
        
        # Try colors in random order for diversity
        color_list = list(range(k_colors))
        random.shuffle(color_list)
        
        for color in color_list:
            # Forward checking: verify this assignment doesn't immediately violate
            # any K4 constraint
            feasible = True
            for k4_idx in edge_to_k4s[edge_idx]:
                if not is_feasible(k4_idx, color, edge_idx):
                    feasible = False
                    break
            
            if not feasible:
                continue
            
            assign(pos, edge_idx, color)
            
            if backtrack(pos + 1):
                return True
            
            unassign(pos, edge_idx, color)
        
        return False
    
    # Try multiple restarts with different random seeds
    max_restarts = 50
    for restart in range(max_restarts):
        if time.time() - start_time > _DEADLINE:
            break
        random.seed(42 + restart)
        # Reset state
        for i in range(num_edges):
            colors[i] = -1
        for i in range(len(k4_edges)):
            k4_cc[i] = {}
        
        if backtrack(0):
            return colors[:]
    
    return None
