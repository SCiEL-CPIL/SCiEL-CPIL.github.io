"""Improved program for edge-colourings of K16 with every K4 seeing five colours."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _edge_index(i, j):
    """Return linear index for edge (i,j) in lexicographic order."""
    return i * 15 - i * (i - 1) // 2 + (j - i - 1)

def _build_k4_edges():
    """Precompute all K4 edge index sets and incidence."""
    k4_edges = []
    k4s_for_edge = [[] for _ in range(120)]
    for idx, quad in enumerate(combinations(range(16), 4)):
        a, b, c, d = quad
        edges = (
            _edge_index(a, b), _edge_index(a, c), _edge_index(a, d),
            _edge_index(b, c), _edge_index(b, d), _edge_index(c, d)
        )
        k4_edges.append(edges)
        for e in edges:
            k4s_for_edge[e].append(idx)
    return k4_edges, k4s_for_edge

def _k4_is_valid(k4_idx, coloring, k4_edges):
    """Check if a single K4 has at least 5 distinct colours."""
    edges = k4_edges[k4_idx]
    c0 = coloring[edges[0]]
    c1 = coloring[edges[1]]
    c2 = coloring[edges[2]]
    c3 = coloring[edges[3]]
    c4 = coloring[edges[4]]
    c5 = coloring[edges[5]]
    
    # Use a set for simplicity in this helper, performance is dominated by call frequency
    # But since we only call this for affected K4s, it's fast enough.
    return len({c0, c1, c2, c3, c4, c5}) >= 5

def _get_violations(coloring, k4_edges):
    """Return set of indices of K4s that violate the constraint."""
    violations = set()
    for idx in range(len(k4_edges)):
        if not _k4_is_valid(idx, coloring, k4_edges):
            violations.add(idx)
    return violations

def _count_colors(coloring):
    return len(set(coloring))

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    k4_edges, k4s_for_edge = _build_k4_edges()
    num_k4s = len(k4_edges)
    
    # Initial coloring: (i+j) mod 17
    best = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    best_colors = _count_colors(best)
    best_violations = _get_violations(best, k4_edges)
    
    # If initial is invalid, fallback (should not happen for mod 17)
    if best_violations:
        best = list(range(120))
        best_colors = 120
        best_violations = _get_violations(best, k4_edges)
        
    current = best[:]
    current_colors = best_colors
    current_violations = best_violations.copy()
    
    # Helper to check if a move is valid and update violations incrementally
    def try_move(coloring, violations, edge_idx, new_color):
        """
        Attempts to recolor edge_idx to new_color.
        Returns (new_coloring, new_violations) if valid, else (None, None).
        """
        if coloring[edge_idx] == new_color:
            return None, None
            
        old_color = coloring[edge_idx]
        coloring[edge_idx] = new_color
        
        affected_k4s = k4s_for_edge[edge_idx]
        new_violations = set()
        
        # Check only affected K4s
        for k4_idx in affected_k4s:
            if not _k4_is_valid(k4_idx, coloring, k4_edges):
                new_violations.add(k4_idx)
        
        # The new total violations are:
        # (old violations not in affected) U (new violations in affected)
        # Since we only care if the result is valid (0 violations),
        # we can just check if new_violations is empty AND no old violations remain?
        # No, old violations outside affected_k4s remain.
        # So if current_violations was not empty, and we don't fix all of them, it's invalid.
        # But we are looking for a valid state (0 violations).
        # So we only accept if new_violations is empty AND (current_violations - set(affected_k4s)) is empty.
        
        # Actually, simpler:
        # If current_violations is not empty, a single edge change might not fix all violations.
        # We only want to accept moves that result in a VALID coloring (0 violations).
        
        if new_violations:
            coloring[edge_idx] = old_color
            return None, None
            
        # Check if any old violations remain outside affected K4s
        # If there were violations not in affected_k4s, they still exist.
        remaining_old_violations = current_violations.difference(set(affected_k4s))
        if remaining_old_violations:
            coloring[edge_idx] = old_color
            return None, None
            
        # If we reach here, the move is valid (0 violations)
        return coloring, set()

    # Phase 1: Systematic color merging
    # We use a simpler check for merging: try the merge, then check validity.
    # Merging changes many edges, so incremental check is harder to set up quickly.
    # But we can optimize _is_valid for the merge case by checking only K4s affected by the merged edges.
    
    def is_valid_merge(coloring, target, other, k4_edges, k4s_for_edge):
        """Check if merging 'target' to 'other' in 'coloring' results in valid coloring."""
        # Find all edges with target color
        affected_edges = [i for i, c in enumerate(coloring) if c == target]
        if not affected_edges:
            return True
            
        # Union of K4s affected by these edges
        affected_k4s = set()
        for e in affected_edges:
            affected_k4s.update(k4s_for_edge[e])
            
        # Temporarily apply merge
        for e in affected_edges:
            coloring[e] = other
            
        # Check affected K4s
        for k4_idx in affected_k4s:
            if not _k4_is_valid(k4_idx, coloring, k4_edges):
                # Revert
                for e in affected_edges:
                    coloring[e] = target
                return False
                
        # Revert
        for e in affected_edges:
            coloring[e] = target
        return True

    # Phase 1: Merge colors
    for iteration in range(100):
        if time.time() > deadline:
            break
        
        used_colors = sorted(set(current))
        if len(used_colors) <= current_colors:
            break
            
        freq = {}
        for c in current:
            freq[c] = freq.get(c, 0) + 1
        sorted_colors = sorted(used_colors, key=lambda c: freq.get(c, 0))
        
        improved = False
        for target in sorted_colors:
            if time.time() > deadline:
                break
            if target not in freq or freq[target] == 0:
                continue
                
            for other in sorted_colors:
                if other == target or other not in freq or freq[other] == 0:
                    continue
                
                if is_valid_merge(current, target, other, k4_edges, k4s_for_edge):
                    # Apply merge
                    for i in range(len(current)):
                        if current[i] == target:
                            current[i] = other
                    current_violations = set() # Merges are checked to be valid
                    new_count = _count_colors(current)
                    if new_count < current_colors:
                        current_colors = new_count
                        best = current[:]
                        best_colors = new_count
                        improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    # Phase 2: Local search with incremental validity
    # We are now at a state with 0 violations (hopefully).
    # We want to reduce color count by recoloring single edges.
    
    random.seed(42)
    
    for iteration in range(1000):
        if time.time() > deadline:
            break
        
        used_colors = sorted(set(current))
        if len(used_colors) <= current_colors:
            break
            
        freq = {}
        for c in current:
            freq[c] = freq.get(c, 0) + 1
        sorted_colors = sorted(used_colors, key=lambda c: freq.get(c, 0))
        
        improved = False
        # Try to eliminate the rarest color
        target = sorted_colors[0]
        if freq.get(target, 0) == 0:
            continue
            
        edges_with_target = [i for i, c in enumerate(current) if c == target]
        random.shuffle(edges_with_target)
        
        for ei in edges_with_target:
            if time.time() > deadline:
                break
                
            for other in sorted_colors:
                if other == target:
                    continue
                
                # Try to recolor ei to other
                new_coloring, new_violations = try_move(current, current_violations, ei, other)
                if new_coloring is not None:
                    # Move is valid
                    current = new_coloring
                    current_violations = new_violations
                    new_count = _count_colors(current)
                    if new_count < current_colors:
                        current_colors = new_count
                        best = current[:]
                        best_colors = new_count
                        improved = True
                        break
            
            if improved:
                break
        
        if not improved:
            # Try a random perturbation to escape local minimum
            # Pick a random edge and recolor it to a random color, allowing invalid states?
            # No, we must maintain validity.
            # So we try random valid moves that don't reduce color count but might enable it.
            found_move = False
            for _ in range(10):
                ei = random.randint(0, 119)
                old_c = current[ei]
                new_c = random.choice(used_colors)
                if new_c == old_c:
                    continue
                new_coloring, new_violations = try_move(current, current_violations, ei, new_c)
                if new_coloring is not None:
                    current = new_coloring
                    current_violations = new_violations
                    found_move = True
                    break
            
            if not found_move:
                break
    
    return best
# EVOLVE-BLOCK-END
