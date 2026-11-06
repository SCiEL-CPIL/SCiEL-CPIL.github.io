"""Improved program: construct a 5-colouring of K_n with no monochromatic C4."""
import time
import random

def _index(i, j, n):
    """Lexicographic index of pair (i,j) with i<j in the flat list."""
    return i * (2 * n - i - 1) // 2 + (j - i - 1)

def _build_adj(colors, n):
    """Build adjacency bitsets for each color. adj[c][v] is a bitset of neighbors of v in color c."""
    adj = [[0] * n for _ in range(5)]
    for i in range(n):
        for j in range(i + 1, n):
            idx = _index(i, j, n)
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
    return adj

def _count_c4s(adj, n):
    """Count total number of C4 violations across all colors."""
    count = 0
    for c in range(5):
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[c][u] & adj[c][v]
                if common & (common - 1):
                    count += 1
    return count

def _find_violation(adj, n):
    """Find one C4 violation. Returns (c, u, v, n1, n2) or None."""
    for c in range(5):
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[c][u] & adj[c][v]
                if common & (common - 1):
                    # Get two common neighbors
                    bits = common
                    b1 = bits & (-bits)
                    bits ^= b1
                    b2 = bits & (-bits)
                    n1 = b1.bit_length() - 1
                    n2 = b2.bit_length() - 1
                    return (c, u, v, n1, n2)
    return None

def _is_valid(colors, n):
    """Check if coloring is valid (no monochromatic C4)."""
    adj = _build_adj(colors, n)
    return _count_c4s(adj, n) == 0

def construct():
    deadline = time.time() + 130  # safety margin before 150s kill
    random.seed(42)
    
    # Start with known 11-vertex construction
    best_n = 11
    best_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]
    
    # Try to find larger valid colorings via local search
    # Start from higher n and work down, as larger n is better
    for target_n in range(20, 11, -1):
        if time.time() > deadline - 15:
            break
        
        num_edges = target_n * (target_n - 1) // 2
        found = False
        
        # Multiple attempts with different random seeds
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            # Random initial coloring
            colors = [random.randint(0, 4) for _ in range(num_edges)]
            adj = _build_adj(colors, target_n)
            current_violations = _count_c4s(adj, target_n)
            
            # Local search: iteratively reduce violations
            max_iters = 20000
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                
                if current_violations == 0:
                    found = True
                    break
                
                # Find a violation
                viol = _find_violation(adj, target_n)
                if viol is None:
                    found = True
                    break
                
                c, u, v, n1, n2 = viol
                
                # Try recoloring one of the edges in the C4
                # The C4 is u - n1 - v - n2 - u (in color c)
                # We can try recoloring (u, n1), (v, n1), (u, n2), or (v, n2)
                candidate_edges = [(u, n1), (v, n1), (u, n2), (v, n2)]
                random.shuffle(candidate_edges)
                
                improved = False
                for (a, b) in candidate_edges:
                    if a > b:
                        a, b = b, a
                    
                    idx = _index(a, b, target_n)
                    old_c = colors[idx]
                    
                    # Try each new color
                    best_new_c = -1
                    best_count = current_violations
                    
                    for new_c in range(5):
                        if new_c == old_c:
                            continue
                        
                        # Simulate the change
                        adj[old_c][a] &= ~(1 << b)
                        adj[old_c][b] &= ~(1 << a)
                        adj[new_c][a] |= (1 << b)
                        adj[new_c][b] |= (1 << a)
                        
                        new_count = _count_c4s(adj, target_n)
                        
                        # Revert
                        adj[new_c][a] &= ~(1 << b)
                        adj[new_c][b] &= ~(1 << a)
                        adj[old_c][a] |= (1 << b)
                        adj[old_c][b] |= (1 << a)
                        
                        if new_count < best_count:
                            best_count = new_count
                            best_new_c = new_c
                    
                    if best_new_c != -1:
                        # Apply the best change
                        adj[old_c][a] &= ~(1 << b)
                        adj[old_c][b] &= ~(1 << a)
                        adj[best_new_c][a] |= (1 << b)
                        adj[best_new_c][b] |= (1 << a)
                        colors[idx] = best_new_c
                        current_violations = best_count
                        improved = True
                        break
                
                if not improved:
                    # No improvement found, try a random move to escape local minimum
                    idx = random.randint(0, num_edges - 1)
                    # Find which edge this is
                    for i in range(target_n):
                        for j in range(i + 1, target_n):
                            if _index(i, j, target_n) == idx:
                                old_c = colors[idx]
                                new_c = random.randint(0, 4)
                                if new_c != old_c:
                                    adj[old_c][i] &= ~(1 << j)
                                    adj[old_c][j] &= ~(1 << i)
                                    adj[new_c][i] |= (1 << j)
                                    adj[new_c][j] |= (1 << i)
                                    colors[idx] = new_c
                                    current_violations = _count_c4s(adj, target_n)
                                break
                        else:
                            continue
                        break
            
            if found:
                # Verify validity
                if _is_valid(colors, target_n):
                    best_n = target_n
                    best_colors = colors[:]
                    break
        
        if found:
            break
    
    return {'n': best_n, 'colors': best_colors}
