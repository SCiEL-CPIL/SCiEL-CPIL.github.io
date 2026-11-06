import time
import random
import sys

def _check_c4_fast(colors, n, c):
    """Check if colour class c contains a C4 using bitsets."""
    adj = [0] * n
    # Build adjacency bitsets for colour c
    # Index mapping: i < j -> idx = i * (2*n - i - 1) // 2 + (j - i - 1)
    for i in range(n):
        base_idx = i * (2 * n - i - 1) // 2
        for j in range(i + 1, n):
            idx = base_idx + (j - i - 1)
            if colors[idx] == c:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Check for C4: for each pair (u, v), common neighbours >= 2
    for u in range(n):
        for v in range(u + 1, n):
            common = adj[u] & adj[v]
            if common & (common - 1):
                return True
    return False

def _find_violation(colors, n, c):
    """Find a specific C4 in colour c. Returns (u, v, n1, n2) or None."""
    adj = [0] * n
    for i in range(n):
        base_idx = i * (2 * n - i - 1) // 2
        for j in range(i + 1, n):
            idx = base_idx + (j - i - 1)
            if colors[idx] == c:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    for u in range(n):
        for v in range(u + 1, n):
            common = adj[u] & adj[v]
            if common & (common - 1):
                # Extract two bits
                bits = common
                b1 = bits & (-bits)
                bits ^= b1
                b2 = bits & (-bits)
                n1 = b1.bit_length() - 1
                n2 = b2.bit_length() - 1
                return (u, v, n1, n2)
    return None

def _get_index(i, j, n):
    if i > j: i, j = j, i
    return i * (2 * n - i - 1) // 2 + (j - i - 1)

def _is_valid_fast(colors, n):
    for c in range(5):
        if _check_c4_fast(colors, n, c):
            return False
    return True

def construct():
    deadline = time.time() + 140  # Safety margin
    
    # Start with known valid n=11
    best_n = 11
    best_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]
    
    # Try to extend to n=12, 13, ..., 26
    # We will try multiple random restarts for each n
    max_n_target = 26
    
    for target_n in range(12, max_n_target + 1):
        if time.time() > deadline - 15:
            break
        
        found = False
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() > deadline - 15:
                break
            
            # Initialize colors
            # Use a heuristic: start with random, or extend previous if possible
            if target_n == 12:
                # Extend n=11 by adding vertex 11
                # We need to add 11 edges from 11 to 0..10
                # And keep existing edges
                colors = best_colors[:]
                # Add edges for new vertex
                for i in range(11):
                    colors.append(random.randint(0, 4))
            else:
                # Random initialization
                colors = [random.randint(0, 4) for _ in range(target_n * (target_n - 1) // 2)]
            
            # Local Search
            iters = 0
            max_iters = 2000
            stuck = False
            while iters < max_iters:
                if time.time() > deadline - 10:
                    stuck = True
                    break
                iters += 1
                
                # Check validity
                if _is_valid_fast(colors, target_n):
                    best_n = target_n
                    best_colors = colors[:]
                    found = True
                    break
                
                # Find a violation
                violation = None
                for c in range(5):
                    v = _find_violation(colors, target_n, c)
                    if v:
                        violation = (c, v)
                        break
                
                if not violation:
                    # Should be valid
                    if _is_valid_fast(colors, target_n):
                        best_n = target_n
                        best_colors = colors[:]
                        found = True
                    break
                
                c, (u, v, n1, n2) = violation
                # Recolor one of the edges in the C4
                # Edges are (u, n1), (u, n2), (v, n1), (v, n2)
                # Actually the C4 is u-n1-v-n2-u? No, common neighbours of u,v are n1,n2.
                # So edges (u,n1), (v,n1), (u,n2), (v,n2) are all colour c.
                # The cycle is u-n1-v-n2-u.
                # We can recolor one of these edges to break the cycle.
                candidates = []
                for a, b in [(u, n1), (v, n1), (u, n2), (v, n2)]:
                    if a < b:
                        candidates.append((a, b))
                    else:
                        candidates.append((b, a))
                
                random.shuffle(candidates)
                improved = False
                for a, b in candidates:
                    idx = _get_index(a, b, target_n)
                    old_c = colors[idx]
                    # Try other colors
                    new_c = random.randint(0, 4)
                    if new_c != old_c:
                        colors[idx] = new_c
                        # Check if new color creates a C4
                        if not _check_c4_fast(colors, target_n, new_c):
                            improved = True
                            break
                        else:
                            colors[idx] = old_c
                
                if not improved:
                    # Try harder: random recolor of a random edge in the C4
                    a, b = random.choice(candidates)
                    idx = _get_index(a, b, target_n)
                    colors[idx] = random.randint(0, 4)
                    
                # If we didn't improve, we might be stuck. 
                # To avoid infinite loops, we can accept worse solutions occasionally (simulated annealing)
                # but for simplicity, we just continue.
            
            if found:
                break
            
    return {'n': best_n, 'colors': best_colors}
